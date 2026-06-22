"""ComfyUI client — image-to-video via LTX-Video 2B on local Comfy instance.

Flow: upload mascot image → submit i2v workflow → poll history → download
output webp → ffmpeg convert to mp4.

Default endpoint: http://localhost:8188 (override via env COMFY_URL).
LTX-Video 2B v0.9.5 must be present at
models/checkpoints/ltx-video-2b-v0.9.5.safetensors. ungated, ~6GB.

LTX accepts a text prompt → real motion controlled by mascot_action_en.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import httpx


COMFY_URL = os.getenv("COMFY_URL", "http://localhost:8188").rstrip("/")
COMFY_TIMEOUT_SEC = float(os.getenv("COMFY_TIMEOUT_SEC", "600"))
WORKFLOW_PATH = Path(__file__).parent.parent / "comfy_workflows" / "ltx_i2v.json"

# LTX latent has 8x temporal compression — total frame count must satisfy 8n+1.
def _round_to_ltx_frames(target_frames: int, min_f: int = 25, max_f: int = 257) -> int:
    f = max(min_f, min(max_f, target_frames))
    return ((f - 1) // 8) * 8 + 1


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


async def _upload_image(client: httpx.AsyncClient, image_path: Path) -> str:
    """POST /upload/image — returns the filename Comfy assigned (may rename on collision)."""
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {"overwrite": "true"}
        r = await client.post(f"{COMFY_URL}/upload/image", files=files, data=data)
    r.raise_for_status()
    j = r.json()
    return j.get("name") or image_path.name


def _build_workflow(image_filename: str, *, seed: int, video_frames: int,
                    fps: int, width: int, height: int,
                    positive_prompt: str, filename_prefix: str) -> dict:
    """Load ltx_i2v.json template and inject runtime params."""
    with WORKFLOW_PATH.open() as f:
        wf = json.load(f)
    wf.pop("_meta", None)

    wf["2"]["inputs"]["image"] = image_filename
    wf["3"]["inputs"]["text"] = positive_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["length"] = video_frames
    wf["6"]["inputs"]["frame_rate"] = fps
    wf["10"]["inputs"]["noise_seed"] = seed
    wf["12"]["inputs"]["fps"] = float(fps)
    wf["14"]["inputs"]["filename_prefix"] = filename_prefix
    return wf


async def _submit(client: httpx.AsyncClient, workflow: dict) -> str:
    """POST /prompt — returns prompt_id."""
    r = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    if r.status_code >= 400:
        raise RuntimeError(f"Comfy /prompt failed {r.status_code}: {r.text[:500]}")
    j = r.json()
    pid = j.get("prompt_id")
    if not pid:
        raise RuntimeError(f"Comfy returned no prompt_id: {j}")
    return pid


async def _wait_for_done(client: httpx.AsyncClient, prompt_id: str,
                         timeout: float = COMFY_TIMEOUT_SEC) -> dict:
    """Poll /history/{prompt_id} until present (= completed) or timeout."""
    deadline = time.monotonic() + timeout
    delay = 1.0
    while time.monotonic() < deadline:
        r = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        if r.status_code == 200:
            data = r.json()
            entry = data.get(prompt_id)
            if entry:
                status = (entry.get("status") or {}).get("status_str")
                if status == "error":
                    msgs = (entry.get("status") or {}).get("messages") or []
                    raise RuntimeError(f"Comfy execution error: {msgs}")
                outputs = entry.get("outputs") or {}
                if outputs:
                    return entry
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 5.0)
    raise TimeoutError(f"Comfy prompt {prompt_id} did not complete in {timeout}s")


def _extract_output_files(entry: dict) -> list[dict]:
    """history entry → flat list of output {filename, subfolder, type}.

    SaveVideo emits {"videos": [{"filename": ...}]}; older save nodes use
    "images" or "gifs". We prefer videos when available.
    """
    videos: list[dict] = []
    others: list[dict] = []
    for _node_id, node_out in (entry.get("outputs") or {}).items():
        for key in ("videos", "images", "gifs"):
            for item in node_out.get(key, []) or []:
                if not item.get("filename"):
                    continue
                row = {
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                }
                (videos if key == "videos" else others).append(row)
    return videos or others


async def _download(client: httpx.AsyncClient, file: dict, out_path: Path) -> None:
    params = {
        "filename": file["filename"],
        "subfolder": file.get("subfolder", ""),
        "type": file.get("type", "output"),
    }
    r = await client.get(f"{COMFY_URL}/view", params=params)
    r.raise_for_status()
    out_path.write_bytes(r.content)


async def _normalize_mp4(src_path: Path, dst_path: Path, fps: int,
                         target_w: int = 1080, target_h: int = 1920,
                         target_fps: int = 30) -> None:
    """Re-encode Comfy mp4 (480x832 @ 25fps) → 1080x1920 @ 30fps yuv420p h264.
    Output fps is forced to 30 to match sprite/cut_segment so concat -c copy
    works without re-encoding.
    """
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1"
    )
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "18",
        "-r", str(target_fps),
        "-vf", vf,
        "-an",
        "-movflags", "+faststart",
        str(dst_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dst_path.exists():
        raise RuntimeError(f"ffmpeg normalize mp4 failed: {err.decode()[:300]}")


async def healthcheck(timeout: float = 5.0) -> bool:
    """Return True if ComfyUI is reachable."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{COMFY_URL}/system_stats")
            return r.status_code == 200
    except Exception:
        return False


async def i2v(image_path: Path, duration_sec: float, out_mp4: Path,
              *, action_prompt: str = "",
              fps: int = 25,
              width: int = 480, height: int = 832,
              seed: int | None = None) -> dict:
    """Run LTX-Video i2v — returns {prompt_id, frames, fps, duration_sec}.

    LTX-specific:
      - frame count must be 8n+1, in [25, 257] for 2B model
      - 25fps is native; common widths/heights divisible by 32
      - action_prompt drives motion (e.g. "praying with hands together,
        hopeful expression, looking up at sky")
    """
    image_path = Path(image_path)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    target_frames = max(25, round(duration_sec * fps))
    frames = _round_to_ltx_frames(target_frames)
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    prefix = f"banbaji_{int(time.time())}_{seed}"

    # Mascot prompt prefix anchors character — keeps generation focused on
    # the mascot performing the action rather than drifting to random subjects.
    full_prompt = (
        f"a 2D mascot character, {action_prompt}, smooth animation, "
        f"clear focus, vibrant colors, dynamic pose"
    ) if action_prompt else "a 2D mascot character, idle animation, smooth motion"

    async with httpx.AsyncClient(timeout=COMFY_TIMEOUT_SEC) as client:
        uploaded = await _upload_image(client, image_path)
        wf = _build_workflow(uploaded, seed=seed, video_frames=frames,
                             fps=fps, width=width, height=height,
                             positive_prompt=full_prompt,
                             filename_prefix=prefix)
        prompt_id = await _submit(client, wf)
        entry = await _wait_for_done(client, prompt_id)
        files = _extract_output_files(entry)
        if not files:
            raise RuntimeError(f"Comfy {prompt_id} returned no output files")

        raw_path = out_mp4.with_name(out_mp4.stem + ".raw" + Path(files[0]["filename"]).suffix)
        await _download(client, files[0], raw_path)

    await _normalize_mp4(raw_path, out_mp4, fps)
    raw_path.unlink(missing_ok=True)

    return {
        "prompt_id": prompt_id,
        "frames": frames,
        "fps": fps,
        "duration_sec": round(frames / fps, 3),
        "seed": seed,
        "prompt": full_prompt,
    }
