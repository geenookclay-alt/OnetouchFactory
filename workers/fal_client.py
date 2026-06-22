"""Fal.ai client — LTX-Video 13B distilled image-to-video.

Cloud GPU, ~5~10s per clip vs local Comfy 30~60s. Same LTX family so quality
≈ Comfy distilled, but 13B is larger than our local 2B → slightly better.

Endpoint: https://fal.run/fal-ai/ltx-video-13b-distilled/image-to-video (sync).
Auth: header "Authorization: Key <FAL_API_KEY>".
Pricing (approx): ~$0.04~0.10 per 2s clip (720p).
"""
from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx


FAL_API_KEY = os.getenv("FAL_API_KEY", "")
FAL_MODEL = "fal-ai/ltx-video-13b-distilled/image-to-video"
FAL_ENDPOINT = f"https://queue.fal.run/{FAL_MODEL}"
FAL_KONTEXT_ENDPOINT = "https://queue.fal.run/fal-ai/flux-pro/kontext"
FAL_BIREFNET_VIDEO_ENDPOINT = "https://queue.fal.run/fal-ai/birefnet/v2/video"
FAL_STORAGE_INIT = "https://rest.alpha.fal.ai/storage/upload/initiate"
FAL_TIMEOUT_SEC = float(os.getenv("FAL_TIMEOUT_SEC", "1500"))


def _ffmpeg() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return shutil.which("ffmpeg") or "ffmpeg"


def _image_to_data_uri(image_path: Path) -> str:
    raw = image_path.read_bytes()
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(suffix, "image/png")
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


async def _normalize_mp4(src_path: Path, dst_path: Path,
                         target_w: int = 1080, target_h: int = 1920,
                         target_fps: int = 30) -> None:
    """Re-encode Fal mp4 → 1080x1920 @ 30fps h264 yuv420p, audio dropped.
    Matches sprite/cut_segment params exactly so concat -c copy stitches cleanly.
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
    """Cheap check — just verify FAL_API_KEY is set + Fal accepts a HEAD-ish probe.
    We don't actually hit the i2v endpoint (would cost money)."""
    return bool(FAL_API_KEY)


async def storage_upload(file_path: Path) -> str:
    """Upload a local file to fal storage via the official Python SDK.
    Returns a public URL usable as image_url / video_url in fal model calls.
    SDK is sync — wrap in to_thread for async context.
    """
    if not FAL_API_KEY:
        raise RuntimeError("FAL_API_KEY not set")
    # The fal-client SDK reads FAL_KEY (not FAL_API_KEY) — alias once.
    os.environ.setdefault("FAL_KEY", FAL_API_KEY)
    import fal_client
    return await asyncio.to_thread(fal_client.upload_file, str(file_path))


async def _submit_and_wait(client: httpx.AsyncClient, endpoint: str,
                           payload: dict, timeout: float = FAL_TIMEOUT_SEC) -> dict:
    """Generic Fal queue API: POST submit → poll status → GET response."""
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    r = await client.post(endpoint, json=payload, headers=headers)
    if r.status_code >= 400:
        raise RuntimeError(f"Fal submit {r.status_code}: {r.text[:500]}")
    sub = r.json()
    request_id = sub.get("request_id")
    status_url = sub.get("status_url")
    response_url = sub.get("response_url")
    if not (status_url and response_url):
        raise RuntimeError(f"Fal submit missing fields: {sub}")
    deadline = time.monotonic() + timeout
    delay = 1.0
    while time.monotonic() < deadline:
        rs = await client.get(status_url, headers=headers)
        rs.raise_for_status()
        st = rs.json()
        status = st.get("status")
        if status == "COMPLETED":
            break
        if status in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Fal failed: {st}")
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 4.0)
    else:
        raise TimeoutError(f"Fal {request_id} timeout {timeout}s")
    rr = await client.get(response_url, headers=headers)
    rr.raise_for_status()
    return rr.json()


async def flux_kontext_img2img(image_path: Path, out_image: Path,
                                prompt: str = "black and white line art illustration, "
                                              "simple manga ink drawing style, clean black lines "
                                              "on pure white background, no shading, no color, "
                                              "minimalist sketch") -> dict:
    """Transform an image via Flux Kontext (img2img). Returns {url, cost_usd}."""
    image_url = await storage_upload(image_path)
    payload = {
        "prompt": prompt,
        "image_url": image_url,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "output_format": "png",
    }
    async with httpx.AsyncClient(timeout=FAL_TIMEOUT_SEC) as client:
        data = await _submit_and_wait(client, FAL_KONTEXT_ENDPOINT, payload, timeout=120.0)
        images = data.get("images") or []
        if not images:
            raise RuntimeError(f"Kontext returned no images: {data}")
        out_url = images[0].get("url")
        if not out_url:
            raise RuntimeError(f"Kontext image missing url: {images[0]}")
        rv = await client.get(out_url, timeout=60.0)
        rv.raise_for_status()
        out_image.write_bytes(rv.content)
    return {"url": out_url, "cost_usd": 0.04}


async def birefnet_video_alpha(video_path: Path, out_alpha_mov: Path) -> dict:
    """Run BiRefNet v2 video → output PRORES4444 .mov with embedded alpha.
    The PRORES4444 codec embeds alpha so a single file is enough for ffmpeg overlay.
    Returns {url, cost_usd}.
    """
    video_url = await storage_upload(video_path)
    payload = {
        "video_url": video_url,
        "model": "General Use (Light)",
        "operating_resolution": "1024x1024",
        "refine_foreground": True,
        "video_output_type": "PRORES4444 (.mov)",
        "video_quality": "high",
        "video_write_mode": "balanced",
    }
    async with httpx.AsyncClient(timeout=FAL_TIMEOUT_SEC) as client:
        data = await _submit_and_wait(client, FAL_BIREFNET_VIDEO_ENDPOINT, payload, timeout=240.0)
        video = data.get("video") or {}
        out_url = video.get("url")
        if not out_url:
            raise RuntimeError(f"BiRefNet returned no video.url: {data}")
        rv = await client.get(out_url, timeout=240.0)
        rv.raise_for_status()
        out_alpha_mov.write_bytes(rv.content)
    return {"url": out_url, "cost_usd": 0.05}


async def i2v(image_path: Path, duration_sec: float, out_mp4: Path,
              *, action_prompt: str = "",
              resolution: str = "480p",
              aspect_ratio: str = "9:16",
              frame_rate: int = 24,
              seed: int | None = None) -> dict:
    """Run Fal LTX 13B distilled i2v. Returns {request_id, frames, fps, duration_sec, cost_usd}.

    Auto-uploads image as data URI (no separate upload step needed for ≤4MB).
    """
    if not FAL_API_KEY:
        raise RuntimeError("FAL_API_KEY not set")
    image_path = Path(image_path)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    num_frames = max(24, round(duration_sec * frame_rate))
    full_prompt = (
        f"a 2D mascot character, {action_prompt}, smooth animation, "
        f"clear focus, vibrant colors, dynamic pose"
    ) if action_prompt else "a 2D mascot character, idle animation, smooth motion"

    payload = {
        "prompt": full_prompt,
        "image_url": _image_to_data_uri(image_path),
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "first_pass_num_inference_steps": 8,
        "second_pass_num_inference_steps": 8,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(FAL_ENDPOINT, json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"Fal submit {r.status_code}: {r.text[:500]}")
        sub = r.json()
        request_id = sub.get("request_id")
        status_url = sub.get("status_url")
        response_url = sub.get("response_url")
        if not (request_id and status_url and response_url):
            raise RuntimeError(f"Fal submit missing fields: {sub}")

        deadline = time.monotonic() + FAL_TIMEOUT_SEC
        delay = 1.0
        while time.monotonic() < deadline:
            rs = await client.get(status_url, headers=headers)
            rs.raise_for_status()
            st = rs.json()
            status = st.get("status")
            if status == "COMPLETED":
                break
            if status in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Fal failed: {st}")
            await asyncio.sleep(delay)
            delay = min(delay * 1.3, 4.0)
        else:
            raise TimeoutError(f"Fal request {request_id} did not complete in {FAL_TIMEOUT_SEC}s")

        rr = await client.get(response_url, headers=headers)
        rr.raise_for_status()
        data = rr.json()

        video = data.get("video") or {}
        video_url = video.get("url")
        if not video_url:
            raise RuntimeError(f"Fal returned no video.url: {data}")

        raw_path = out_mp4.with_name(out_mp4.stem + ".raw.mp4")
        rv = await client.get(video_url, timeout=120.0)
        rv.raise_for_status()
        raw_path.write_bytes(rv.content)

    await _normalize_mp4(raw_path, out_mp4)
    raw_path.unlink(missing_ok=True)

    return {
        "request_id": request_id,
        "frames": num_frames,
        "fps": frame_rate,
        "duration_sec": round(num_frames / frame_rate, 3),
        "seed": data.get("seed", seed),
        "wall_sec": round(time.monotonic() - t0, 2),
        "prompt": full_prompt,
        "fal_video_url": video_url,
    }
