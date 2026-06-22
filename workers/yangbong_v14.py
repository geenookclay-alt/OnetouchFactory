"""yangbong_kontext v14 — 모든 화풍 흑백 웹툰 + 마스코트 narrative motion.

흐름 (검증 완료):
  1. Kontext img2img — frame → 흑백 웹툰 sketch (사람/배경 통합, 마스코트 X)
  2. LoRA banbaji-* × N 자세 (얼굴 정면 보이는 narrative 자세)
  3. Kontext img2img × N (sprite tone match — 흑백 웹툰 톤 통일)
  4. PIL flood-fill alpha + thicken outline (검은 line 강조)
  5. ffmpeg PRORES4444 sprite cycle 8fps
  6. ffmpeg compose: bg 정지 + sprite overlay
     - alpha pop in (t=0.3s 갑자기 나타남)
     - 위치 path (center → target_x linear 이동)
     - 자세 cycle 8fps

⚠️ 카메라 zoom/pan/자막 X (캡컷 후처리 영역).
"""
from __future__ import annotations
import asyncio, os, shutil, subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import httpx

from . import kie_client


KONTEXT_BG_PROMPT_DEFAULT = (
    "Convert this scene to rough hand-drawn black ink sketch line drawing with thin wobbly outlines. "
    "Preserve all the people and objects exactly. Simplify clothing to plain unpatterned line outlines. "
    "DO NOT add any angel, mascot, or character. Keep ONLY the original scene. "
    "Completely black and white, no shading, B-grade Korean web comic doodle aesthetic"
)
KONTEXT_TONE_PROMPT = (
    "convert to extremely simple thin black ink line drawing, "
    "single thin pen outline only, no shading no fill no color, "
    "minimal childlike doodle sketch, very thin wobbly hand-drawn lines, "
    "completely black and white on pure white background"
)


def _ffmpeg() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists(): return p
    return shutil.which("ffmpeg") or "ffmpeg"


def _flood_fill_alpha(img: Image.Image, threshold: int = 240) -> Image.Image:
    from scipy import ndimage
    img = img.convert("RGBA")
    arr = np.array(img)
    rgb_sum = arr[:, :, :3].sum(axis=2)
    mask = rgb_sum >= threshold * 3
    labels, _ = ndimage.label(mask)
    ext = set()
    for s in (labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]):
        ext.update(s.tolist())
    ext.discard(0)
    arr[np.isin(labels, list(ext)), 3] = 0
    return Image.fromarray(arr)


def _thicken_outline(img: Image.Image, iterations: int = 2) -> Image.Image:
    from scipy import ndimage
    arr = np.array(img.convert("RGBA"))
    rgb_sum = arr[:, :, :3].sum(axis=2)
    is_black = (rgb_sum < 250) & (arr[:, :, 3] > 0)
    dilated = ndimage.binary_dilation(is_black, iterations=iterations)
    arr[dilated, 0] = 0; arr[dilated, 1] = 0; arr[dilated, 2] = 0
    arr[dilated, 3] = 255
    return Image.fromarray(arr)


async def _extract_frame(orig: Path, sec: float, out: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y", "-ss", str(sec), "-i", str(orig),
        "-frames:v", "1", "-q:v", "2", str(out),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"frame extract: {err.decode()[:200]}")


async def _replicate_lora(character: str, prompt: str, out_png: Path, seed: int) -> None:
    token = os.getenv("REPLICATE_API_TOKEN", "")
    slug_map = {"angel": "dongret-max/banbaji-angel", "reaper": "dongret-max/banbaji-reaper"}
    slug = slug_map.get(character, slug_map["angel"])
    body = {"input": {"prompt": prompt[:2000], "aspect_ratio": "1:1",
        "num_inference_steps": 28, "guidance_scale": 3.5, "lora_scale": 1.0,
        "seed": seed, "output_format": "png", "output_quality": 95}}
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(f"https://api.replicate.com/v1/models/{slug}",
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        ver = r.json()["latest_version"]["id"]
        r = await c.post(
            f"https://api.replicate.com/v1/models/{slug}/versions/{ver}/predictions",
            json=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        r.raise_for_status()
        pid = r.json()["id"]
        for _ in range(80):
            await asyncio.sleep(3)
            r = await c.get(f"https://api.replicate.com/v1/predictions/{pid}",
                            headers={"Authorization": f"Bearer {token}"})
            d = r.json()
            if d["status"] == "succeeded":
                url = d["output"][0] if isinstance(d["output"], list) else d["output"]
                rv = await c.get(url); out_png.write_bytes(rv.content); return
            if d["status"] in ("failed", "canceled"): raise RuntimeError(d.get("error"))
        raise RuntimeError("LoRA timeout")


# 기본 face-visible 자세 prompts
# 머리/외형 일관 강제 prompt — cycle 시 깜빡임 방지
_HAIR_ANGEL = "with short brown bangs hairstyle and round chibi face"
_HOOD_REAPER = "with black hood covering the head, only skull face visible"

DEFAULT_POSES_ANGEL = [
    f"facing forward, {_HAIR_ANGEL}, sad crying expression with big tear drops falling, "
    f"mouth open in despair, both hands at the sides, face clearly visible",
    f"facing forward, {_HAIR_ANGEL}, sad pouting expression with frowning eyebrows mouth slightly open, "
    f"both hands raised next to body palms up, face clearly visible",
]
DEFAULT_POSES_REAPER = [
    f"facing forward, {_HOOD_REAPER}, holding scythe upright, ominous menacing expression",
    f"facing forward, {_HOOD_REAPER}, swinging scythe overhead, evil grin",
]


async def make_yangbong_v14_clip(
    orig_video: Path,
    start_sec: float,
    duration_sec: float,
    out_clip: Path,
    work_dir: Path,
    *,
    character: str = "angel",
    bg_prompt: str | None = None,
    pose_prompts: list[str] | None = None,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    fps: int = 30,
    sprite_fps: int = 8,
    mascot_w: int = 480,
    pop_in_sec: float = 0.3,
    base_y: int = 1100,
    seed: int = 20260509,
) -> dict:
    """v14 흐름: bg sketch + sprite overlay (alpha pop + motion path + 자세 cycle)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cost_usd = 0.0

    # 1. Frame extract + Kontext bg
    frame = work_dir / "frame.png"
    await _extract_frame(orig_video, start_sec, frame)
    bg_kontext = work_dir / "bg_kontext.png"
    fr = await kie_client.flux_kontext_img2img(
        frame, bg_kontext,
        prompt=bg_prompt or KONTEXT_BG_PROMPT_DEFAULT,
        aspect_ratio="9:16",
    )
    cost_usd += float(fr.get("cost_usd") or 0.025)

    # 2. bg → 1080x1920 canvas
    bg_img = Image.open(bg_kontext).convert("RGB")
    sw, sh = bg_img.size
    scale = max(canvas_w/sw, canvas_h/sh)
    nw, nh = int(sw*scale), int(sh*scale)
    bg_img = bg_img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(bg_img, ((canvas_w-nw)//2, (canvas_h-nh)//2))
    bg_path = work_dir / "bg_canvas.png"
    canvas.save(bg_path)

    # 3. LoRA poses + Kontext tone match
    poses = pose_prompts or (DEFAULT_POSES_ANGEL if character == "angel" else DEFAULT_POSES_REAPER)
    sprite_paths_raw = []
    for i, p in enumerate(poses):
        full = (
            f"TOK_BANBAJI_{character.upper()} chibi {character} character with halo "
            f"and small wings, {p}, "
            f"rough thin black ink sketch line on pure white background, chibi proportions"
        )
        out_lora = work_dir / f"lora_{i+1}.png"
        await _replicate_lora(character, full, out_lora, seed=seed+i*100)
        cost_usd += 0.04
        # tone match
        out_match = work_dir / f"matched_{i+1}.png"
        try:
            fr = await kie_client.flux_kontext_img2img(
                out_lora, out_match, prompt=KONTEXT_TONE_PROMPT, aspect_ratio="1:1")
            cost_usd += float(fr.get("cost_usd") or 0.025)
            sprite_paths_raw.append(out_match)
        except Exception:
            sprite_paths_raw.append(out_lora)

    # 4. flood-fill alpha + thicken + resize normalize
    sprite_paths = []
    sizes = []
    for i, p in enumerate(sprite_paths_raw):
        m = _flood_fill_alpha(Image.open(p))
        m = _thicken_outline(m, iterations=2)
        bbox = m.getbbox()
        if bbox: m = m.crop(bbox)
        aspect = m.size[1] / m.size[0]
        m = m.resize((mascot_w, int(mascot_w * aspect)), Image.LANCZOS)
        sp = work_dir / f"sprite_{i+1}.png"
        m.save(sp); sprite_paths.append(sp); sizes.append(m.size)
    if not sprite_paths:
        raise RuntimeError("no sprites")
    max_h = max(s[1] for s in sizes)
    for sp, sz in zip(sprite_paths, sizes):
        if sz[1] < max_h:
            img = Image.open(sp)
            c2 = Image.new("RGBA", (mascot_w, max_h), (255,255,255,0))
            c2.paste(img, (0, max_h-sz[1]), img); c2.save(sp)

    # 5. PRORES sprite cycle
    mascot_mov = work_dir / "mascot.mov"
    cmd = [_ffmpeg(), "-y", "-framerate", str(sprite_fps),
           "-loop", "1", "-i", str(work_dir / "sprite_%d.png"),
           "-t", str(duration_sec), "-vf", f"fps={sprite_fps}",
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
           str(mascot_mov)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"sprite mov: {err.decode()[-300:]}")

    # 6. ffmpeg compose: bg + sprite overlay (alpha pop + motion path)
    center_x = (canvas_w - mascot_w) // 2
    target_x = canvas_w - mascot_w - 60
    expr_x = (
        f"if(lt(t\\,{pop_in_sec + 0.3})\\,{center_x}\\,"
        f"{center_x}+({target_x}-{center_x})*(t-{pop_in_sec + 0.3})/({duration_sec}-{pop_in_sec + 0.3}))"
    )
    expr_y = f"{base_y}+15*sin(t*2)"
    fc = (
        f"[0:v]loop=loop=-1:size=1:start=0,trim=duration={duration_sec},setpts=PTS-STARTPTS[bg];"
        f"[1:v]format=yuva444p10le[mv];"
        f"[bg][mv]overlay=x='{expr_x}':y='{expr_y}':enable='gte(t\\,{pop_in_sec})':eval=frame:format=auto[out]"
    )
    cmd = [_ffmpeg(), "-y",
           "-loop", "1", "-t", str(duration_sec), "-r", str(fps), "-i", str(bg_path),
           "-i", str(mascot_mov),
           "-filter_complex", fc, "-map", "[out]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
           "-r", str(fps), str(out_clip)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"compose: {err.decode()[-500:]}")

    return {"cost_usd": round(cost_usd, 4), "n_poses": len(sprite_paths)}
