"""지브리 화풍 정적 테스트 — 변곡점 frame + 지브리 배경 + 지브리 angel 표정컷 정지.

흐름:
1. frame@start_sec 추출
2. Kontext img2img → frame을 지브리 화풍으로 (사람/배경 그대로 보존)
3. angel reference image → Kontext img2img로 지브리 + 표정 변환
4. PIL flood-fill alpha (외부 흰 배경만 투명)
5. ffmpeg static composite (배경 정지 + angel 정지, alpha pop in 0.3s만, zoom/motion/자막 X)

화면 zoom in/out은 형님이 캡컷에서 직접.
"""
from __future__ import annotations
import asyncio, subprocess, shutil
from pathlib import Path
from PIL import Image
import numpy as np

from . import kie_client


GHIBLI_BG_PROMPT = (
    "Transform this scene to Studio Ghibli hand-painted anime style. "
    "Soft watercolor backgrounds, warm cinematic Miyazaki lighting, "
    "painterly soft edges with subtle gentle outlines (NOT harsh black ink), "
    "muted earth-tone palette, lush detail, dreamy atmospheric depth. "
    "Preserve every person and object in their exact original positions and poses. "
    "DO NOT add any angel, halo, wings, mascot, or fantasy character to the scene. "
    "Keep ONLY what's already in the photo, just repainted."
)

GHIBLI_ANGEL_PROMPT_TEMPLATE = (
    "Transform this small angel character drawing to Studio Ghibli hand-painted anime style. "
    "Soft watercolor brush strokes, painterly gentle outlines, "
    "warm subtle Miyazaki lighting, dreamy soft edges. "
    "Keep the character's identity: a small chibi-style angel with halo and feathered wings. "
    "Change facial expression to: {expression}. "
    "Full body visible, isolated character on pure pristine white background, "
    "no scenery, no other characters, no objects around."
)


def _ffmpeg() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return "ffmpeg"


def _alpha_isolate(img_path: Path, white_threshold: int = 245) -> Image.Image:
    """외부 흰 배경만 투명. 내부 흰색 (옷, 흰 깃털 등) 보존."""
    from scipy import ndimage
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    is_white = (rgb >= white_threshold).all(axis=2)
    labels, _ = ndimage.label(is_white)
    ext = set()
    for s in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        ext.update(s.tolist())
    ext.discard(0)
    arr[np.isin(labels, list(ext)), 3] = 0
    return Image.fromarray(arr)


async def make_ghibli_static_clip(
    orig_video: Path,
    start_sec: float,
    duration_sec: float,
    out_clip: Path,
    work_dir: Path,
    *,
    angel_ref_path: Path,
    expression: str = (
        "shocked dismay, mouth slightly open in surprise, "
        "both hands raised to face in worried disbelief, eyes wide"
    ),
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    fps: int = 24,
    mascot_w: int = 520,
    base_y: int = 1100,
    pop_in_sec: float = 0.3,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0

    # 1. frame extract
    frame = work_dir / "frame.png"
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y", "-ss", str(start_sec), "-i", str(orig_video),
        "-frames:v", "1", "-q:v", "2", str(frame),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not frame.exists():
        raise RuntimeError(f"frame extract: {err.decode()[-300:]}")

    # 2. Kontext bg → 지브리
    bg_ghibli = work_dir / "bg_ghibli.png"
    fr = await kie_client.flux_kontext_img2img(
        frame, bg_ghibli, prompt=GHIBLI_BG_PROMPT, aspect_ratio="9:16")
    cost += float(fr.get("cost_usd") or 0.025)

    # 3. resize bg → 1080x1920
    bg_img = Image.open(bg_ghibli).convert("RGB")
    sw, sh = bg_img.size
    scale = max(canvas_w / sw, canvas_h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    bg_img = bg_img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(bg_img, ((canvas_w - nw) // 2, (canvas_h - nh) // 2))
    bg_path = work_dir / "bg_canvas.png"
    canvas.save(bg_path)

    # 4. angel ref → 지브리 + 표정
    angel_ghibli_raw = work_dir / "angel_ghibli_raw.png"
    angel_prompt = GHIBLI_ANGEL_PROMPT_TEMPLATE.format(expression=expression)
    fr = await kie_client.flux_kontext_img2img(
        angel_ref_path, angel_ghibli_raw,
        prompt=angel_prompt, aspect_ratio="1:1")
    cost += float(fr.get("cost_usd") or 0.025)

    # 5. alpha (외부 흰 배경만 투명) + crop + resize
    angel_alpha = _alpha_isolate(angel_ghibli_raw, white_threshold=245)
    bbox = angel_alpha.getbbox()
    if bbox:
        angel_alpha = angel_alpha.crop(bbox)
    aspect = angel_alpha.size[1] / max(1, angel_alpha.size[0])
    angel_alpha = angel_alpha.resize((mascot_w, int(mascot_w * aspect)), Image.LANCZOS)
    angel_sprite = work_dir / "angel_sprite.png"
    angel_alpha.save(angel_sprite)

    # 6. ffmpeg static composite
    center_x = (canvas_w - mascot_w) // 2
    fc = (
        f"[0:v]loop=loop=-1:size=1:start=0,trim=duration={duration_sec},setpts=PTS-STARTPTS[bg];"
        f"[1:v]format=rgba,loop=loop=-1:size=1:start=0,trim=duration={duration_sec},setpts=PTS-STARTPTS[mv];"
        f"[bg][mv]overlay=x={center_x}:y={base_y}:enable='gte(t\\,{pop_in_sec})':eval=init[out]"
    )
    cmd = [
        _ffmpeg(), "-y",
        "-loop", "1", "-t", str(duration_sec), "-r", str(fps), "-i", str(bg_path),
        "-loop", "1", "-t", str(duration_sec), "-r", str(fps), "-i", str(angel_sprite),
        "-filter_complex", fc, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
        "-r", str(fps), str(out_clip),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"compose: {err.decode()[-500:]}")

    return {
        "cost_usd": round(cost, 4),
        "bg_ghibli_path": str(bg_ghibli),
        "angel_sprite_path": str(angel_sprite),
        "out_clip": str(out_clip),
    }
