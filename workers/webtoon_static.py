"""webtoon_static motion_mode — 변곡점 정적컷 흐름.

흐름 (정체성 99% 검증 완료 — 2026-05-10):
  1. frame@start_sec 추출
  2. Kontext img2img → frame을 2D 한국 웹툰 화풍으로 (사람·구도 보존, 자막 제거)
  3. baseline (per-mascot, savior/victim 둘 중 한 쪽) → GPT-image-2 i2i로 표정 변경
     (baseline reference + lockdown character_spec prompt → 정체성 99% 보존)
  4. PIL flood-fill alpha (외곽 흰 배경만 투명, 내부 흰색 보존)
  5. ffmpeg static composite — 배경 정지 + 캐릭터 정지 + alpha pop in 0.3s
     (motion 없음, zoom 없음, 자막 없음 — 캡컷 후처리 영역)
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import kie_client


WEBTOON_BG_PROMPT = (
    "Repaint this scene as a 2D Korean webtoon illustration (네이버웹툰 style). "
    "Flat clean colors, clear black ink line outlines, simple soft cell shading, "
    "vibrant but balanced palette, 2D digital anime aesthetic. "
    "Keep all the people and the room layout exactly as shown — same poses, same outfits, "
    "same interior — but rendered in webtoon style. "
    "*** ABSOLUTELY NO TEXT IN THE OUTPUT IMAGE ***. "
    "Do NOT draw any letters, numbers, captions, subtitles, watermarks, or written words of any kind. "
    "Remove ALL text — English, Korean, Japanese, Chinese, symbols, anything resembling characters. "
    "If the input image has subtitles or watermarks or any text overlay, COMPLETELY REMOVE them by inpainting "
    "the underlying scene. The final output must be 100% text-free, just the people and the background. "
    "NO photographic textures, NO watercolor blur, NO 3D rendering — clean 2D digital line "
    "art with flat color fills."
)


def _ffmpeg() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return shutil.which("ffmpeg") or "ffmpeg"


def _alpha_isolate(img_path: Path, white_threshold: int = 235) -> Image.Image:
    """외곽 흰 배경만 투명. 내부 흰색 (옷, 깃털) 보존 — flood-fill from edges."""
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


def _ffprobe() -> str:
    for _p in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"):
        if Path(_p).exists():
            return _p
    return "ffprobe"


async def _video_dur(orig: Path) -> float:
    try:
        proc = await asyncio.create_subprocess_exec(
            _ffprobe(), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(orig),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        o, _ = await proc.communicate()
        return float(o.decode().strip() or 0)
    except Exception:
        return 0.0


async def _extract_frame(orig: Path, sec: float, out: Path) -> None:
    # 영상 길이 초과 clamp (오래된 spec의 끝 초과 clip 방어 — no packets 에러).
    dur = await _video_dur(orig)
    if dur and dur > 0 and sec > dur - 0.2:
        sec = max(0.0, dur - 0.2)
    # 1차: -ss AFTER -i (accurate seek). 실패 시 2차: -ss 앞 (fast seek).
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y", "-i", str(orig), "-ss", str(sec),
        "-frames:v", "1", "-q:v", "2", str(out),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode == 0 and out.exists():
        return
    # 2차 — fast seek (-ss 앞)
    proc2 = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y", "-ss", str(sec), "-i", str(orig),
        "-frames:v", "1", "-q:v", "2", str(out),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err2 = await proc2.communicate()
    if proc2.returncode != 0 or not out.exists():
        raise RuntimeError(f"frame extract (t={sec:.1f}, dur={dur:.1f}): {err2.decode()[-200:]}")


async def make_webtoon_static_clip(
    orig_video: Path,
    start_sec: float,
    duration_sec: float,
    out_clip: Path,
    work_dir: Path,
    *,
    baseline_path: Path,           # webtoon baseline (savior or victim)
    expression_prompt: str,        # mascot.expression_prompt_for() 결과
    bg_prompt: str | None = None,  # default = WEBTOON_BG_PROMPT
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    fps: int = 30,
    mascot_w: int = 480,
    base_y: int = 1100,            # 세로 위치 (마스코트 top 좌표). 1100 = 하단 부근
    pop_in_sec: float = 0.3,
    x_offset: int = 100,           # 우측으로 살짝 (음수면 좌측). x_center 있으면 무시
    y_offset: int | None = None,   # 세로 직접 지정 (base_y 무시). y_center 있으면 무시
    x_center: int | None = None,   # 마스코트 가운데 가로 좌표 (1080 캔버스 기준). 있으면 x_offset 무시
    y_center: int | None = None,   # 마스코트 가운데 세로 좌표 (1920 캔버스 기준). 있으면 y_offset 무시
    mirror: bool = False,          # True면 마스코트 좌우 반전 (예: 왼쪽 향한 동작)
    progress_cb=None,              # 단계별 진행률 콜백 (pct, msg)
) -> dict:
    """변곡점 1컷 webtoon static 영상 generate.

    baseline은 GPT-image-2 baseline 1장 (per-mascot, role: savior/victim).
    expression_prompt는 lockdown 형식 (character_spec + 바뀌는 표정만 명시).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cost_usd = 0.0

    async def _emit(pct: int, msg: str):
        if progress_cb:
            await progress_cb(pct, msg)

    # 1. frame extract
    # JPEG로 추출 (PNG 1MB → JPEG ~300KB). Kie 서버가 image 받을 때 connection
    # broken 발생 빈도 줄임 (전송 사이즈 70% 감소).
    await _emit(10, "🎬 1/5 원본 프레임 추출")
    frame = work_dir / "frame.jpg"
    await _extract_frame(orig_video, start_sec, frame)

    # 2 + 3: Kontext bg + GPT-image-2 i2i sprite를 PARALLEL 호출 (둘 다 별개라 안전).
    # baseline + turnaround → multi-ref anchor (정체성 ~95% lockdown).
    # baseline_path 옆에 webtoon_<role>_turnaround_*.png 있으면 모두 ref로 사용.
    import re
    bg_painted = work_dir / "bg_webtoon.png"
    sprite_raw = work_dir / "sprite_raw.png"
    bp = Path(baseline_path)
    refs: list[Path] = [bp]
    m = re.match(r"webtoon_(.+)_baseline\.png$", bp.name)
    if m:
        role_id = m.group(1)
        ta_paths = sorted(bp.parent.glob(f"webtoon_{role_id}_turnaround_*.png"))
        refs.extend(ta_paths)
    await _emit(25, f"🎨 2/5 배경 + 마스코트 표정 동시 생성 중 (~60초)")
    print(f"[webtoon_static] parallel: Kontext bg + i2i with {len(refs)} refs", flush=True)

    # 자동 재시도 wrapper — Kie 서버 일시 에러 (422, "task id is blank" 등) 시 최대 3번 시도
    async def _with_retry(coro_fn, name: str, max_attempts: int = 3):
        last_err = None
        for attempt in range(max_attempts):
            try:
                return await coro_fn()
            except Exception as e:
                last_err = e
                msg = str(e)[:200]
                if attempt < max_attempts - 1:
                    delay = 5 * (attempt + 1)  # 5초, 10초, 15초
                    print(f"[webtoon_static] {name} 실패 ({attempt+1}/{max_attempts}): {msg} — {delay}초 후 다시 시도", flush=True)
                    await asyncio.sleep(delay)
                else:
                    print(f"[webtoon_static] {name} 최종 실패 ({max_attempts}번): {msg}", flush=True)
                    raise last_err

    fr_bg, fr_sp = await asyncio.gather(
        _with_retry(lambda: kie_client.flux_kontext_img2img(
            frame, bg_painted,
            prompt=bg_prompt or WEBTOON_BG_PROMPT,
            aspect_ratio="9:16",
        ), "Kontext bg"),
        _with_retry(lambda: kie_client.nano_banana_img2img(
            refs[0], sprite_raw,
            prompt=expression_prompt,
            extra_image_paths=refs[1:] if len(refs) > 1 else None,
            aspect_ratio="1:1",
            timeout=90.0,  # Nano Banana 보통 8~30초. 90초 = 충분 여유 + 빠른 fail
        ), "nano-banana sprite", max_attempts=1),
    )
    cost_usd += float(fr_bg.get("cost_usd") or 0.025) + 0.02  # Nano Banana ~$0.02/img
    await _emit(70, "✂️ 3/5 마스코트 배경 제거 + 크기 조정")

    # 4. alpha (외곽 흰 배경 → transparent) + crop + resize
    sprite_alpha = _alpha_isolate(sprite_raw, white_threshold=235)
    bbox = sprite_alpha.getbbox()
    if bbox:
        sprite_alpha = sprite_alpha.crop(bbox)
    aspect = sprite_alpha.size[1] / max(1, sprite_alpha.size[0])
    sprite_alpha = sprite_alpha.resize((mascot_w, int(mascot_w * aspect)), Image.LANCZOS)
    sprite_path = work_dir / "sprite.png"
    sprite_alpha.save(sprite_path)

    await _emit(80, "🖼 4/5 배경 캔버스 만드는 중")
    # 5. resize bg → 1080x1920 canvas
    bg_img = Image.open(bg_painted).convert("RGB")
    sw, sh = bg_img.size
    scale = max(canvas_w / sw, canvas_h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    bg_img = bg_img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(bg_img, ((canvas_w - nw) // 2, (canvas_h - nh) // 2))
    bg_canvas_path = work_dir / "bg_canvas.png"
    canvas.save(bg_canvas_path)

    await _emit(90, "🎞 5/5 영상 합성")
    # 6. ffmpeg static composite — bg 정지 + sprite 정지 + alpha pop in + silent audio
    #    합본 호환을 위해 silent aac stream 추가 (이어붙일 때 redo 인코딩 안 거치게).
    #    정적 컷이라 preset ultrafast로도 화질 동일. crf 23.
    # sprite 실제 크기 (resize 끝난 후) — 가운데 좌표 변환에 사용
    sprite_w, sprite_h = sprite_alpha.size
    # 가로 위치 결정 — x_center 우선 (sprite 가운데 기준, ffmpeg overlay는 왼쪽 위 기준이라 변환)
    if x_center is not None:
        center_x = int(x_center) - sprite_w // 2
    else:
        center_x = (canvas_w - mascot_w) // 2 + int(x_offset)
    # 세로 위치 결정: y_center 우선 (가운데), 아니면 y_offset (top), 아니면 base_y
    if y_center is not None:
        final_y = int(y_center) - sprite_h // 2
    elif y_offset is not None:
        final_y = int(y_offset)
    else:
        final_y = int(base_y)
    # 좌우 반전 옵션 — 마스코트가 반대쪽 보고 있어야 자연스러울 때
    mirror_filter = "hflip," if mirror else ""
    fc = (
        f"[0:v]loop=loop=-1:size=1:start=0,trim=duration={duration_sec},setpts=PTS-STARTPTS[bg];"
        f"[1:v]format=rgba,{mirror_filter}loop=loop=-1:size=1:start=0,trim=duration={duration_sec},setpts=PTS-STARTPTS[mv];"
        f"[bg][mv]overlay=x={center_x}:y={final_y}:enable='gte(t\\,{pop_in_sec})':eval=init[outv]"
    )
    cmd = [
        _ffmpeg(), "-y",
        "-loop", "1", "-t", str(duration_sec), "-r", str(fps), "-i", str(bg_canvas_path),
        "-loop", "1", "-t", str(duration_sec), "-r", str(fps), "-i", str(sprite_path),
        # silent audio stream — 합본 호환 + 클립 단독 재생 가능 (소리는 무음)
        "-f", "lavfi", "-t", str(duration_sec),
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex", fc,
        "-map", "[outv]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23",
        "-r", str(fps),
        "-g", str(fps),  # 기준 컷 간격 1초 — 합본 시 이어붙이기 안전
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-t", str(duration_sec),
        "-movflags", "+faststart",
        str(out_clip),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"compose: {err.decode()[-500:]}")

    await _emit(100, "✅ 클립 완료")
    return {
        "cost_usd": round(cost_usd, 4),
        "bg_path": str(bg_painted),
        "sprite_path": str(sprite_path),
        "out_clip": str(out_clip),
    }
