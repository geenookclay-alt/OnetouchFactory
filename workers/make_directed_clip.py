#!/usr/bin/env python3
"""DIRECTED BY [이름] 검은 화면 클립 만들기

마이클베이 밈 마지막 크레딧 스타일 모방:
- 1080x1920 (세로 쇼츠)
- 검은 배경
- 흰색 영화 크레딧 폰트
- "DIRECTED BY" (작게, 위) + "이름" (크게, 아래)
- 3초 (fade in 0.3s + 유지 1.9s + fade out 0.8s)
"""
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920

# 폰트 후보
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Trajan Pro Regular.ttf",
    "/System/Library/Fonts/Optima.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_image(name: str, out_path: Path):
    img = Image.new("RGB", (W, H), color="black")
    draw = ImageDraw.Draw(img)

    # 폰트 크기
    small_font = find_font(60)   # DIRECTED BY
    big_font = find_font(140)     # 이름

    # 텍스트
    title = "DIRECTED BY"
    name_upper = name.upper()

    # 위치 계산 (중앙 정렬)
    t_bbox = draw.textbbox((0, 0), title, font=small_font)
    t_w = t_bbox[2] - t_bbox[0]
    t_h = t_bbox[3] - t_bbox[1]

    n_bbox = draw.textbbox((0, 0), name_upper, font=big_font)
    n_w = n_bbox[2] - n_bbox[0]
    n_h = n_bbox[3] - n_bbox[1]

    # 중앙
    gap = 40
    total_h = t_h + gap + n_h
    top_y = (H - total_h) // 2

    t_x = (W - t_w) // 2
    n_x = (W - n_w) // 2

    # 그리기
    draw.text((t_x, top_y), title, font=small_font, fill="white")
    draw.text((n_x, top_y + t_h + gap), name_upper, font=big_font, fill="white")

    img.save(out_path, "PNG")


def make_clip(name: str, out_mp4: Path, duration: float = 3.0,
               fade_in: float = 0.3, fade_out: float = 0.8):
    # 1) 이미지 만들기
    tmp_png = out_mp4.parent / f".{out_mp4.stem}.png"
    make_image(name, tmp_png)

    # 2) 이미지 → 영상 (fade in/out 포함)
    fade_out_start = duration - fade_out
    vf = f"fade=t=in:st=0:d={fade_in},fade=t=out:st={fade_out_start}:d={fade_out}"

    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-loop", "1",
        "-i", str(tmp_png),
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", vf,
        "-r", "30",
        "-preset", "fast",
        "-y",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fail: {proc.stderr.decode()[-300:]}")

    # 3) png 삭제
    tmp_png.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="감독 이름")
    p.add_argument("--out", required=True, help="출력 mp4 경로")
    p.add_argument("--duration", type=float, default=3.0)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_clip(args.name, out, args.duration)
    print(f"OK: {out}")
