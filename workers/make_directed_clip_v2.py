#!/usr/bin/env python3
"""DIRECTED BY + STORY BY 영화 크레딧 밈 — 마이클베이 원본 디테일 모방.

원본 분석:
- 폰트: sans-serif condensed (Trade Gothic / DIN Condensed)
- 색상: 회색 톤 (#DCDCDC, 완전 흰색 아님)
- 자간: 매우 크다 (~글자 폭의 15~20%)
- 위치: 정중앙
- 두 줄: 상단 작게 ("DIRECTED BY") + 하단 크게 (이름)
- fade in/out 부드러움
- 4초 = DIRECTED BY 2초 + STORY BY 2초
"""
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
BG = "#000000"
FG = "#DCDCDC"          # 약간 회색 (영화 크레딧 톤)
SMALL_SIZE = 50          # "DIRECTED BY" / "STORY BY" 작은 폰트
BIG_SIZE = 110           # 이름 큰 폰트
LETTER_SPACING_RATIO = 0.22  # 글자 폭의 22% 자간 (영화 크레딧 매우 크다)
LINE_GAP = 60            # 두 줄 사이 간격

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Trade Gothic LT Std Bold Cn No 20.ttf",
    "/System/Library/Fonts/Avenir Next Condensed.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def text_width_spaced(draw, text: str, font, spacing: int) -> int:
    """자간 포함 전체 텍스트 폭"""
    if not text:
        return 0
    widths = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        widths.append(bbox[2] - bbox[0])
    return sum(widths) + spacing * (len(text) - 1)


def draw_spaced_centered(draw, text: str, font, color, center_x: int, baseline_y: int, spacing: int):
    """자간 적용 + 중앙 정렬"""
    total = text_width_spaced(draw, text, font, spacing)
    x = center_x - total // 2
    for ch in text:
        draw.text((x, baseline_y), ch, font=font, fill=color)
        bbox = draw.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + spacing


MAX_TEXT_WIDTH = int(W * 0.88)  # 화면 폭의 88% 이내


def fit_big_font(draw, text: str, spacing_ratio: float = LETTER_SPACING_RATIO) -> tuple:
    """긴 텍스트면 폰트 사이즈 자동 축소. (font, spacing) 반환."""
    size = BIG_SIZE
    while size >= 50:
        font = find_font(size)
        spacing = int(size * spacing_ratio)
        width = text_width_spaced(draw, text.upper(), font, spacing)
        if width <= MAX_TEXT_WIDTH:
            return font, spacing
        size -= 5
    # 못 줄여도 가장 작은 거 반환
    return find_font(50), int(50 * spacing_ratio)


def make_card(top_label: str, name: str, out_path: Path):
    """한 카드 — 상단 작은 라벨 + 하단 큰 이름"""
    img = Image.new("RGB", (W, H), color=BG)
    draw = ImageDraw.Draw(img)

    small_font = find_font(SMALL_SIZE)
    big_font, spacing_big = fit_big_font(draw, name)

    spacing_small = int(SMALL_SIZE * LETTER_SPACING_RATIO)

    # 두 줄 높이 계산
    sbbox = draw.textbbox((0, 0), "Ag", font=small_font)
    s_h = sbbox[3] - sbbox[1]
    bbbox = draw.textbbox((0, 0), "Ag", font=big_font)
    b_h = bbbox[3] - bbbox[1]

    total_h = s_h + LINE_GAP + b_h
    top_y = (H - total_h) // 2 - sbbox[1]  # bbox top offset 보정

    center_x = W // 2

    # 그리기
    draw_spaced_centered(
        draw, top_label.upper(), small_font, FG, center_x, top_y, spacing_small
    )
    draw_spaced_centered(
        draw, name.upper(), big_font, FG,
        center_x, top_y + s_h + LINE_GAP - bbbox[1], spacing_big
    )

    img.save(out_path, "PNG")


def make_clip(director: str, writers: str, out_mp4: Path,
               card_duration: float = 2.0, fade: float = 0.5):
    """4초 클립: DIRECTED BY [감독] + STORY BY [작가]"""
    tmp_dir = out_mp4.parent / f".{out_mp4.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) 카드 2장
    card1 = tmp_dir / "card1.png"
    card2 = tmp_dir / "card2.png"
    make_card("DIRECTED BY", director, card1)
    make_card("STORY BY", writers, card2)

    # 2) 카드 1 → 클립 1
    clip1 = tmp_dir / "clip1.mp4"
    fade_out_start = card_duration - fade
    cmd1 = [
        "/opt/homebrew/bin/ffmpeg",
        "-loop", "1", "-t", str(card_duration),
        "-i", str(card1),
        "-vf", f"fade=t=in:st=0:d={fade},fade=t=out:st={fade_out_start}:d={fade}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "30", "-preset", "fast", "-y",
        str(clip1),
    ]
    subprocess.run(cmd1, capture_output=True, check=True)

    # 3) 카드 2 → 클립 2
    clip2 = tmp_dir / "clip2.mp4"
    cmd2 = [
        "/opt/homebrew/bin/ffmpeg",
        "-loop", "1", "-t", str(card_duration),
        "-i", str(card2),
        "-vf", f"fade=t=in:st=0:d={fade},fade=t=out:st={fade_out_start}:d={fade}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "30", "-preset", "fast", "-y",
        str(clip2),
    ]
    subprocess.run(cmd2, capture_output=True, check=True)

    # 4) concat
    concat_list = tmp_dir / "list.txt"
    concat_list.write_text(f"file '{clip1.resolve()}'\nfile '{clip2.resolve()}'\n")
    cmd_concat = [
        "/opt/homebrew/bin/ffmpeg",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy", "-y",
        str(out_mp4),
    ]
    subprocess.run(cmd_concat, capture_output=True, check=True)

    # 5) tmp 정리
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--director", required=True)
    p.add_argument("--writers", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_clip(args.director, args.writers, out)
    print(f"OK: {out}")
