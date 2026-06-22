#!/usr/bin/env python3
"""DIRECTED BY + STORY BY 영화 크레딧 밈 — 마이클베이 원본 정밀 모방 v3

원본 픽셀 분석 결과:
- 폰트: Helvetica Neue Condensed Bold / Avenir Next Condensed Bold (angular sans-serif)
- 색상: RGB(219,219,219) ~ (190,190,190) = #C8C8C8 회색 톤
- 외곽: 약한 outer glow (자연 anti-aliasing + soft halo)
- 자간: 글자 폭의 ~28% (매우 넓다)
- 단어 사이 공백: 폭 ~글자 크기의 0.7배
- 가장자리 점진 fade out (배경 검은색과 자연 합성)
"""
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
BG = "#000000"
FG = "#D2D2D2"          # 약간 회색 (원본 #C8C8C8 ~ #DBDBDB 평균)
GLOW_COLOR = "#888888"   # 외곽 글로우 (회색 약하게)
GLOW_RADIUS = 4          # blur 반경 (글자 외부 부드러움)

SMALL_SIZE = 52          # "DIRECTED BY" 작은 폰트
BIG_SIZE = 100           # 이름 큰 폰트
LETTER_SPACING_RATIO = 0.28  # 자간 28%
LINE_GAP = 55            # 두 줄 사이

FONT_CANDIDATES_CONDENSED_BOLD = [
    # 영화 크레딧 비슷 — HelveticaNeue Condensed Bold 최우선
    ("/System/Library/Fonts/HelveticaNeue.ttc", 4),    # Condensed Bold ★
    ("/System/Library/Fonts/HelveticaNeue.ttc", 9),    # Condensed Black
    ("/System/Library/Fonts/Avenir Next Condensed.ttc", 0),  # Bold
    ("/System/Library/Fonts/Avenir Next Condensed.ttc", 8),  # Heavy
    ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", 0),
]


def find_font(size: int, prefer_index: int = None):
    """Condensed Bold 후보 시도"""
    for path, idx in FONT_CANDIDATES_CONDENSED_BOLD:
        p = Path(path)
        if not p.exists():
            continue
        try:
            font = ImageFont.truetype(str(p), size, index=idx)
            return font
        except Exception:
            try:
                font = ImageFont.truetype(str(p), size)
                return font
            except Exception:
                continue
    return ImageFont.load_default()


def text_width_spaced(draw, text: str, font, spacing: int) -> int:
    if not text:
        return 0
    widths = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        widths.append(bbox[2] - bbox[0])
    return sum(widths) + spacing * (len(text) - 1)


def draw_spaced_centered_glow(base_img, text: str, font, color, center_x: int, baseline_y: int,
                                spacing: int, glow_color: str, glow_radius: int):
    """자간 + 외곽 글로우 + 중앙 정렬"""
    # 1) 글로우 레이어 (큰 stroke + blur)
    glow_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_layer)

    total = text_width_spaced(g_draw, text, font, spacing)
    x_start = center_x - total // 2

    # 글로우 stroke를 위해 약간 굵게
    x = x_start
    for ch in text:
        # stroke로 글자 그리기 (외곽 확장)
        g_draw.text((x, baseline_y), ch, font=font, fill=glow_color,
                     stroke_width=2, stroke_fill=glow_color)
        bbox = g_draw.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + spacing

    # blur
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))

    # 2) 글로우 합성
    base_img.paste(glow_layer, (0, 0), glow_layer)

    # 3) 본 텍스트 그리기 (밝은 색)
    main_draw = ImageDraw.Draw(base_img)
    x = x_start
    for ch in text:
        main_draw.text((x, baseline_y), ch, font=font, fill=color)
        bbox = main_draw.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + spacing


MAX_TEXT_WIDTH = int(W * 0.82)   # margin 9% 양쪽


def actual_text_width(draw, text: str, font, spacing: int) -> int:
    """실제 렌더 폭 — stroke margin 4 추가 (글로우 stroke_width=2 양쪽)"""
    return text_width_spaced(draw, text, font, spacing) + 4


def fit_big_font(draw, text: str, initial_spacing_ratio: float = LETTER_SPACING_RATIO):
    """size + spacing_ratio 동시 축소. 안전하게 글자 안 잘림"""
    text_upper = text.upper()
    size = BIG_SIZE
    spacing_ratio = initial_spacing_ratio
    while size >= 38:
        font = find_font(size)
        spacing = int(size * spacing_ratio)
        width = actual_text_width(draw, text_upper, font, spacing)
        if width <= MAX_TEXT_WIDTH:
            return font, spacing
        # 자간부터 먼저 축소 (가독성 유지)
        if spacing_ratio > 0.12:
            spacing_ratio -= 0.04
        else:
            size -= 4
            spacing_ratio = initial_spacing_ratio  # 자간 복귀
    # 최후 안전판
    font = find_font(38)
    spacing = int(38 * 0.12)
    return font, spacing


def make_card(top_label: str, name: str, out_path: Path):
    img = Image.new("RGB", (W, H), color=BG).convert("RGBA")
    draw = ImageDraw.Draw(img)

    small_font = find_font(SMALL_SIZE)
    big_font, spacing_big = fit_big_font(draw, name)

    spacing_small = int(SMALL_SIZE * LETTER_SPACING_RATIO)

    sbbox = draw.textbbox((0, 0), "Ag", font=small_font)
    s_h = sbbox[3] - sbbox[1]
    bbbox = draw.textbbox((0, 0), "Ag", font=big_font)
    b_h = bbbox[3] - bbbox[1]

    total_h = s_h + LINE_GAP + b_h
    top_y = (H - total_h) // 2 - sbbox[1]

    center_x = W // 2

    # 상단 작은 라벨
    draw_spaced_centered_glow(
        img, top_label.upper(), small_font, FG, center_x, top_y,
        spacing_small, GLOW_COLOR, GLOW_RADIUS
    )
    # 하단 큰 이름
    draw_spaced_centered_glow(
        img, name.upper(), big_font, FG,
        center_x, top_y + s_h + LINE_GAP - bbbox[1],
        spacing_big, GLOW_COLOR, GLOW_RADIUS
    )

    img.convert("RGB").save(out_path, "PNG")


def make_clip(director: str, writers: str, out_mp4: Path,
               card_duration: float = 2.0, fade: float = 0.5):
    tmp_dir = out_mp4.parent / f".{out_mp4.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    card1 = tmp_dir / "card1.png"
    card2 = tmp_dir / "card2.png"
    make_card("DIRECTED BY", director, card1)
    make_card("STORY BY", writers, card2)

    fade_out_start = card_duration - fade

    def png_to_clip(png_path, mp4_path):
        cmd = [
            "/opt/homebrew/bin/ffmpeg",
            "-loop", "1", "-t", str(card_duration),
            "-i", str(png_path),
            "-vf", f"fade=t=in:st=0:d={fade},fade=t=out:st={fade_out_start}:d={fade}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", "30", "-preset", "fast", "-y",
            str(mp4_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    clip1 = tmp_dir / "clip1.mp4"
    clip2 = tmp_dir / "clip2.mp4"
    png_to_clip(card1, clip1)
    png_to_clip(card2, clip2)

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
