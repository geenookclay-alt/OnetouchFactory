"""캐리커처 워커 — 영상에서 주요 등장 인물 자동 식별 → 인물별 동화책 삽화체 캐리커처 PNG 생성.

흐름:
1. Gemini Pro로 영상 분석 → 등장 인물 N명 (최대 3명) + best frame 시점 + 외모 특징
2. ffmpeg으로 인물별 best frame 추출
3. Nano Banana(Kie)로 사진 → 캐리커처 PNG 변환
4. characters/{idx}_{role}.png 저장

대표님 룰:
- 한 영상 캐리커처 최대 3장 (2명이면 2장, 3+명이면 주요 3명)
- 동화책 삽화체 (영상2 reference 결)
- 흰 배경 (캡컷에서 chromakey로 따거나 그대로 사용)
"""
import json
import re
import sys
import asyncio
import subprocess
from pathlib import Path

sys.path.insert(0, ".")
from workers.auto_subtitle import (
    call_gemini, upload_video_to_gemini, ensure_inline_video, GEMINI_PRO_MODEL,
)
from workers.kie_client import nano_banana_img2img


CHARACTER_ANALYSIS_PROMPT = """이 영상에서 등장하는 주요 인물들을 분석해줘.
캐리커처 일러스트를 만들 거니까 정확히 식별이 필요해.

규칙:
- **최대 __MAXCHARS__명**까지만 (주요 인물 위주)
- 영상에 인물이 적게 나오면 나온 인원만큼만
- 인물이 많으면 가장 주요한 __MAXCHARS__명만 (가장 자주 잡히는 / 핵심 화자 / 주인공)
- 각 인물의 **best frame 시점** 1개씩 (얼굴 가장 잘 보이는, 정면 가까운, 중심에 있는 시점 — 초 단위 소수 1자리)
- 옷색·외모·헤어·안경·수염 등 **특징 정확히** 기록

[출력 JSON만]
{
  "characters": [
    {
      "role": "MC 또는 게스트 또는 호스트 등 역할",
      "name_or_nickname": "이름 모르면 '안경 쓴 게스트' 같은 별명",
      "appearance": "옷색·헤어·안경·수염·체형 등 외모 한 줄",
      "distinctive_features": "이 사람만의 특징 (예: 폭탄머리/뿔테 안경/콧수염)",
      "best_frame_sec": 25.5
    }
  ]
}

[중요] JSON만 출력. 인물 정확히 식별. best_frame_sec은 그 인물이 정면으로 가장 잘 보이는 시점."""


# 대표님 가이드: 단순 prompt + 원본 보존 강조 (자세/표정/옷 그대로, 그림체만 변환).
# 복잡한 prompt는 모델이 자세·표정 임의 변경 → 단순할수록 원본 보존 잘 됨.
CARICATURE_PROMPT_TPL = (
    "Turn this photo into a children's book watercolor illustration. "
    "Cut out only the person: {appearance}. "
    "Keep the exact same pose, expression, hair, and clothing as in the photo. "
    "White background. No other people, no text."
)


async def analyze_characters(video_path: Path, max_chars: int = 3) -> list[dict]:
    """영상 → 주요 등장 인물 (최대 max_chars명) + best frame 시점 + 특징.

    max_chars: 예능/드라마=3 (기존), 영화=6 (출연자 많음, 대표님 2026-05-31)."""
    analysis_video = await ensure_inline_video(video_path)
    file_uri = await upload_video_to_gemini(analysis_video)
    prompt = CHARACTER_ANALYSIS_PROMPT.replace("__MAXCHARS__", str(max_chars))
    data = await call_gemini(GEMINI_PRO_MODEL, file_uri,
                              prompt, temperature=0.2)
    if not isinstance(data, dict):
        return []
    chars = data.get("characters") or []
    return chars[:max_chars]


def extract_face_frame(video_path: Path, sec: float, out_jpg: Path) -> None:
    """ffmpeg로 특정 시점 frame 추출 (1920x1080 또는 원본 해상도)."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(max(0.0, sec)), "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(out_jpg),
    ], check=True, capture_output=True)


async def generate_caricature(face_jpg: Path, out_png: Path,
                                character: dict) -> dict:
    """Nano Banana로 동화책 일러스트 PNG 생성. 원본 자세/표정/옷 보존."""
    appearance = character.get("appearance") or character.get("name_or_nickname") or "person"
    prompt = CARICATURE_PROMPT_TPL.format(appearance=appearance)
    return await nano_banana_img2img(
        image_path=face_jpg,
        out_png=out_png,
        prompt=prompt,
        aspect_ratio="3:4",  # 인물 비율에 더 fit
    )


_SAFE_NAME_RE = re.compile(r"[^\w가-힣]+")


def _safe_filename(role: str, idx: int) -> str:
    s = _SAFE_NAME_RE.sub("_", role or "")[:20].strip("_")
    return s or f"char{idx}"


async def run_character_generation(video_path: Path,
                                    out_dir: Path,
                                    max_chars: int = 3) -> list[dict]:
    """전체 흐름: 등장 인물 분석 → 캐리커처 생성. characters/ 폴더에 PNG 저장.

    max_chars: 캐리커처 최대 인원 (예능/드라마=3, 영화=6).
    반환: [{"role", "name_or_nickname", "png", "best_frame_sec", ...}, ...]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        chars = await analyze_characters(video_path, max_chars=max_chars)
    except Exception as e:
        print(f"  ⚠️ 캐리커처 인물 분석 실패: {e}", flush=True)
        return []
    if not chars:
        print(f"  ⚠️ 인물 분석 결과 0명 — 캐리커처 스킵", flush=True)
        return []

    print(f"  캐리커처 인물 {len(chars)}명 식별", flush=True)
    results = []
    for i, ch in enumerate(chars, 1):
        try:
            sec = float(ch.get("best_frame_sec") or 0)
            safe = _safe_filename(ch.get("role", ""), i)
            # 임시 frame은 한글 없이 ASCII만 (fal storage 한글 파일명 못 다룸)
            face_jpg = out_dir / f"_face_{i:02d}.jpg"
            extract_face_frame(video_path, sec, face_jpg)
            png = out_dir / f"{i:02d}_{safe}.png"
            r = await generate_caricature(face_jpg, png, ch)
            face_jpg.unlink(missing_ok=True)
            results.append({
                **ch,
                "png": str(png),
                "wall_sec": r.get("wall_sec"),
                "cost_credits": r.get("cost_credits"),
            })
            print(f"    ✅ {i}/{len(chars)} {ch.get('role')} → {png.name} "
                  f"({r.get('wall_sec'):.1f}s, {r.get('cost_credits')} cr)",
                  flush=True)
        except Exception as e:
            print(f"    ❌ {i}/{len(chars)} 실패: {e}", flush=True)
            results.append({**ch, "error": str(e)})
    return results


if __name__ == "__main__":
    # CLI: python -m workers.character_generator <video> <out_dir>
    async def _cli():
        vp = Path(sys.argv[1])
        od = Path(sys.argv[2])
        r = await run_character_generation(vp, od)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    asyncio.run(_cli())
