"""자막 메뉴 잡에 효과음(SFX) mp3 자동 첨부.

run_auto_subtitle 끝에 호출:
1. 영상 분석 (효과음 시점)
2. 라이브러리 메타에서 효과음 매칭
3. ffmpeg으로 효과음 mp3 생성 → out_dir/bgm_mix.mp3

라이브러리 메타 없으면 (학습 전) 조용히 스킵. (음악 없음 — 효과음만, 대표님 0614)
"""
import json
from pathlib import Path

from workers.auto_subtitle import (
    call_gemini, upload_video_to_gemini, ensure_inline_video, GEMINI_PRO_MODEL,
)
from workers.bgm_sfx_selector import (
    load_meta, match_bgm, match_sfx_multi, make_bgm_sfx_mix,
)


MOOD_SFX_PROMPT = """이 영상의 자막 옆에 깔 효과음(SFX) 자동 매칭용 분석.

[JSON 출력만]
{
  "duration_sec": 0.0,
  "video_genre_hint": "예능/정보/감성/뷰티/동물/액션/푸드/일상 등 중에서 1",
  "mood_keywords": ["귀여움/감동/슬픔/긴장/공포/멋짐/신남/잔잔/벅참/유머/엉뚱/매드무비 중 3개"],
  "sfx_points": [
    {"time_sec": 2.0, "action": "어떤 액션/리액션", "sfx_keyword": "팝/두둥탁/와우/짝/와장창/퍽/지이잉/오우예/타다/멈춰/등장/꿀꿀/박수 등에서"}
  ]
}

규칙:
- mood_keywords 정확히 3개 (효과음 톤 매칭용)
- sfx_points 5~8개 (액션 시점에)
- 일반 한국 의성어 풀에서 sfx_keyword 선택
"""


async def attach_bgm_mix(video_path: Path, out_dir: Path) -> dict | None:
    """자막 잡에 효과음 믹스 mp3 첨부. 메타 없으면 None.

    Returns: {"path", "sfx_count", "duration", "sfx_points"} or None
    """
    meta = load_meta()
    if not meta:
        print("  ⚠️ 효과음 라이브러리 메타 없음 (학습 전) — 효과음 스킵",
              flush=True)
        return None

    print("  효과음 매칭용 영상 분석...", flush=True)
    inline = await ensure_inline_video(Path(video_path))
    file_uri = await upload_video_to_gemini(inline)
    data = await call_gemini(GEMINI_PRO_MODEL, file_uri,
                              MOOD_SFX_PROMPT, temperature=0.2)
    if not isinstance(data, dict):
        return None
    mood = data.get("mood_keywords", []) or []
    sfx_points = data.get("sfx_points", []) or []
    dur = float(data.get("duration_sec") or 0)

    sfx_matches = match_sfx_multi(sfx_points, meta, limit=8)
    print(f"  효과음 {len(sfx_matches)}개 자동 매칭", flush=True)

    out_mp3 = Path(out_dir) / "bgm_mix.mp3"
    result = await make_bgm_sfx_mix(
        duration_sec=dur,
        bgm_match=None,   # 음악 미사용 — 효과음만
        sfx_matches=sfx_matches,
        out_mp3=out_mp3,
    )
    result["sfx_points"] = sfx_points
    return result
