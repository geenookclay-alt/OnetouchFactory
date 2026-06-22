"""Channel classifier — decide whether a candidate KR channel is on-DNA.

Logic: fetch ~30 recent shorts from the channel, ask Gemini Flash to classify
each as on-DNA (Y/N) in a single batched call. Channel is a reference iff
matching_count >= MIN_MATCHES OR matching_ratio >= MIN_RATIO.
"""
import json
from typing import Any
from . import llm
from . import youtube_client
from . import dissection as _dissection_mod


MIN_MATCHES = 5
MIN_RATIO = 0.30
SAMPLE_SIZE = 30


CLASSIFY_SYSTEM_PROMPT = """너는 쇼츠 결 판별 전문가야.
주어진 타겟 DNA와 영상 메타들을 보고, 각 영상이 그 DNA와 같은 결인지 Y/N으로만 답해.
설명/주석/마크다운 절대 X. 순수 JSON만."""


CLASSIFY_USER_TEMPLATE = """=== 타겟 DNA ===
{dna_summary}
주요 소재: {topics}
바이럴 요인: {viral_factors}
타겟층: {target}

=== 판별 영상 {n}개 ===
{video_table}

=== 출력 형식 ===
각 영상에 대해 같은 결인지 Y/N. JSON만:
{{"results": [{{"video_id": "...", "match": "Y", "reason": "한 줄"}}, ...]}}

기준:
- 같은 소재/구조/포맷이면 Y
- 단순 비슷한 키워드만으론 N
- AI/CGI/RC/어린이 영상은 무조건 N
- 편집본/모음/랭킹은 N"""


def _format_video_row(v: dict) -> str:
    title = (v.get("title") or "")[:80]
    caption = (v.get("caption") or "")[:120].replace("\n", " ")
    return f"- id={v.get('video_id')} | views={v.get('view_count', 0)} | dur={v.get('duration', 0)}s | title={title} | desc={caption}"


def _dna_summary_payload(dna: dict) -> dict:
    items = dna.get("items") or dna
    return {
        "dna_summary": dna.get("summary_kr", ""),
        "topics": (items.get("6_topics_list", {}) or {}).get("items", [])[:15],
        "viral_factors": (items.get("5_viral_reasons", {}) or {}).get("factors", []),
        "target": items.get("10_target_demographic", {}),
    }


async def classify_channel(channel_id: str, target_dna: dict,
                           sample_size: int = SAMPLE_SIZE) -> dict:
    """Return {is_reference, matching_count, matching_ratio, videos}."""
    videos = await youtube_client.get_channel_videos(channel_id,
                                                     max_results=sample_size)
    if not videos:
        return {"is_reference": False, "matching_count": 0,
                "matching_ratio": 0.0, "videos": [], "reason": "no_videos"}

    # Filter to shorts only (<= 60s) for fair classification
    shorts = [v for v in videos if 0 < (v.get("duration") or 0) <= 65][:sample_size]
    if not shorts:
        return {"is_reference": False, "matching_count": 0,
                "matching_ratio": 0.0, "videos": videos, "reason": "no_shorts"}

    payload = _dna_summary_payload(target_dna)
    prompt = CLASSIFY_USER_TEMPLATE.format(
        dna_summary=payload["dna_summary"],
        topics=", ".join(payload["topics"]),
        viral_factors=", ".join(payload["viral_factors"]),
        target=json.dumps(payload["target"], ensure_ascii=False),
        n=len(shorts),
        video_table="\n".join(_format_video_row(v) for v in shorts),
    )

    try:
        resp = await llm.gemini_chat(prompt, system=CLASSIFY_SYSTEM_PROMPT,
                                     model="gemini-3.5-flash",
                                     max_tokens=4096, json_mode=True)
        text = resp.text
    except Exception:
        resp = await llm.ollama_chat(prompt, system=CLASSIFY_SYSTEM_PROMPT,
                                     json_mode=True)
        text = resp.text

    parsed = _dissection_mod._safe_parse_json(text)
    results = []
    if isinstance(parsed, dict):
        results = parsed.get("results", []) or []

    by_id = {r.get("video_id"): r for r in results if isinstance(r, dict)}
    matching_count = 0
    enriched = []
    for v in shorts:
        vid = v.get("video_id")
        r = by_id.get(vid, {})
        match = (r.get("match") or "").upper().startswith("Y")
        if match:
            matching_count += 1
        enriched.append({
            **v,
            "match": match,
            "match_reason": r.get("reason", ""),
        })

    n = len(shorts)
    ratio = matching_count / n if n else 0.0
    is_ref = (matching_count >= MIN_MATCHES) or (ratio >= MIN_RATIO)
    return {
        "is_reference": is_ref,
        "matching_count": matching_count,
        "total_sampled": n,
        "matching_ratio": round(ratio, 3),
        "videos": enriched,
    }
