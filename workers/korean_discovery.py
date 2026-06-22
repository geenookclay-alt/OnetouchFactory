"""Korean channel discovery — DNA → KR keywords → YouTube search → channel pool.

Given a target DNA (the dissection items dict), generate ~10 Korean keywords,
search YouTube Shorts in the KR region for each, and return a deduped channel
pool of 50~100 entries.
"""
import asyncio
import json
from typing import Any
from . import llm
from . import youtube_client
from . import dissection as _dissection_mod  # for _safe_parse_json reuse


KR_KEYWORD_SYSTEM_PROMPT = """너는 한국어 쇼츠 발굴 전문가야.
주어진 채널 DNA를 보고 한국 YouTube 쇼츠에서 같은 결의 영상을 찾을 수 있는
**한국어 검색 키워드 10개**만 JSON 배열로 출력해. 설명/주석/마크다운 절대 X."""


KR_KEYWORD_USER_TEMPLATE = """다음 채널 DNA를 보고 같은 결의 한국 채널을 찾을
한국어 검색어 10개를 만들어줘.

=== 채널 DNA ===
{dna_payload}

=== 요구사항 ===
- 한국어로만 (영어/한자 X)
- 메인 소재 + 변형 표현 + 상황/감정어 + 포맷어 골고루 섞기
- "랭킹", "탑", "베스트", "모음" 같은 편집본 키워드 제외
- 중복 X
- 출력 우선순위: 적중률 높은 순

=== 출력 ===
{{"keywords": ["키워드1", "키워드2", ..., "키워드10"]}}"""


async def generate_korean_keywords(dna: dict, n: int = 10) -> list[str]:
    """DNA dict (dissection items) → list of Korean search keywords."""
    items = dna.get("items") or dna
    summary = dna.get("summary_kr", "")
    payload = {
        "summary": summary,
        "topics": (items.get("6_topics_list", {}) or {}).get("items", []),
        "viral_factors": (items.get("5_viral_reasons", {}) or {}).get("factors", []),
        "view_drivers": (items.get("4_view_drivers", {}) or {}).get("drivers", []),
        "megahit_pattern": items.get("13_megahit_pattern", {}),
        "target": items.get("10_target_demographic", {}),
    }
    prompt = KR_KEYWORD_USER_TEMPLATE.format(
        dna_payload=json.dumps(payload, ensure_ascii=False)[:5000]
    )
    try:
        resp = await llm.gemini_chat(prompt, system=KR_KEYWORD_SYSTEM_PROMPT,
                                     model="gemini-3.5-flash",
                                     max_tokens=2048, json_mode=True)
        text = resp.text
    except Exception:
        resp = await llm.ollama_chat(prompt, system=KR_KEYWORD_SYSTEM_PROMPT,
                                     json_mode=True)
        text = resp.text

    parsed = _dissection_mod._safe_parse_json(text)
    keywords: list[str] = []
    if isinstance(parsed, dict):
        kws = parsed.get("keywords") or []
        for kw in kws:
            if isinstance(kw, str) and kw.strip() and kw not in keywords:
                keywords.append(kw.strip())
    return keywords[:n]


async def discover_korean_channels(dna: dict,
                                   keywords: list[str] | None = None,
                                   per_keyword: int = 30,
                                   max_channels: int = 100) -> list[dict]:
    """Run KR keyword search and return a deduped channel pool.

    Each entry: {channel_id, handle, name, subs_estimate, sample_video_ids}.
    """
    if keywords is None:
        keywords = await generate_korean_keywords(dna)
    if not keywords:
        return []

    # Run searches in parallel
    tasks = [
        youtube_client.search_youtube(kw, max_results=per_keyword,
                                      duration="short", region="KR")
        for kw in keywords
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    channels: dict[str, dict] = {}
    for res in results:
        if isinstance(res, Exception) or not res:
            continue
        for v in res:
            ch_id = v.get("channel_id")
            if not ch_id:
                continue
            entry = channels.setdefault(ch_id, {
                "channel_id": ch_id,
                "name": v.get("channel_name", ""),
                "handle": "",
                "subs_estimate": 0,
                "sample_video_ids": [],
            })
            vid = v.get("video_id")
            if vid and vid not in entry["sample_video_ids"]:
                entry["sample_video_ids"].append(vid)
            if len(channels) >= max_channels * 2:
                break

    # Sort by sample_count desc (channels appearing in more searches = stronger signal)
    pool = sorted(channels.values(), key=lambda c: -len(c["sample_video_ids"]))
    return pool[:max_channels]
