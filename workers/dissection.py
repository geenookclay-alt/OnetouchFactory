"""Channel dissection — extract 13-item DNA from a YouTube channel.

Workflow:
  1. Fetch all/top videos via YouTube Data API
  2. Build aggregated payload (titles + view counts + durations + descriptions)
  3. Send to Gemini 2.5 Flash for full 13-item analysis
  4. Return structured dissection dict
"""
import os
import json
import re
import asyncio
from typing import Any
from . import youtube_client
from . import llm


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


DISSECTION_SYSTEM_PROMPT = """너는 YouTube/TikTok/Instagram 쇼츠 채널을 "참치 해체"하듯 완전히 분석하는 전문 분석가야.
주어진 채널의 영상 데이터를 보고 **반드시 한국어로** 13개 항목을 깊이 있게 분석해서 JSON으로만 출력해.
설명, 주석, 마크다운 절대 X. 순수 JSON만."""


DISSECTION_USER_TEMPLATE = """다음은 채널 "{channel_name}"의 분석 대상 영상 {n_videos}개야.
조회수 상위순으로 정렬되어있어.

=== 영상 데이터 ===
{video_table}

=== 채널 메타 ===
구독자: {subs}
총 영상 수: {total_videos}
국가: {country}
채널 설명: {channel_desc}

=== 분석 요청 ===
이 채널을 13개 항목으로 "참치 해체"해. 메가히트 영상(상위 20%)에 가중치를 둬서 분석해.
JSON 스키마:

{{
  "channel_name": "채널 이름",
  "summary_kr": "한 줄 요약 (이 채널이 무엇으로 터졌는지)",
  "items": {{
    "1_narrative_structure": {{
      "label": "기승전결 구조",
      "value": "기/승/전/결 각 단계 한 줄로",
      "examples": ["메가히트 영상 1개 제목 + 그 구조 적용"]
    }},
    "2_topic_distribution": {{
      "label": "주제 분포",
      "value": "주요 주제 카테고리들",
      "breakdown": [{{"category": "이름", "count": 5, "percentage": 25}}]
    }},
    "3_writing_formula": {{
      "label": "터지는 글쓰기 조건",
      "value": "공통 카피라이팅 패턴",
      "patterns": ["반복되는 표현 1", "반복되는 표현 2"]
    }},
    "4_view_drivers": {{
      "label": "조회수 견인 요소",
      "value": "어떤 요소가 조회수 폭발시키는지",
      "drivers": ["요소 1 (평균 X M)", "요소 2 (평균 Y M)"]
    }},
    "5_viral_reasons": {{
      "label": "바이럴 이유",
      "value": "왜 바이럴 됐는지 핵심 진단",
      "factors": ["요인 1", "요인 2", "요인 3"]
    }},
    "6_topics_list": {{
      "label": "소재 리스트",
      "value": "사용된 모든 구체 소재들",
      "items": ["소재1", "소재2", "..."]
    }},
    "7_hook_structure": {{
      "label": "후킹 구성",
      "value": "도입 0~5초 패턴",
      "patterns": ["0~2초: ...", "3~5초: ..."]
    }},
    "8_tone_voice": {{
      "label": "말투/톤",
      "value": "전반적 화법",
      "characteristics": ["특징 1", "특징 2"]
    }},
    "9_recurring_phrases": {{
      "label": "공통 대사",
      "value": "반복되는 표현/어미/대사",
      "phrases": ["표현 1", "표현 2"]
    }},
    "10_target_demographic": {{
      "label": "타겟 성별/연령대",
      "value": "주 타겟층",
      "primary_gender": "남성/여성/혼합",
      "primary_age": "예: 25~40",
      "interests": ["관심사 1", "관심사 2"]
    }},
    "11_conjunctions": {{
      "label": "자주 쓰는 접속사/연결어",
      "value": "공통 연결어",
      "conjunctions": ["근데", "그런데", "..."]
    }},
    "12_emotional_arc": {{
      "label": "감정 유발 구조",
      "value": "감정 트리거 단계",
      "arc": ["호기심", "긴장", "반전", "안도"]
    }},
    "13_megahit_pattern": {{
      "label": "메가히트 공통 패턴",
      "value": "조회수 상위 영상의 공통점",
      "common_traits": ["특징 1", "특징 2", "특징 3"]
    }}
  }}
}}

JSON만 출력. 다른 텍스트 절대 X."""


def _safe_parse_json(text: str) -> Any:
    """Robust JSON parsing with multiple fallback strategies."""
    if not text:
        return None
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract balanced { ... } block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        # Try truncating to last balanced brace
        depth = 0
        last_close = -1
        for i, ch in enumerate(block):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_close = i
        if last_close > 0:
            try:
                return json.loads(block[: last_close + 1])
            except json.JSONDecodeError:
                pass
    return None


def _build_video_table(videos: list[dict], max_n: int = 50) -> str:
    """Build a compact table of videos for the LLM prompt."""
    rows = []
    for i, v in enumerate(videos[:max_n], 1):
        title = (v.get("title") or "")[:120]
        views = v.get("view_count") or 0
        dur = v.get("duration") or 0
        rows.append(f"{i}. [{views:>10,}뷰 · {dur}s] {title}")
    return "\n".join(rows)


async def fetch_channel_data(channel_url_or_id: str, max_videos: int = 50) -> dict | None:
    """Resolve channel URL/handle → channelId and fetch videos + metadata.
    youtube_client에 위임 (key round-robin/fallback 적용된 함수들 사용)."""
    handle = youtube_client.extract_channel_handle(channel_url_or_id)

    # 1. handle/URL → channelId (key round-robin)
    channel_id = await youtube_client.resolve_channel_id(channel_url_or_id)
    if not channel_id:
        print(f"⚠️ fetch_channel_data: channelId 못 찾음: {channel_url_or_id}", flush=True)
        return None

    # 2. channel meta (key round-robin)
    meta = (await youtube_client.get_channel_meta(channel_id)) or {}

    # 3. videos (인기순 default, key round-robin 이미 들어있음)
    videos = await youtube_client.get_channel_videos(channel_id, max_results=max_videos,
                                                     order="viewCount")
    # 안전망 — view count desc
    videos.sort(key=lambda v: v.get("view_count") or 0, reverse=True)

    return {
        "channel_id": channel_id,
        "channel_name": meta.get("snippet", {}).get("title", handle),
        "channel_desc": (meta.get("snippet", {}).get("description") or "")[:500],
        "country": meta.get("snippet", {}).get("country", ""),
        "subs": int(meta.get("statistics", {}).get("subscriberCount", 0)),
        "total_videos": int(meta.get("statistics", {}).get("videoCount", 0)),
        "videos": videos,
    }


def _filter_videos_by_topic(videos: list[dict], topic_hint: str) -> tuple[list[dict], int]:
    """주제 hint 키워드 (콤마 구분)로 영상 필터링.
    title + caption + channel_name 메타에 키워드 하나라도 포함 → 통과.
    return (필터된 영상 list, 원본 수)
    """
    if not topic_hint or not topic_hint.strip():
        return videos, len(videos)
    keywords = [k.strip().lower() for k in topic_hint.split(",") if k.strip()]
    if not keywords:
        return videos, len(videos)
    filtered = []
    for v in videos:
        blob = " ".join([
            str(v.get("title") or ""),
            str(v.get("caption") or "")[:500],
            str(v.get("channel_name") or ""),
        ]).lower()
        if any(kw in blob for kw in keywords):
            filtered.append(v)
    return filtered, len(videos)


async def dissect_channel(channel_url_or_id: str, max_videos: int = 50,
                          topic_hint: str = "") -> dict | None:
    """Run full dissection on a single channel. Returns the 13-item analysis.

    topic_hint: 콤마 구분 키워드. 채널 영상 중 키워드 매칭되는 영상만 분석에 사용.
                빈 문자열이면 모든 영상 사용 (기존 동작).
    """
    # topic_hint 쓰면 메타 필터 위해 max_videos 키워서 가져옴 (default 200)
    fetch_n = max(max_videos * 4, 200) if topic_hint else max_videos
    data = await fetch_channel_data(channel_url_or_id, max_videos=fetch_n)
    if not data or not data.get("videos"):
        return None

    filtered_videos, total_n = _filter_videos_by_topic(data["videos"], topic_hint)
    # 필터 결과 5개 미만이면 fallback (원본 전체 사용 + 경고)
    if topic_hint and len(filtered_videos) < 5:
        print(f"⚠️ topic_hint '{topic_hint[:50]}' 필터 결과 {len(filtered_videos)}개 (5개 미만) — fallback to 전체", flush=True)
        filtered_videos = data["videos"][:max_videos]
        filter_used = False
    else:
        filtered_videos = filtered_videos[:max_videos]
        filter_used = bool(topic_hint and len(filtered_videos) >= 5)
    data["videos"] = filtered_videos

    video_table = _build_video_table(data["videos"], max_n=max_videos)
    prompt = DISSECTION_USER_TEMPLATE.format(
        channel_name=data["channel_name"],
        n_videos=len(data["videos"]),
        video_table=video_table,
        subs=data["subs"],
        total_videos=data["total_videos"],
        country=data["country"] or "?",
        channel_desc=data["channel_desc"] or "(없음)",
    )

    # Use Gemini 2.5 Flash (fast, cheap, large context, native JSON mode)
    try:
        resp = await llm.gemini_chat(prompt, system=DISSECTION_SYSTEM_PROMPT,
                                     model="gemini-3.5-flash",
                                     max_tokens=16384, json_mode=True)
        text = resp.text.strip()
    except Exception as e:
        # Fallback to Ollama Qwen 14B
        resp = await llm.ollama_chat(prompt, system=DISSECTION_SYSTEM_PROMPT,
                                     json_mode=True)
        text = resp.text

    result = _safe_parse_json(text)
    if not isinstance(result, dict):
        return {"error": "JSON parse failed",
                "raw": text[:3000] if isinstance(text, str) else str(text)[:3000]}

    result["_meta"] = {
        "channel_id": data["channel_id"],
        "channel_name": data["channel_name"],
        "subs": data["subs"],
        "total_videos": data["total_videos"],
        "scanned_videos": len(data["videos"]),
        "country": data["country"],
        "topic_hint": topic_hint or None,
        "topic_filter_applied": filter_used,
        "videos_before_filter": total_n if topic_hint else None,
    }
    return result


async def dissect_multiple(channel_urls: list[str], max_videos_each: int = 30,
                            topic_hint: str = "") -> dict:
    """Dissect multiple reference channels and synthesize common DNA.

    topic_hint 전달 시 각 채널에서 해당 주제 영상만 골라 분석.
    """
    if not channel_urls:
        return {"channels": [], "common_dna": None}

    # Run dissections in parallel
    results = await asyncio.gather(
        *[dissect_channel(url, max_videos=max_videos_each, topic_hint=topic_hint)
          for url in channel_urls],
        return_exceptions=True,
    )
    valid = [r for r in results if isinstance(r, dict) and "error" not in r]

    if len(valid) <= 1:
        return {
            "channels": valid,
            "common_dna": valid[0] if valid else None,
        }

    # Synthesize common DNA across multiple channels
    summary_payload = json.dumps([
        {"channel": r.get("_meta", {}).get("channel_name"),
         "items": r.get("items", {})}
        for r in valid
    ], ensure_ascii=False)[:8000]

    synth_prompt = f"""다음은 {len(valid)}개 채널의 해체 분석 결과들이야.
이들의 공통 DNA를 뽑아서 단일 통합 분석으로 합쳐줘.
형식은 동일한 13개 항목 JSON. 다만 각 항목 끝에 "channels_with_this": [채널명들] 추가.

=== 입력 ===
{summary_payload}

=== 출력 형식 ===
{{ "summary_kr": "...", "items": {{ "1_narrative_structure": {{...}}, ... }} }}

JSON만 출력."""

    try:
        resp = await llm.gemini_chat(
            synth_prompt, system=DISSECTION_SYSTEM_PROMPT,
            model="gemini-3.5-flash", max_tokens=16384, json_mode=True,
        )
        common = _safe_parse_json(resp.text)
        if common is None:
            common = {"error": "Synthesis JSON parse failed",
                      "raw": resp.text[:2000]}
    except Exception as e:
        common = {"error": f"Synthesis failed: {e}",
                  "channels": [r.get("_meta") for r in valid]}

    return {"channels": valid, "common_dna": common}
