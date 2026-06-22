"""Multi-language keyword generator from a channel dissection."""
import json
import re
from . import llm


KEYWORD_SYSTEM_PROMPT = """You are a multi-platform keyword expert for short-form video discovery.
Given a channel's content DNA, generate viral search keywords in multiple languages.
Output JSON only — no commentary, no markdown."""


KEYWORD_USER_TEMPLATE = """Based on this channel's DNA analysis, generate **30 search keywords** for short-form video discovery on TikTok, Instagram Reels, and YouTube Shorts.

=== Channel DNA ===
{dna_payload}

=== Requirements ===
- 10 keywords in **English**
- 10 keywords in **Chinese (Simplified)**
- 10 keywords in **Japanese**

**KEYWORD ORDER (매우 중요)**:
- 1st keyword (each lang): CORE CATEGORY WORD (대키워드) — 카테 핵심 단어. 예: 레고 카테 → "lego" (en), "乐高" (zh), "レゴ" (ja)
- 2nd keyword: CORE 단어 + 짧은 modifier. 예: "lego shorts", "lego moc"
- 3rd~6th: 시그니처 키워드 (예: "lego speed build", "minifigure", "afol")
- 7th~10th: 결 specific 키워드 (예: "lego experiment", "lego review")

대키워드 (1st)는 broad search용 — 절대 빼지 마.

- Do NOT include "ranking", "top", "compilation", "best of"
- Mix: 대키워드 + topic words + situational + format-specific

=== Output Schema (JSON only) ===
{{
  "english": [
    {{"keyword": "gender reveal", "rationale_kr": "메인 키워드"}},
    {{"keyword": "balloon gender reveal", "rationale_kr": "풍선반전 결매칭"}},
    ...
  ],
  "chinese": [
    {{"keyword": "性别揭晓", "rationale_kr": "..."}}, ...
  ],
  "japanese": [
    {{"keyword": "ジェンダーリビール", "rationale_kr": "..."}}, ...
  ]
}}"""


async def generate_keywords(dissection: dict) -> dict:
    """Generate 30 multi-language keywords from a dissection result."""
    # Extract relevant fields to keep prompt compact
    items = dissection.get("items") or dissection.get("common_dna", {}).get("items", {})
    summary = dissection.get("summary_kr") or dissection.get("common_dna", {}).get("summary_kr", "")
    channel_name = (
        dissection.get("_meta", {}).get("channel_name")
        or dissection.get("channels", [{}])[0].get("_meta", {}).get("channel_name", "")
    )

    payload = {
        "channel_name": channel_name,
        "summary": summary,
        "topics": items.get("6_topics_list", {}).get("items", []),
        "topic_distribution": items.get("2_topic_distribution", {}),
        "viral_factors": items.get("5_viral_reasons", {}).get("factors", []),
        "view_drivers": items.get("4_view_drivers", {}).get("drivers", []),
        "megahit_pattern": items.get("13_megahit_pattern", {}),
    }

    prompt = KEYWORD_USER_TEMPLATE.format(
        dna_payload=json.dumps(payload, ensure_ascii=False)[:6000]
    )

    try:
        resp = await llm.gemini_chat(prompt, system=KEYWORD_SYSTEM_PROMPT,
                                     model="gemini-3.5-flash",
                                     max_tokens=8192, json_mode=True)
        text = resp.text.strip()
    except Exception:
        resp = await llm.ollama_chat(prompt, system=KEYWORD_SYSTEM_PROMPT,
                                     json_mode=True)
        text = resp.text

    # Use the same robust parser
    from . import dissection
    parsed = dissection._safe_parse_json(text)
    if isinstance(parsed, dict):
        return parsed
    return {
        "error": "JSON parse failed",
        "raw": text[:1500],
        "english": [], "chinese": [], "japanese": [],
    }


def flatten_keywords(kw_data: dict) -> list[str]:
    """Flatten the multi-language keyword dict into a single list of strings."""
    out: list[str] = []
    for lang in ("english", "chinese", "japanese"):
        for item in kw_data.get(lang, []):
            kw = item.get("keyword") if isinstance(item, dict) else item
            if kw and kw not in out:
                out.append(kw)
    return out


def ensure_core_keywords(kw_data: dict, cate_name: str) -> dict:
    """카테 이름에서 핵심 영어 단어 추출 → 1st 위치에 자동 추가 (LLM이 빠뜨려도).

    예: "🧱 1분LEGO 벤치마킹" → "lego" 추출 → english 0번째에 추가.
    """
    import re
    # 영문 4글자 이상 단어 추출 (lego, brick, dance 등)
    core_words = []
    text = cate_name or ""
    # 한국어 카테 → 영어 대응 매핑
    KR_TO_EN = {
        "레고": "lego", "브릭": "brick",
        "강아지 미용": "dog grooming", "미용": "grooming",
        "볼링": "bowling", "곤충": "insect",
        "도자기": "pottery", "클라이밍": "climbing",
        "말": "horse", "복원": "restoration",
        "비트박스": "beatbox", "마술": "magic",
        "모델": "model", "배드민턴": "badminton",
        "젤더 리빌": "gender reveal", "젤더리빌": "gender reveal",
        "젠더 리빌": "gender reveal", "젠더리빌": "gender reveal",
        "3d 프린팅": "3d printing", "3d": "3d printing",
    }
    cate_lower = text.lower()
    for kr, en in KR_TO_EN.items():
        if kr in cate_lower or kr.replace(" ", "") in cate_lower.replace(" ", ""):
            if en not in core_words:
                core_words.append(en)
    # 영문 단어 직접 추출
    for m in re.finditer(r"[a-zA-Z]{4,}", text):
        w = m.group().lower()
        if w not in core_words and w not in ("shorts", "쇼츠", "video", "channel"):
            core_words.append(w)
    if not core_words:
        return kw_data
    # english 리스트 앞에 추가 (중복 제거)
    en_list = kw_data.get("english") or []
    existing = {(i.get("keyword") if isinstance(i, dict) else str(i)).lower() for i in en_list}
    new_items = []
    for w in core_words:
        if w.lower() not in existing:
            new_items.append({"keyword": w, "rationale_kr": "카테 대키워드 (자동)"})
            existing.add(w.lower())
    kw_data["english"] = new_items + en_list
    return kw_data
