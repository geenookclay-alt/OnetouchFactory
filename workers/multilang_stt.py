"""다국어 STT 워커 — Whisper + (비한국어시) Gemini Pro 합의.

흐름:
1. mlx-whisper로 1차 받아쓰기 + 언어 자동 감지
2. 한국어 → Whisper 단독 (추가 비용 0)
3. 비한국어(일본어/영어/기타) → Gemini Pro 추가 호출 + 차이 부분 LLM 중재

비용 (분당):
- 한국어: $0 (Whisper만)
- 비한국어: ~$0.003 (Gemini audio + 중재)
"""
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, ".")
import mlx_whisper  # fallback only
from workers.stt_client import transcribe as _stt
from workers.auto_subtitle import (
    call_gemini, upload_video_to_gemini, ensure_inline_video, GEMINI_PRO_MODEL,
)


MODEL_W = "mlx-community/whisper-large-v3-mlx"


def _safe_str(x) -> str:
    """text가 list/None/dict일 때도 안전 string 변환 (Whisper/Gemini 응답 안전)."""
    if isinstance(x, list):
        return " ".join(_safe_str(i) for i in x if i)
    if x is None:
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


def _drop_hallucination(segments: list[dict]) -> list[dict]:
    """Whisper large-v3 환각 cue 제거."""
    out = []
    prev_text = None
    prev_end = -10.0
    for s in segments:
        t = _safe_str(s.get("text")).strip()
        if not t:
            continue
        unique = set(t.replace(" ", "").replace("\n", ""))
        if len(unique) == 1 and len(t) >= 20:
            continue
        if prev_text and t == prev_text and (s["start"] - prev_end) < 1.5:
            continue
        out.append({**s, "text": t})
        prev_text = t
        prev_end = s["end"]
    return out


def transcribe_whisper(media_path: Path, lang: str | None = None) -> dict:
    """Whisper로 받아쓰기. lang=None이면 자동 감지. 환각 cue 자동 제거."""
    # STT: ElevenLabs scribe_v2 메인 + mlx_whisper failover
    r = _stt(str(media_path), language=lang)
    # 자막에 [효과음]/[화자라벨] 노출 방지
    import re as _re
    def _strip_tags(t):
        t = _re.sub(r"\[[^\]]+\]", " ", t)
        return _re.sub(r"\s+", " ", t).strip()
    segs = []
    for s in r.get("segments", []):
        t = _strip_tags(_safe_str(s.get("text")))
        if not t:
            continue
        segs.append({"start": s["start"], "end": s["end"], "text": t})
    segs = _drop_hallucination(segs)
    return {"language": r.get("language", "unknown"), "segments": segs}


GEMINI_TRANSCRIBE_PROMPT = """이 영상의 음성을 정확히 받아쓰기 해줘.

🚨 절대 룰:
- **출연자가 입으로 말한 대사만** 받아써. (괄호) 안에 리액션·웃음·추임새 묘사 절대 X.
  ❌ "(ㅋㅋㅋ)", "(맞아 맞아)", "(당연ㅋㅋ)", "(웃음)", "(박수)" 같은 묘사 X
  ❌ "...", "ㅡ", "—" 같은 양식 장식 X
  ✅ 출연자가 "맞아 맞아"라고 진짜 말했으면 그냥 "맞아 맞아" 라고 박기 (괄호 X)
- 의역 X, 번역 X — 원어 그대로
- 어투/반말/줄임말 보존
- 마침표(.) 박지 마라. 물음표(?) 느낌표(!)는 OK
- **한 segment 짧게**: 1~3초 단위, 한 cue 텍스트는 24자 이하 권장
- 긴 발화는 여러 segment로 나눠
- 각 segment 시작/끝 시간 정확히 (초 단위 소수 2자리)

[출력 JSON만]
{
  "language": "ko" | "ja" | "en" | "...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "받아쓰기 (짧게, 괄호 묘사 X)"}
  ]
}
"""


async def transcribe_gemini(media_path: Path) -> dict:
    """Gemini Pro로 영상 받아쓰기. JSON 반환."""
    inline_vid = await ensure_inline_video(media_path)
    file_uri = await upload_video_to_gemini(inline_vid)
    data = await call_gemini(GEMINI_PRO_MODEL, file_uri,
                              GEMINI_TRANSCRIBE_PROMPT, temperature=0.1)
    return data if isinstance(data, dict) else {"segments": []}


ARBITRATION_PROMPT = """두 다른 STT 엔진이 같은 영상의 음성을 받아썼는데 일부 단어가 다르게 나왔어. 영상의 실제 음성을 듣고 정확한 transcript를 만들어줘.

엔진 A (Whisper) 결과:
{whisper_text}

엔진 B (Gemini) 결과:
{gemini_text}

규칙:
- 두 결과를 비교해서 영상 음성에 가장 정확한 transcript를 만든다
- 일치하는 부분은 그대로, 다른 부분은 영상 들어보고 정답 선택
- 시간(start/end)은 더 정확한 쪽 (보통 Whisper) 따른다
- 원어 그대로 (번역 X)

[출력 JSON만]
{{
  "segments": [
    {{"start": 0.0, "end": 2.5, "text": "정답"}}
  ]
}}
"""


async def arbitrate(whisper_result: dict, gemini_result: dict,
                     media_path: Path) -> list[dict]:
    """두 결과를 영상과 함께 LLM에 던져 합의 transcript 받음."""
    w_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        for s in whisper_result["segments"]
    )
    g_segs = gemini_result.get("segments", []) or []
    g_text = "\n".join(
        f"[{s.get('start', 0):.1f}-{s.get('end', 0):.1f}] {s.get('text', '')}"
        for s in g_segs
    )
    inline_vid = await ensure_inline_video(media_path)
    file_uri = await upload_video_to_gemini(inline_vid)
    prompt = ARBITRATION_PROMPT.format(whisper_text=w_text, gemini_text=g_text)
    data = await call_gemini(GEMINI_PRO_MODEL, file_uri, prompt, temperature=0.1)
    if isinstance(data, dict):
        return data.get("segments", []) or []
    return []


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t * 1000) % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 괄호 묘사 / 양식 장식 정리 패턴
import re as _re

_PARENS_RE = _re.compile(r"\([^)]*\)|\[[^\]]*\]")  # (), [] 안 내용 제거
_TRAILING_DOTS_RE = _re.compile(r"[\.…]+$")
_MID_DOTS_RE = _re.compile(r"\.{2,}")  # ... 같은 줄임표 → 단일 ?


def _clean_text(text) -> str:
    """괄호 묘사 / 양식 장식 / 마침표 제거. text가 list/dict일 때도 안전 처리."""
    t = _safe_str(text).strip()
    # () [] 안 내용 통째로 제거 (출연자 입에서 안 나온 묘사)
    t = _PARENS_RE.sub("", t)
    # ... → 제거 (말줄임표는 어색)
    t = _MID_DOTS_RE.sub("", t)
    # 모든 마침표 제거 (대표님 룰)
    t = t.replace(".", "")
    # 양끝 공백 정리
    t = _re.sub(r"\s+", " ", t).strip()
    return t


def _fill_cue_gaps(segments: list[dict], max_gap: float = 1.5,
                    end_margin: float = 0.05) -> list[dict]:
    """cue end를 다음 cue start까지 연장 (작은 gap만 fill — 자막 깜빡임 방지).
    큰 무음(>1.5초)은 그대로 유지 → 발화 끝나면 자막 사라짐."""
    if not segments:
        return segments
    out = []
    for i, seg in enumerate(segments):
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s + 1))
        if i < len(segments) - 1:
            next_s = float(segments[i + 1].get("start", e))
            gap = next_s - e
            if 0 < gap <= max_gap:
                e = next_s - end_margin
        out.append({**seg, "end": e})
    return out


def _post_process(segments: list[dict]) -> list[dict]:
    """괄호 제거 + 12자/2줄 분할 + cue gap fill (자막 깜빡임 방지)."""
    from workers.audio_subtitle import _split_cue as _sc
    out = []
    for seg in segments:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s + 1))
        t = _clean_text(seg.get("text", "") or "")
        if not t:
            continue
        # _split_cue: 너무 길면 24자×2줄 분할 + 시간 비례
        for ss, ee, tt in _sc(s, e, t):
            out.append({"start": ss, "end": ee, "text": tt})
    # gap fill — 작은 cue 사이 빈공간 메움 (1.5초 이내)
    out = _fill_cue_gaps(out)
    return out


def _write_srt(segments: list[dict], out_path: Path) -> None:
    """후처리(괄호/마침표 제거 + 12자×2줄 분할) → SRT."""
    cleaned = _post_process(segments)
    lines = []
    for i, seg in enumerate(cleaned, 1):
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s + 1))
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        lines.append(str(i))
        lines.append(f"{_ts(s)} --> {_ts(e)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


async def transcribe_with_validation(media_path,
                                       out_srt,
                                       lang_hint: str | None = None,
                                       force_validation: bool = True) -> dict:
    """다국어 STT + 합의 흐름 (옵션 A — 한국어 포함 항상 Gemini 합의).

    Whisper(시간 정확) + Gemini Pro(텍스트 정확, 환각 없음) → 중재 시도.
    중재 실패 시 Gemini Pro fallback (환각 누락 케이스 잡힘).

    force_validation=False 주면 한국어는 Whisper 단독 (비용 0, 옛 동작).
    """
    media_path = Path(media_path)
    out_srt = Path(out_srt)
    out_srt.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [multilang_stt] Whisper 1차 받아쓰기...", flush=True)
    w = transcribe_whisper(media_path, lang=lang_hint)
    lang = w["language"]
    print(f"  [multilang_stt] 언어 감지: {lang} ({len(w['segments'])} segments)",
          flush=True)

    if lang == "ko" and not force_validation:
        print(f"  [multilang_stt] 한국어 → Whisper 단독 (force_validation=False)", flush=True)
        _write_srt(w["segments"], out_srt)
        return {"language": lang, "segments": w["segments"], "validated": False}

    print(f"  [multilang_stt] {lang} → Gemini Pro 합의 호출", flush=True)
    try:
        g = await transcribe_gemini(media_path)
        print(f"  [multilang_stt] Gemini: {len(g.get('segments', []))} segments",
              flush=True)
        final_segs = await arbitrate(w, g, media_path)
        if not final_segs:
            print(f"  [multilang_stt] ⚠️ 중재 실패 — Gemini fallback", flush=True)
            final_segs = g.get("segments", []) or w["segments"]
        else:
            print(f"  [multilang_stt] 중재 완료: {len(final_segs)} segments",
                  flush=True)
        _write_srt(final_segs, out_srt)
        return {"language": lang, "segments": final_segs, "validated": True}
    except Exception as e:
        print(f"  [multilang_stt] ⚠️ Gemini 합의 실패 — Whisper fallback: {e}",
              flush=True)
        _write_srt(w["segments"], out_srt)
        return {"language": lang, "segments": w["segments"],
                "validated": False, "error": str(e)}


if __name__ == "__main__":
    # CLI: python -m workers.multilang_stt <media> <out_srt> [--force]
    async def _cli():
        mp = sys.argv[1]
        op = sys.argv[2]
        force = "--force" in sys.argv
        r = await transcribe_with_validation(mp, op, force_validation=force)
        print(json.dumps({
            "language": r["language"],
            "n_segs": len(r["segments"]),
            "validated": r["validated"],
        }, ensure_ascii=False))
    asyncio.run(_cli())
