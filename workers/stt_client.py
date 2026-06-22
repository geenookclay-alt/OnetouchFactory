"""공통 STT 클라이언트 — ElevenLabs Scribe v2 메인 + mlx_whisper failover.

호출자가 mlx_whisper.transcribe()와 같은 형식의 결과를 받도록 변환:
    {
        "text": str,
        "language": str (ko/en/ja...),
        "segments": [{"start", "end", "text", "words": [{"word","start","end"}]}],
        # Eleven 전용 보너스 필드 (있을 때만):
        "speakers": ["speaker_0", "speaker_1", ...],
        "audio_events": [{"type", "text", "start", "end"}],
        "engine": "elevenlabs" | "whisper",
    }

환경변수:
    STT_ENGINE = "elevenlabs" (default) | "whisper" | "auto"
        - "elevenlabs": Eleven 메인, 실패 시 whisper fallback
        - "whisper": whisper 단독
        - "auto" = "elevenlabs"와 동일
    ELEVENLABS_API_KEY: 필수 (Eleven 사용 시)
"""
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ELEVEN_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVEN_MODEL_DEFAULT = "scribe_v2"
WHISPER_MODEL_DEFAULT = "mlx-community/whisper-large-v3-mlx"

# 언어 코드 변환 (mlx_whisper ISO 639-1 → ElevenLabs ISO 639-3)
_LANG_MAP_TO_ELEVEN = {
    "ko": "kor", "en": "eng", "ja": "jpn", "zh": "zho",
    "es": "spa", "fr": "fra", "de": "deu", "ru": "rus",
    "pt": "por", "it": "ita", "vi": "vie", "th": "tha",
    "id": "ind", "ar": "ara", "hi": "hin",
}
_LANG_MAP_FROM_ELEVEN = {v: k for k, v in _LANG_MAP_TO_ELEVEN.items()}


def _get_env_key() -> str | None:
    """우선순위: 환경변수 → .env 파일."""
    k = os.environ.get("ELEVENLABS_API_KEY")
    if k:
        return k.strip()
    try:
        env = Path.home() / "banbaji-discover" / ".env"
        if env.exists():
            m = re.search(r"^ELEVENLABS_API_KEY=(\S+)", env.read_text(encoding="utf-8"), re.M)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return None


def _eleven_to_whisper_format(d: dict) -> dict:
    """ElevenLabs 응답 → mlx_whisper.transcribe() 호환 dict."""
    words_all = d.get("words", []) or []
    word_items = [w for w in words_all if w.get("type") == "word"]
    audio_events = [
        {"type": w.get("type"), "text": w.get("text", "").strip(),
         "start": w.get("start"), "end": w.get("end")}
        for w in words_all
        if w.get("type") not in ("word", "spacing")
    ]
    speakers = sorted({w.get("speaker_id") for w in word_items if w.get("speaker_id")})

    # segment 묶기: 0.6초 이상 무음 갭 또는 화자 변경 시 새 segment
    segments = []
    cur = None
    GAP = 0.6
    for w in word_items:
        ws = float(w.get("start") or 0)
        we = float(w.get("end") or 0)
        txt = w.get("text") or ""
        spk = w.get("speaker_id")
        if cur is None or (ws - cur["end"]) > GAP or spk != cur.get("_spk"):
            if cur is not None:
                cur["text"] = cur["text"].strip()
                segments.append(cur)
            cur = {"start": ws, "end": we, "text": txt,
                   "words": [{"word": txt, "start": ws, "end": we, "speaker": spk}],
                   "_spk": spk}
        else:
            cur["end"] = we
            cur["text"] += " " + txt
            cur["words"].append({"word": txt, "start": ws, "end": we, "speaker": spk})
    if cur is not None:
        cur["text"] = cur["text"].strip()
        segments.append(cur)
    for s in segments:
        s.pop("_spk", None)

    lang_iso3 = d.get("language_code") or ""
    lang_iso1 = _LANG_MAP_FROM_ELEVEN.get(lang_iso3, lang_iso3[:2] if lang_iso3 else "")

    return {
        "text": d.get("text", ""),
        "language": lang_iso1,
        "segments": segments,
        "speakers": speakers,
        "audio_events": audio_events,
        "engine": "elevenlabs",
        "model": d.get("model_id") or ELEVEN_MODEL_DEFAULT,
    }


def _call_elevenlabs(audio_path: str | Path, language: str | None,
                     model_id: str = ELEVEN_MODEL_DEFAULT,
                     timeout: float = 600.0) -> dict:
    """ElevenLabs scribe_v2 호출."""
    import httpx
    key = _get_env_key()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    lang_code = _LANG_MAP_TO_ELEVEN.get((language or "").lower(), language) if language else None
    data = {
        "model_id": model_id,
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if lang_code:
        data["language_code"] = lang_code
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        headers = {"xi-api-key": key}
        r = httpx.post(ELEVEN_URL, files=files, data=data, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
    return _eleven_to_whisper_format(r.json())


def _call_whisper(audio_path: str | Path, language: str | None,
                  model_repo: str = WHISPER_MODEL_DEFAULT) -> dict:
    """mlx_whisper fallback."""
    import mlx_whisper
    kw = dict(path_or_hf_repo=model_repo, word_timestamps=True,
              condition_on_previous_text=False, verbose=False)
    if language:
        kw["language"] = language
    r = mlx_whisper.transcribe(str(audio_path), **kw)
    r["engine"] = "whisper"
    r["model"] = model_repo
    r.setdefault("speakers", [])
    r.setdefault("audio_events", [])
    return r


def transcribe(audio_path: str | Path,
               language: str | None = None,
               engine: str | None = None,
               model_id: str | None = None) -> dict:
    """STT 통합 함수.

    Args:
        audio_path: wav/mp3 경로
        language: "ko"/"en"/"ja"/None (None = 자동 감지)
        engine: "elevenlabs"/"whisper"/None (None = STT_ENGINE env, default elevenlabs)
        model_id: scribe_v2/scribe_v1 등 ElevenLabs 모델 오버라이드

    Returns:
        mlx_whisper.transcribe() 호환 dict + engine/speakers/audio_events 추가 필드
    """
    eng = (engine or os.environ.get("STT_ENGINE") or "elevenlabs").lower()
    if eng == "auto":
        eng = "elevenlabs"
    t0 = time.time()
    if eng == "elevenlabs":
        try:
            result = _call_elevenlabs(audio_path, language,
                                       model_id=model_id or ELEVEN_MODEL_DEFAULT)
            log.info(f"STT elevenlabs OK in {time.time()-t0:.2f}s "
                     f"(words={sum(len(s.get('words',[])) for s in result['segments'])})")
            return result
        except Exception as e:
            log.warning(f"STT elevenlabs FAIL → whisper fallback: {e!r}")
            t1 = time.time()
            result = _call_whisper(audio_path, language)
            log.info(f"STT whisper(fallback) OK in {time.time()-t1:.2f}s")
            return result
    elif eng == "whisper":
        result = _call_whisper(audio_path, language)
        log.info(f"STT whisper OK in {time.time()-t0:.2f}s")
        return result
    else:
        raise ValueError(f"unknown STT_ENGINE={eng!r}")


# 편의 함수: 결과를 SRT 문자열로 변환
def to_srt(result: dict, max_chars_per_line: int = 24) -> str:
    """{segments} → SRT 문자열."""
    def fmt(t):
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        ms = int(round((t - int(t)) * 1000))
        if ms >= 1000:
            s += 1; ms = 0
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    out = []
    for i, seg in enumerate(result.get("segments", []), 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        out.append(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n")
    return "\n".join(out)


if __name__ == "__main__":
    # CLI 테스트: python -m workers.stt_client <audio_path> [language]
    import sys
    audio = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    r = transcribe(audio, language=lang)
    print(json.dumps({k: v for k, v in r.items() if k != "segments"},
                     ensure_ascii=False, indent=2))
    print("---SRT---")
    print(to_srt(r))
