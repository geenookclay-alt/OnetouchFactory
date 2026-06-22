"""음성-자막/컷 sync 헬퍼 (silence-aware pad).

대표님 룰:
- Whisper word-level timestamps로 화자 첫 단어/마지막 단어 정확히 잡음
- silero-vad로 음성 segment 검출 → silence 구간 식별
- 자막 cue / 영상 컷 boundary를 silence 안에서만 pad
- 인접 음성 침범 X (clamp)
- 침묵 구간에 짧은 자막 X (대사 시점에만)

쓰임:
- 자막 워커: 각 cue의 start/end를 word boundary에 snap + silence-aware pad
- 컷 워커 (쇼츠메이커): cut start/end를 silence 안 boundary로 snap

검증 흐름:
- Whisper로 영상 음성 → word/segment list
- silero-vad로 영상 음성 → speech/silence intervals
- 두 정보로 정확한 boundary 계산
"""
import asyncio
import subprocess
import tempfile
from pathlib import Path


def _ffmpeg_bin() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return "ffmpeg"


def extract_audio_16k_mono(video_path: Path, out_wav: Path) -> Path:
    """Whisper/VAD용 16kHz mono wav 추출."""
    cmd = [
        _ffmpeg_bin(), "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract fail: {r.stderr[:200]}")
    return out_wav


def detect_speech_intervals(audio_wav: Path,
                              sampling_rate: int = 16000,
                              min_silence_duration_ms: int = 300,
                              threshold: float = 0.5) -> list[tuple[float, float]]:
    """silero-vad로 음성 segment 검출. Returns [(start_sec, end_sec), ...]."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

    model = load_silero_vad()
    wav = read_audio(str(audio_wav), sampling_rate=sampling_rate)
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def silence_aware_pad(start: float, end: float,
                       speech_intervals: list[tuple[float, float]],
                       pre_pad: float = 0.30,
                       post_pad: float = 0.30,
                       min_gap: float = 0.05,
                       video_dur: float | None = None) -> tuple[float, float]:
    """주어진 자막/컷 boundary (start, end)를 silence-aware로 pad.

    - start를 앞쪽으로 최대 pre_pad만큼 늘림. 단 이전 speech end + min_gap 이내까지만.
    - end를 뒤쪽으로 최대 post_pad만큼 늘림. 단 다음 speech start - min_gap 이내까지만.
    - 영상 길이(video_dur) 있으면 end <= video_dur 보장.
    - 인접 음성 침범 X.

    speech_intervals: silero-vad 결과 (오름차순 정렬 가정).
    Returns: (new_start, new_end).
    """
    # 이전 speech end 찾기 (현재 start 이전에 끝난 마지막 speech)
    prev_end = 0.0
    next_start = video_dur if video_dur else float("inf")
    for s, e in speech_intervals:
        if e <= start:
            prev_end = max(prev_end, e)
        elif s >= end:
            next_start = min(next_start, s)
            break  # 정렬 가정
    # start pad: prev_end + min_gap 이상으로
    new_start = max(start - pre_pad, prev_end + min_gap, 0.0)
    if new_start > start:
        new_start = start  # 어떤 경우에도 원래 start보다 늦지 않게
    # end pad: next_start - min_gap 이하로
    new_end = min(end + post_pad, next_start - min_gap)
    if new_end < end:
        new_end = end  # 어떤 경우에도 원래 end보다 빠르지 않게
    if video_dur:
        new_end = min(new_end, video_dur)
    if new_end <= new_start:
        new_end = new_start + 0.1  # 안전장치
    return float(new_start), float(new_end)


def snap_to_word_boundary(start: float, end: float,
                            words: list[dict],
                            tolerance: float = 0.5) -> tuple[float, float]:
    """자막 cue를 가장 가까운 Whisper word 시작/끝으로 snap.

    words: Whisper 결과의 word-level list — [{"word":..., "start":..., "end":...}, ...]
    tolerance: 이 거리 이내의 word만 snap (너무 멀면 원래 값 유지).
    """
    if not words:
        return start, end
    # 가장 가까운 word start
    best_start = start
    best_start_d = tolerance
    for w in words:
        d = abs(w.get("start", 0) - start)
        if d < best_start_d:
            best_start = w.get("start", start)
            best_start_d = d
    # 가장 가까운 word end
    best_end = end
    best_end_d = tolerance
    for w in words:
        d = abs(w.get("end", 0) - end)
        if d < best_end_d:
            best_end = w.get("end", end)
            best_end_d = d
    return float(best_start), float(best_end)


def adjust_cues(cues: list[dict],
                speech_intervals: list[tuple[float, float]],
                words: list[dict] | None = None,
                pre_pad: float = 0.20,
                post_pad: float = 0.20,
                min_gap: float = 0.05,
                video_dur: float | None = None) -> list[dict]:
    """자막 cue list에 silence-aware pad + word boundary snap 적용.

    cues: [{"start": ..., "end": ..., "text": ...}, ...]
    Returns: 같은 형식, start/end 보정됨.
    """
    out = []
    for cue in cues:
        s = float(cue.get("start", 0))
        e = float(cue.get("end", s + 1))
        if words:
            s, e = snap_to_word_boundary(s, e, words, tolerance=0.4)
        s, e = silence_aware_pad(s, e, speech_intervals,
                                  pre_pad=pre_pad, post_pad=post_pad,
                                  min_gap=min_gap, video_dur=video_dur)
        new_cue = dict(cue)
        new_cue["start"] = s
        new_cue["end"] = e
        out.append(new_cue)
    # 인접 cue 겹침 제거 — 한 화면에 자막 2개 동시 표시 방지.
    # cue[i].end가 다음 cue.start를 넘으면 다음 start - gap으로 당김.
    out.sort(key=lambda c: c["start"])
    gap = 0.05
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        if cur["end"] > nxt["start"] - gap:
            new_end = nxt["start"] - gap
            # 최소 0.4초는 보장 (너무 짧으면 다음 cue를 살짝 뒤로)
            if new_end <= cur["start"] + 0.4:
                new_end = cur["start"] + 0.4
                if nxt["start"] < new_end + gap:
                    nxt["start"] = new_end + gap
                    if nxt["end"] <= nxt["start"]:
                        nxt["end"] = nxt["start"] + 0.4
            cur["end"] = new_end
    return out


def parse_srt(srt_path: Path) -> list[dict]:
    """SRT 파일 → cue list [{"start", "end", "text"}, ...]."""
    import re
    raw = Path(srt_path).read_text(encoding="utf-8")

    def to_sec(t: str) -> float:
        h, m, s_ms = t.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    pattern = re.compile(
        r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*\n((?:.+\n?)+?)(?=\n\s*\n|\n*$)",
        re.MULTILINE,
    )
    cues = []
    for m in pattern.finditer(raw):
        cues.append({
            "start": to_sec(m.group(2)),
            "end": to_sec(m.group(3)),
            "text": m.group(4).strip(),
        })
    return cues


def write_srt(cues: list[dict], srt_path: Path) -> None:
    """cue list → SRT 파일."""
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec * 1000) % 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt(c['start'])} --> {fmt(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")


def snap_cut_to_silence(start: float, end: float,
                          speech_intervals: list[tuple[float, float]],
                          max_pad: float = 0.50,
                          min_gap: float = 0.05,
                          video_dur: float | None = None) -> tuple[float, float]:
    """영상 컷 boundary를 silence 안에서 자연스럽게 조정.

    - 컷 시작이 speech 중간이면 → 가장 가까운 silence 시작점(이전 speech 끝 +min_gap)으로 snap
    - 컷 끝이 speech 중간이면 → 가장 가까운 silence 끝점(다음 speech 시작 -min_gap)으로 snap
    - 단 max_pad 이내에서만 snap (너무 멀면 원래값 유지)
    """
    new_start, new_end = start, end
    # 시작이 speech 중간인지 확인
    for s, e in speech_intervals:
        if s < start < e:
            # speech 중간 — 이전 silence 끝점으로 snap
            target = s - min_gap
            if start - target <= max_pad:
                new_start = max(0.0, target)
            break
    # 끝이 speech 중간인지 확인
    for s, e in speech_intervals:
        if s < end < e:
            target = e + min_gap
            if target - end <= max_pad:
                new_end = target
            break
    if video_dur:
        new_end = min(new_end, video_dur)
    if new_end <= new_start:
        new_end = new_start + 0.5
    return float(new_start), float(new_end)


# ─────────────────────────────────────────────────────────────
# 대사 SRT 한 줄 길이 분할 — Whisper가 긴 발화를 한 cue에 통째로 넣어
# 화면에서 3줄+로 깨지는 문제 (대표님 지적 2026-05-29).
# 한 cue 텍스트가 max_line_chars 넘으면 시간 비례로 여러 cue로 분할.
# ─────────────────────────────────────────────────────────────
def _split_text_natural(text: str, max_chars: int = 16) -> list[str]:
    """텍스트를 max_chars 이하 조각으로 — 어절(공백) 단위 우선, 문장부호 우선 분할."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    # 1순위: 문장부호(? ! 쉼표)로 끊기
    import re
    parts = re.split(r"(?<=[?!,])\s+", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            # 2순위: 어절(공백) 단위로 쌓기
            words = part.split()
            cur = ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    chunks.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w) if cur else w
            if cur:
                chunks.append(cur)
    # 3순위: 그래도 max 넘으면 강제 글자 분할
    final = []
    for c in chunks:
        while len(c) > max_chars:
            final.append(c[:max_chars])
            c = c[max_chars:]
        if c:
            final.append(c)
    return final or [text]


def split_long_dialogue_srt(srt_path: Path, max_line_chars: int = 18,
                              min_cue_dur: float = 0.6) -> int:
    """대사 SRT 정리 — 한 cue(=한 발화)는 그대로 유지하고, 길면 **줄바꿈(\\n)으로 2줄**.
    cue를 시간 분할하지 X (sync 깨짐·토막남 원인이었음 — 대표님 지적 2026-05-29).
    - 한 줄 max_line_chars 이하로 줄바꿈 (최대 2줄, 2줄 넘으면 그대로 둠)
    Returns cue 수 (변동 없음).
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return 0
    cues = parse_srt(srt_path)
    # ★긴 cue end cap — Whisper가 무음 구간에서 자막을 안 끊어 한 줄이 수십초 떠있는 것 방지
    #   (대표님 2026-06-01 "그렇다면 이번에는/뭐라? 가 40초 늘어짐"). 한국어 읽기속도 + 여유.
    #   다음 cue 침범 X, end 역전(end<start)도 보정.
    for i, c in enumerate(cues):
        t0 = (c.get("text") or "").replace("\n", " ").strip()
        max_dur = len(t0) * 0.33 + 1.8
        cap_end = c["start"] + max_dur
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if nxt is not None:
            cap_end = min(cap_end, nxt - 0.05)
        if c["end"] < c["start"] or (c["end"] - c["start"]) > max_dur:
            c["end"] = max(c["start"] + 0.4, cap_end)
    for c in cues:
        text = (c.get("text") or "").replace("\n", " ").strip()
        if len(text) <= max_line_chars:
            c["text"] = text
            continue
        # 어절 단위로 2줄 균형 분할 (가운데 가까운 공백에서 한 번만 끊음)
        words = text.split()
        if len(words) <= 1:
            c["text"] = text
            continue
        best_i, best_diff = 1, 10**9
        for i in range(1, len(words)):
            left = len(" ".join(words[:i]))
            right = len(" ".join(words[i:]))
            # 두 줄 다 max 이하 우선, 그 중 길이 차 최소
            if left <= max_line_chars and right <= max_line_chars:
                diff = abs(left - right)
                if diff < best_diff:
                    best_diff, best_i = diff, i
        if best_diff == 10**9:
            # 2줄로도 max 못 맞추면 그냥 가운데 어절에서 끊음 (3줄보단 2줄)
            best_i = len(words) // 2
        line1 = " ".join(words[:best_i])
        line2 = " ".join(words[best_i:])
        c["text"] = line1 + "\n" + line2
    write_srt(cues, srt_path)
    return len(cues)


# ─────────────────────────────────────────────────────────────
# Whisper 대사 맞춤법/오인식 교정 — Gemini로 텍스트만 교정 (timestamp 보존).
# 대표님 지적 2026-05-29: 위스퍼 대사 맞춤법/sync 안 맞음.
# cue 텍스트 list만 Gemini에 던져 자연스러운 한국어로 교정. cue 수/순서/시간 그대로.
# ─────────────────────────────────────────────────────────────
_SPELL_CHAIN = ("gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash")


async def correct_dialogue_srt(srt_path: Path, api_key: str | None = None) -> int:
    """대사 SRT의 cue 텍스트를 Gemini로 맞춤법/오인식 교정. timestamp 보존.
    cue 수가 바뀌면(모델 실수) 원본 유지. Returns 교정된 cue 수 (0=skip/fail).
    """
    import os
    import json as _json
    import httpx
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return 0
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        return 0
    cues = parse_srt(srt_path)
    if not cues:
        return 0
    texts = [c.get("text", "").replace("\n", " ").strip() for c in cues]
    prompt = (
        "다음은 한국 영상(드라마·예능·전래동화 등)의 음성 받아쓰기(STT) 결과 cue 목록이야. "
        "STT가 발음 비슷한 다른 단어로 잘못 들은 부분을 앞뒤 문맥에 맞게 적극적으로 자연스러운 한국어로 교정해줘.\n"
        "규칙:\n"
        "- cue 개수·순서 절대 바꾸지 마 (입력 N개 → 출력 N개)\n"
        "- 각 cue는 화자가 실제 말한 대사. 의미 유지하되 자연스럽게.\n"
        "- 내용 추가/삭제 X. 단 **앞뒤 문맥상 명백히 안 맞는 단어는 적극 교정**(발음이 비슷해 잘못 들은 것 — 예: 마지막 쌀을 공양하는 장면인데 '사례'로 들렸으면 '쌀'로).\n"
        "- **인물·등장 이름은 cue마다 일관되게**: 같은 사람·이름이 cue마다 다르게 들렸으면 가장 그럴듯한 하나로 통일.\n"
        "- 욕설·비속어는 그대로 (대사라 살림)\n"
        '출력은 JSON만: {"lines": ["교정1", "교정2", ...]} — lines 길이 = 입력 cue 수.\n\n'
        "[입력 cue]\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096,
                                  "responseMimeType": "application/json"}}
    async with httpx.AsyncClient(timeout=120) as c:
        for model in _SPELL_CHAIN:
            for attempt in range(2):
                try:
                    r = await c.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                        json=body)
                    if r.status_code == 200:
                        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                        data = _json.loads(txt)
                        lines = data.get("lines") or []
                        if len(lines) == len(cues):
                            for i, ln in enumerate(lines):
                                cues[i]["text"] = str(ln).strip()
                            write_srt(cues, srt_path)
                            return len(cues)
                        return 0  # 개수 안 맞으면 원본 유지
                    if r.status_code in (429, 500, 502, 503, 504):
                        if attempt < 1:
                            await asyncio.sleep(3); continue
                        break  # 다음 model
                    return 0
                except Exception:
                    if attempt < 1:
                        await asyncio.sleep(2); continue
                    break
    return 0


# ─────────────────────────────────────────────────────────────
# 비한국어 대사 번역 — 일본어/영어 등 원어 대사 SRT → 자연스러운 한국어.
# 블리치(일본어 원본) 같은 애니/해외 영화용. Whisper는 원어 그대로 받아쓰므로
# 제작 단계에서 한국어로 번역해야 자막이 한국어로 박힘. cue 수·timestamp 보존.
# 발굴 후보(제목/요약)는 Gemini가 이미 한국어로 만들지만, 대사 SRT는 별도.
# ─────────────────────────────────────────────────────────────
def detect_srt_lang(srt_path: Path) -> str:
    """SRT 텍스트의 문자 구성으로 언어 추정. 'ko'/'ja'/'en'.
    히라가나·가타카나가 있으면 일본어 확정. 한글 음절이 전체 글자의 절반 이상이면
    한국어. 그 외 라틴 위주면 'en'(영어 등). 한자 위주면 'ja'(번역 대상)."""
    try:
        cues = parse_srt(srt_path)
    except Exception:
        return "ko"
    text = "".join(c.get("text", "") for c in cues)
    if not text.strip():
        return "ko"
    kana = sum(1 for ch in text if "぀" <= ch <= "ヿ")   # 히라가나+가타카나
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    han = sum(1 for ch in text if "一" <= ch <= "鿿")     # CJK 한자
    total = kana + hangul + latin + han
    if total == 0:
        return "ko"
    if hangul / total >= 0.5:
        return "ko"
    if kana > 0 or han > latin:
        return "ja"
    if latin > 0:
        return "en"
    return "ko"


async def translate_dialogue_srt(srt_path: Path, src_lang: str = "ja",
                                  api_key: str | None = None) -> int:
    """비한국어 대사 SRT를 자연스러운 한국어 더빙 대사로 번역. cue 수·timestamp 보존.
    cue 수가 바뀌면(모델 실수) 원본(원어) 유지. Returns 번역된 cue 수 (0=skip/fail).
    """
    import os
    import json as _json
    import httpx
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return 0
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        return 0
    cues = parse_srt(srt_path)
    if not cues:
        return 0
    texts = [c.get("text", "").replace("\n", " ").strip() for c in cues]
    lang_name = {"ja": "일본어", "en": "영어", "zh": "중국어"}.get(src_lang, "외국어")
    prompt = (
        f"다음은 {lang_name} 애니메이션/영화 영상의 음성 받아쓰기(STT) 대사 cue 목록이야. "
        "각 cue를 자연스러운 한국어 더빙 대사로 번역해줘.\n"
        "규칙:\n"
        "- cue 개수·순서 절대 바꾸지 마 (입력 N개 → 출력 N개)\n"
        "- 직역 X. 한국 더빙판처럼 자연스러운 구어체로.\n"
        "- 짧은 외침·감탄사도 한국어로 (예: 「やめろ」→「그만해」)\n"
        "- 인물명·기술명·고유명사는 한국 정발/통용 표기로 (예: 야마모토, 퀸시, 만해, 소울 소사이어티)\n"
        "- 거친 말투·기세는 살림 (작품 톤 유지)\n"
        "- 내용 추가/삭제 X. 들린 대사만 번역.\n"
        '출력은 JSON만: {"lines": ["번역1", "번역2", ...]} — lines 길이 = 입력 cue 수.\n\n'
        "[입력 cue]\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192,
                                  "responseMimeType": "application/json"}}
    async with httpx.AsyncClient(timeout=180) as c:
        for model in _SPELL_CHAIN:
            for attempt in range(2):
                try:
                    r = await c.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                        json=body)
                    if r.status_code == 200:
                        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                        data = _json.loads(txt)
                        lines = data.get("lines") or []
                        if len(lines) == len(cues):
                            for i, ln in enumerate(lines):
                                cues[i]["text"] = str(ln).strip()
                            write_srt(cues, srt_path)
                            return len(cues)
                        return 0  # 개수 안 맞으면 원본 유지
                    if r.status_code in (429, 500, 502, 503, 504):
                        if attempt < 1:
                            await asyncio.sleep(3); continue
                        break  # 다음 model
                    return 0
                except Exception:
                    if attempt < 1:
                        await asyncio.sleep(2); continue
                    break
    return 0
