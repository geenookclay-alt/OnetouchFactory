"""음성 자막 워커: mlx-whisper(large-v3) 받아쓰기 + Gemini(어투보존 맞춤법교정)
   + 2줄 제한 + 빈공간 제거 → segments_json(검수용) + SRT.
   고유명사는 STT 한계로 검수 화면에서 사람이 최종 수정."""
import json, pathlib, datetime
import mlx_whisper  # fallback only
from workers.stt_client import transcribe as _stt
from google import genai
from google.genai import types
from api import database as db
from workers.auto_subtitle import apply_user_gemini_key, _get_gemini_key, GEMINI_FLASH_MODEL

MODEL_W = "mlx-community/whisper-large-v3-mlx"   # 로컬 무료, 오인식 적음
MAXLINE = 24   # 한 줄 최대 글자(롱폼), 한 자막 최대 2줄

CORRECT_PROMPT = """다음은 한국어 음성 받아쓰기 자막 조각 배열이다. 각 항목의 t를 '들리는 그대로' 유지하면서 한국어 맞춤법·띄어쓰기·문장부호만 정확히 교정하라.

반드시 지킬 것:
- 화자의 어투·반말·구어체를 절대 바꾸지 말 것. (예: "긴장해야 된다"를 "긴장해야 합니다"로 바꾸지 마라. "된다"는 "된다" 그대로.)
- 줄임말·약어를 풀어쓰지 말 것. (예: "도기본"을 "도시기반시설본부"로 바꾸지 마라.)
- 단어를 추가·삭제·대체하지 말 것. 명백한 오타와 띄어쓰기, 문장부호(. , ? !)만 손본다.
- 내용·의미·길이를 바꾸지 말 것. 합치거나 나누지 말 것. 입력 개수와 출력 개수가 반드시 같아야 한다. i는 그대로 유지.
출력은 JSON 배열만: [{"i":0,"t":"교정된 자막"}]"""


def _fmt(t):
    if t < 0: t = 0
    h = int(t//3600); m = int(t%3600//60); s = int(t%60); ms = int(round((t-int(t))*1000))
    if ms == 1000: s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap2(text):
    w = text.split()
    if not w: return text
    l1 = w[0]; i = 1
    while i < len(w) and len(l1 + " " + w[i]) <= MAXLINE:
        l1 += " " + w[i]; i += 1
    return l1 if i >= len(w) else l1 + "\n" + " ".join(w[i:])


def _split_cue(s, e, text):
    if len(text) <= MAXLINE * 2:
        return [(s, e, _wrap2(text))]
    w = text.split()
    if len(w) < 2:
        return [(s, e, _wrap2(text))]
    tot = len(text); acc = 0; cut = 1
    for j, x in enumerate(w):
        acc += len(x) + 1
        if acc >= tot / 2:
            cut = j + 1; break
    cut = max(1, min(cut, len(w) - 1))
    t1 = " ".join(w[:cut]); t2 = " ".join(w[cut:])
    mid = s + (e - s) * len(t1) / (len(t1) + len(t2) or 1)
    return _split_cue(s, mid, t1) + _split_cue(mid, e, t2)


def write_srt(segments, path):
    lines = [f"{i}\n{_fmt(c['start'])} --> {_fmt(c['end'])}\n{c['text']}\n"
             for i, c in enumerate(segments, 1)]
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


def _clean_segs(segs, dur):
    """large-v3 환각 제거 — 음성 길이 넘는 자막 drop + 연속 동일 반복(무음 환각) 제거."""
    out = []
    for s, e, t in segs:
        if dur and s >= dur - 0.1:        # 음성 끝 넘는 환각 ("아." 반복 등) drop
            continue
        if out and out[-1][2].strip() == t.strip() and (s - out[-1][1]) < 1.5:
            continue                       # 직전과 동일 텍스트 + 1.5초 내 → 반복 환각 skip
        out.append((s, e, t))
    return out


def _correct(client, texts):
    corrected = texts[:]
    B = 60
    for i in range(0, len(texts), B):
        batch = [{"i": j, "t": texts[j]} for j in range(i, min(i + B, len(texts)))]
        res = None
        for _ in range(3):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=[CORRECT_PROMPT + "\n\n" + json.dumps(batch, ensure_ascii=False)],
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       max_output_tokens=8000, temperature=0.1),
                )
                res = json.loads(resp.text); break
            except Exception:
                continue
        if res is None:
            res = batch
        for rr in res:
            try:
                idx = int(rr["i"])
                if 0 <= idx < len(corrected) and str(rr.get("t", "")).strip():
                    corrected[idx] = str(rr["t"]).strip()
            except Exception:
                pass
    return corrected


def run_audio_subtitle(job_id: int):
    """동기 함수 — main에서 asyncio.to_thread로 실행."""
    job = db.get_audio_subtitle_job(job_id)
    if not job:
        return
    try:
        apply_user_gemini_key(job.get("user_id"))
        db.update_audio_subtitle_job(job_id, status="transcribing", progress=10,
                                     progress_message="받아쓰기 중 (large-v3)")
        # STT: ElevenLabs scribe_v2 메인 + mlx_whisper(MODEL_W) failover (env STT_ENGINE=whisper로 강제 가능)
        r = _stt(job["audio_path"], language="ko")
        # 자막에 [효과음]/[화자라벨] 노출 방지 — text에서 대괄호 태그 제거
        import re as _re
        def _strip_tags(t):
            t = _re.sub(r"\[[^\]]+\]", " ", t)
            return _re.sub(r"\s+", " ", t).strip()
        segs = [(s["start"], s["end"], _strip_tags(s["text"]))
                for s in r.get("segments", []) if _strip_tags(s["text"])]
        segs = _clean_segs(segs, job.get("duration_sec") or 0)
        if not segs:
            db.update_audio_subtitle_job(job_id, status="failed", error="받아쓰기 결과 없음(무음?)")
            return
        db.update_audio_subtitle_job(job_id, status="correcting", progress=60,
                                     progress_message="맞춤법 교정 중 (어투 보존)")
        client = genai.Client(api_key=_get_gemini_key())
        corrected = _correct(client, [t for _, _, t in segs])
        # 2줄 제한 + 빈공간 제거
        all_cues = []
        for i, (s, e, orig) in enumerate(segs):
            all_cues.extend(_split_cue(s, e, corrected[i] or orig))
        for k in range(len(all_cues) - 1):
            gap = all_cues[k+1][0] - all_cues[k][1]
            if 0 < gap <= 1.5:   # 작은 틈만 메워 연속, 큰 무음(>1.5초)은 유지 → 싱크 보존
                all_cues[k] = (all_cues[k][0], all_cues[k+1][0], all_cues[k][2])
        segments = [{"start": round(s, 2), "end": round(e, 2), "text": t} for s, e, t in all_cues]
        srt_path = pathlib.Path(job["audio_path"]).parent / f"job_{job_id}.srt"
        write_srt(segments, srt_path)
        db.update_audio_subtitle_job(job_id, status="review", progress=100,
                                     segments_json=segments, srt_path=str(srt_path),
                                     progress_message="검수 대기",
                                     completed_at=datetime.datetime.utcnow().isoformat())
    except Exception as e:
        db.update_audio_subtitle_job(job_id, status="failed", error=str(e)[:500])
        raise


def rebuild_srt(job_id: int):
    """검수 저장 후 SRT 재생성 (수정된 segments_json → srt)."""
    job = db.get_audio_subtitle_job(job_id)
    if not job:
        return
    segs = job["segments_json"]
    if isinstance(segs, str):
        segs = json.loads(segs)
    if job.get("srt_path"):
        write_srt(segs, job["srt_path"])
