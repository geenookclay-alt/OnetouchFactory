"""대본+더빙 워커 — 영상 1개 → Gemini(대본+메타데이터) → 타입캐스트 TTS → 12자/2줄 SRT.

흐름:
1. 영상 → Gemini Pro (영상 보고 딸기우유+오풍 스타일 한국어 더빙 대본 + 메타 생성)
   결과 JSON: hook_title(상단 박제 타이틀), youtube_title, description, tags, lines(대본)
2. 대본 라인별 → 타입캐스트 TTS (필재, ko-kr, 피치+1, mp3)
3. ffmpeg atempo=1.18 + 무음 제거 → 라인 음성 (무음제거 포함 실효 ~1.4배)
4. 자막은 12자/2줄 청크로 쪼개 음성 길이 글자수 비례 배분 → SRT (TTS 100% 싱크)
5. 라인 음성 concat → tts.mp3
출력: tts.mp3 + subtitle.srt (캡컷용) + result_json
"""
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path

import httpx

from api import database as db
from workers.auto_subtitle import (
    ensure_inline_video, upload_video_to_gemini, call_gemini,
    GEMINI_PRO_MODEL, apply_user_gemini_key,
)

OUT_DIR = Path(__file__).parent.parent / "data" / "tts_dub"
TYPECAST_URL = "https://api.typecast.ai/v1/text-to-speech"
DEFAULT_VOICE = "tc_68257f68bc6e3c161ab5078d"  # 필재(Piljae) 한국 남성
SPEED = 1.4  # (대표님 지시 2026-06-06) atempo 1.4. 무음제거 포함 실효 ~1.66배
              # (1.4로 두면 실효 1.66배라 "말이 너무 빨라" — 2026-05-25 정상화)
PITCH = 1     # 타입캐스트 피치 +1

# atempo 1.18 + 앞뒤+내부 무음 제거 → 파일=순수 발화 (대표님: 무음구간 삭제). 무음제거로 실효 ~1.4배.
TRIM = (f"atempo={SPEED},"
        "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0:"
        "stop_periods=-1:stop_threshold=-40dB:stop_silence=0.08:detection=peak,"
        "areverse,"
        "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0:detection=peak,"
        "areverse")

DUB_PROMPT = """이 영상을 보고, 한국 쇼츠용 **한국어 보이스오버 더빙 대본 + 업로드 메타데이터**를 만들어줘.
톤: 실화/CCTV/반전/사이다/참교육/감동 스토리텔링 쇼츠 (한국 인기 쇼츠 결).
🚫 절대 금지: 참고/출처 채널명이나 특정 유튜버 이름(예: 딸기우유, 오풍 등)을 태그·제목·설명·대본 어디에도 넣지 마라. 태그는 일반 키워드만.

[톤·문체]
- 3인칭 내레이션, '~습니다 / ~했죠 / ~했습니다'체. 차분하기보다 **긴장감 있고 몰입되게**.
- 짧고 빠른 문장(쇼츠 호흡). 첫 문장 = 강력한 후크(궁금증/긴장/황당/충격) — 보자마자 "헉".
- 중반 빠른 전개 → 마지막 사이다(통쾌한 반전)/감동/교훈 중 영상에 맞는 것.
- 이모지/신조어 남발 금지. 자연스러운 구어체.
- ⚠️ 마지막 줄(클로징, 필수): "~했던 실화입니다" 형태로 마무리.
- 각 줄 text는 화면 자막 한 칸(최대 2줄, 합쳐 약 30자)에 들어갈 짧은 분량으로. 더 길면 줄(line)을 더 쪼개 각각 start를 매겨라. ⚠️ 따옴표 대사("거기 있니?")나 짧은 외마디는 절대 끊지 말고 한 줄(text) 안에 통째로 — "거기", "있니?" 처럼 토막 내지 마라.

[🔥 자극성·몰입 — 매우 중요 (대표님 요청)]
대본을 **더 자극적으로** 써라. 욕설·비속어가 아니라 **내용의 긴장과 충격**으로 끌어당겨라:
- **후크 극대화**: 첫 1~2줄에서 충격적 상황·위기·금기·미스터리를 정면으로 던져 "헉" 하게. 밋밋한 소개 X.
- **판돈 키우기**: 위험·갈등·반전의 무게를 부각 — "한순간의 실수로 모든 게 끝날 뻔했죠" 식으로 조마조마하게.
- **서스펜스(정보 감추기)**: 결정적 사실을 한 박자 늦게. "그런데 진짜 충격은 그다음이었습니다" / "아무도 예상 못 한 일이 벌어집니다" 식 떡밥·클리프행어를 중간중간 박아라.
- **생생한 묘사**: 밋밋한 서술 대신 긴박·소름·아찔하게. 시청자의 충격·분노·공포·소름·통쾌함 같은 감정을 정확히 건드려라.
- **반전·결정적 순간**을 최대한 극적으로 띄워라. youtube_title·hook_title도 가장 자극적인 포인트를 후크로.

━━ 더 세게 — 구체 무기 (대표님 "더 자극적으로" 재요청) ━━
🚫 **첫 줄 도입형 금지** — "오늘은 ~의 이야기입니다", "이번 사건은 ~", "한 ~가 있었습니다", "~한 적이 있죠" 류 절대 X. **첫 단어부터 사건 한가운데로** 던져라.
- 좋은/나쁜 첫 줄 예 (감 잡으라고):
  ❌ "버스에서 한 청년이 노인에게 폭행을 당한 사건이 있었습니다."
  ✅ "갑자기 날아온 노인의 손이, 청년의 뺨을 후려쳤습니다."
  ❌ "오늘은 한 마트에서 일어난 황당한 일을 소개합니다."
  ✅ "CCTV에 잡힌 그 장면, 보고도 믿을 수 없었습니다."
- **자극 어휘 적극**: "충격", "소름", "경악", "참혹", "끔찍", "도저히 이해 안 가는", "어이없는", "황당한", "믿기지 않는", "한순간에", "그 순간", "정적이 흘렀습니다", "공기가 얼어붙었습니다" 같은 분위기 끌어올리는 단어.
- **호기심 갭 길게**: 결정적 사실(범인 정체·반전·결말)은 **가능한 한 늦게** 풀어라. 끝까지 안 풀면 끝까지 본다.
- **떡밥 더 촘촘**: 2~3줄마다 "그런데", "그게 끝이 아니었습니다", "진짜 충격은 그다음이었죠", "아무도 예상 못 한 일이 벌어집니다", "그 순간이 마지막일 줄은 몰랐습니다" 식. 너무 적으면 늘어진다.
- **감각 디테일**: "때렸다" X → "뺨을 후려치는 둔탁한 소리, 손이 부들부들 떨렸습니다", "시간이 멈춘 듯했죠" 식으로 청각·시각·체감 박아라.
- **마지막 1~2줄은 한 방**: 시청자가 댓글에 "헐..." 쓰게 만들 임팩트 한 방.

🚫 **자극 = 과장된 긴장 연출이지 사실 날조가 아니다** — 영상에 없는 사건·인물·수치는 지어내지 마라(아래 [분량·흐름] 날조 금지 유지). 보이는 걸 '더 강렬하게 전달'하는 것.
🚫 욕설·비속어·성적·혐오 표현 금지. 자극은 오직 스토리의 긴장으로만.

[🎯 분량·흐름 — 매우 중요]
- ⚠️ 이 대본은 **쉼 없이 처음부터 끝까지 이어서 낭독**된다(줄 사이 공백·무음 없음). 분량은 **영상 길이에 딱 맞게** — 너무 짧으면 뒤가 비고, 너무 길면 음성을 빠르게 눌러야 해 부자연스럽다. 아래 [⏱] 글자수 목표를 지켜라.
- 영상에서 **실제로 보이는 사건만 시간 순서대로** 서술(없는 사건·인물 날조 금지). 단, 분량은 **보이는 장면을 더 풍부하게 묘사·심리·긴장·해설**로 자연스럽게 채워라 — 새 사건을 지어내는 게 아니라, 보이는 것에 대한 설명을 두텁고 생생하게 늘리는 것.
- 각 줄은 자막 한 칸 단위로 짧게 끊고, 영상 시간 순서(start 오름차순)로. start는 그 내용이 영상에 나오는 대략 시각(초)일 뿐 — 낭독은 줄 사이 끊김 없이 바로 이어진다.
- 영상에 원래 내레이션/자막이 있으면 그 의미에 맞춰 한국어로 각색.

[출력 — 반드시 이 JSON 형식만]
{
  "hook_title": ["[라벨]", "윗줄", "아랫줄"],   // 영상 상단 박제용. 라벨(예 "[반전 실화]") + 2줄 후크. 마지막에 '실화' 뉘앙스
  "youtube_title": "업로드용 제목 (궁금증 후크, 60자 이내)",
  "description": "스토리 2~3줄 요약 + 줄바꿈 + '※ 실제 사건을 바탕으로 각색했으며, 인물·세부는 허구입니다.' + 줄바꿈 + 해시태그 5개",
  "tags": ["실화","반전", ...],   // 8~15개
  "lines": [ {"start": 0.0, "text": "첫 문장"}, {"start": 4.5, "text": "다음 장면 문장"} ]
          // start=영상 내 시각(초), 시간 오름차순. 마지막 text는 '~했던 실화입니다' 클로징
}
JSON 외 다른 텍스트 금지."""


import contextvars
# 작업 만든 사용자의 개인 타입캐스트 키 (비용 분리) — apply_user_typecast_key로 설정
_user_typecast_key = contextvars.ContextVar("_user_typecast_key", default=None)


def apply_user_typecast_key(user_id) -> bool:
    """user_id의 저장된 개인 타입캐스트 키가 있으면 contextvar에 설정. 더빙 워커 시작 시 호출."""
    if not user_id:
        return False
    try:
        from api import auth
        u = auth.get_user_by_id(int(user_id))
        key = (u or {}).get("typecast_api_key")
        if key and str(key).strip():
            _user_typecast_key.set(str(key).strip())
            print(f"🔑 사용자 {user_id} 개인 타입캐스트 키 사용", flush=True)
            return True
    except Exception as e:
        print(f"⚠️ 개인 타입캐스트 키 조회 실패(글로벌 사용): {e}", flush=True)
    return False


def _get_typecast_key() -> str:
    # 사용자 개인 키 우선 (작업 실행 컨텍스트에 설정돼 있으면)
    uk = _user_typecast_key.get()
    if uk:
        return uk
    k = os.environ.get("TYPECAST_API_KEY")
    if k:
        return k
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPECAST_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPECAST_API_KEY not set")


# 자막 청크 — 한 줄 ~19자(사진 수준), 의미(조사) 경계 우선
_JOSA = "은는이가을를에도만의로와과께"
_JOSA_END = ("에서", "으로", "에게", "한테", "라고", "라는", "다고", "지만")


def _split_chunks(text: str, maxc: int = 38) -> list[str]:
    """문장을 어절 단위로 한 청크 최대 maxc자(2줄×19)로 분할 + 짧은 조각 병합."""
    words = text.replace(",", ", ").split()
    out, cur = [], ""
    for w in words:
        if cur and len((cur + " " + w).strip()) > maxc:
            out.append(cur.strip())
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur.strip():
        out.append(cur.strip())
    # 짧은 조각을 인접 청크와 병합 — "있니?"·"단계입니다.'" 같은 외톨이 자막 방지
    # · 여유 있으면(<=maxc) 짧은(<10) 조각 흡수
    # · 토막(<8자)은 maxc 살짝 초과(+10, 2줄 ~24자)해도 무조건 흡수 — 외톨이 방지 우선
    res = []
    for c in out:
        vis = len(c.replace(" ", ""))
        if res and (
            (len(res[-1]) + 1 + len(c) <= maxc and (vis < 10 or len(res[-1].replace(" ", "")) < 10))
            or (vis < 8 and len(res[-1]) + 1 + len(c) <= maxc + 10)
        ):
            res[-1] = (res[-1] + " " + c).strip()
        else:
            res.append(c)
    # 마지막 청크가 짧으면(<12자) 직전 청크에서 어절을 당겨와 균형 — "번..."같은 꼬리/("세 번")분리 방지
    if len(res) >= 2:
        while len(res[-1].replace(" ", "")) < 12:
            pw = res[-2].split()
            if len(pw) < 2:
                break
            cand_prev, cand_last = " ".join(pw[:-1]), pw[-1] + " " + res[-1]
            if len(cand_prev.replace(" ", "")) < 8 or len(cand_last) > maxc + 10:
                break
            res[-2], res[-1] = cand_prev, cand_last
    return res or [text]


def _wrap2(c: str, per: int = 19) -> str:
    """한 청크를 한 줄 최대 per자로, 의미(조사) 경계+균형 우선 2줄 분할."""
    if len(c) <= per:
        return c
    words = c.split()
    if len(words) < 2:
        return c
    best, best_score = 1, 10 ** 9
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        prev = words[i - 1].rstrip(",.!?")
        is_josa = (prev != "" and prev[-1] in _JOSA) or prev.endswith(_JOSA_END)
        over = max(0, len(l1) - per) + max(0, len(l2) - per)
        score = abs(len(l1) - len(l2)) + over * 1000 + (0 if is_josa else 40)
        if score < best_score:
            best_score, best = score, i
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def _fmt(s: float) -> str:
    h = int(s // 3600); m = int(s % 3600 // 60)
    sec = int(s % 60); ms = int(round((s - int(s)) * 1000))
    if ms == 1000:
        sec += 1; ms = 0
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


async def _tts_line(text: str, voice_id: str, out_mp3: Path) -> None:
    key = _get_typecast_key()
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(
            TYPECAST_URL,
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={
                "text": text, "model": "ssfm-v30", "voice_id": voice_id,
                "language": "ko-kr", "emotion": "normal", "pitch": PITCH,
                "output": {"audio_format": "mp3"},
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"Typecast {r.status_code}: {r.text[:200]}")
        out_mp3.write_bytes(r.content)


async def _ff(*args) -> None:
    p = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", *args)
    await p.wait()


async def _dur(path: Path) -> float:
    p = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await p.communicate()
    try:
        v = float(out.strip())
        if v > 0:
            return v
    except Exception:
        pass
    # MediaRecorder webm 등 duration 헤더 없는 파일 → 디코드해서 실제 길이 측정
    import re
    p = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", str(path), "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await p.communicate()
    ts = re.findall(rb"time=(\d+):(\d+):(\d+(?:\.\d+)?)", err or b"")
    if ts:
        h, m, s = ts[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


_BANNED_REF = ("딸기우유", "오풍")


def _strip_banned_meta(meta: dict) -> dict:
    """참고 채널명이 태그·메타·대본에 새어나가지 않게 제거 (이중 안전장치)."""
    if not isinstance(meta, dict):
        return meta

    def clean(s):
        s = str(s)
        for b in _BANNED_REF:
            s = s.replace("#" + b, "").replace(b, "")
        return s.strip()

    if isinstance(meta.get("tags"), list):
        meta["tags"] = [t for t in meta["tags"]
                        if not any(b in str(t) for b in _BANNED_REF)]
    for k in ("youtube_title", "description"):
        if meta.get(k) is not None:
            meta[k] = clean(meta[k])
    if isinstance(meta.get("hook_title"), list):
        meta["hook_title"] = [clean(x) for x in meta["hook_title"]]
    if isinstance(meta.get("lines"), list):
        nl = []
        for x in meta["lines"]:
            if isinstance(x, dict):
                x["text"] = clean(x.get("text", ""))
                if x["text"]:
                    nl.append(x)
            else:
                c = clean(x)
                if c:
                    nl.append(c)
        meta["lines"] = nl
    return meta


async def run_tts_dub(job_id: int, make_tts: bool = True) -> None:
    """대본+더빙 전체 파이프라인. make_tts=False면 TTS 음성 생략(대본+메타+자막SRT만, 대표님 0614)."""
    job = db.get_tts_dub_job(job_id)
    if not job:
        return
    out = OUT_DIR / f"job_{job_id}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        apply_user_gemini_key(job.get("user_id"))     # 대본 = 개인 Gemini 키
        apply_user_typecast_key(job.get("user_id"))   # TTS = 개인 타입캐스트 키
    except Exception:
        pass
    voice = job.get("voice_id") or DEFAULT_VOICE
    try:
        vp = Path(job["video_path"])
        if not vp.exists():
            raise RuntimeError(f"영상 파일 없음: {vp}")

        # 1. Gemini — 대본 + 메타데이터
        db.update_tts_dub_job(job_id, status="analyzing", progress=15,
                              progress_message="영상 분석·대본 생성 중 (60~90초)..")
        dur_sec = await _dur(vp)  # 영상 길이 → 대본 분량 목표 계산
        analysis_video = await ensure_inline_video(vp)
        file_uri = await upload_video_to_gemini(analysis_video)
        # 글자율 = 자연(≈5.9자/초) × 실효배속. atempo 1.18 + 무음제거(÷0.84) = 실효 ~1.4배 → ≈8.3자/초.
        # (이전 ×9.8은 실효 1.66배라 "말이 너무 빨라" — 2026-05-25 ×8.3으로 정상화. atempo·배율 둘 다 낮춤.)
        # ×8.3 = 영상길이에 '맞춤'. 넘치면 길이가드가 압축=다시 빨라지니 살짝 under-fill 쪽으로.
        target_chars = max(80, round(dur_sec * 8.3)) if dur_sec else 0
        length_note = ""
        if dur_sec:
            length_note = (
                f"\n\n[⏱ 영상 길이 = 약 {dur_sec:.0f}초 — 이 길이에 '맞춰라' / 매우 중요]\n"
                f"대본은 줄 사이 끊김 없이 쭉 이어 읽는다. 또박또박 낭독 기준 **총 길이가 영상({dur_sec:.0f}초)과 비슷**해야 한다.\n"
                f"→ 공백 제외 글자수 **약 {target_chars}자**(±10%). ⚠️ 이보다 많이 넘기지 마라 — 넘치면 음성을 억지로 빠르게 눌러야 해서 부자연스럽다. 너무 짧아도 뒤가 빈다.\n"
                f"짧은 줄 여러 개로 나눠 쓰고, 마지막 클로징은 영상 끝 무렵에. start는 영상 내 대략 시각(초, 오름차순)."
            )
        meta = await call_gemini(GEMINI_PRO_MODEL, file_uri, DUB_PROMPT + length_note,
                                 temperature=0.5)
        meta = _strip_banned_meta(meta)  # 참고 채널명(딸기우유/오풍 등) 제거
        # lines 정규화: [{start, text}] (구버전 문자열 리스트도 fallback)
        norm = []
        for x in (meta.get("lines") or []):
            if isinstance(x, dict):
                txt = str(x.get("text", "")).strip()
                try:
                    st = float(x.get("start"))
                except Exception:
                    st = None
            else:
                txt, st = str(x).strip(), None
            if txt:
                norm.append({"start": st, "text": txt})
        if not norm:
            raise RuntimeError("Gemini 대본 lines 비어있음")
        # start 없는 줄은 영상 길이에 균등 분배
        if any(n["start"] is None for n in norm):
            span = dur_sec or (len(norm) * 3.0)
            for i, n in enumerate(norm):
                if n["start"] is None:
                    n["start"] = span * i / max(1, len(norm))
        norm.sort(key=lambda n: n["start"])
        if dur_sec:  # Gemini가 영상 길이 넘는 start 주는 것 방지
            for n in norm:
                n["start"] = min(max(0.0, n["start"]), dur_sec)
        # 외톨이(아주 짧은) 줄을 인접 줄과 병합 — "거기"/"있니?" 같은 단독 자막 방지
        merged = []
        for n in norm:
            if merged and len(n["text"].replace(" ", "")) < 6:
                merged[-1]["text"] = (merged[-1]["text"] + " " + n["text"]).strip()
            else:
                merged.append(n)
        if len(merged) >= 2 and len(merged[0]["text"].replace(" ", "")) < 6:
            merged[1]["text"] = (merged[0]["text"] + " " + merged[1]["text"]).strip()
            merged[1]["start"] = merged[0]["start"]
            merged = merged[1:]
        norm = merged

        N = len(norm)
        if make_tts:
            # 2. 줄별 TTS + 무음제거 → **연속(gapless) 배치** (줄 사이 무음 없음) + 청크 SRT
            #    ⚠️ start로 adelay 무음 넣던 방식 폐기(대표님: 무음구간 생김). 그냥 쭉 이어붙인다.
            #    길이는 Gemini가 영상길이만큼 채우게 유도 + 끝에서 미세 균일압축으로만 보정.
            db.update_tts_dub_job(job_id, status="synthesizing", progress=45,
                                  progress_message="타입캐스트 TTS 생성 중..")
            srt, listf, cursor, idx = [], [], 0.0, 0
            for i in range(N):
                n = norm[i]
                text = n["text"]
                raw = out / f"raw_{i + 1:02d}.mp3"
                wav = out / f"line_{i + 1:02d}.wav"
                await _tts_line(text, voice, raw)
                await _ff("-i", str(raw), "-af", TRIM, "-ar", "44100", "-ac", "1", str(wav))
                d = await _dur(wav)
                start = cursor  # 연속 배치 — 직전 줄 바로 뒤에 붙음(무음 없음)
                listf.append(wav)
                chunks = _split_chunks(text)
                chars = [max(1, len(c.replace(" ", ""))) for c in chunks]
                tot = sum(chars)
                lt = start
                for c, ch in zip(chunks, chars):
                    cd = d * ch / tot
                    idx += 1
                    srt.append((idx, lt, lt + cd, _wrap2(c)))
                    lt += cd
                cursor = start + d
                db.update_tts_dub_job(job_id, progress=45 + int(45 * (i + 1) / N))

            # 3. concat → 전체 길이 측정 → 영상 초과 시 균일 압축(오디오+SRT 동시) → 출력
            listtxt = out / "list.txt"
            listtxt.write_text("".join(f"file '{p}'\n" for p in listf), encoding="utf-8")
            full = out / "full.wav"
            await _ff("-f", "concat", "-safe", "0", "-i", str(listtxt), str(full))
            total = await _dur(full)
            final_len = total
            if dur_sec and total > dur_sec + 0.15:
                # 전체가 영상보다 길면 → 균일 배속으로 영상에 딱 맞춤 (자막 시간도 동일 비율 스케일)
                factor = min(1.6, total / dur_sec)
                fitted = out / "full_fit.wav"
                await _ff("-i", str(full), "-af", f"atempo={factor:.4f}", str(fitted))
                full = fitted
                srt = [(n, a / factor, b / factor, tx) for (n, a, b, tx) in srt]
                final_len = await _dur(full)
            srt_path = out / "subtitle.srt"
            srt_path.write_text(
                "".join(f"{n}\n{_fmt(a)} --> {_fmt(b)}\n{tx}\n\n" for n, a, b, tx in srt),
                encoding="utf-8")
            tts_mp3 = out / "tts.mp3"
            await _ff("-i", str(full), "-b:a", "192k", str(tts_mp3))
        else:
            # 2'. TTS 생략 (대표님 0614) — 대본 줄 start 기반으로만 SRT 생성 (음성 없음)
            db.update_tts_dub_job(job_id, status="synthesizing", progress=75,
                                  progress_message="자막 생성 중 (TTS 생략)..")
            srt, idx = [], 0
            for i in range(N):
                n = norm[i]
                start = float(n["start"] or 0)
                end = float(norm[i + 1]["start"]) if i + 1 < N else ((dur_sec or 0) or start + 3.0)
                if end <= start:
                    end = start + 2.0
                chunks = _split_chunks(n["text"])
                chars = [max(1, len(c.replace(" ", ""))) for c in chunks]
                tot = sum(chars)
                span = end - start
                lt = start
                for c, ch in zip(chunks, chars):
                    cd = span * ch / tot
                    idx += 1
                    srt.append((idx, lt, lt + cd, _wrap2(c)))
                    lt += cd
            srt_path = out / "subtitle.srt"
            srt_path.write_text(
                "".join(f"{n}\n{_fmt(a)} --> {_fmt(b)}\n{tx}\n\n" for n, a, b, tx in srt),
                encoding="utf-8")
            tts_mp3 = None
            final_len = dur_sec or 0

        # BGM 믹스 (Runaway trim + SFX 자동 매핑) — 캡컷에서 별도 트랙
        # 길이는 실제 영상 길이 기준 (final_len은 atempo 후 TTS 길이라 영상보다
        # 짧을 수 있음 — 영상보다 짧으면 영상 끝 BGM 비어버림)
        bgm_mix_path = None
        try:
            from workers.dub_bgm_mixer import make_dub_bgm_mix
            bgm_out = out / "bgm_mix.mp3"
            try:
                video_real_dur = await _dur(vp)
            except Exception:
                video_real_dur = 0.0
            # 영상 길이와 TTS 길이 중 큰 값 기준 + CCTV 2초는 mixer가 자동 추가
            bgm_base_dur = max(float(final_len or 0), float(video_real_dur or 0))
            r = await make_dub_bgm_mix(bgm_base_dur, bgm_out, video_path=vp)
            bgm_mix_path = r["path"]
            print(f"  BGM 믹스 생성: {bgm_mix_path} "
                  f"(영상={video_real_dur:.1f}s / TTS={final_len:.1f}s "
                  f"/ base={bgm_base_dur:.1f}s / 출력={r.get('total_dur')}s "
                  f"/ SFX {r.get('sfx_count', 0)}개)",
                  flush=True)
        except Exception as e:
            print(f"  ⚠️ BGM 믹스 실패: {e}", flush=True)

        # 마지막 장면 CCTV 실사 변환 png (캡컷에서 끝장면 컷)
        try:
            from workers.cctv_last_frame import attach_cctv_frame
            cr = await attach_cctv_frame(vp, out)
            if cr:
                print(f"  ✅ CCTV 변환 생성: {cr['path']} "
                      f"(timestamp={cr.get('timestamp')}, "
                      f"{cr.get('wall_sec')}s)", flush=True)
        except Exception as e:
            print(f"  ⚠️ CCTV 변환 실패 (더빙 잡 OK): {e}", flush=True)

        db.update_tts_dub_job(
            job_id, status="completed", progress=100, progress_message="끝",
            result_json=meta, srt_path=str(srt_path),
            tts_path=(str(tts_mp3) if tts_mp3 else None),
            duration_sec=final_len, cost_usd=0.06,
            completed_at=datetime.utcnow().isoformat())
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.update_tts_dub_job(job_id, status="failed", error=str(e)[:500],
                              progress_message=f"실패: {str(e)[:200]}")
