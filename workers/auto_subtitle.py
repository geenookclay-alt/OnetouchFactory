"""자막 자동 생성 — Gemini 5중 교차검증 + MZ 밈 톤.

흐름:
1. 영상 mp4 → Gemini Files API 업로드
2. 5중 호출 병렬:
   a. Gemini Flash 자세한 분석 (시간별 흐름)
   b. Gemini Flash 단순 분석 (핵심 액션만)
   c. Gemini Pro 분석 (더 정밀)
   d. Frame 추출 + 별도 분석 (시작/끝/마스코트 시점)
   e. (URL 있으면) NexLev 댓글 분석
3. 결과 비교 → 일치 확인
4. 일치 → srt + 제목 생성
5. 불일치 → needs_review=1, 사람 검수 대기

자막 톤: 1분기악 스타일 + MZ + 밈 + 1분기악 시청자 공감 키워드
- 한 줄 12자 미만
- "시전", "광탈", "X됨", "갓복", "ㄷㄷ", "ㅋ", "ㄹㅇ" 같이 신조어
- 대화 형식 ("A: 너냐? / B: 아닌데요?")
- 캐릭터 별명 ("뚱이", "갓비둘", "차주인")
- 마스코트 등장 시 무조건 자막 들어감
"""
import os
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Any
import httpx

from api import database as db

# Gemini API
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta"  # resumable upload용 별도
GEMINI_FLASH_MODEL = "gemini-3.5-flash"
GEMINI_PRO_MODEL = "gemini-3.5-flash"

# 🔴 동시 Gemini 호출 제한 (대표님 0608: 6시 429 rate limit로 자막 17~25분 → 한꺼번에 최대 4개만).
#   자막메뉴(call_gemini)·영상업로드·쇼츠 영상분석 공유. 또 429 잦으면 3으로, 느리면 6으로 조절.
_GEMINI_SEM = asyncio.Semaphore(4)

# 자막 생성 prompt — MZ + 밈 + 1분기악 스타일
SUBTITLE_GENERATION_PROMPT = """이 영상으로 한국 1분 쇼츠 자막 + 쨉쨉이 + 제목 만들어줘.
1분기악 / 빌런참교육 / 도그한판 같은 채널 톤.

═══════════════════════════════════════
[너는 누구]
═══════════════════════════════════════

297개 터진 쇼츠 자막 본 MZ 친구 + 영상 주제에 대한 모든 지식 동원하는 작가.
영상 보고 받아쓰는 거 = 평범 죽음.
영상 + 영상 너머 외부 맥락 = 바이럴.

═══════════════════════════════════════
[1단계] 영상 주제에 대한 모든 외부 정보 동원
═══════════════════════════════════════

영상 보고 무슨 주제인가 먼저 파악.
그 다음 그 주제에 대한 **가능한 모든 외부 지식** 떠올림:

- **이유** (왜 이걸 하는가? / 왜 만들어졌는가?)
- **회사/만든 사람** (이거 만든 회사명, 창립자, 어디서 시작)
- **숫자/통계** (몇 명 살렸는가, 몇 % 성공률, 얼마나 빠른가)
- **가격** (이거 한 개에 얼마, 비용)
- **역사** (몇 년 됐는가, 처음 만든 시기)
- **반전** (사람들이 모르는 숨은 사실)
- **위험/한계** (한 번 쓰면 못 씀, 사용 후 부작용)
- **궁금증** (이건 어떻게 작동하는가, 왜 이런 모양)
- **유명 사례** (이걸로 살아남은 유명한 사람, 큰 사건)
- **비교** (다른 거랑 비교, 더 빠름/싸/안전)

예 (사출 좌석 영상):
- 이유: 비상 시 조종사 살리려고
- 회사: Martin-Baker (영국 회사, 80년 역사)
- 통계: 전 세계 7,700명+ 살림
- 가격: ~30억원 / 1개
- 역사: 1944년 시작
- 반전: 한 번 쓰면 다시 못 씀
- 위험: 작동 시 척추/갈비뼈 부러질 충격 (20G+)
- 궁금: 0.3초 만에 발사
- 유명 사례: 우크라이나 전쟁에서도 사용
- 비교: 헬멧만으로 못 살림 — 이게 진짜 답

═══════════════════════════════════════
[2단계] 자막 4종 만들기
═══════════════════════════════════════

1. **상단 고정 타이틀** (title): 12~16자. 외부 맥락 후크.
2. **상황 설명** (situation_subtitles): 한 줄 12자 미만. 1~3초 단위.
   ⚠️ **모든 청크에 외부 맥락 / 위트 / 별명 / 정보 1개 이상 박기**.
   ⚠️ "갑자기 의자가 발사" 같은 평범 받아쓰기 절대 X.
3. **쨉쨉이** (jjap_jjap_i_subtitles): 중앙 강조 + **3초에 1개 의무**:
   - **영상 길이 ÷ 3 = 최소 개수** (9초 영상 = 최소 3개 / 25초 = 8개 / 40초 = 13개)
   - 각 쨉쨉이 길이 1~1.5초 (짧고 임팩트)
   - ⚠️ 1개만 박고 6초 길게 끄는 거 절대 X
   - 패턴:
     - `* 단어 *` (리액션/감정/관찰만: "* 긴장 *", "* 시무룩 *", "* 소름 *", "* 대박 *", "* 초집중 *", "* 해맑 *")
     - ⚠️ 효과음(쾅·펑·슈우웅·탁·짠·휘청 같은 의성어)은 절대 만들지 마. 감정·리액션·관찰만. 괄호( )는 쓰지 말고 전부 `* 단어 *` 별표로 표기.
     - `?? ???` (의문/충격: "??", "???")
     - `ㄷㄷ ㅋㅋ ㄹㅇ` (약자)
     - `A: ~ / B: ~` (대화 밈)
4. **대사** (dialogue_subtitles): 영어 있을 때만 한국어 의역.

═══════════════════════════════════════
[자막 본문 청크 — 외부 맥락 박기 예시]
═══════════════════════════════════════

평범 (❌):
- "갑자기 의자가 발사"
- "엄청난 화염 폭발"
- "하늘로 솟구쳐 오름"
- "낙하산 펼쳐지고"
- "안전하게 착지 완료"

외부 맥락 + 위트 (✅):
- "30억짜리 한 방에 폭발"        ← 가격
- "0.3초 만에 발사"              ← 통계
- "100m 위까지 솟구침"           ← 숫자
- "갈비뼈 부러질 각오로"         ← 위험
- "이래서 7,700명이 살았음"      ← 통계 + 결과
- "Martin-Baker 갓의자"          ← 회사 + 별명
- "한 번 쓰면 다시 못 씀 ㄷㄷ"   ← 반전 + 충격

═══════════════════════════════════════
[절대 룰 2개]
═══════════════════════════════════════

① 첫 자막 (situation_subtitles[0]) 무조건 "~데/~인데/~하는데/~다는데"로 끝남.
   영상 끝까지 본 후 + **외부 맥락 활용 후킹**.
   ⚠️ 평범 묘사 X. 외부 맥락 / 반전 / 호기심 박기.

② 자막 시간 = 영상 진짜 액션 시점 정확 매칭 (어긋나면 범죄).
   - 자막 사이 0.05~0.1초 gap
   - 마지막 자막 영상 끝까지 fill

═══════════════════════════════════════
[그 외는 자유]
═══════════════════════════════════════

- 알잘딱깔센으로 자유.
- 신박/위트/MZ 단어 자연 박기 (강제 X — 어울리면).
- 평범 한국어도 어울리면 OK.
- ⚠️ **확신 없는 사실 X** — 추측 박지 X (예: "마네킹이라는데" 확신 없으면 X).
- ⚠️ **이상한 강제 신조어 X**.
- 매너리즘 (같은 단어 / 같은 어미 연속) 피해.

═══════════════════════════════════════
[영상 종류별 톤 — 후킹 방향]
═══════════════════════════════════════

위 "외부 맥락(가격/통계/회사)" 후크는 강력하지만 **영상 종류에 맞을 때만** 쓴다.
- **강아지·반려동물·동물 영상** → **돈/몸값으로만 후킹하지 마라** (대표님 지적: 돈 후킹 경향이 과함):
  - 이런 영상의 진짜 후크 = 귀여움·표정·행동·성격·반전·교감·사연·웃긴 순간. **그 매력으로 후킹**해라.
  - 🚫 "몸값 얼마", "분양가 ○○○만원", "○천만원짜리 댕댕이" 식 **가격 후크를 디폴트로 깔지 X**.
  - 가격·품종값은 **그게 진짜 영상의 핵심일 때만** (드묾). 평소엔 행동·감정·반전으로.
  - 외부 맥락을 쓰더라도 견종 특성·습성·행동 이유 같은 **'결에 맞는 정보'**로 (돈 X).

═══════════════════════════════════════
[좋은 자막 예시]
═══════════════════════════════════════

영상: 사출 좌석 테스트 (9초)
✅ 자막:
1. "조종사 살리는 30억짜리 의자라는데"   ← 가격 후크
2. "Martin-Baker 갓의자 발동"             ← 회사 + 별명
3. "0.3초 만에 폭발 발사"                 ← 통계
4. "갈비뼈 부러질 충격으로"               ← 위험
5. "100m 위에서 낙하산 펼침"              ← 숫자
6. "이래서 7,700명이 살았음"              ← 통계 + 결과

영상: 여드름 패치
✅ 자막:
1. "여드름 짜는 진짜 방법 있다는데"
2. "맨손 X (감염 위험)"                   ← 위험
3. "도구도 X (흉터 남음)"                 ← 결과
4. "정답은 하이드로콜로이드 패치"         ← 정보
5. "병원에서도 쓰는 방법"                 ← 신뢰

═══════════════════════════════════════
[영상 분석 정확히]
═══════════════════════════════════════

- 옷색/외모로 누구인지 식별
- 카메라 시점 = 누구의 시점
- 영상 끝 (마지막 1초) 누가 승/패
- 핵심 액션 주체 + 대상
- 마스코트 등장 구간 식별

═══════════════════════════════════════
[출력 — JSON만]
═══════════════════════════════════════

```json
{
  "duration_sec": 9.06,
  "summary": "줄거리 + 외부 맥락 한 줄",
  "external_context": "이 주제에 대한 외부 지식 모음 (회사/가격/역사/통계/위험 등 떠올린 거)",
  "characters": [{"role": "역할", "appearance": "옷색", "nickname": "별명"}],
  "mascot_appearances": [],
  "title": "상단 고정 (12~16자, 외부 맥락 후크)",
  "youtube_upload_title": "YouTube 업로드용 (35~60자 + #shorts)",
  "youtube_upload_title_candidates": ["후보1", "후보2", "후보3", "후보4", "후보5"],
  "youtube_description": "유튜브 설명 + 해쉬태그",
  "hashtags": ["#쇼츠", "...8~12개"],
  "title_candidates": [
    "후보1", "후보2", "후보3", "후보4",
    "후보5", "후보6", "후보7", "후보8"
  ],
  "situation_subtitles": [
    {"start": 0.0, "end": 1.5, "text": "첫 자막 ~데 + 외부 맥락"}
  ],
  "jjap_jjap_i_subtitles": [
    {"start": 1.5, "end": 2.5, "text": "* 강조 *"}
  ],
  "dialogue_subtitles": [],
  "key_actions": [
    {"time_sec": 2.0, "action": "핵심 액션", "subject": "주체", "target": "대상"}
  ]
}
```

[중요] JSON만 출력. 자연어 X. 자막 12자 미만. **모든 자막 청크에 외부 맥락 박기**.
"""

# 단순 검증 prompt (2차)
SIMPLE_VERIFY_PROMPT = """이 영상을 보고 다음 4가지만 정확히 한 줄씩 답해줘:

1. 영상 시점 = 어떤 캐릭터 시점 (옷 색/외모로)
2. 영상 끝 (마지막 1초) — 누가 살아있고 누가 쓰러져 있는지
3. 핵심 액션 = 누가 누구에게 (예: "가짜 NPC가 차주인을 사살")
4. 마스코트 (chibi 캐릭터) 등장 시간 — 시작~끝초

JSON으로:
{
  "camera_pov": "옷색/외모 묘사",
  "ending_alive": "살아있는 캐릭터",
  "ending_dead": "쓰러진 캐릭터",
  "key_action_subject": "주체",
  "key_action_target": "대상",
  "mascot_times": [{"start": 2.0, "end": 3.5}]
}
"""


import contextvars
# 작업 실행 시 해당 사용자의 개인 Gemini 키 (있으면 글로벌 대신 이걸 씀 — 프리랜서 비용 분리)
_user_gemini_key = contextvars.ContextVar("_user_gemini_key", default=None)


def apply_user_gemini_key(user_id) -> bool:
    """user_id의 저장된 개인 Gemini 키가 있으면 contextvar에 설정. 작업 워커 시작 시 호출."""
    if not user_id:
        return False
    try:
        from api import auth
        u = auth.get_user_by_id(int(user_id))
        key = (u or {}).get("gemini_api_key")
        if key and str(key).strip():
            _user_gemini_key.set(str(key).strip())
            print(f"🔑 사용자 {user_id} 개인 Gemini 키 사용", flush=True)
            return True
    except Exception as e:
        print(f"⚠️ 개인 키 조회 실패(글로벌 사용): {e}", flush=True)
    return False


def _get_gemini_key() -> str:
    # 사용자 개인 키 우선 (작업 실행 컨텍스트에 설정돼 있으면)
    uk = _user_gemini_key.get()
    if uk:
        return uk
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # .env 파일에서 읽기
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GEMINI_API_KEY not set")

def _get_youtube_key() -> str | None:
    """YouTube Data API key — .env에서 받음"""
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        return key
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("YOUTUBE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _extract_youtube_video_id(url: str) -> str | None:
    """YouTube URL에서 video_id 추출"""
    import re
    if not url:
        return None
    # https://www.youtube.com/watch?v=XXXX
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # https://youtu.be/XXXX
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # https://www.youtube.com/shorts/XXXX
    m = re.search(r"/shorts/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return None


async def fetch_youtube_comments_for_urls(urls: list[str], per_video: int = 20) -> dict:
    """YouTube URL list → 각 영상별 제목 + 인기 댓글 top N개.

    Returns: {video_id: {"url": url, "title": "...", "description": "...",
                          "channel": "...", "comments": [...]}}
    """
    api_key = _get_youtube_key()
    if not api_key or not urls:
        return {}

    # URL → video_id 매핑
    url_to_vid = {}
    for url in urls:
        vid = _extract_youtube_video_id(url)
        if vid:
            url_to_vid[vid] = url
    if not url_to_vid:
        return {}

    result: dict = {}
    async with httpx.AsyncClient(timeout=30.0) as c:
        # 1) 영상 metadata (제목/설명/채널) 한 번에 받기 (max 50개)
        try:
            ids_param = ",".join(list(url_to_vid.keys())[:50])
            r_meta = await c.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "snippet",
                    "id": ids_param,
                    "key": api_key,
                },
            )
            if r_meta.status_code == 200:
                for item in r_meta.json().get("items", []):
                    vid = item.get("id")
                    sn = item.get("snippet", {})
                    result[vid] = {
                        "url": url_to_vid.get(vid, ""),
                        "title": (sn.get("title") or "")[:300],
                        "description": (sn.get("description") or "")[:500],
                        "channel": sn.get("channelTitle", ""),
                        "comments": [],
                    }
        except Exception:
            pass

        # 2) 영상별 댓글 받기
        for vid, url in url_to_vid.items():
            if vid not in result:
                result[vid] = {"url": url, "title": "", "description": "",
                                "channel": "", "comments": []}
            try:
                r = await c.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params={
                        "part": "snippet",
                        "videoId": vid,
                        "maxResults": per_video,
                        "order": "relevance",
                        "key": api_key,
                    },
                )
                if r.status_code != 200:
                    continue
                items = r.json().get("items", [])
                comments = []
                for it in items:
                    sn = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    comments.append({
                        "text": (sn.get("textOriginal") or sn.get("textDisplay") or "")[:500],
                        "author": sn.get("authorDisplayName", ""),
                        "likes": sn.get("likeCount", 0),
                    })
                result[vid]["comments"] = comments
            except Exception:
                continue
    return result


def format_comments_for_prompt(comments_dict: dict) -> str:
    """원본 영상 metadata + 댓글 → Gemini prompt inject용 text"""
    if not comments_dict:
        return ""
    lines = ["\n\n[원본 영상들 정보 — 제목·채널·댓글 참고로 영상 흐름 유추]"]
    for vid, data in comments_dict.items():
        url = data.get("url", "")
        title = data.get("title", "")
        channel = data.get("channel", "")
        description = data.get("description", "")
        cmts = data.get("comments", [])
        if not (title or cmts):
            continue
        lines.append(f"\n📺 {url}")
        if channel:
            lines.append(f"  채널: {channel}")
        if title:
            lines.append(f"  제목: {title}")
        if description and len(description) > 5:
            desc_short = description.replace("\n", " ")[:200]
            lines.append(f"  설명: {desc_short}")
        if cmts:
            lines.append(f"  인기 댓글:")
            for i, c in enumerate(cmts, start=1):
                t = (c.get("text") or "").replace("\n", " ")[:200]
                likes = c.get("likes", 0)
                lines.append(f"    {i}. ({likes}👍) {t}")
    return "\n".join(lines)


async def upload_video_to_gemini(file_path: Path, max_retries: int = 6) -> str:
    """Gemini 영상 업로드. 18MB 미만은 inline 우선 (Files API 불안정 회피), 큰 영상만 Files API."""
    # 18MB 미만 → inline 우선 (Files API 안 거치고 base64 직접 — 장애 무관, 빠름)
    try:
        size = file_path.stat().st_size
    except Exception:
        size = 0
    if 0 < size < 18_000_000:
        return f"inline:{file_path}"

    # 18MB 이상만 Files API (inline 요청 크기 한계 초과)
    last_err = None
    for attempt in range(max_retries):
        try:
            async with _GEMINI_SEM:
                return await asyncio.wait_for(_upload_video_once(file_path), timeout=60)
        except (RuntimeError, asyncio.TimeoutError) as e:
            msg = str(e) or "TimeoutError"
            last_err = e
            if isinstance(e, asyncio.TimeoutError):
                msg = "503 timeout"  # timeout도 일시 에러 취급
            # 일시 에러 (서버 과부하) → backoff 후 재시도
            if any(w in msg.lower() for w in ("depleted", "prepay")):
                raise RuntimeError("Gemini 결제 크레딧 소진 — 충전 필요: " + msg[:120])
            if any(code in msg for code in ("503", "500", "502", "429", "UNAVAILABLE", "overloaded")):
                wait = min(2 ** attempt, 30)  # 1,2,4,8,16
                print(f"⚠️ Gemini 업로드 재시도 {attempt+1}/{max_retries} ({wait}s 후): {msg[:80]}", flush=True)
                await asyncio.sleep(wait)
                continue
            raise  # 일시 에러 아니면 즉시 실패
    # Files API 다 실패 → inline data fallback (18MB 미만 영상만)
    try:
        size = file_path.stat().st_size
    except Exception:
        size = 0
    if 0 < size < 18_000_000:
        print(f"⚠️ Files API 실패 → inline 모드 fallback ({size//1024//1024}MB)", flush=True)
        return f"inline:{file_path}"
    raise last_err or RuntimeError(f"Gemini 업로드 실패 (영상 {size//1024//1024}MB, inline 한계 초과)")


async def ensure_inline_video(video_path: Path) -> Path:
    """18MB 넘는 영상은 저화질 압축본(오디오 보존) 생성 → inline 업로드 가능.
    Files API(불안정) 회피. 분석은 저화질로 충분 (자막 타이밍/내용/노래 식별 OK).
    길이로 비트레이트 산정해 어떤 길이든 ~13MB 목표."""
    try:
        size = video_path.stat().st_size
    except Exception:
        size = 0
    if 0 < size < 17_000_000:
        return video_path   # 이미 inline 가능

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        dur = float(out.decode().strip() or 0)
    except Exception:
        dur = 0.0
    if dur <= 0:
        dur = 60.0

    small = video_path.parent / f"_inline_{video_path.stem}.mp4"
    target_bytes = 13_000_000
    audio_k = 96
    audio_bytes = audio_k * 1000 / 8 * dur
    vbit = int(max(150, min(1500, (target_bytes - audio_bytes) * 8 / 1000 / dur)))
    scale = 480 if dur < 600 else 360
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path), "-vf", f"scale=-2:{scale}",
            "-c:v", "libx264", "-b:v", f"{vbit}k", "-maxrate", f"{vbit}k",
            "-bufsize", f"{vbit}k", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", f"{audio_k}k", "-y", str(small),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=900)
    except Exception as e:
        print(f"⚠️ 인라인 압축 실패: {e} → 원본 사용", flush=True)
        return video_path
    if small.exists() and 0 < small.stat().st_size < 18_000_000:
        print(f"📦 자막 분석본: {size//1024//1024}MB → {small.stat().st_size//1024//1024}MB "
              f"({scale}p {vbit}k, {dur:.0f}s)", flush=True)
        return small
    print("⚠️ 압축본 초과/실패 → 원본 사용", flush=True)
    return video_path


async def _upload_video_once(file_path: Path) -> str:
    """Gemini Files API 업로드 1회 시도."""
    api_key = _get_gemini_key()
    file_size = file_path.stat().st_size
    async with httpx.AsyncClient(timeout=90.0) as c:
        # Start resumable upload (upload subdomain 사용 필요)
        start = await c.post(
            f"{GEMINI_UPLOAD_URL}/files",
            headers={
                "x-goog-api-key": api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": "video/mp4",
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": file_path.name}},
        )
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError(
                f"Gemini upload URL fail: status={start.status_code} "
                f"headers={dict(start.headers)} body={start.text[:200]}"
            )
        # Upload body
        data = file_path.read_bytes()
        up = await c.post(
            upload_url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=data,
        )
        if up.status_code != 200:
            raise RuntimeError(f"Gemini upload fail: {up.status_code} {up.text[:200]}")
        info = up.json().get("file", {})
        uri = info.get("uri")
        state = info.get("state")
        # Wait for ACTIVE
        if state != "ACTIVE":
            for _ in range(40):
                await asyncio.sleep(2)
                check = await c.get(uri, headers={"x-goog-api-key": api_key})
                state = check.json().get("state")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise RuntimeError("Gemini file processing FAILED")
        return uri


def _gemini_media_part(file_uri: str) -> dict:
    """file_uri가 'inline:<path>'면 base64 inline_data, 아니면 file_data."""
    if file_uri.startswith("inline:"):
        import base64
        vpath = file_uri[len("inline:"):]
        vdata = base64.b64encode(Path(vpath).read_bytes()).decode()
        return {"inline_data": {"mime_type": "video/mp4", "data": vdata}}
    return {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}}


async def call_gemini(model: str, file_uri: str, prompt: str,
                       temperature: float = 0.3, max_retries: int = 6,
                       fallback_chain: tuple = ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash")) -> dict:
    """Gemini 호출 + JSON 응답 받기. JSON 파싱 fail 시 자동 retry.
    503/429 retry 끝나도 fail이면 fallback_chain 순서대로 다른 모델 시도 (2.5 Pro→2.5 Flash→1.5 Pro→1.5 Flash)."""
    api_key = _get_gemini_key()
    last_err: Exception | None = None
    _original_model = model
    # 시도별 max tokens 증가 (응답 잘림 대응)
    max_tokens_per_try = [16384, 24576, 32768]
    for attempt in range(max_retries):
        max_tokens = max_tokens_per_try[min(attempt, len(max_tokens_per_try) - 1)]
        # 재시도 시 temperature 약간 변동 (다른 응답 유도)
        actual_temp = temperature + (0.05 * attempt)
        try:
            async with _GEMINI_SEM, httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [
                                _gemini_media_part(file_uri),
                                {"text": prompt},
                            ]
                        }],
                        "generationConfig": {
                            "temperature": actual_temp,
                            "maxOutputTokens": max_tokens,
                            "responseMimeType": "application/json",
                        },
                    },
                )
                if r.status_code != 200:
                    # 503/429/500 일시 에러는 대기 후 재시도 (over capacity / rate limit)
                    if r.status_code == 429 and any(w in (r.text or "").lower() for w in ("depleted", "prepay", "billing")):
                        raise RuntimeError("Gemini 결제 크레딧 소진 — 충전 필요(AI Studio billing). 재시도 무의미: " + (r.text or "")[:120])
                    if r.status_code in (429, 500, 502, 503, 504):
                        if attempt < max_retries - 1:
                            wait = min(5.0 * (2 ** attempt), 60.0)  # 지수백오프 5,10,20,40,60
                            print(f"  [gemini] {r.status_code} 일시 에러 — {wait:.0f}초 대기 후 재시도 ({attempt+1}/{max_retries})", flush=True)
                            last_err = RuntimeError(f"Gemini {r.status_code} retrying")
                            await asyncio.sleep(wait)
                            continue
                        # 마지막 attempt도 retryable status — fallback으로 흐름
                        last_err = RuntimeError(f"Gemini call fail: {r.status_code} {r.text[:300]}")
                        break
                    # non-retryable (400, 404 등) — 즉시 raise
                    raise RuntimeError(f"Gemini call fail: {r.status_code} {r.text[:300]}")
                result = r.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    # ```json ``` 둘러싸인 경우 처리
                    cleaned = text
                    if "```" in cleaned:
                        cleaned = cleaned.split("```")[1].lstrip("json\n").rstrip("`").strip()
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        # 마지막 시도면 실패. 아니면 retry
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Gemini JSON 파싱 실패 (retry {max_retries}회 모두 fail): {e} "
                                f"— text[:200]={text[:200]}"
                            ) from e
                        last_err = e
                        continue
        except RuntimeError:
            raise
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break  # fallback으로
            continue
    # max_retries 다 소진 — fallback_chain 순서대로 다른 모델 시도
    for fb_model in (fallback_chain or ()):
        if fb_model == _original_model:
            continue
        print(f"  [gemini] {_original_model} fail → fallback {fb_model} 시도", flush=True)
        try:
            return await call_gemini(fb_model, file_uri, prompt,
                                       temperature=temperature,
                                       max_retries=max_retries,
                                       fallback_chain=())  # 무한 fallback 방지
        except Exception as e:
            print(f"  [gemini] fallback {fb_model} 도 fail: {str(e)[:120]}", flush=True)
            last_err = e
            continue
    if last_err:
        raise RuntimeError(f"Gemini 모든 모델 fail: {last_err}") from last_err
    raise RuntimeError("Gemini 호출 알 수 없는 실패")


async def extract_key_frames(video_path: Path, work_dir: Path) -> dict:
    """영상의 핵심 frame 추출 (시작 0.5초, 중간, 끝).
    Returns: {"start": Path, "middle": Path, "end": Path, "duration": float}
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # 영상 길이
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    duration = float(out.decode().strip() or "0")
    times = {
        "start": 0.5,
        "middle": duration / 2,
        "end": max(duration - 0.5, 0.1),
    }
    out_paths = {"duration": duration}
    for name, t in times.items():
        out_path = work_dir / f"frame_{name}.jpg"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "2",
            str(out_path),
        )
        await proc.wait()
        out_paths[name] = out_path
    return out_paths


async def extract_scene_changes(video_path: Path) -> dict:
    """ffmpeg로 객관 시점 추출 — scene change + keyframe + audio peak.
    Returns: {
        "duration": float,
        "keyframes": [0.0, 6.5, ...],       # I-frame 시점 (압축 keyframe)
        "scene_changes": [1.2, 4.8, ...],    # 픽셀 변화 큰 시점 (scene > 0.3)
        "audio_peaks": [0.5, 2.1, ...],      # 음성 큰 시점 (silencedetect 역)
        "all_anchors": [0.0, 0.5, 1.2, ...], # 위 3개 합치고 dedup·정렬
    }
    이 anchor 시점들이 자막 시작/끝 후보로 진짜 정확 (±0.05초).
    """
    result = {"duration": 0.0, "keyframes": [], "scene_changes": [], "audio_peaks": [], "all_anchors": []}
    # 영상 길이 — format(컨테이너)은 오디오가 비디오보다 길면 더 길게 나옴
    #   → 자막이 영상보다 더 이어지는 문제. video stream 길이를 우선 사용.
    fmt_dur = 0.0
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        fmt_dur = float(out.decode().strip() or "0")
    except Exception:
        return result
    vid_dur = 0.0
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        vid_dur = float((out.decode().strip() or "0").replace("N/A", "0") or "0")
    except Exception:
        vid_dur = 0.0
    # 비디오 스트림 길이가 유효하면 그걸로(영상 실제 끝), 아니면 format
    result["duration"] = vid_dur if vid_dur > 0 else fmt_dur

    # Keyframe 시점 (I-frame)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        kfs = []
        for line in out.decode().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2 and "K" in parts[1]:
                try: kfs.append(round(float(parts[0]), 3))
                except: pass
        result["keyframes"] = sorted(set(kfs))
    except Exception:
        pass

    # Scene change 시점 (픽셀 변화 큰 시점)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path), "-filter:v",
            "select='gt(scene,0.3)',showinfo", "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        scenes = []
        for line in stderr.decode().splitlines():
            if "pts_time:" in line:
                try:
                    t = line.split("pts_time:")[1].split()[0]
                    scenes.append(round(float(t), 3))
                except: pass
        result["scene_changes"] = sorted(set(scenes))
    except Exception:
        pass

    # Audio peak (silence 끝 = 소리 시작 시점)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path), "-af",
            "silencedetect=noise=-30dB:d=0.2", "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        peaks = []
        for line in stderr.decode().splitlines():
            if "silence_end:" in line:
                try:
                    t = line.split("silence_end:")[1].split()[0]
                    peaks.append(round(float(t), 3))
                except: pass
        result["audio_peaks"] = sorted(set(peaks))
    except Exception:
        pass

    # 합쳐서 dedup·정렬 (0.05초 이내는 묶음)
    all_t = sorted(set(result["keyframes"] + result["scene_changes"] + result["audio_peaks"]))
    anchors = []
    for t in all_t:
        if not anchors or t - anchors[-1] > 0.05:
            anchors.append(t)
    result["all_anchors"] = anchors
    return result


def snap_to_anchor(t: float, anchors: list[float], tol: float = 0.25) -> float:
    """시간 t를 가까운 anchor 시점에 snap (tol 이내). 없으면 t 그대로."""
    if not anchors:
        return t
    nearest = min(anchors, key=lambda a: abs(a - t))
    if abs(nearest - t) <= tol:
        return nearest
    return t


def validate_subtitle_timing(subtitles: list[dict], duration: float,
                              anchors: list[float] = None,
                              fill_last: bool = True, no_gap: bool = False) -> list[dict]:
    """자막 후처리 — Gemini 시간 그대로 + duration 초과 clamp + 청크 순서/겹침/최소 길이 강제.
    ⚠️ anchor snap 제거 (2026-05-19) — anchor (keyframe/audio peak)가 진짜 액션 시점 보장 X.
       Gemini가 영상 보고 박은 시간이 진짜 액션에 가까움. snap이 오히려 자막 어긋나게 만듦.
    fill_last: 마지막 자막을 영상 끝까지 늘릴지. 끊김없는 사연/상황설명은 True,
       노래 가사는 False (가사 한 줄이 영상 끝까지 늘어나면 안 됨).
    """
    if not subtitles:
        return subtitles
    MIN_DUR = 0.5      # 자막 최소 길이
    MIN_GAP = 0.03     # 청크 사이 최소 gap

    # 1. 시간 순서 정렬
    subs = sorted([dict(s) for s in subtitles], key=lambda s: s.get("start", 0))
    out = []
    for s in subs:
        start = max(0.0, float(s.get("start", 0)))
        end = float(s.get("end", start + 1))
        # 2. duration 초과 clamp (Gemini 시간 그대로 — snap X)
        if start >= duration:
            continue  # 영상 끝 넘은 자막 drop
        if end > duration:
            end = duration
        # 3. start < end 강제
        if end - start < MIN_DUR:
            end = min(duration, start + MIN_DUR)
        s["start"], s["end"] = start, end
        out.append(s)

    # 5. 청크 사이 겹침 X (앞 청크 end > 다음 start이면 앞 end 조정 — 앞 end 줄임)
    for i in range(len(out) - 1):
        if out[i]["end"] > out[i + 1]["start"] - MIN_GAP:
            # 앞 청크 end를 다음 start - gap으로 줄임
            limit = out[i + 1]["start"] - MIN_GAP
            # 단 너무 짧아지면 (MIN_DUR 미만) 다음 청크 start를 뒤로 밀음
            if limit < out[i]["start"] + MIN_DUR:
                # 앞 청크 보존, 다음 청크 start를 뒤로 밀음
                out[i + 1]["start"] = out[i]["end"] + MIN_GAP
                if out[i + 1]["end"] < out[i + 1]["start"] + MIN_DUR:
                    out[i + 1]["end"] = out[i + 1]["start"] + MIN_DUR
            else:
                out[i]["end"] = limit
    # 5.5 빈공간 제거 (no_gap) — 각 자막 끝을 다음 자막 시작까지 연장 (상황설명 끊김없이 연속)
    if no_gap:
        for i in range(len(out) - 1):
            if out[i + 1]["start"] > out[i]["end"]:
                out[i]["end"] = out[i + 1]["start"]
    # 6. 마지막 청크 end 영상 끝까지 fill (90% 이상) — 끊김없는 사연만 (가사는 X)
    if fill_last and out and out[-1]["end"] < duration * 0.9 and duration - out[-1]["end"] > 0.3:
        out[-1]["end"] = duration
    return out


def cross_validate(primary: dict, simple: dict, frame_check: dict,
                    pro: dict | None = None, comments_dict: dict | None = None) -> dict:
    """5중 결과 비교 — 객관 자료 (영상 길이)만 점검.

    근본 룰 (2026-05-18 대표님 결정):
    - POV/마스코트/액션 주체 비교 X — 자연 언어 표현 다양해서 false alarm 양산
    - 영상 길이 차이 5초 이상만 conflict (영상 분석 자체 잘못된 경우)
    - 그 외는 다 자동 completed
    """
    conflicts = []

    # 영상 길이 차이 5초 이상이면 분석 자체 잘못. 그 외는 모두 통과.
    primary_dur = primary.get("duration_sec", 0)
    frame_dur = frame_check.get("duration", 0)
    if frame_dur > 0 and abs(primary_dur - frame_dur) > 5.0:
        conflicts.append(
            f"영상 길이 불일치: 분석 {primary_dur:.1f}초 vs 실제 {frame_dur:.1f}초"
        )

    comments_video_count = len(comments_dict) if comments_dict else 0
    confidence = 1.0 if not conflicts else 0.5
    return {
        "consistent": len(conflicts) == 0,
        "conflicts": conflicts,
        "confidence": confidence,
        "comments_video_count": comments_video_count,
    }


def _sanitize_subtitle_text(t: str) -> str:
    """SRT에 쓰기 전 자막 텍스트 최종 정리.
    - 양쪽 공백 있는 슬래시(' / ') 제거: Gemini가 "A: ~ / B: ~" 대화 패턴
      학습으로 단독 자막에도 "나보다 팔자 좋은 / 댕댕이가 있다는데"처럼 박는
      사례 → 공백으로 치환. 대화 분기(": " 두 화자)면 줄바꿈으로 치환.
    - 양쪽 공백 없는 슬래시("5/10", "AC/DC")는 유지.
    - 다중 공백 정리.
    """
    import re
    if not t:
        return t
    t = t.rstrip()
    # 대화 분기 ("X: ... / Y: ...") → 줄바꿈
    if re.search(r":\s*\S[^/]*?\s+/\s+\S+\s*:", t):
        t = re.sub(r"\s+/\s+", "\n", t)
    else:
        # 양쪽 공백 슬래시만 공백으로 (5/10 같은 붙은 슬래시는 유지)
        t = re.sub(r"\s+/\s+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def write_srt(subtitles: list[dict], out_path: Path, text_key: str = "text") -> None:
    """자막 list → srt 파일.
    subtitles: [{"start": 0.5, "end": 2.0, "text": "..."}, ...]
    """
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec * 1000) % 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, sub in enumerate(subtitles, 1):
        start = sub.get("start", 0)
        end = sub.get("end", start + 1)
        text = sub.get(text_key) or sub.get("text") or sub.get("korean", "")
        if not text:
            continue
        text = _sanitize_subtitle_text(text)
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# 감성 12자 분할 시 줄 끝에 홀로 두면 어색한 '뒷말 꾸미는' 짧은 단어 → 다음 줄로
_LEADING_MOD = {
    "이", "그", "저", "이런", "그런", "저런", "어떤", "무슨", "웬",
    "너무", "더", "좀", "조금", "아주", "매우", "가장", "제일", "훨씬",
    "정말", "진짜", "참", "다시", "또", "문득", "가만히", "조용히",
    "천천히", "서서히", "안", "못", "막", "딱", "꽤", "꼭", "늘", "항상",
    "그저", "마치", "어쩌면", "왠지",
    # 관형사·수식 (뒷말과 붙어야 자연스러움)
    "첫", "옛", "새", "헌", "온", "전", "매", "각", "여러", "모든", "온갖",
    "어느", "갖은", "뭇", "딴", "한", "두", "세", "네", "다섯", "여섯",
    "일곱", "여덟", "아홉", "열",
}


def _wrap_meaning(text: str, max_chars: int) -> list[str]:
    """한 문장을 max_chars 이하 여러 줄로 — 균형 분할 + 어색한 끊김 방지.
    ① 줄 길이를 고르게 (한 줄만 꽉 채우고 토막 남기지 않음)
    ② 마지막 줄 외톨이(≤4자)면 앞 줄에서 한 어절 당겨 합침 (앞줄이 4자 미만 되면 중단)
    ③ 줄 끝 꾸밈말('이/그/첫/너무'…)이 홀로 남으면 다음 줄로 — 모든 줄 ≤max_chars."""
    raw = text.split()
    if not raw:
        return [text]
    words = []
    for w in raw:
        while len(w) > max_chars:        # 한 단어가 너무 길면 강제 분할
            words.append(w[:max_chars]); w = w[max_chars:]
        if w:
            words.append(w)
    total = len(" ".join(words))
    n = max(1, (total + max_chars - 1) // max_chars)   # 필요한 줄 수
    target = total / n                                  # 줄당 목표 길이 (균형용)
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
            continue
        cand = cur + " " + w
        if len(cand) > max_chars:
            lines.append(cur); cur = w
        elif len(cur) >= target and len(lines) < n - 1:
            lines.append(cur); cur = w        # 목표 채웠고 줄 여유 있으면 균형 위해 끊음
        else:
            cur = cand
    if cur:
        lines.append(cur)
    # ② 마지막 줄 외톨이 합치기
    if len(lines) >= 2 and len(lines[-1]) <= 4:
        prev = lines[-2].split()
        if len(prev) >= 2:
            remain = " ".join(prev[:-1])
            cand = prev[-1] + " " + lines[-1]
            if len(cand) <= max_chars and len(remain) >= 4:
                lines[-2] = remain
                lines[-1] = cand
    # ③ 줄 끝 꾸밈말 다음 줄로 (꼬리합치기 뒤에 — 새로 생긴 끝 꾸밈말까지)
    for i in range(len(lines) - 1):
        toks = lines[i].split()
        if len(toks) >= 2 and toks[-1] in _LEADING_MOD:
            cand = toks[-1] + " " + lines[i + 1]
            if len(cand) <= max_chars:
                lines[i] = " ".join(toks[:-1])
                lines[i + 1] = cand
    return [l for l in lines if l]


def _merge_tiny_cues(subs: list[dict], min_chars: int = 5, max_merged: int = 24) -> list[dict]:
    """Gemini가 과하게 토막낸 짧은 꼬리 자막을 앞(없으면 뒤) 자막에 재병합.
    감성 연속 내레이션 전용 — 이후 _split_long_cues가 균형 재분할.
    (Gemini가 '이 깊이를'을 '…이' + '깊이를'로 갈라 보내는 문제 복구.)"""
    if not subs:
        return subs
    out = []
    for s in subs:
        text = (s.get("text") or "").strip()
        if out:
            prev = out[-1]
            merged = (prev["text"] + " " + text).strip()
            if len(text) <= min_chars and len(merged) <= max_merged:
                prev["text"] = merged
                prev["end"] = s.get("end", prev.get("end"))
                continue
        out.append({**s, "text": text})
    # 첫 칸이 짧으면 다음 칸으로 흡수
    if len(out) >= 2 and len(out[0]["text"]) <= min_chars:
        merged = (out[0]["text"] + " " + out[1]["text"]).strip()
        if len(merged) <= max_merged:
            out[1]["text"] = merged
            out[1]["start"] = out[0].get("start", out[1].get("start"))
            out = out[1:]
    return out


def _split_long_cues(subs: list[dict], max_chars: int = 12) -> list[dict]:
    """자막 한 칸이 max_chars 넘으면 의미(어절) 단위로 '균형' 분할 — 어색한 끊김 방지.
    시간은 글자 수 비례 배분. 감성 스타일 '한 칸 12자' 픽스용."""
    out = []
    for s in subs:
        text = (s.get("text") or "").strip()
        start = float(s.get("start", 0))
        end = float(s.get("end", start))
        if len(text) <= max_chars:
            out.append({**s, "text": text})
            continue
        pieces = _wrap_meaning(text, max_chars)
        if not pieces:
            out.append({**s, "text": text})
            continue
        total = sum(len(p) for p in pieces) or 1
        dur = max(0.0, end - start)
        t = start
        for idx, p in enumerate(pieces):
            pe = end if idx == len(pieces) - 1 else t + dur * len(p) / total
            out.append({**s, "text": p, "start": round(t, 2), "end": round(pe, 2)})
            t = pe
    return out



EMOTION_STORY_PROMPT = """이 영상의 **오디오(노래)**를 듣고 감성 스토리텔링 자막 + 제목 + 설명 만들어줘.

═══ [🚨 가장 중요 — 영상 화면 무시, 오직 노래(오디오)로만 판단] ═══

이 영상의 **화면(영상)은 노래와 무관하게 아무거나 끼워 넣은 영상**이야.
심지어 지금 들리는 노래와 **전혀 다른 노래의 공연 영상**일 수도 있어.
→ **절대 화면(영상)을 보고 판단하지 마.** 화면 속 인물/가수/장면은 이 노래와 무관해.
→ 오직 **귀로 들리는 노래(오디오)** 만으로 모든 걸 판단해.

═══ [1단계 — 오디오로 노래 식별] ═══

들리는 노래를 듣고:
- **노래 제목** 파악 (song_title) — 멜로디 / 가사 / 창법으로
- **이 노래를 부른 가수가 누구인지** 파악
- 잘 모르겠으면 들리는 가사·분위기·시대감으로 최대한 추정 (단 확신 없는 사실을 단정 X)

═══ [2단계 — 감성 상황설명 자막 (메인, situation_subtitles)] ═══

식별한 **노래·가수 이야기를 최대한 풍부하게** 써라. 이게 이 자막의 핵심이다.
**막연한 감성보다 '그 노래·그 가수'에 대한 진짜 이야기가 훨씬 더 많이 들어가야 해** (대표님 요청 — 노래·가수 얘기 더 써줘).

**① 가수 이야기 (구체적으로 — 아는 만큼 깊게)**:
- 이 가수가 누구인지 / 어느 시대·어떤 장르의 가수인지
- 데뷔·전성기·대표곡 / 어떤 삶을 살아온 사람인지 (굴곡·일화·지금 근황)
- 그 가수만의 목소리·창법·특유의 매력

**② 노래 이야기 (구체적으로)**:
- 언제 나온 노래인지 / 그 시절 시대 배경 / 발표 당시 반응·인기
- 이 노래가 만들어진 계기·담긴 사연·숨은 이야기
- 가사가 말하는 정서, 이 노래가 오래 사랑받은 이유

**③ 듣는 사람 이야기**:
- 그 시절 어떤 세대가, 어떤 순간(이별·군 시절·첫사랑·부모님 생각 등)에 들었는지
- 지금 다시 들으면 누가 뭉클한지

→ situation_subtitles의 **대부분을 위 ①②③ '노래·가수 구체 이야기'로 채워라.**
   "세월이 가도…" 같은 막연한 보편 감성은 양념 정도로만, 너무 많이 깔지 마.
→ 화면 내용은 한 마디도 언급 X (영상 무관). 오직 이 '노래·가수' 이야기.
⚠ 단, **확신 없는 사실을 지어내지 마.** 가수·노래가 확실히 파악되면 그 **진짜 정보**를 풍부하게 쓰고,
   정말 모를 때만 그 노래의 정서·분위기·가사가 주는 감정을 구체적으로 (뻔한 클리셰 말고).

═══ [⏱ 길이 — 반드시 '영상(클립)' 길이에 맞춤 (가장 자주 틀리는 부분)] ═══

⚠️ 이 영상은 노래 전체가 아니라 **짧은 클립**이야. 노래 전체 분량으로 쓰면 절대 안 돼.
- 각 자막은 **자연스러운 의미 단위(한 호흡)**로 써 — 단어·조사·꾸밈말("이/그/첫/너무" 등)을 어색하게 토막내지 마. (한 칸 24자 이내 권장)
- ⚠️ **억지로 12자로 미리 자르지 마.** 12자 넘는 칸은 시스템이 자동으로 자연스럽게 나눠준다. 너는 의미(어절) 단위로만 끊으면 돼.
- 자막은 촘촘하고 많아짐 (영상 길이만큼 끝까지 이어지게).
- **마지막 자막 end = 영상 길이 또는 그 직전. 영상 길이 절대 초과 X.**
- 영상 길이 내내 빈 구간 없이 균등 분배 (청크 사이 gap 0.1초 이내)
- 영상 객관정보의 scene change / 액션 시점은 **무시** (영상 무관). 노래 흐름 따라 영상 길이에 균등하게.

═══ [🎵 노래 가사 (lyrics_subtitles) — 클립에서 들리는 부분만] ═══

- 이 클립에서 **실제로 들리는 가사만** 받아쓰기 (노래 전체 X — 클립에 안 나오는 가사 넣지 마)
- 각 가사 줄의 시작/끝 시간 = 실제 그 가사가 들리는 시점 (클립 timeline 기준)
- 마지막 가사 end도 영상 길이 초과 X
- 가사가 안 들리는(반주만 나오는) 클립이면 lyrics_subtitles = 빈 배열 []

⚠️ lyrics_subtitles(가사)와 situation_subtitles(감성 사연)는 **별개**. 둘 다 채우기.

═══ [⚠️ 자막 끝 마침표 금지] ═══

각 자막 text 끝에 마침표(.) 절대 X. (자막에 마침표 = 안 어울림)
- ❌ "온 세상을 적셨던 노래였죠."  ✅ "온 세상을 적셨던 노래였죠"
- 단, 물음표(?) / 말줄임표(...) 는 OK.

═══ [톤 — 감성 / 사연 / 감동] ═══

- 40~60대 감성. 뭉클 + 아련 + 향수. 시처럼, 편지처럼.
- **쨉쨉이 만들지 마** (jjap_jjap_i_subtitles = 빈 배열 []).
- ⚠ 신조어 / 밈 / ㄷㄷ / 갓X 절대 X.
- **각 자막은 자연스러운 의미 단위로** (12자 분할은 시스템이 자동 — 단어 토막 X). 첫 자막 ~데/~인데 어미.

═══ [제목 + 설명] ═══

- 제목 8개: 감성 클릭 유발 (노래/가수/추억 기반 — "30년 전 그 사람 생각나는..." 류)
- 유튜브 설명: 노래 사연 + 감성 + 해쉬태그

═══ [출력 — JSON만] ═══

```json
{
  "duration_sec": 43.93,
  "summary": "감성 스토리 한 줄",
  "external_context": "노래/가수 외부 맥락",
  "characters": [],
  "mascot_appearances": [],
  "title": "상단 고정 타이틀 (14~18자, 감성)",
  "youtube_upload_title": "YouTube 업로드용 (35~60자, 감성 + #shorts)",
  "youtube_upload_title_candidates": ["후보1", "후보2", "후보3", "후보4", "후보5"],
  "youtube_description": "유튜브 설명 (노래 사연 + 감성 + 해쉬태그)",
  "hashtags": ["#감성", "...8~12개"],
  "title_candidates": ["후보1", "후보2", "후보3", "후보4", "후보5", "후보6", "후보7", "후보8"],
  "situation_subtitles": [
    {"start": 0.0, "end": 3.5, "text": "첫 자막 ~데 (노래 이야기 시작)"},
    {"start": 3.6, "end": 7.0, "text": "이어지는 자막2 ... 영상 길이까지만"}
  ],
  "jjap_jjap_i_subtitles": [],
  "dialogue_subtitles": [],
  "song_title": "노래 제목 (오디오로 파악, 아니면 빈 문자열)",
  "singer": "가수 이름 (파악되면, 아니면 빈 문자열)",
  "lyrics_subtitles": [
    {"start": 0.0, "end": 3.5, "text": "클립에 들리는 가사 한 줄"},
    {"start": 3.6, "end": 7.0, "text": "다음 들리는 가사 ... 클립에 들리는 데까지만"}
  ],
  "key_actions": []
}
```

[중요] JSON만. 영상 화면 무시·오디오로만. **situation_subtitles 대부분을 노래·가수 구체 이야기로 풍부하게** (막연한 보편 감성 최소화, 단 확신 없는 사실 날조 X). 자막은 영상 길이에 맞춤 (초과 X). 쨉쨉이 빈 배열. 감성 톤. 신조어 X.
"""


# 🚫 절대 금지어 — 모든 스타일에 항상 append
BANNED_WORDS_RULE = """

═══════════════════════════════════════
[🚫 절대 금지어 — 무조건 사용 X]
═══════════════════════════════════════

다음 단어 절대 자막/제목에 박지 X (진부 + 매너리즘 + 센스 없음):
- "시전"  ← 절대 X
- "출격"  ← 절대 X

→ 위 단어 대신 영상에 진짜 어울리는 신선한 동사/표현 쓸 것.
→ 예: "시전" X → 그냥 그 동작 자연 표현 ("불러봄", "다가감", "끄덕임")
→ 예: "출격" X → "달려감", "냅다 감", "직진"
→ 위 금지어 1개라도 들어가면 자막 무효. 다시 만들 것.
"""


# 유머 스타일 — 쇼츠 prompt + 유머 강화 inject (2026-05-21 정식 추가)
HUMOR_INJECT = """

═══════════════════════════════════════
[🤣 유머 스타일 — 유머가 1순위, 세게 빵 터지게]
═══════════════════════════════════════

목표: 보는 사람이 진짜로 "ㅋㅋㅋㅋ" 빵 터지는 자막. 평범한 상황 묘사는 실패.
근데 **억지 X** — 안 웃긴데 우기면 더 안 웃김. 진짜 센스로, 단 **세게**.

━━ 강도: 세게 (대표님 요청 — 더 공격적으로) ━━
- **밍밍한 줄 0개**: 모든 자막이 한 방씩 친다. 단순 설명/연결용 자막 X.
- **펀치 강하게**: 점잖게 묘사하지 말고 드립을 한 단계 더 세게 (촌철살인 / 팩폭 / 디스).
- **디스·자폭 개그 허용**: 영상 속 상황·인물의 허당/굴욕 포인트를 위트로 후벼파기.
  (단 혐오·인신공격·외모비하·선정성은 X — 어디까지나 웃기려고)
- **밈 적극**: 지금 한국에서 진짜 통하는 밈/유행어를 상황에 딱 맞게 (기계적 반복은 X).
- **과장 극대화**: 낙차를 더 크게 (사소 → 우주급, 대단 → 시큰둥).
- **첫 자막 강펀치**: 인트로 설명 말고 첫 줄부터 세게 친다.

━━ 진짜 웃긴 자막 기법 (한 영상 안에서 여러 개 섞어 써) ━━
1. **빌드업 → 펀치라인**: 앞 자막에서 기대 깔고, 다음 자막에서 한 방에 뒤집기
2. **속마음 더빙**: 등장인물/동물 1인칭 속마음 (그 표정·동작에 딱 맞는 진짜 속내)
3. **기대 배신(반전)**: 뻔한 전개 예상시키고 정반대로 꺾기
4. **과장 ↔ 축소**: 사소한 걸 세계멸망급으로, 대단한 걸 시큰둥하게 (낙차로 웃김)
5. **디테일 집착**: 남들 안 보는 작은 디테일 하나를 집요하게 파기
6. **콜백**: 앞에서 깐 드립을 뒤 자막에서 다시 소환 (반복 아님, 변주)
7. **공감 개그**: "내 얘기네" 류 — 단 뻔한 거 말고 구체적인 상황으로
8. **신선한 의인화/비유**: 그 영상만의 포인트로 (뻔한 비유 X)

━━ 후크 & 마무리 (제일 중요) ━━
- **첫 자막** = 웃음 텐션 + 호기심 동시에. 밍밍한 소개 X, 보자마자 픽 하게.
- **마지막 자막** = 빵 터지는 한 방 (반전/콜백/촌철살인). 여운보다 웃음으로 끝.

━━ 변주 (매너리즘 방지 — 중요) ━━
- 한 영상 안에서 **같은 드립 패턴 반복 X** (위 기법들 섞기)
- 신조어/밈은 **진짜 어울릴 때만** (갓X·X됨·ㄷㄷ 기계적 반복 X)
- 학습 표현 풀에서 상황에 맞는 신선한 표현 골라 쓰기 (많이 쓴 표현 자제)
- **영상 결 맞추기**: 영상 종류에 안 맞는 드립 X (예: 군사 영상에 꿀피부 X)

━━ 영상 종류별 톤 (중요) ━━
- **피부관리·뷰티 (스킨케어/메이크업/시술/꿀팁 등)** → 상황을 **긍정적으로** 풀어라:
  - 개선·변화·만족·글로우업·"이거 사야 돼" 같은 **위로 올라가는 결**. 보는 사람이 "오 해보고 싶다 / 부럽다 / 속 시원하다"가 되게.
  - 🚫 피부 고민·트러블·'전(before)' 상태를 비하·조롱·혐오스럽게 묘사 X. 자폭·굴욕·디스 개그를 피부/외모에 들이대지 X (= 위 humor 디스 룰의 예외).
  - 깎아내리는 웃음 X → **띄워주는·통쾌한·공감되는 웃음 O**. (예: 트러블 비하 X → "내 피부도 좀 나눠주라" / "각질이 짐 싸서 나가네" / "장바구니 이미 담음")
  - 유머 강도·센스는 유지하되 **방향만 긍정으로**.

━━ 절대 피하기 ━━
- 그냥 상황 묘사 + "ㅋㅋ"만 붙이기 (제일 안 웃김)
- 뻔한 표현 남발 ("역대급", "실화냐", "레전드")
- 안 웃긴데 억지로 우기기 / 영상이랑 따로 노는 드립

⚠️ **쨉쨉이 2~3초마다 1개** (영상 ÷ 2.5 = 개수): 리액션 흐름으로 ("(흠칫)", "??", "(현타)" 등). 쨉쨉이도 웃기게, 상황설명이랑 안 겹치게.
⚠️ 유머 1순위지만 영상 내용/타이밍은 정확. **위 예시 표현을 그대로 베끼지 말고 그 '기법'을 이 영상에 맞게 적용**.
"""



# 자막 스타일 — 영상 올릴 때 선택
SUBTITLE_STYLES = {
    "shorts": {
        "label": "🎬 쇼츠 (MZ 위트 + 외부 맥락)",
        "prompt": SUBTITLE_GENERATION_PROMPT,
        "use_learning": True,
    },
    "humor": {
        "label": "🤣 유머 (드립 + 밈 + 빽빽한 쨉쨉이)",
        "prompt": SUBTITLE_GENERATION_PROMPT + HUMOR_INJECT,
        "use_learning": True,
    },
    "emotion": {
        "label": "💔 감성 스토리텔링 (아재아줌마)",
        "prompt": EMOTION_STORY_PROMPT,
        "use_learning": False,
    },
}


def get_learning_inject(mannerism_only: bool = False) -> str:
    """학습 자료 inject. 기본=표현풀(655)+매너리즘 신호.
    🔴mannerism_only=True면 표현풀(유도) 빼고 '매너리즘 자제 신호'만 준다.
       (2026-06-06 대표님: 3.1-pro 등 똑똑한 모델은 표현풀 리스트가 오히려 창의력 족쇄 →
        '식상한 표현만 피하고 나머지는 자유'가 더 신선. 표현 자유도 살리고 반복만 방지.)
    """
    try:
        from collections import defaultdict
        pool = defaultdict(list)
        if not mannerism_only:
            # 의미별 표현 풀 (mannerism_only면 skip — 유도 안 함)
            with db.get_db() as conn:
                rows = conn.execute("""
                    SELECT meaning_key, meaning_name, expression, use_count
                    FROM subtitle_expression_pool ORDER BY meaning_key, RANDOM()
                """).fetchall()
            for r in rows:
                pool[(r["meaning_key"], r["meaning_name"])].append((r["expression"], r["use_count"] or 0))

        # 매너리즘 신호 — 가장 많이 사용된 top 10 표현 (use_count >= 30)
        with db.get_db() as conn:
            top_used = conn.execute("""
                SELECT phrase, use_count FROM subtitle_phrase_frequency
                WHERE use_count >= 30 ORDER BY use_count DESC LIMIT 10
            """).fetchall()

        lines = []
        if mannerism_only:
            lines += ["", "[🎯 자막 매너리즘 방지 (297영상 학습 — 표현은 영상보고 자유롭게 새로, 아래만 피해라)]", ""]
        else:
            lines += ["", "[🎯 학습된 표현 풀 (297영상 + 655표현 학습 자료)]", "",
                      "✅ 의미별 사용 가능 표현 (영상 분위기에 맞게 선택):"]
            for (mk, mn), exprs in sorted(pool.items()):
                exprs_sorted = sorted(exprs, key=lambda x: x[1])[:8]
                lines.append(f"- {mn}: {' / '.join(e for e, _ in exprs_sorted)}")
            lines.append("")
        if top_used:
            lines.append("⛔ 너무 많이 쓰여 식상한 표현 (이번엔 자제):")
            for r in top_used:
                lines.append(f"  - \"{r['phrase']}\" ({r['use_count']}회)")
            lines.append("→ 위 표현만 피하고, 나머지는 영상에 딱 맞게 자유롭고 신선하게 새로 만들어라.")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ 학습 자료 inject 실패: {e}", flush=True)
        return ""


async def run_auto_subtitle(job_id: int, video_path: Path,
                              original_urls: list = None,
                              review_note: str | None = None,
                              style: str = "shorts",
                              song_title: str | None = None,
                              prompt_override: str | None = None) -> None:
    """2중 검증 (Pro=3.1 메인 + Flash=3.5 검증) + srt 3개 + 제목 후보 자동 생성. 자료에 결과 저장.
    review_note: 사람 검수 정정 메모 (있으면 prompt에 inject)
    song_title: 감성 스타일에서 사용자가 직접 입력한 노래 제목 (있으면 오디오 식별 대신 이걸 신뢰)
    prompt_override: 자막 메뉴의 SUBTITLE_STYLES 무시하고 외부에서 직접 prompt 전달.
                     쇼츠메이커 등 다른 메뉴가 자체 prompt로 자막 생성 시 사용
                     ([[menu-prompt-separation]] 룰). 학습 inject도 생략 (호출자가 직접 관리).
    """
    SUBTITLES_DIR = Path(__file__).parent.parent / "data" / "subtitles"
    out_dir = SUBTITLES_DIR / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이 작업 만든 사용자의 개인 Gemini 키 있으면 적용 (프리랜서 비용 분리)
    try:
        apply_user_gemini_key((db.get_subtitle_job(job_id) or {}).get("user_id"))
    except Exception:
        pass

    # 스타일 선택 — prompt_override 있으면 외부 메뉴 자체 prompt 사용 (자막 메뉴 SUBTITLE_STYLES 무시)
    if prompt_override:
        base_prompt = prompt_override
        learning = ""
        style_def = {"prompt": prompt_override, "use_learning": False}
    else:
        style_def = SUBTITLE_STYLES.get(style) or SUBTITLE_STYLES["shorts"]
        base_prompt = style_def["prompt"]
        learning = get_learning_inject(mannerism_only=True) if style_def.get("use_learning") else ""   # 🔴3.1-pro는 표현풀 족쇄 → 매너리즘 자제만(대표님 0606)

    # 정정 메모를 prompt 맨 앞에 박기 (있으면)
    main_prompt = base_prompt + learning + BANNED_WORDS_RULE
    if review_note and review_note.strip():
        review_inject = (
            f"⚠️ 사람 검수 정정 메모 (반드시 반영):\n{review_note.strip()}\n\n"
            f"이전 분석에서 위 내용이 잘못 분석됐음. 위 메모를 핵심 사실로 두고 다시 분석.\n\n"
        )
        main_prompt = review_inject + base_prompt + learning + BANNED_WORDS_RULE

    # 감성 — 사용자가 노래 제목 직접 입력 시 오디오 식별 대신 100% 신뢰 (오인식 방지)
    if style == "emotion" and song_title and song_title.strip():
        st = song_title.strip()
        song_inject = (
            f"\n\n═══ [🎵 노래 제목 확정 — 사용자 입력, 절대 신뢰] ═══\n"
            f"이 영상의 노래는 **'{st}'** 이다 (사용자가 직접 알려준 확정 정보).\n"
            f"- 오디오로 노래를 다시 식별하지 마라. 무조건 '{st}'로 간주.\n"
            f"- 이 노래/가수 기준으로 사연(가수가 누군지/추억/들은 사람/지금 공감)을 써라.\n"
            f"- song_title 필드에는 정확히 '{st}' 를 넣어라.\n"
        )
        main_prompt = main_prompt + song_inject

    try:
        db.update_subtitle_job(
            job_id, status="uploading", progress=10,
            progress_message="영상 준비 중 (큰 영상은 압축)..",
        )
        # 큰 영상(18MB+)은 저화질 압축본으로 inline 업로드 (Files API 불안정 회피)
        analysis_video = await ensure_inline_video(video_path)
        file_uri = await upload_video_to_gemini(analysis_video)

        db.update_subtitle_job(
            job_id, status="analyzing", progress=30,
            progress_message="Pro+Flash 2중 분석 중 (3.1-pro + 3.5-flash)..",
        )

        # 1단계: 댓글/제목 + Frame 먼저 (병렬, 빠름)
        db.update_subtitle_job(
            job_id, progress=35, progress_message="댓글/제목/frame 수집 중..",
        )
        frame_task = extract_key_frames(video_path, out_dir / "frames")
        scene_task = extract_scene_changes(video_path)
        comments_task = fetch_youtube_comments_for_urls(original_urls or [], per_video=20)
        pre_results = await asyncio.gather(
            frame_task, scene_task, comments_task, return_exceptions=True,
        )
        frame_check = pre_results[0] if not isinstance(pre_results[0], Exception) else {}
        scene_info = pre_results[1] if not isinstance(pre_results[1], Exception) else {}
        comments_dict = pre_results[2] if not isinstance(pre_results[2], Exception) else {}

        # 객관 시점 inject — Gemini에 ffmpeg가 찾은 진짜 anchor 시점 알려줌
        timing_inject = ""
        if scene_info.get("all_anchors"):
            anchors = scene_info["all_anchors"]
            kfs = scene_info.get("keyframes", [])
            sc = scene_info.get("scene_changes", [])
            ap = scene_info.get("audio_peaks", [])
            dur = scene_info.get("duration", 0)
            if style == "emotion":
                # 감성(음악) — 화면은 노래와 무관. 액션 타이밍 X, 영상 길이 균등 분배만.
                timing_inject = (
                    "\n\n[⏱ 영상 길이 정보]\n"
                    f"영상 길이: {dur:.2f}초\n"
                    "🚨 이 영상은 노래의 짧은 클립이고 화면은 노래와 무관함:\n"
                    f"- 자막(상황설명/가사)은 영상 길이({dur:.2f}초)에 맞춰 균등 분배\n"
                    f"- 마지막 자막 end = 영상 길이({dur:.2f}초) 또는 그 직전. 절대 초과 X\n"
                    "- scene change/액션 시점은 무시 (영상 무관). 노래 흐름만 따름\n"
                )
            else:
                timing_inject = (
                    "\n\n[⏱ 영상 객관 정보 — 참고용 (강제 매칭 X)]\n"
                    f"영상 길이: {dur:.2f}초\n"
                    f"📍 Scene change (씬 변화 시점): {sc}\n"
                    f"📍 Audio peak (소리/음성 시작): {ap}\n\n"
                    "🚨 자막 타이밍 룰:\n"
                    "1. 영상 보고 진짜 액션 시점에 자막 박음 (anchor에 강제 snap X)\n"
                    "2. 위 시점은 참고 — 진짜 액션과 다를 수 있음. 영상 직접 보고 결정\n"
                    "3. 자막 start = 그 액션 시작 시점 (±0.1초)\n"
                    "4. 자막 end는 다음 자막 start 0.05초 전\n"
                    f"5. 마지막 자막 end = 영상 길이 ({dur:.2f}초) 또는 그 직전 (절대 초과 X)\n"
                    "6. 자막 사이 빈 구간 0.5초 이상 X (자연 흐름)\n"
                )

        # 2단계: 모든 Gemini 호출에 영상 제목/댓글 inject (검증 일관성 ↑)
        comments_text = format_comments_for_prompt(comments_dict) if comments_dict else ""
        main_prompt_full = main_prompt + timing_inject + comments_text
        simple_prompt_full = SIMPLE_VERIFY_PROMPT + timing_inject + comments_text

        db.update_subtitle_job(
            job_id, progress=55,
            progress_message="Pro 분석 (메인) + Flash 검증 (병렬) 중..",
        )

        # 3단계: Pro 메인 + Flash 단순 병렬 (메인 = Pro 진짜 sharp / simple = Flash 검증용)
        primary_task = call_gemini(GEMINI_PRO_MODEL, file_uri,
                                     main_prompt_full, temperature=0.3, max_retries=3)  # Pro 과부하면 ~15초만 시도 후 Flash 폴백(대표님 0610)
        simple_task = call_gemini(GEMINI_FLASH_MODEL, file_uri,
                                    simple_prompt_full, temperature=0.1)
        analysis_results = await asyncio.gather(
            primary_task, simple_task, return_exceptions=True,
        )
        primary = analysis_results[0] if not isinstance(analysis_results[0], Exception) else {}
        simple_verify = analysis_results[1] if not isinstance(analysis_results[1], Exception) else {}

        if not primary:
            raise RuntimeError(f"메인 분석 실패: {analysis_results[0]}")

        db.update_subtitle_job(
            job_id, progress=70, progress_message="교차 검증 중..",
        )

        # 4중 교차 검증
        validation = cross_validate(primary, simple_verify, frame_check, None, comments_dict)

        # ⚠️ Pro 조건부 호출 — Flash 분석이 의심스러우면 Pro로 재검증
        pro_verify = {}
        pro_triggered = False
        # Pro 메인 이미 사용 — 조건부 Pro 재호출 무용지물 (비활성화 2026-05-19)
        if False and (not validation.get("consistent") or validation.get("confidence", 1.0) < 0.6):
            pro_triggered = True
            db.update_subtitle_job(
                job_id, progress=75,
                progress_message=f"⚠️ Flash 분석 의심 (conflict {len(validation.get('conflicts', []))}건) — Pro 재검증 중..",
            )
            try:
                # Pro로 단순 검증 + 메인 분석 다시 (Pro 자체가 메인이 되도록)
                pro_simple = await call_gemini(GEMINI_PRO_MODEL, file_uri,
                                                 SIMPLE_VERIFY_PROMPT + timing_inject, temperature=0.1)
                pro_verify = pro_simple if isinstance(pro_simple, dict) else {}

                # Pro 시점·액션이 Flash와 다르면 Pro로 메인 자막도 재생성
                if pro_verify:
                    pro_main = await call_gemini(
                        GEMINI_PRO_MODEL, file_uri,
                        (main_prompt + timing_inject
                         + (format_comments_for_prompt(comments_dict) if comments_dict else "")),
                        temperature=0.3,
                    )
                    if isinstance(pro_main, dict) and pro_main.get("situation_subtitles"):
                        primary = pro_main  # Pro로 교체 (더 정확)

                # 재검증
                validation = cross_validate(primary, simple_verify, frame_check, pro_verify, comments_dict)
                validation["pro_triggered"] = True
            except Exception as e:
                validation["pro_error"] = str(e)[:200]
                validation["pro_triggered"] = True

        # YouTube 업로드 메타 자동 fallback — primary에 누락된 경우 추가 호출
        if not primary.get("youtube_upload_title") or not primary.get("youtube_description"):
            db.update_subtitle_job(
                job_id, progress=82,
                progress_message="YouTube 메타 자동 생성 중..",
            )
            existing_title = primary.get("title", "")
            existing_summary = primary.get("summary", "")
            title_cands_existing = primary.get("title_candidates", [])
            cands_str = "\n".join([f"  - {t}" for t in title_cands_existing[:3]])
            yt_prompt = f"""이 영상을 보고 YouTube 업로드용 메타 (제목+설명+해쉬태그)만 만들어줘.

[이미 분석된 자료 — 참고]
- 영상 안 자막용 제목: {existing_title}
- 줄거리: {existing_summary}
- 영상 안 자막 제목 후보 (따라하지 X, YouTube용은 다름):
{cands_str}

[YouTube 업로드용 룰]
- **YouTube 업로드용 제목**과 **영상 안 자막용 제목**은 다름!
- YouTube용: 35~60자, SEO + 호기심 + 인기 키워드 + #shorts inline
- 영상 직접 보고 시청자가 클릭할만한 제목 만들기

[출력 — JSON만]
```json
{{
  "youtube_upload_title": "YouTube 업로드용 메인 제목 (35~60자) #shorts",
  "youtube_upload_title_candidates": [
    "(SEO+호기심) 후보1",
    "(키워드+밈) 후보2",
    "(반전 떡밥) 후보3",
    "(자극+숫자) 후보4",
    "(질문) 후보5"
  ],
  "youtube_description": "YouTube 설명 (200~500자, 줄거리 + 호기심 + inline 해쉬태그 5개)",
  "hashtags": ["#쇼츠", "#shorts", "...총 8~12개"]
}}
```
다른 텍스트 X. JSON만.
"""
            try:
                yt_meta = await call_gemini(
                    GEMINI_FLASH_MODEL, file_uri, yt_prompt,
                    temperature=0.3, max_retries=3,
                )
                if isinstance(yt_meta, dict):
                    primary["youtube_upload_title"] = yt_meta.get("youtube_upload_title", "")
                    primary["youtube_upload_title_candidates"] = yt_meta.get("youtube_upload_title_candidates", [])
                    primary["youtube_description"] = yt_meta.get("youtube_description", "")
                    primary["hashtags"] = yt_meta.get("hashtags", [])
            except Exception as e:
                print(f"[auto_subtitle] YouTube fallback 실패 (skip): {e}", flush=True)

        # srt 3개 파일 만들기
        db.update_subtitle_job(
            job_id, status="generating", progress=85,
            progress_message="srt 파일 만드는 중..",
        )

        # 자막 타이밍 후처리 — duration 초과 clamp / 순서 / 겹침 / anchor snap
        # 영상 길이 = ffprobe 실측 우선. Gemini duration_sec 추정은 부정확 → 17.6초 영상을
        # 20초로 추정하면 자막이 영상 밖(18~20초)까지 안 잘림 (대표님 2026-05-29 racing 영상).
        video_duration = 0
        try:
            import subprocess as _sp
            _r = _sp.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", str(video_path)],
                         capture_output=True, text=True, timeout=20)
            video_duration = float(_r.stdout.strip() or 0)
        except Exception:
            video_duration = 0
        if not video_duration or video_duration <= 0:
            video_duration = (scene_info.get("duration") or frame_check.get("duration")
                              or primary.get("duration_sec") or 0)
        anchors = scene_info.get("all_anchors") or []

        # 감성 스타일 — 자막 끝 마침표(.) 제거 (단 ... ? 는 유지)
        def _strip_period(subs):
            for s in subs:
                t = (s.get("text") or "").rstrip()
                if t.endswith(".") and not t.endswith("..."):
                    s["text"] = t[:-1].rstrip()
            return subs

        srt_paths = {}
        # 1. 상황 설명
        situations = primary.get("situation_subtitles", [])
        if situations:
            situations = validate_subtitle_timing(situations, video_duration, anchors, no_gap=True)
            # 상황설명 한 줄이 6초 넘게 떠있지 않게 (대표님: 700cc 자막이 12초로 영상 절반 차지).
            # 감성(emotion 사연)은 길게 떠야 하므로 제외 — humor/일반만 적용.
            if style != "emotion":
                for _s in situations:
                    if float(_s.get("end", 0)) - float(_s.get("start", 0)) > 6.0:
                        _s["end"] = round(float(_s.get("start", 0)) + 6.0, 2)
            if style == "emotion":
                situations = _strip_period(situations)
                situations = _merge_tiny_cues(situations)       # Gemini 과토막 재병합 (clause 복원)
                situations = _split_long_cues(situations, 12)   # 균형 분할 (어절 어색하게 안 끊기게)
            primary["situation_subtitles"] = situations
            p = out_dir / "01_상황설명.srt"
            write_srt(situations, p)
            srt_paths["situation"] = str(p)

        # 감성 스타일 — 노래 가사 srt (있으면). 가사는 끝까지 늘리지 X (fill_last=False)
        if style == "emotion":
            lyrics = primary.get("lyrics_subtitles", [])
            if lyrics:
                lyrics = validate_subtitle_timing(lyrics, video_duration, anchors, fill_last=False)
                primary["lyrics_subtitles"] = lyrics
                lp = out_dir / "05_가사.srt"
                write_srt(lyrics, lp)
                srt_paths["lyrics"] = str(lp)

        # 2. 쨉쨉이 — 리액션이라 마지막 줄을 영상 끝까지 늘리면 안 됨(대표님 0605: 사이다 22초로 늘어나 상황설명 자막들을 통째로 덮음)
        jjap = primary.get("jjap_jjap_i_subtitles", [])
        if jjap:
            jjap = validate_subtitle_timing(jjap, video_duration, anchors, fill_last=False)
            # 대표님 0612: 쨉쨉이 괄호()->별표, 효과음(의성어) 통째 제거
            import re as _re
            _SFX = ("쾅","펑","슈우웅","위이잉","두둥탁","와장창","퍽","팡","쿵","텅","탁","짠","휘청","철컥","챱","쨍","빵","툭","딱","우당탕")
            _clean = []
            for _j in jjap:
                _txt = _re.sub(r"\(([^)]+)\)", r"* \1 *", _j.get("text",""))   # (긴장)->* 긴장 *
                _core = _txt.replace("*"," ").strip()                            # 별표 떼고 알맹이
                if _core in _SFX:                                                # 순수 효과음 의성어면 버림
                    continue
                _txt = _re.sub(r"\*\s*\*","",_txt).strip()                    # 빈 별표 정리
                _j["text"] = _txt
                _clean.append(_j)
            jjap = _clean
            primary["jjap_jjap_i_subtitles"] = jjap
            p = out_dir / "02_쨉쨉이.srt"
            write_srt(jjap, p)
            srt_paths["jjap_jjap_i"] = str(p)

        # 3. 대사 번역 — 대사는 띄엄띄엄 말하는 거라 마지막 줄을 영상 끝까지 늘리면 안 됨
        #    (fill_last=False — 가사와 동일. 말 끝나면 자막도 끝남)
        dialogue = primary.get("dialogue_subtitles", [])
        if dialogue:
            dialogue = validate_subtitle_timing(dialogue, video_duration, anchors, fill_last=False)
            primary["dialogue_subtitles"] = dialogue
            p = out_dir / "03_대사번역.srt"
            write_srt(dialogue, p, text_key="korean")
            srt_paths["dialogue"] = str(p)

        # 제목 후보 괄호 후크태그 제거 (대표님 0609: (위험)(반전) 같은 괄호 안 나오게)
        import re as _re
        _PAREN_TAG = _re.compile(r"\s*[\(（][^()（）]{1,10}[\)）]\s*$")
        def _strip_tag(_s):
            return _PAREN_TAG.sub("", str(_s)).strip()
        for _lk in ("title_candidates", "youtube_upload_title_candidates"):
            if isinstance(primary.get(_lk), list):
                primary[_lk] = [_strip_tag(_t) for _t in primary[_lk]]
        for _sk in ("title", "youtube_upload_title"):
            if primary.get(_sk):
                primary[_sk] = _strip_tag(primary[_sk])

        # 제목 후보 텍스트
        titles = primary.get("title_candidates", [])
        if titles:
            (out_dir / "04_제목후보.txt").write_text(
                "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)]),
                encoding="utf-8",
            )

        # 자료 저장
        # review_note 있으면 사람이 이미 정정한 것 → conflict 무시하고 completed
        if review_note and review_note.strip():
            needs_review = 0
            status = "completed"
            progress_msg = "끝 (정정 메모 반영)"
        else:
            needs_review = 0 if validation["consistent"] else 1
            status = "completed" if validation["consistent"] else "needs_review"
            progress_msg = "끝" if not needs_review else "검수 필요 (불일치)"
        db.update_subtitle_job(
            job_id, status=status, progress=100,
            progress_message=progress_msg,
            subtitle_paths=srt_paths,
            title_candidates=titles,
            gemini_results={"primary": primary, "simple": simple_verify},
            cross_validation=validation,
            needs_review=needs_review,
            duration_sec=frame_check.get("duration", 0),
            cost_usd=0.05,
            completed_at=__import__("datetime").datetime.utcnow().isoformat(),
        )
        # BGM/SFX mp3 첨부 (실패해도 자막 잡은 OK)
        try:
            from workers.bgm_for_subtitle import attach_bgm_mix
            r = await attach_bgm_mix(video_path, out_dir)
            if r:
                _rec = ", ".join(r.get("bgm_recommend") or []) or "-"
                print(f"  ✅ 효과음 믹스 생성: {r['path']} "
                      f"(효과음 {r.get('sfx_count')}개)",
                      flush=True)
        except Exception as e:
            print(f"  ⚠️ BGM 믹스 실패 (자막 잡 OK): {e}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.update_subtitle_job(
            job_id, status="failed", error=str(e)[:500],
            progress_message=f"실패: {str(e)[:200]}",
        )
