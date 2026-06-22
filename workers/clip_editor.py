"""클립 편집 워커 (주제형) — 영상 다운 → Gemini로 '주제(인물/역사/사건 등)' 나오는 구간 선택
+ 노빠꾸패밀리식 매드무비 내레이션 자막 동시 생성 → 스토리 순서로 컷 편집 → SRT 출력.
- 주제형: song_title(노래) 대신 topic. 그 대상이 화면에 나오는 구간만 골라 스토리로 엮음.
- 자막: 노빠꾸 톤(짧은 단정 문장, '~었음/~음' 종결, 후크 시작). 영상엔 안 박고 SRT로만 출력(캡컷 후처리).
- BGM(옵션)·TTS(옵션)는 별도 단계에서 결합. 긴 영상은 저화질 분석본 inline 분석(결과물은 원본 화질).
"""
import asyncio
import math
import random
import httpx
from pathlib import Path

from api import database as db
from workers.auto_subtitle import (upload_video_to_gemini, call_gemini, GEMINI_PRO_MODEL,
                                    GEMINI_FLASH_MODEL, apply_user_gemini_key, write_srt,
                                    _get_gemini_key, GEMINI_API_URL)
from workers.bgm_for_subtitle import attach_bgm_mix

import sys as _sys
import shutil as _shutil
# 실행 중인 venv python의 yt-dlp (테스트=1호점venv·배포=딸깍venv 둘 다 호환). 없으면 PATH 탐색. (대표님 0614)
YTDLP = Path(_sys.executable).parent / ("yt-dlp.exe" if _sys.platform == "win32" else "yt-dlp")
if not YTDLP.exists():
    _w = _shutil.which("yt-dlp")
    if _w:
        YTDLP = Path(_w)
# ── YouTube 추출용 Deno(JS런타임)+EJS 솔버 (2026-06: yt-dlp가 YouTube JS챌린지 솔버 필요.
#    없으면 부실 폴백(android_vr)→영상에 따라 0개. deno+ejs면 제대로 추출) ──
import platform as _platform
def _deno_path():
    base = Path(__file__).resolve().parent.parent / "bin"   # 앱 동봉 bin/ (윈도우는 yt-dlp가 venv에 있어 앱폴더에 둠)
    if _sys.platform == "win32":
        c = base / "deno.exe"
    elif _sys.platform == "darwin":
        c = base / ("deno_arm" if _platform.machine() == "arm64" else "deno_x64")
    else:
        c = base / "deno"
    return c if c.exists() else None
_DENO = _deno_path()
# yt-dlp YouTube 공통 옵션: EJS 솔버 자동 다운(1회 캐시) + deno 런타임 지정(번들돼 있으면)
YT_YT = ["--remote-components", "ejs:github"] + (["--js-runtimes", f"deno:{_DENO}"] if _DENO else [])
OUT_DIR = Path(__file__).parent.parent / "data" / "clip_edits"

SEG_LEN = 3             # 컷 1개 길이(초) — 3초마다 컷 변화 (대표님 요청)
MAX_PER_VIDEO = 15      # 영상 1개에서 뽑을 구간 최대 개수 (롱폼 상한)
INLINE_LIMIT = 18_000_000   # Gemini inline 한계 (이 미만이면 Files API 안 거침)
FACE_MODEL = Path(__file__).parent.parent / "data" / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = Path(__file__).parent.parent / "data" / "models" / "face_recognition_sface_2021dec.onnx"  # 얼굴 식별(누구인지) — 주인공만 남기기
ARCFACE_ONNX = Path(__file__).parent.parent / "data" / "models" / "w600k_r50.onnx"  # ArcFace 인식(onnxruntime 직접 — insightface 패키지 불필요, 윈도우 설치 문제 회피. 대표님 0613)
MIN_FACE_FRAC = 0.12    # 얼굴높이/화면높이 — 이 미만이면 단독샷 아님(뉴스 PIP·군중샷) → 구간 드랍
FACE_SHARP_MIN = 20.0   # 얼굴부위 라플라시안 분산 — 이 미만이면 모자이크/블러 얼굴 → 구간 드랍
CUT_LEN = 3.5           # 노빠꾸식 빠른 컷 — 한 컷 길이(초). 매 컷 다른 영상으로 번갈아 (대표님 0613)
FIXED_BGM = Path(__file__).parent.parent / "data" / "bgm_library" / "fixed" / "clip_bgm.mp3"  # 고정 BGM(Holding Out For A Hero, 7초~ 트림본. 대표님 0613)


# ── 🔴 한글 경로 방어 (대표님 2026-06-16) ──
# 윈도우 OpenCV/onnxruntime/ffmpeg(C++)는 경로에 한글이 있으면 파일을 못 읽음(맥/리눅스는 UTF-8이라 무관).
# 폴더·사용자명에 한글이 있어도 클립편집이 되도록 ① 작업파일은 ASCII 임시폴더에서 ② 모델은 8.3 단축경로(or ASCII 복사)로.
def _safe_path(p):
    """비ASCII(한글 등) 경로를 윈도우 8.3 단축경로(ASCII)로 변환. 이미 ASCII거나 변환 실패 시 원본."""
    p = str(p)
    if _sys.platform != "win32":
        return p
    try:
        p.encode("ascii"); return p
    except UnicodeEncodeError:
        pass
    try:
        import ctypes
        from ctypes import wintypes
        _g = ctypes.windll.kernel32.GetShortPathNameW
        _g.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        buf = ctypes.create_unicode_buffer(1024)
        if _g(p, buf, 1024):
            return buf.value
    except Exception:
        pass
    return p


def _clip_workroot():
    """클립 작업 폴더 루트. 윈도우에서 앱 경로에 한글이 있으면 ASCII 임시폴더(결과물도 여기 남고 API가
    result_path로 제공)로, 아니면 기존 위치. 맥/리눅스는 UTF-8이라 기존 위치 그대로."""
    if _sys.platform != "win32":
        return OUT_DIR
    try:
        str(OUT_DIR).encode("ascii"); return OUT_DIR   # 앱 경로가 이미 ASCII면 그대로
    except UnicodeEncodeError:
        pass
    import os, tempfile
    t = tempfile.gettempdir()
    try:
        t.encode("ascii"); root = Path(t) / "ddalkkak_clips"
    except UnicodeEncodeError:
        root = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp" / "ddalkkak_clips"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_model_path(p):
    """모델 파일을 cv2/onnx가 읽을 ASCII 경로로. 단축경로 우선, 8.3 비활성 등으로 안 되면 ASCII 폴더에 복사."""
    sp = _safe_path(p)
    if _sys.platform != "win32":
        return sp
    try:
        sp.encode("ascii"); return sp
    except UnicodeEncodeError:
        pass
    try:
        dst = _clip_workroot() / "models" / Path(p).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or dst.stat().st_size != Path(p).stat().st_size:
            _shutil.copy2(str(p), str(dst))
        return str(dst)
    except Exception:
        return sp


async def _run(*args, timeout: float | None = None):
    """비동기 subprocess 실행 — 서버 블로킹 방지 (롱폼은 다운/인코딩이 길어 필수)."""
    p = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.communicate()
        raise RuntimeError(f"명령 시간초과 ({timeout}s): {args[0]}")
    return p.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")


async def _probe_duration(path: Path) -> float:
    try:
        _, out, _ = await _run(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path), timeout=30,
        )
        return float(out.strip() or 0)
    except Exception:
        return 0.0


async def _probe_dims(path: Path) -> tuple[int, int]:
    """영상 (width, height). 실패 시 (0,0)."""
    try:
        _, out, _ = await _run(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            str(path), timeout=30,
        )
        w, h = out.strip().split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _detect_face_center(src_path: str, start: float, dur: float, n: int = 5):
    """구간에서 가장 큰 얼굴의 평균 중심+크기 — YuNet(DNN, 측면·작은얼굴 OK) 우선, Haar 폴백.
    반환 (fx_frac, fy_frac, size_frac). size_frac=얼굴높이/화면높이 (단독샷 게이트용).
    동기 함수 (cv2 블로킹) → asyncio.to_thread로 호출. 얼굴 못 찾으면 None."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return None
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    if w <= 0 or h <= 0:
        cap.release()
        return None
    det = None
    if FACE_MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            det = cv2.FaceDetectorYN.create(_safe_model_path(FACE_MODEL), "", (320, 320), 0.6)
        except Exception:
            det = None
    cascade = None
    if det is None:   # 폴백: Haar (YuNet 모델 없을 때만)
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if cascade.empty():   # headless OpenCV엔 haar xml이 없을 수 있음 → 빈 분류기로 detectMultiScale 시 크래시
                cascade = None
        except Exception:
            cascade = None
    def _lap(img):
        return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                    cv2.CV_64F).var())

    def _roi(frame, fx, fy, fw, fh):
        y0, y1 = max(0, int(fy)), max(0, int(fy + fh))
        x0, x1 = max(0, int(fx)), max(0, int(fx + fw))
        r = frame[y0:y1, x0:x1]
        return _lap(r) if r.size else 0.0

    hits = []
    grays = []   # 컷 모션 측정용 (얼굴 유무 무관, 모든 샘플 프레임)
    for i in range(n):
        t = start + dur * (i + 0.5) / max(n, 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        grays.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)))
        if det is not None:
            det.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = det.detect(frame)
            if faces is not None and len(faces):
                biggest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = biggest[:4]
                # 비슷한 크기(최대의 절반+) 얼굴 수 — 2개+면 콜라주/단체샷 (단독샷 아님)
                nbig = sum(1 for f in faces if f[3] >= fh * 0.5)
                hits.append(((fx + fw / 2) / w, (fy + fh / 2) / h, fh / h,
                             _roi(frame, fx, fy, fw, fh), _lap(frame), nbig))
        elif cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                nbig = sum(1 for f in faces if f[3] >= fh * 0.5)
                hits.append(((fx + fw / 2) / w, (fy + fh / 2) / h, fh / h,
                             _roi(frame, fx, fy, fw, fh), _lap(frame), nbig))
    cap.release()
    if len(hits) < 2:        # 최소 2프레임에서 잡혀야 신뢰 (오탐 방지)
        return None
    # 컷 모션 (연속 샘플 평균 차이) — 포스터/정지화면/사진슬라이드면 ~0
    motion = 99.0
    if len(grays) >= 2:
        motion = sum(float(cv2.absdiff(grays[k], grays[k + 1]).mean())
                     for k in range(len(grays) - 1)) / (len(grays) - 1)
    nn = len(hits)
    return tuple(sum(c[k] for c in hits) / nn for k in range(6)) + (motion,)


def _face_crop_vf(src_w: int, src_h: int, face) -> str:
    """1080x1920 cover scale + 크롭. 얼굴 있으면 그 중심으로, 없으면 가운데."""
    OUT_W, OUT_H = 1080, 1920
    if src_w <= 0 or src_h <= 0:
        return ("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")
    sf = max(OUT_W / src_w, OUT_H / src_h)
    sw = round(src_w * sf); sw += sw % 2
    sh = round(src_h * sf); sh += sh % 2
    if face:
        cx = int(max(0, min(sw - OUT_W, face[0] * sw - OUT_W / 2)))
        cy = int(max(0, min(sh - OUT_H, face[1] * sh - OUT_H / 2)))
    else:
        cx = max(0, (sw - OUT_W) // 2)
        cy = max(0, (sh - OUT_H) // 2)
    return f"scale={sw}:{sh},crop={OUT_W}:{OUT_H}:{cx}:{cy}"


_ARCFACE = None


def _get_arcface():
    """ArcFace 인식 모델(w600k_r50) — onnxruntime 직접 로드. insightface 패키지 불필요(윈도우
    C++빌드 설치 실패 회피, 대표님 0613). 검출=YuNet 5점, 정렬=arcface_dst, 인식=이 onnx.
    insightface와 cosine 0.90~0.97 동일 검증 완료. lazy 1회 로드. 실패 시 식별 스킵."""
    global _ARCFACE
    if _ARCFACE is None:
        try:
            import onnxruntime as ort
            if not ARCFACE_ONNX.exists():
                print(f"[arcface] 모델 없음(식별 스킵): {ARCFACE_ONNX}", flush=True)
                _ARCFACE = False
            else:
                _ARCFACE = ort.InferenceSession(_safe_model_path(ARCFACE_ONNX), providers=["CPUExecutionProvider"])
                print("[arcface] onnxruntime ArcFace 로드 (insightface 불필요)", flush=True)
        except Exception as e:
            print(f"[arcface] 로드 실패(식별 스킵): {e}", flush=True)
            _ARCFACE = False
    return _ARCFACE if _ARCFACE else None


def _face_embedding(src_path: str, start: float, dur: float):
    """구간 대표 프레임의 주얼굴 ArcFace normed embedding (인물 식별용). 실패 시 None.
    YuNet 검출+5점 랜드마크 → arcface_dst 112x112 정렬 → w600k_r50 onnx 추론 → L2정규화."""
    sess = _get_arcface()
    if sess is None:
        return None
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(src_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + dur / 2) * 1000)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = fr.shape[:2]
        det = cv2.FaceDetectorYN.create(_safe_model_path(FACE_MODEL), "", (w, h), 0.6)
        det.setInputSize((w, h))
        _, faces = det.detect(fr)
        if faces is None or len(faces) == 0:
            return None
        big = max(faces, key=lambda f: f[2] * f[3])
        lm = big[4:14].reshape(5, 2).astype(np.float32)   # YuNet 5점(우눈·좌눈·코·우입·좌입)
        dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)  # arcface 표준
        M, _ = cv2.estimateAffinePartial2D(lm, dst)
        if M is None:
            return None
        aligned = cv2.warpAffine(fr, M, (112, 112))
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
        return emb / (float(np.linalg.norm(emb)) or 1.0)
    except Exception:
        return None


async def _search_recent_news(topic: str) -> str:
    """Gemini 구글검색(grounding)으로 주제의 최근 뉴스·근황을 텍스트 요약 (노빠꾸 '마무리 한 방'용).
    grounding은 JSON 응답과 호환 안 돼 텍스트로 받음. 실패 시 빈 문자열(뉴스 없이 진행)."""
    try:
        api_key = _get_gemini_key()
    except Exception:
        return ""
    prompt = (f"'{topic}'에 대한 최근 뉴스와 근황을 검색해서 한국어로 정리해줘.\n"
              f"- 최신 사건·성과·활동·화제가 된 일 위주 (시점/연도 포함)\n"
              f"- 노빠꾸패밀리식 쇼츠 매드무비의 '마무리 한 방'에 쓸 최신 반전·성공·근황 사실 위주\n"
              f"- 검증된 사실만. 짧은 항목 5~8개로.")
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "tools": [{"google_search": {}}],
                        "generationConfig": {"temperature": 0.3},
                    },
                )
            if r.status_code != 200:
                print(f"[news] {model} {r.status_code}: {(r.text or '')[:160]}", flush=True)
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if text:
                print(f"[news] '{topic}' 최근 뉴스 {len(text)}자 수집", flush=True)
                return text
        except Exception as e:
            print(f"[news] {model} 검색 실패: {e}", flush=True)
    return ""


async def _career_timeline(topic: str) -> list:
    """인물/주제의 연도별 타임라인(데뷔~현재) — 연도별 영상 발굴 검색어 생성. grounding으로 최신까지.
    반환: [{"year":2007,"event":"...","query":"검색어"}] (연도 오름차순). 실패 시 []."""
    try:
        api_key = _get_gemini_key()
    except Exception:
        return []
    prompt = (f"'{topic}'의 커리어/일대기를 데뷔(첫 작품/등장)부터 현재까지 연도순으로 정리해줘.\n"
              f"연도별 대표 작품·활동·전환점 위주. 각 항목에 3가지:\n"
              f"  · event: 그 시기에 무슨 일이 있었는지 (짧게)\n"
              f"  · query: 그 시기 영상이 유튜브에서 잘 나올 검색어 ('{topic}'+작품/사건, 예고편·명장면·인터뷰)\n"
              f"  · subtitle: 그 장면에 깔 노빠꾸패밀리 매드무비 자막 1줄 ('~었음/~음' 등, 공백포함 22자 이내, 사이다·성공·반전 톤)\n"
              f"반드시 JSON 배열만 출력: "
              f'[{{"year":2007,"event":"영화 데뷔작 OOO","query":"{topic} OOO","subtitle":"무명 시절 OOO으로 시작했음"}}]\n'
              f"- 8~14개, 연도 오름차순. 마지막 항목은 가장 최신 근황(성공/반전)으로 마무리")
    import json as _json
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "tools": [{"google_search": {}}],
                        "generationConfig": {"temperature": 0.3},
                    },
                )
            if r.status_code != 200:
                print(f"[timeline] {model} {r.status_code}: {(r.text or '')[:140]}", flush=True)
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if "[" in text and "]" in text:
                text = text[text.index("["):text.rindex("]") + 1]
            tl = _json.loads(text)
            tl = [x for x in tl if isinstance(x, dict) and x.get("query")]
            tl.sort(key=lambda x: x.get("year", 0))
            if tl:
                print(f"[timeline] '{topic}' {len(tl)}개 시기 발굴", flush=True)
                return tl
        except Exception as e:
            print(f"[timeline] {model} 실패: {e}", flush=True)
    return []


async def _discover_videos(topic: str, target_duration: int) -> list:
    """주제 → 연도별 커리어 타임라인 → 각 시기 영상 1개씩 검색 → 연도순 발굴 리스트.
    반환: [{"url","year","event"}] — year/event는 자막 생성 컨텍스트로 사용.
    8분 이하 영상만(다운·분석 속도, 긴 영화풀영상은 분석 화질 저하로 인물식별도 약함)."""
    timeline = await _career_timeline(topic)
    if not timeline:
        return []
    # 작품 수 — 구간당 ~4초, 작품당 1~2구간 가정. 최소 4, 최대 12
    n_works = max(4, min(12, round(target_duration / 4)))
    if len(timeline) > n_works:   # 데뷔~최신 균등 샘플 (서사 흐름 유지)
        step = len(timeline) / n_works
        timeline = [timeline[int(i * step)] for i in range(n_works)]
    found = []
    for item in timeline:
        q = (item.get("query") or "").strip()
        if not q:
            continue
        try:
            _, out, _ = await _run(
                str(YTDLP), *YT_YT, f"ytsearch5:{q}", "--print",
                "%(webpage_url)s|%(duration)s|%(height)s",
                "--no-download", "--no-warnings", timeout=60,
            )
            cands = []
            for line in (out or "").strip().split("\n"):
                parts = line.split("|")
                if len(parts) < 3 or not parts[0].startswith("http"):
                    continue
                try:
                    d = float(parts[1])
                except ValueError:
                    d = 0
                try:
                    hh = int(float(parts[2]))
                except ValueError:
                    hh = 0
                # 20초~8분만 (쇼츠 너무 짧음 제외, 긴 풀영상 제외 — 속도+분석화질)
                if 20 <= d <= 480:
                    cands.append((parts[0].strip(), d, hh))
            if cands:
                hd = [c for c in cands if c[2] >= 720]   # HD 우선 (저화질 소스=결과물 소프트)
                url, d, hh = (hd or cands)[0]
                found.append({"url": url,
                              "year": item.get("year"),
                              "event": str(item.get("event", ""))[:80]})
                print(f"[discover] {item.get('year')} '{q}' → {url} ({d:.0f}s {hh}p)", flush=True)
        except Exception as e:
            print(f"[discover] '{q}' 검색 실패: {e}", flush=True)
    return found


async def _narration_script(topic: str, n_lines: int, news: str = "") -> list:
    """주제 인물의 '가장 임팩트 있는 단 하나의 사건'을 기승전결로 n줄 생성 (일대기 시간순 나열 X — 노빠꾸 실제 방식).
    대본이 핵심 — 화면은 인물만 나오면 됨(대표님 0613). grounding으로 최신까지. 반환 [자막,...] 기승전결 순."""
    try:
        api_key = _get_gemini_key()
    except Exception:
        return []
    import json as _json
    news_block = f"\n[최근 뉴스·근황 — 중심 사건 또는 마지막 여운에 활용]\n{news[:1200]}\n" if news else ""
    prompt = (f"'{topic}'를 노빠꾸패밀리 매드무비 '자막 대본'으로 써줘.\n"
              f"(영상 화면 설명이 아니라 독립 내레이션. 화면은 인물만 나오면 됨)\n\n"
              f"★★★노빠꾸의 진짜 공식 (노빠꾸가 직접 만든 실제 영상 분석) — 이게 전부다:\n"
              f"❌ 인물 일대기를 시간순 나열하지 마라(데뷔작→다음작→…→최신작 식 작품 나열 = 위키요약 = 최악, 기승전결 없음).\n"
              f"✅ 그 인물의 **가장 임팩트 있는 단 하나의 사건·관계·선택**을 잡아 그것만 기승전결로 깊게 파라.\n\n"
              f"[실제 노빠꾸 정답 — 이 방식 그대로]\n"
              f"· 피오 → '사비 털어 극단 차린 사건' 하나로만: \"연기는 안 된다 아이돌 출신이다\"(편견)→10년 따라다닌 꼬리표→주연 맡아도 혹평뿐→\"회사에 도움 안 되는 거 안다\"(각오)→그래도 사비 털어 극단 차림→결국 27개국 1위로 증명. ※해병대·블락비컴백·다른작품 하나도 안 넣음!\n"
              f"· 김무열 → '실수 트윗이 결혼까지' 하나로만: 술 취해 DM이 트위터 전체공개로 날아감→결별 각오→윤승아 \"공개연애 하자\"→근데 병역논란 터짐→윤승아가 묵묵히 기다림→5년 연애 끝 결혼·아들. ※범죄도시4·존시나 안 넣음!\n\n"
              f"[기승전결 — 정확히 {n_lines}줄을 '하나의 사건' 흐름으로]\n"
              f"· 기(앞 2~3줄): 강력 후크 + 그 사건의 발단/처지. 후크는 궁금증 유발하되 무슨 상황인지(주어·대상)는 명확히\n"
              f"· 승(3~4줄): 갈등·위기·시련이 점점 심화\n"
              f"· 전(3~5줄): 결정적 선택·행동·반전 — 영상의 핵심 사건\n"
              f"· 결(끝 2~3줄): 결과 + 맨 마지막 한 줄은 여운(메시지·현재, 정보 나열 X)\n\n"
              f"[필수 규칙]\n"
              f"- ⚠️⚠️한 줄이 앞 줄을 인과로 받아라(그래서·근데·결국·알고보니). 뜬금없는 사건 점프 절대 금지(A사건 다음 갑자기 무관한 B사건 X — '블락비 컴백한다' 다음 '그 기세로 영화' 같은 비약 X)\n"
              f"- 대사 인용 1~2줄(큰따옴표, 본인/주변 말) — 감정 확 올림\n"
              f"- 구체적 숫자·고유명사(연도·금액·횟수)로 신빙성\n"
              f"- 각 줄 22자 이내, 종결 '~었음/~았음/~음' 구어 단정체. '~이다/~다/~했다/~함/~됨' 문어체·명사형 금지(대사·체언 마무리는 허용)\n"
              f"- ⚠️한 줄만 떼어 읽어도 뜻이 명확해야. 의미불명 압축·말장난 금지. 화면 묘사('수트핏·비주얼') 금지\n"
              f"- ⚠️⚠️⚠️팩트 최우선: grounding 구글검색으로 '확인된 사실'만 써라. 추측·각색·과장으로 없는 사건/수치/연도/일화 지어내기 절대 금지. "
              f"조금이라도 불확실하면 그 줄을 빼고 확실한 사실로 채워라. 거짓이 단 한 줄도 있으면 안 됨.{news_block}\n"
              f'반드시 JSON 문자열 배열만: ["자막1","자막2", ...]')
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "tools": [{"google_search": {}}],
                          "generationConfig": {"temperature": 0.4}},
                )
            if r.status_code != 200:
                print(f"[script] {model} {r.status_code}", flush=True)
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if "[" in text and "]" in text:
                text = text[text.index("["):text.rindex("]") + 1]
            lines = [str(x).strip()[:60] for x in _json.loads(text) if str(x).strip()]
            if lines:
                print(f"[script] '{topic}' 대본 {len(lines)}줄 생성", flush=True)
                return lines
        except Exception as e:
            print(f"[script] {model} 실패: {e}", flush=True)
    return []


async def _factcheck_script(topic: str, lines: list) -> list:
    """생성된 대본을 grounding 구글검색으로 2차 팩트체크 — 사실과 다르거나 확인 안 되는 줄을
    검증된 사실로 교정. 줄 수·기승전결 흐름·말투 유지. 대표님 0613 '거짓말 안 하고 팩트체크'."""
    if not lines:
        return lines
    try:
        api_key = _get_gemini_key()
    except Exception:
        return lines
    import json as _json
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    prompt = (f"아래는 '{topic}' 쇼츠 자막 대본 {len(lines)}줄이다. 구글 검색으로 한 줄씩 팩트체크해라.\n"
              f"[대본]\n{numbered}\n\n"
              f"- 사실과 다르거나 검색으로 확인 안 되는 줄 → 검증된 사실로 고쳐라(앞뒤 기승전결 흐름 유지)\n"
              f"- 확실한 사실인 줄 → 그대로 둬라\n"
              f"- ⚠️정확히 {len(lines)}줄 유지. 말투 '~었음/~음', 각 줄 22자 이내, 의미 명확. 추측·각색·과장 금지\n"
              f'반드시 JSON 문자열 배열만: ["줄1","줄2",...]')
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "tools": [{"google_search": {}}],
                          "generationConfig": {"temperature": 0.3}},
                )
            if r.status_code != 200:
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if "[" in text and "]" in text:
                text = text[text.index("["):text.rindex("]") + 1]
            checked = [str(x).strip()[:60] for x in _json.loads(text) if str(x).strip()]
            if len(checked) == len(lines):
                print(f"[factcheck] '{topic}' {len(checked)}줄 팩트체크 완료", flush=True)
                return checked
            print(f"[factcheck] 줄수 불일치({len(checked)}≠{len(lines)}) → 원본 유지", flush=True)
            return lines
        except Exception as e:
            print(f"[factcheck] {model} 실패: {e}", flush=True)
    return lines


async def _generate_clip_meta(topic: str, script: list) -> dict:
    """대본 기반 유튜브 메타 생성 — 상단제목/유튜브제목/설명/태그 (바로 복사용, 노빠꾸 스타일).
    대표님 0613. 대본에 있는 사실만 활용(지어내기 금지)."""
    if not script:
        return {}
    try:
        api_key = _get_gemini_key()
    except Exception:
        return {}
    import json as _json
    body = "\n".join(script)
    prompt = (f"아래는 '{topic}' 노빠꾸패밀리 매드무비 쇼츠의 자막 대본이다.\n[대본]\n{body}\n\n"
              f"이 영상 유튜브 업로드용 메타데이터를 노빠꾸패밀리 스타일로 만들어라:\n"
              f"1) top_title: 영상 상단에 큰 글씨로 박을 후크 제목 한 줄(공백포함 18자 이내). 예) '사비 털어 무대 만든 표지훈'\n"
              f"2) yt_title: 유튜브 제목. 후크 문구 + 해시태그 5~7개. 예) '사비 털어 무대 만든 #표지훈 #피오 #참교육 #넷플릭스'\n"
              f"3) description: 유튜브 설명란. 대본 내용을 자연스러운 문장으로 풀어쓰고 줄바꿈(\\n) 넣고, 맨 끝 줄에 해시태그 10개 내외\n"
              f"4) tags: 검색 태그 12~15개 (쉼표로 구분, # 없이)\n"
              f"⚠️대본에 있는 사실만 활용. 없는 내용 지어내기 절대 금지.\n"
              f'반드시 JSON만: {{"top_title":"...","yt_title":"...","description":"...","tags":"..."}}')
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0.6}},
                )
            if r.status_code != 200:
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            if "{" in text and "}" in text:
                text = text[text.index("{"):text.rindex("}") + 1]
            meta = _json.loads(text)
            if isinstance(meta, dict) and meta.get("yt_title"):
                print(f"[meta] '{topic}' 메타 생성 완료", flush=True)
                return {"top_title": str(meta.get("top_title", "")).strip(),
                        "yt_title": str(meta.get("yt_title", "")).strip(),
                        "description": str(meta.get("description", "")).strip(),
                        "tags": str(meta.get("tags", "")).strip()}
        except Exception as e:
            print(f"[meta] {model} 실패: {e}", flush=True)
    return {}


async def _finale_line(topic: str, news: str) -> str:
    """뉴스 최신근황으로 노빠꾸식 '마무리 한 방' 자막 1줄 생성 (실패 시 빈 문자열).
    최신작 영상이 발굴 안 돼도 마지막 자막엔 최신 근황이 박히도록 보장."""
    try:
        api_key = _get_gemini_key()
    except Exception:
        return ""
    prompt = (f"'{topic}' 쇼츠 매드무비의 마지막 자막 1줄을 써줘.\n"
              f"[최근 뉴스·근황]\n{news[:1500]}\n\n"
              f"규칙: 노빠꾸패밀리 톤 '마무리 한 방'. 가장 최신·가장 큰 성과로 끝맺음. "
              f'예: "결국 2024년, 첫 천만 영화의 주인공이 됐음" / "데뷔 20년 만에 글로벌 1위까지 접수했다". '
              f"30자 이내, 자막 텍스트만 출력 (설명·따옴표 없이).")
    for model in (GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite"):
        try:
            async with httpx.AsyncClient(timeout=45.0) as c:
                r = await c.post(
                    f"{GEMINI_API_URL}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0.6}},
                )
            if r.status_code != 200:
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            line = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip().strip('"')
            if line:
                return line.split("\n")[0][:60]
        except Exception:
            pass
    return ""


def _topic_segment_prompt(topic: str, seg_dur: int, max_segs: int, news: str = "",
                           era: str = "") -> str:
    news_block = ""
    if news:
        news_block = f"""
[{topic} 최근 뉴스·근황 (검색된 최신 사실)]
{news}
→ 이 영상이 매드무비의 '마지막(마무리)' 파트다. 마지막 구간 자막에 위 최신 근황을
   노빠꾸식 '한 방'으로 녹여라 (예: "결국 2026년 OO까지 해냈음"). 마무리 멘트는 딱 한 번만.
"""
    era_block = ""
    if era:
        guard = "" if news else " 마무리 멘트(결국 ~해냈다류)는 쓰지 마라 — 마무리는 마지막 영상 담당."
        era_block = f"""
[이 영상의 시기 맥락] {era}
→ 자막에 이 시기·작품 맥락을 자연스럽게 녹여라 (예: "2007년, 별순검으로 처음 얼굴을 알렸음").
   전체 매드무비는 연도순 인생 서사다. 이 영상은 그중 위 시기를 담당한다.{guard}
"""
    return f"""이 영상으로 '{topic}'에 대한 쇼츠 매드무비를 만들 거야.
('{topic}'이 인물이면 그 인물, 사건/역사면 그 주제)
{era_block}

[1단계] '{topic}'이(가) 화면에 '크게·단독으로' 나오는 구간을 시간순으로 최대 {max_segs}개 골라줘.
- ⭐가장 중요: 인물이 화면에 크게(클로즈업/바스트샷/단독샷) 잡힌 컷만. 인물이 화면에서 작거나 구석에 있으면 절대 제외
- ❌제외: 뉴스 그래픽·자막으로 꽉 찬 화면·PIP(작은 원형/사각 삽입)·앵커/기자 화면·빈 배경·로고·여러 명이 작게 나오는 군중샷·인트로/아웃트로/광고
- 각 구간 약 {seg_dur}초 (3~6초). 시간 순서대로, 겹치지 않게
- 표정·리액션·결정적 행동이 크게 보이는 장면 우선

[2단계] 각 구간마다, 그 장면에 깔 '자막'을 노빠꾸패밀리 매드무비 톤으로 1줄 써줘:
- ⭐길이: 한 컷이 3.5초라 3~4초에 읽히게 공백 포함 22자 이내로 짧게 (길면 못 읽음)
- 짧고 단정적이되, ⭐'~음/었음'으로만 끝내지 마. 매 줄 같은 어미면 단조로움. 아래를 섞어서:
  · 음슴체 '~었음/~음' (전체의 절반 정도만)
  · 실제 대사·평가 인용 (예: "색깔이 없다는 말만 들었음", 따옴표 활용)
  · 명사·체언 종결 (예: "데뷔 9년차의 반전", "역대급 빌런의 탄생")
  · 현재형·감탄 (예: "이게 그 유명한 장면이다", "결국 해냈다")
- 사이다·성공신화·반전 정서. 영상에 보이는 사실에 근거 (마무리 구간만 아래 최근 뉴스 반영 가능, 그 외 지어내기 금지)
{news_block}
JSON 배열만 출력(다른 텍스트 X), 시간순:
[{{"start": 12.5, "end": 17.0, "subtitle": "이 장면 자막 1줄"}}]
"""


async def _analysis_copy(clip: Path, dur: float) -> Path:
    """18MB 넘는 영상 → inline 가능한 저화질 분석본.
    비트레이트 캡으로 영상 길이와 무관하게 ~13MB 목표 (롱폼 긴 영상도 inline 보장)."""
    try:
        size = clip.stat().st_size
    except Exception:
        size = 0
    if 0 < size < 17_000_000:
        return clip   # 이미 inline 가능

    small = clip.parent / f"small_{clip.stem}.mp4"
    target_bytes = 13_000_000
    vbit = int(target_bytes * 8 / 1000 / max(dur, 1))   # kbps
    vbit = max(120, min(900, vbit))
    scale = "scale=-2:480" if dur < 600 else "scale=-2:360"
    rc, _, err = await _run(
        "ffmpeg", "-i", str(clip), "-vf", f"{scale},fps=10",
        "-c:v", "libx264", "-b:v", f"{vbit}k", "-maxrate", f"{vbit}k",
        "-bufsize", f"{vbit}k", "-preset", "veryfast", "-an", "-y", str(small),
        timeout=600,
    )
    if small.exists() and 0 < small.stat().st_size < INLINE_LIMIT:
        print(f"{clip.name}: 분석본 {small.stat().st_size // 1024 // 1024}MB "
              f"({vbit}k, {dur:.0f}s)", flush=True)
        return small
    print(f"{clip.name}: 분석본 생성 실패/초과 → 원본 사용 (err={err[:120]})", flush=True)
    return clip


def _parse_segments(r, dur: float, per_video: int, seg_dur: int, clip_name: str) -> list:
    """Gemini 응답(배열 또는 객체) → 정리된 구간 리스트 (clamp/겹침 제거)."""
    if isinstance(r, list):
        raw = r
    elif isinstance(r, dict):
        raw = r.get("segments") if isinstance(r.get("segments"), list) else [r]
    else:
        raw = []

    segs = []
    for item in raw[:per_video]:
        if not isinstance(item, dict):
            continue
        try:
            s = max(0.0, float(item.get("start", 0)))
            e = float(item.get("end", s + seg_dur))
        except (TypeError, ValueError):
            continue
        if dur > 0:
            s = min(s, max(0.0, dur - 2))
            e = min(e, dur)
        # 너무 긴 구간 제한 — 한 컷에 오래 머무르면 몽타주가 늘어짐 (요청 길이 근처로)
        max_len = max(seg_dur + 5, int(seg_dur * 1.5))
        if e - s > max_len:
            e = s + max_len
        if e - s < 3:                      # 너무 짧으면 seg_dur로 보정
            e = s + seg_dur
            if dur > 0:
                e = min(e, dur)
        if e - s < 3:
            continue
        segs.append({"clip": clip_name, "start": round(s, 1), "end": round(e, 1),
                     "subtitle": str(item.get("subtitle", item.get("reason", "")))[:120]})

    segs.sort(key=lambda x: x["start"])
    dedup = []
    for sg in segs:                        # 겹침 제거
        if dedup and sg["start"] < dedup[-1]["end"]:
            continue
        dedup.append(sg)
    return dedup


async def run_clip_edit(job_id: int, urls: list, topic: str,
                         target_duration: int = 50, make_tts: bool = False,
                         crf: int = 18) -> None:
    """여러 영상 → '주제(topic)' 나오는 구간 + 노빠꾸식 자막 → 스토리 순서 컷 편집 + SRT 출력."""
    out_dir = _clip_workroot() / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이 작업 만든 사용자 개인 Gemini 키 있으면 적용 (프리랜서 비용 분리)
    try:
        apply_user_gemini_key((db.get_clip_edit_job(job_id) or {}).get("user_id"))
    except Exception:
        pass

    try:
        # 0. URL 없으면 주제로 연도별 영상 자동 발굴 (대표님: 키워드만으로 인생 서사)
        vid_meta = []   # 발굴 모드: urls와 동일 인덱스의 {year,event} (자막 시기 컨텍스트)
        if not urls:
            db.update_clip_edit_job(
                job_id, status="downloading", progress=3,
                progress_message=f"'{topic}' 연도별 영상 발굴 중 (커리어 타임라인)..",
            )
            found = await _discover_videos(topic, target_duration)
            if not found:
                raise RuntimeError(f"'{topic}' 관련 영상을 찾지 못함 (발굴 실패)")
            urls = [f["url"] for f in found]
            vid_meta = found
            db.update_clip_edit_job(job_id, urls=urls)
            print(f"[discover] '{topic}' {len(urls)}개 영상 발굴 → 연도순 편집", flush=True)
        n = len(urls)
        # 1. 다운로드 (원본 화질, 비동기)
        clips = []
        clip_meta = []   # clips와 동일 인덱스 (다운 실패 스킵 반영)
        for i, url in enumerate(urls):
            db.update_clip_edit_job(
                job_id, status="downloading", progress=5 + int(20 * i / max(n, 1)),
                progress_message=f"영상 다운 중 ({i + 1}/{n})..",
            )
            cp = out_dir / f"clip{i}.mp4"
            _, _, err = await _run(
                str(YTDLP), *YT_YT, "-f", "best[ext=mp4]/best", "-o", str(cp), url,
                timeout=900,
            )
            if cp.exists():
                clips.append(cp)
                clip_meta.append(vid_meta[i] if i < len(vid_meta) else None)
            else:
                print(f"다운 실패 {url}: {err[:200]}", flush=True)
        if not clips:
            raise RuntimeError("다운된 영상이 하나도 없음 — " + (err or "")[-300:].strip())
        nclips = len(clips)

        # 구간 계획 — 주제형은 구간 ~4초(자막 한두 줄 읽힘), 영상별로 스토리 구간 추출
        seg_dur = 4
        # 게이트/식별 폐기 대비 영상당 구간 넉넉히 추출 (통과분으로 target 길이 채우기)
        # target/CUT_LEN 컷 필요 → 영상수로 나눠 영상당 구간, 여유 +3 (대표님 0613: 50초 분량 확보)
        per_video = min(MAX_PER_VIDEO,
                        max(8, math.ceil((target_duration / CUT_LEN) / nclips) + 3))

        # 1.5 최근 뉴스 검색 (Gemini 구글검색 grounding) — 노빠꾸식 '최신 근황 마무리'용
        db.update_clip_edit_job(
            job_id, status="analyzing", progress=28,
            progress_message=f"'{topic}' 최근 뉴스 검색 중..",
        )
        news = await _search_recent_news(topic)

        # 2. 영상별 '주제' 구간 + 노빠꾸식 자막 — 3개 동시 병렬 분석 (대표님 0613: 속도)
        #    뉴스(마무리 한 방)는 '마지막 영상'에만 — 영상마다 마무리 멘트 반복 방지
        sem = asyncio.Semaphore(3)
        done = {"n": 0}

        async def _analyze_one(i: int, c: Path) -> list:
            m = clip_meta[i] if i < len(clip_meta) else None
            era = f"{m.get('year')}년 — {m.get('event')}" if m else ""
            nb = news if i == nclips - 1 else ""   # 마지막 영상만 뉴스 마무리
            async with sem:
                dur = await _probe_duration(c)
                try:
                    av = await _analysis_copy(c, dur)
                    uri = await upload_video_to_gemini(av)
                    r = await call_gemini(GEMINI_PRO_MODEL, uri,
                                          _topic_segment_prompt(topic, seg_dur, per_video, nb, era),
                                          temperature=0.5)
                    segs = _parse_segments(r, dur, per_video, seg_dur, c.name)
                    print(f"{c.name}: {len(segs)}개 '{topic}' 구간", flush=True)
                except Exception as ex:
                    print(f"{c.name} 분석 실패: {ex}", flush=True)
                    segs = []
                done["n"] += 1
                db.update_clip_edit_job(
                    job_id, status="analyzing", progress=30 + int(35 * done["n"] / nclips),
                    progress_message=f"'{topic}' 구간·자막 분석 ({done['n']}/{nclips}) · Gemini 3병렬..",
                )
                return segs

        results = await asyncio.gather(*[_analyze_one(i, c) for i, c in enumerate(clips)])
        all_segs = [s for segs in results for s in segs if s.get("subtitle")]   # 영상(연도)순 유지
        if not all_segs:
            raise RuntimeError(f"'{topic}' 관련 구간을 찾지 못함 (영상에 안 나오거나 분석 실패)")

        cpath = {c.name: c for c in clips}

        # 2.5 단독샷 검증 게이트 (대표님 0613: 뒤통수·모자이크·PIP 불량 차단)
        #     소스 원본 해상도에서 YuNet 얼굴검사 → 얼굴크기·선명도 게이트. 편집 전에 불량 폐기.
        db.update_clip_edit_job(job_id, progress=66,
                                progress_message=f"단독샷 검증 중 ({len(all_segs)}구간 얼굴검사)..")

        async def _gate(sg):
            sg["_face"] = await asyncio.to_thread(
                _detect_face_center, str(cpath[sg["clip"]]),
                sg["start"], sg["end"] - sg["start"])
            return sg
        all_segs = list(await asyncio.gather(*[_gate(s) for s in all_segs]))

        # 판별식 — f = (cx, cy, 얼굴크기, 얼굴선명도, 화면선명도, 비슷한크기 얼굴수)
        def _mosaic(f):    # 모자이크 시그니처: 얼굴만 유독 흐림 (시네마틱=화면 전체 소프트는 통과)
            return f and f[3] < FACE_SHARP_MIN and f[3] < 0.5 * max(f[4], 1.0)

        def _why_bad(f):
            if not f:
                return "얼굴없음(뒤통수/풍경)"
            if _mosaic(f):
                return f"모자이크 얼굴{f[3]:.0f}/화면{f[4]:.0f}"
            if f[2] < MIN_FACE_FRAC:
                return f"얼굴작음 {f[2]:.2f}"
            if f[5] >= 2.0:
                return f"단독샷아님(얼굴 {f[5]:.1f}개)"
            return None   # 정지화면(포스터/사진)은 드랍하지 않고 아래 컷편집에서 Ken Burns 줌인으로 살림

        ok_face = [s for s in all_segs if s["_face"] and not _why_bad(s["_face"])]
        # 인물모드 자동판정 — 구간 40%+에서 멀쩡한 단독샷이 잡히면 인물 주제로 보고 얼굴 필수
        person_mode = len(ok_face) >= max(2, int(0.4 * len(all_segs)))
        if person_mode:
            for s in all_segs:
                if s not in ok_face:
                    print(f"[gate] 드랍 {s['clip']} {s['start']}s — {_why_bad(s['_face'])} "
                          f"| {s['subtitle'][:30]}", flush=True)
            kept = ok_face
        else:   # 사건/역사 주제 (얼굴 위주 아님) — 모자이크 얼굴만 드랍
            kept = [s for s in all_segs if not _mosaic(s["_face"])]

        # (오염소스 통째폐기 제거 — 대표님 0613: 큐레이션 영상을 모자이크 컷 몇 개 때문에 통째 버리면 안 됨.
        #  불량 컷은 위에서 이미 개별 드랍됐고, 타인은 아래 ArcFace 주인공식별이 거르므로 영상 자체는 유지)
        all_segs = kept
        print(f"[gate] 인물모드={person_mode} — 통과 {len(all_segs)}구간", flush=True)
        if not all_segs:
            raise RuntimeError(f"단독샷 검증 통과 구간이 없음 — 소스 영상에 '{topic}' 단독샷 부족")

        # 2.7 주인공 얼굴 식별 (ArcFace) — 여러 영상에 공통으로 가장 많이 나오는 얼굴 = 주제 인물.
        #     상대역·진행자 등 '다른 인물' 제거. SFace→ArcFace 강화 (대표님 0613: 이수근 등 타인 섞임)
        if person_mode and len(all_segs) >= 4 and _get_arcface() is not None:
            import numpy as np
            db.update_clip_edit_job(job_id, progress=68,
                                    progress_message=f"주인공('{topic}') 얼굴 식별 중 (ArcFace)..")
            embs = list(await asyncio.gather(*[
                asyncio.to_thread(_face_embedding, str(cpath[s["clip"]]),
                                  s["start"], s["end"] - s["start"])
                for s in all_segs]))
            idxs = [i for i, e in enumerate(embs) if e is not None]
            if len(idxs) >= 4:
                # ArcFace normed embedding 군집 — 임계 0.42 (같은 인물이 영상 화질·각도차로
                # 0.5 못 넘어 갈라지던 것 묶음. 진짜 타인은 ~0.3이라 여전히 배제. 대표님 0613)
                # rep는 멤버 centroid로 갱신 → 한 컷 우연 매칭 흔들림 방지
                clusters = []
                for i in idxs:
                    best, bs = None, 0.42
                    for c in clusters:
                        sc = float(np.dot(embs[i], c["rep"]))
                        if sc > bs:
                            best, bs = c, sc
                    if best:
                        best["mem"].append(i)
                        best["clips"].add(all_segs[i]["clip"])
                        m = len(best["mem"])
                        rep = best["rep"] * ((m - 1) / m) + embs[i] * (1 / m)
                        n = float(np.linalg.norm(rep)) or 1.0
                        best["rep"] = rep / n
                    else:
                        clusters.append({"rep": embs[i], "mem": [i],
                                         "clips": {all_segs[i]["clip"]}})
                # 주인공 = 가장 많은 '서로 다른 영상'에 등장한 군집 (1영상 전용 타인 배제)
                main = max(clusters, key=lambda c: (len(c["clips"]), len(c["mem"])))
                keep = set(main["mem"])
                if len(keep) >= 3:   # 주인공 군집 충분할 때만 적용 (오판 방지)
                    kept, dropped = [], []
                    for i, s in enumerate(all_segs):
                        if i in keep:   # ArcFace로 주인공(주제인물) 얼굴이 확인된 컷만 통과
                            kept.append(s)
                        else:           # 타인 OR 얼굴 미검출(브이로그·포스터·풍경 등) → 드랍
                            dropped.append(s)
                            why = "타인" if embs[i] is not None else "얼굴없음"
                            print(f"[face] {why} 드랍 {s['clip']}@{s['start']:.0f} | "
                                  f"{s['subtitle'][:22]}", flush=True)
                    all_segs = kept
                    print(f"[face] 주인공(영상 {len(main['clips'])}개 공통·{len(main['mem'])}컷) "
                          f"식별 후 {len(all_segs)}구간 (제거 {len(dropped)})", flush=True)

        # 3. 노빠꾸식 빠른 컷 선별 — 매 컷 '다른 영상'으로 번갈아(같은 영상 연속 금지), 영상당 최대 2컷.
        #    큰 흐름=연도순(발굴순), 마지막 컷=최신작(뉴스 마무리). 정렬 안 함(번갈이 순서 유지 = 화면 계속 바뀜).
        n_need = max(8, round(target_duration / CUT_LEN))   # 50초 ÷ 3.5 ≈ 14컷
        by_clip = {}
        for sg in all_segs:
            by_clip.setdefault(sg["clip"], []).append(sg)
        clip_order = [c.name for c in clips if c.name in by_clip]   # 발굴=연도순
        finale = by_clip[clip_order[-1]][-1]
        picked = []
        # 영상당 최대 컷 = target 채우게 동적 (통과 영상 적으면 영상당 더 뽑아 50초 분량 확보)
        max_per = max(2, math.ceil(n_need / max(len(clip_order), 1)) + 1)
        for round_i in range(max_per):
            for cn in clip_order:      # 영상 번갈아 → 매 컷 화면이 다른 작품으로 바뀜
                lst = by_clip[cn]
                if round_i < len(lst) and lst[round_i] is not finale and len(picked) < n_need - 1:
                    picked.append(lst[round_i])
        picked.append(finale)          # 마지막 컷 = 최신작 (뉴스 마무리 자막)
        chosen = picked
        # 컷 길이 동적 — 통과 컷이 적어도 target(50초) 채우게 늘림 (3.0~5.0초, 대표님 0613: 50초 분량)
        cut_len = max(3.0, min(5.0, target_duration / max(len(chosen), 1)))

        # 4. 컷 편집 — 스토리 순서대로(셔플 X), 얼굴 중앙 정렬 + 미세 색보정. SRT 타임라인 누적.
        db.update_clip_edit_job(
            job_id, status="editing", progress=70,
            progress_message=f"컷 편집 중 ({len(chosen)}개 구간)..",
            segments_json=[{k: v for k, v in s.items() if k != "_face"} for s in chosen],
        )
        parts = []
        clip_dims = {}      # clip명 → (w,h) 캐시
        face_hits = 0
        srt_items = []      # [{start,end,text}] — 최종 영상 누적 타임라인 기준
        t_cursor = 0.0
        for idx, seg in enumerate(chosen):
            src = cpath.get(seg["clip"])
            if not src:
                continue
            out = out_dir / f"seg{idx}.mp4"
            avail = round(seg["end"] - seg["start"], 2)
            dur_seg = round(cut_len, 2)   # 동적 컷 길이 (target 채우게). 영상 길이 넘으면 ffmpeg가 알아서 자름
            # 얼굴 중앙 정렬 — 게이트에서 검사한 얼굴 재사용 (주제 인물 얼굴을 화면 정중앙으로)
            dims = clip_dims.get(src.name)
            if dims is None:
                dims = await _probe_dims(src)
                clip_dims[src.name] = dims
            # 크롭용 얼굴은 실제 컷 구간(start~dur_seg)으로 재검출 — 게이트는 구간 전체 평균이라
            # 인물이 움직이면 컷 위치와 어긋나 한쪽으로 치우침 (대표님 0613: 얼굴 왼쪽 박힘)
            gate_face = seg.pop("_face", None)
            face = await asyncio.to_thread(
                _detect_face_center, str(src), seg["start"], dur_seg) or gate_face
            if face:
                face_hits += 1
            crop_vf = _face_crop_vf(dims[0], dims[1], face)
            # 미세 색보정 — 중복 감지 회피 (과하지 않게, 구간마다 살짝 다르게)
            hue = random.uniform(-5, 5)        # 색조 ±5도
            sat = random.uniform(1.02, 1.06)   # 채도 +2~6%
            bri = random.uniform(0.008, 0.022) # 밝기 +0.8~2.2%
            con = random.uniform(1.01, 1.035)  # 대비 +1~3.5%
            color = f"hue=h={hue:.1f}:s={sat:.3f},eq=brightness={bri:.3f}:contrast={con:.3f}"
            # 정지화면(포스터·사진, 모션<1.5)은 Ken Burns 서서히 줌인으로 생동감 부여 (노빠꾸식)
            if face and len(face) >= 7 and face[6] < 1.5:
                kb = ("zoompan=z='min(1+0.0022*on,1.32)':d=1:fps=30:s=1080x1920"
                      ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'")
                vf = f"{crop_vf},{kb},setsar=1,{color}"
            else:
                vf = f"{crop_vf},setsar=1,fps=30,{color}"
            await _run(
                "ffmpeg", "-ss", str(seg["start"]), "-i", str(src), "-t", str(dur_seg),
                "-vf", vf, "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-y", str(out), timeout=600,
            )
            if out.exists():
                parts.append(out)
                srt_items.append({"start": round(t_cursor, 2),
                                  "end": round(t_cursor + dur_seg, 2),
                                  "text": seg["subtitle"]})
                t_cursor += dur_seg
            db.update_clip_edit_job(
                job_id, progress=70 + int(18 * (idx + 1) / len(chosen)),
                progress_message=f"컷 편집 중 ({idx + 1}/{len(chosen)}) · 얼굴정렬 {face_hits}..",
            )
        print(f"얼굴 중앙 정렬: {face_hits}/{len(chosen)} 구간", flush=True)
        if not parts:
            raise RuntimeError("컷 편집 결과가 없음")

        # 5. concat (무손실 우선, 실패 시 재인코딩)
        concat_list = out_dir / "concat.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in parts))
        final = out_dir / "clip_edit_final.mp4"
        db.update_clip_edit_job(job_id, progress=90, progress_message="합치는 중..")
        rc, _, _ = await _run(
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-y", str(final), timeout=300,
        )
        if rc != 0 or not final.exists():
            await _run(
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-y", str(final), timeout=1800,
            )
        if not final.exists():
            raise RuntimeError("최종 영상 생성 실패")

        # 6. 클립편집은 무음 컷영상 — 음악 입히는 기능 제거(대표님 0614). 자막(SRT)·메타만 출력.

        # 6.5 대본 = 핵심 (대표님 0613: 화면 묘사 자막 X). 노빠꾸 인생서사 대본을 독립 생성해
        #     화면 컷에 순서대로 배분. 대본↔화면 매칭 불필요(인물만 나오면 됨).
        if srt_items:
            script = await _narration_script(topic, len(srt_items), news)
            if script:
                script = await _factcheck_script(topic, script)   # 2차 팩트체크 (거짓·각색 교정, 대표님 0613)
                for i in range(len(srt_items)):
                    if i < len(script):
                        srt_items[i]["text"] = script[i]
                print(f"[script] 노빠꾸 대본 {len(script)}줄 → {len(srt_items)}컷 배분", flush=True)
            elif news:   # 대본 실패 시 최소한 마무리 한 방이라도
                fin = await _finale_line(topic, news)
                if fin:
                    srt_items[-1]["text"] = fin

        # 7. SRT 출력 — 노빠꾸식 내레이션 자막 (영상엔 안 박음, 캡컷 후처리용)
        srt_path = out_dir / "clip_edit_narration.srt"
        try:
            write_srt(srt_items, srt_path)
        except Exception as ex:
            print(f"SRT 작성 실패: {ex}", flush=True)

        # 7.5 유튜브 메타 — 상단제목/유튜브제목/설명/태그 (바로 복사용, 대표님 0613)
        try:
            meta = await _generate_clip_meta(topic, [it["text"] for it in srt_items])
            if meta:
                import json as _jm
                (out_dir / "meta.json").write_text(
                    _jm.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as ex:
            print(f"메타 생성 실패: {ex}", flush=True)

        dur_out = await _probe_duration(final)
        db.update_clip_edit_job(
            job_id, status="completed", progress=100,
            progress_message=f"끝 ({dur_out:.1f}초, {len(parts)}컷, 자막 {len(srt_items)}줄)",
            result_path=str(final),
            srt_path=str(srt_path) if srt_path.exists() else None,
            cost_usd=nclips * 0.04, completed_at_now=True,
        )
    except Exception as e:
        db.update_clip_edit_job(
            job_id, status="failed", progress=0,
            progress_message=f"실패: {str(e)[:200]}", error=str(e)[:500],
        )
        raise
