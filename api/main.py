"""FastAPI server — REST + WebSocket for the discovery system."""
import os
import json
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from . import database as db
from . import auth

# ── [배포판] 경로 로컬화: 설치 위치 기준 상대 경로 ──
from pathlib import Path as _BBPath
import sys as _bb_sys, shutil as _bb_shutil
_BB_ROOT = _BBPath(__file__).resolve().parent.parent
_BB_DATA = _BB_ROOT / "data"

def _bb_ytdlp():
    cand = _BBPath(_bb_sys.executable).parent / ("yt-dlp.exe" if _bb_sys.platform == "win32" else "yt-dlp")
    if cand.exists():
        return str(cand)
    return _bb_shutil.which("yt-dlp") or "yt-dlp"

from workers import pipeline
from workers import korean_pool as kr_pool
from workers import qdrant_index as kr_qdrant
from workers import visual_match as kr_visual
from workers import mascot as mascot_worker
from workers import ai_remix
from workers import comfy_client
from workers import fal_client
from workers import kie_client


BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
PUBLIC_PATHS = {"/health", "/", "/manifest.json", "/sw.js", "/favicon.ico",
                "/docs", "/redoc", "/openapi.json", "/api/auth/login"}


# ===== Lifespan =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    auth.bootstrap_admin()
    # Cleanup orphaned remix tasks left over from previous server crash/reload
    try:
        with db.get_db() as conn:
            cur = conn.execute(
                "UPDATE remixes SET status='failed', "
                "error = COALESCE(error, '') || ' / orphaned (server reload)' "
                "WHERE status IN ('rendering','composing','generating')"
            )
            if cur.rowcount:
                print(f"🧹 cleaned up {cur.rowcount} orphan remix")
    except Exception as e:
        print(f"⚠️ orphan cleanup fail: {e}")
    # 서버 reload/재시작으로 인프로세스 워커가 죽은 자막·음성자막 잡 자동 재개.
    # startup 시점엔 이전 process가 죽었으므로 in-flight 상태 = 전부 orphan.
    # 영상/오디오 파일이 디스크에 있으면 처음부터 재처리, 없으면 failed.
    try:
        from workers.auto_subtitle import run_auto_subtitle
        with db.get_db() as conn:
            rows = conn.execute(
                "SELECT id, video_path, original_urls, style FROM subtitle_jobs "
                "WHERE status IN ('pending','uploading','analyzing')"
            ).fetchall()
        resumed = 0
        for r in rows:
            vp = r["video_path"]
            if vp and Path(vp).exists():
                try:
                    urls = json.loads(r["original_urls"]) if r["original_urls"] else []
                except Exception:
                    urls = []
                if not isinstance(urls, list):
                    urls = []
                asyncio.create_task(run_auto_subtitle(
                    r["id"], Path(vp), urls, None, r["style"] or "shorts", ""))
                resumed += 1
            else:
                db.update_subtitle_job(
                    r["id"], status="failed",
                    progress_message="영상 파일 없음 (서버 재시작 — 재업로드 필요)")
        if rows:
            print(f"🔁 자막 잡 재개: {resumed}/{len(rows)}건 (나머지는 파일 없음→failed)")
    except Exception as e:
        print(f"⚠️ 자막 잡 재개 실패: {e}")
    # 음성자막 잡 재개 (sync 워커 → to_thread). review/done 은 건드리지 않음.
    try:
        from workers.audio_subtitle import run_audio_subtitle
        with db.get_db() as conn:
            arows = conn.execute(
                "SELECT id, audio_path FROM audio_subtitle_jobs "
                "WHERE status IN ('pending','transcribing','correcting')"
            ).fetchall()
        a_resumed = 0
        for r in arows:
            ap = r["audio_path"]
            if ap and Path(ap).exists():
                asyncio.create_task(asyncio.to_thread(run_audio_subtitle, r["id"]))
                a_resumed += 1
            else:
                db.update_audio_subtitle_job(
                    r["id"], status="failed",
                    error="오디오 파일 없음 (서버 재시작 — 재업로드 필요)")
        if arows:
            print(f"🔁 음성자막 잡 재개: {a_resumed}/{len(arows)}건")
    except Exception as e:
        print(f"⚠️ 음성자막 잡 재개 실패: {e}")
    # 대본+더빙 잡 재개 (async 워커). 영상 있으면 처음부터 재처리.
    try:
        from workers.tts_dub import run_tts_dub
        with db.get_db() as conn:
            drows = conn.execute(
                "SELECT id, video_path FROM tts_dub_jobs "
                "WHERE status IN ('pending','analyzing','synthesizing')"
            ).fetchall()
        d_resumed = 0
        for r in drows:
            vp = r["video_path"]
            if vp and Path(vp).exists():
                asyncio.create_task(run_tts_dub(r["id"]))
                d_resumed += 1
            else:
                db.update_tts_dub_job(
                    r["id"], status="failed",
                    error="영상 파일 없음 (서버 재시작 — 재업로드 필요)")
        if drows:
            print(f"🔁 더빙 잡 재개: {d_resumed}/{len(drows)}건")
    except Exception as e:
        print(f"⚠️ 더빙 잡 재개 실패: {e}")
    # 쇼츠 메이커 잡 정리 (Gemini Pro+Whisper 비용 무거움 → 자동 재개 X, failed 표시 후 UI에서 재시작)
    try:
        with db.get_db() as conn:
            srows = conn.execute(
                "SELECT id FROM shorts_jobs "
                "WHERE status IN ('pending','downloading','picking','processing')"
            ).fetchall()
        for r in srows:
            db.update_shorts_job(
                r["id"], status="failed",
                error="서버 재시작으로 중단됨 — UI에서 '재시작' 버튼을 눌러주세요",
            )
        if srows:
            print(f"🩹 쇼츠 잡 정리: {len(srows)}건 failed (수동 재시작 대기)")
    except Exception as e:
        print(f"⚠️ 쇼츠 잡 정리 실패: {e}")
    print("✅ DB initialized")
    print(f"🔑 Backend API key: {'set' if BACKEND_API_KEY else 'NOT SET (open access!)'}")
    # Telegram 양방향 봇 polling 시작 (백그라운드)
    try:
        if os.getenv("SOLO_MODE","") != "1":
            from workers.telegram_bot import poll_loop
            asyncio.create_task(poll_loop())
            print("🤖 Telegram bot polling 시작")
    except Exception as e:
        print(f"⚠️ Telegram bot 시작 실패: {e}")
    # 디스크 자동 정리 + 모니터링 (1시간마다)
    try:
        if os.getenv("SOLO_MODE","") != "1":
            from workers.disk_janitor import janitor_loop
            asyncio.create_task(janitor_loop())
            print("🧹 Disk janitor 시작")
    except Exception as e:
        print(f"⚠️ Disk janitor 시작 실패: {e}")
    # 서버 reload 시 죽은 task가 남긴 stuck status (generating/rendering) 즉시 정리
    # startup 호출은 force=True — 디스크 파일 0개여도 무조건 정리 (이전 process 죽었으니).
    # 정리한 게 있으면 텔레그램 알림.
    try:
        if os.getenv("SOLO_MODE","") == "1": raise RuntimeError("solo-skip")
        from workers.disk_janitor import cleanup_stuck_statuses
        n_mascot, n_remix = await cleanup_stuck_statuses(force=True)
        if n_mascot or n_remix:
            print(f"🩹 startup cleanup: 마스코트 {n_mascot}건 + 합본 {n_remix}건 마무리")
    except Exception as e:
        print(f"⚠️ startup cleanup 실패: {e}")
    # 마스코트 자료 즉시 백업 (1시간 기다리지 말고 startup 직후 한 번)
    try:
        if os.getenv("SOLO_MODE","") == "1": raise RuntimeError("solo-skip")
        from workers.disk_janitor import backup_mascot_data
        backup_file = await backup_mascot_data()
        if backup_file:
            print(f"💾 startup 백업: {backup_file}")
    except Exception as e:
        print(f"⚠️ startup 백업 실패: {e}")
    yield


app = FastAPI(
    title="딸깍공장",
    description="멀티 플랫폼 쇼츠 발굴 시스템",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for PWA + iOS shortcuts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API Key middleware =====

_MAINTENANCE_MODE = {"on": False, "allow_token": None}


@app.middleware("http")
async def maintenance_block(request: Request, call_next):
    """점검 모드 — 외부 사용자 503 반환. 형님 (admin)만 token 일치 시 통과."""
    if _MAINTENANCE_MODE["on"]:
        path = request.url.path
        # health + 점검 해제 endpoint + 정적 자원 차단 안 함
        if path in ("/health", "/api/admin/maintenance/off"):
            return await call_next(request)
        # token 일치 시 통과 (형님 admin 토큰)
        auth_header = request.headers.get("authorization", "")
        token_q = request.query_params.get("token", "")
        if _MAINTENANCE_MODE["allow_token"] and (
            _MAINTENANCE_MODE["allow_token"] in auth_header or
            _MAINTENANCE_MODE["allow_token"] == token_q
        ):
            return await call_next(request)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "🔧 점검 중 — 잠시 후 다시"},
        )
    return await call_next(request)


@app.post("/api/admin/maintenance/on")
async def maintenance_on(current=Depends(auth.admin_only)):
    """점검 모드 시작 — 외부 사용자 503. 형님 (호출자) token만 허용."""
    # 호출자 token 추출은 별도 method. 단순화: admin은 다 통과되게 username 기반
    _MAINTENANCE_MODE["on"] = True
    # 호출자 (admin)의 username을 allow_marker로
    _MAINTENANCE_MODE["allow_token"] = f'"username":"{current["username"]}"'
    return {"ok": True, "on": True, "allow_marker": _MAINTENANCE_MODE["allow_token"]}


@app.post("/api/admin/maintenance/off")
async def maintenance_off(current=Depends(auth.admin_only)):
    """점검 모드 해제."""
    _MAINTENANCE_MODE["on"] = False
    _MAINTENANCE_MODE["allow_token"] = None
    return {"ok": True, "on": False}


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Frontend (index.html / sw.js / static assets) 강제 no-cache — 형님 브라우저가
    옛 코드 잡고 있어서 새 변경 안 보이는 issue 회피."""
    response = await call_next(request)
    p = request.url.path
    # HTML / JS / CSS / SW — 매번 fresh fetch
    if p == "/" or p.endswith(("/index.html", "/sw.js", "/manifest.json")) \
       or p.endswith(".js") or p.endswith(".css") or p.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Allow JWT Bearer or legacy X-API-Key. Public paths bypass."""
    # [배포판] SOLO_MODE: 개인용 — 인증 전부 통과
    if os.getenv("SOLO_MODE", "") == "1":
        return await call_next(request)
    path = request.url.path
    method = request.method

    if (
        method == "OPTIONS"
        or path in PUBLIC_PATHS
        or path.startswith("/static")
        or path.startswith("/assets")
        or path.startswith("/icons")
        or path.startswith("/mascots")
        or path.startswith("/remixes")
        or ("/api/shorts/" in path and ("/file" in path or "/zip/" in path))
    ):
        return await call_next(request)

    if path.startswith("/api") or path.startswith("/ws"):
        # Path 1: Bearer JWT (new auth)
        authz = request.headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            token = authz.split(" ", 1)[1].strip()
            try:
                auth.decode_token(token)  # raises HTTPException(401) if bad
                return await call_next(request)
            except HTTPException as e:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)

        # Path 2: query ?token=... — for WebSockets AND <img>/<video> tags that
        # cannot send Authorization headers (e.g., preview-frame, source-video).
        tok = request.query_params.get("token")
        if tok:
            try:
                auth.decode_token(tok)
                return await call_next(request)
            except HTTPException as e:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)

        # Path 3: legacy X-API-Key (backward compat for existing PWA + iOS shortcuts)
        if BACKEND_API_KEY:
            provided = (
                request.headers.get("x-api-key")
                or request.query_params.get("api_key")
            )
            if provided == BACKEND_API_KEY:
                return await call_next(request)

        return JSONResponse(
            {"detail": "Unauthorized — provide Bearer JWT or X-API-Key"},
            status_code=401,
        )

    return await call_next(request)


# ===== WebSocket connection manager =====

class WSManager:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        if job_id in self.connections:
            try:
                self.connections[job_id].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, job_id: str, message: dict):
        for ws in self.connections.get(job_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                pass


ws_manager = WSManager()


# ===== Pydantic models =====

class DiscoverRequest(BaseModel):
    name: str = Field(..., description="발굴 작업 이름")
    keywords: list[str] = Field(..., description="검색 키워드 목록")
    reference_channel: str | None = Field(None, description="레퍼런스 채널 URL or @handle")
    platforms: list[str] = Field(default=["youtube", "tiktok", "instagram"])
    min_views: int = Field(default=5_000_000)
    max_duration: int = Field(default=55, description="초 단위 최대 길이")
    excluded_keywords: list[str] = Field(default_factory=list)
    excluded_channels: list[str] = Field(default_factory=list)
    notion_database_id: str | None = Field(None)


class JobResponse(BaseModel):
    job_id: str
    status: str
    name: str
    progress: int = 0
    progress_message: str = ""
    candidate_count: int = 0


class CandidateResponse(BaseModel):
    id: int
    platform: str
    video_id: str
    url: str
    title: str | None = None
    channel_name: str | None = None
    view_count: int | None = None
    duration: float | None = None  # TikTok returns fractional durations (e.g. 6.8s)
    classification: str | None = "pending"
    dna_match_score: float | None = None
    notes: str | None = None
    used: int = 0  # 1=실제 채널에 업로드한 영상, 0=대기/안 씀
    used_at: str | None = None
    used_by_username: str | None = None  # 누가 마크했는지
    memo_kr: str | None = None  # 양봉여리 한국어 메모
    published_at: str | None = None
    thumbnail_url: str | None = None
    visual_match_verdict: str | None = None
    visual_match_score: float | None = None
    is_manual: int = 0  # 1=수동 URL 추가, 0=자동 발굴 (정렬 + 화면 색상 구분용)
    on_hold: int = 0  # 1=보류 (미사용 탭에 남아있되 색상 다름, 나중에 사용 예정)


# ===== Routes =====

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "db": str(db.DB_PATH.exists()),
    }


@app.post("/api/discover", response_model=JobResponse)
async def start_discovery(req: DiscoverRequest, bg: BackgroundTasks):
    """Start a new discovery job (runs in background)."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    db.insert_job(
        job_id=job_id,
        name=req.name,
        reference_channel=req.reference_channel,
        keywords=json.dumps(req.keywords, ensure_ascii=False),
        platforms=json.dumps(req.platforms),
        min_views=req.min_views,
        max_duration=req.max_duration,
        excluded_keywords=json.dumps(req.excluded_keywords, ensure_ascii=False),
        excluded_channels=json.dumps(req.excluded_channels, ensure_ascii=False),
        notion_page_id=req.notion_database_id,
    )

    async def progress_cb(pct: int, msg: str):
        await ws_manager.broadcast(job_id, {
            "type": "progress",
            "progress": pct,
            "message": msg,
        })

    async def runner():
        try:
            await pipeline.run_pipeline(
                job_id=job_id,
                keywords=req.keywords,
                reference_channel=req.reference_channel,
                platforms=req.platforms,
                min_views=req.min_views,
                max_duration=req.max_duration,
                excluded_keywords=req.excluded_keywords,
                excluded_channels=req.excluded_channels,
                notion_database_id=req.notion_database_id,
                progress_cb=progress_cb,
            )
            await ws_manager.broadcast(job_id, {"type": "completed"})
        except Exception as e:
            db.update_job(job_id, status="failed", error=str(e))
            await ws_manager.broadcast(job_id, {"type": "error", "message": str(e)})

    bg.add_task(asyncio.create_task, runner())
    asyncio.create_task(runner())  # actually start
    return JobResponse(job_id=job_id, status="running", name=req.name)


@app.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs(current=Depends(auth.authenticate)):
    rows = db.list_jobs(limit=500)
    if current.get("role") != "admin":
        # Freelancers only see jobs assigned to them
        assigned = set(auth.list_assigned_jobs(current["id"]))
        rows = [r for r in rows if r["id"] in assigned]
    # Attach candidate counts in one batch query (avoids N+1)
    counts: dict[str, int] = {}
    if rows:
        with db.get_db() as conn:
            ids_sql = ",".join("?" * len(rows))
            for r in conn.execute(
                f"SELECT job_id, COUNT(*) AS n FROM candidate_videos "
                f"WHERE job_id IN ({ids_sql}) GROUP BY job_id",
                [r["id"] for r in rows],
            ).fetchall():
                counts[r["job_id"]] = r["n"]
    return [JobResponse(**{k: r.get(k) for k in
            ["job_id", "status", "name", "progress", "progress_message"]
            if r.get(k) is not None}
            | {"job_id": r["id"], "candidate_count": counts.get(r["id"], 0)})
            for r in rows]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(404, "job not found")
    return JobResponse(
        job_id=row["id"],
        status=row.get("status", "pending"),
        name=row.get("name", ""),
        progress=row.get("progress", 0),
        progress_message=row.get("progress_message", ""),
    )


@app.get("/api/jobs/{job_id}/candidates", response_model=list[CandidateResponse])
async def get_candidates(job_id: str, classification: str | None = None,
                         platform: str | None = None,
                         used: int | None = None, limit: int = 200,
                         include_excluded: bool = False,
                         current=Depends(auth.authenticate)):
    """List candidates for a job. Default hides classification='제외'.
    Pass include_excluded=true to see them.
    Freelancers only see jobs assigned to them."""
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "이 작업에 대한 권한 없음")
    rows = db.list_candidates(
        job_id=job_id,
        classification=classification,
        platform=platform,
        used=used,
        limit=limit,
    )
    if not include_excluded and classification != "제외":
        rows = [r for r in rows if (r.get("classification") or "") != "제외"]
    return [CandidateResponse(**{k: r.get(k) for k in CandidateResponse.model_fields})
            for r in rows]


class AddVideoUrlRequest(BaseModel):
    url: str
    note: str | None = None


class AddFromChannelRequest(BaseModel):
    channel_url: str       # TikTok / YouTube 채널 URL
    min_views: int = 1000000   # 최소 조회수 (기본 100만)
    since_days: int = 7        # 최근 N일 안 (0 = 전체 기간)
    max_videos: int = 50       # 채널에서 몇 개까지 스캔할지
    note: str | None = None


class PreviewChannelRequest(BaseModel):
    channel_url: str
    min_views: int = 1000000
    since_days: int = 0
    max_videos: int = 1000


class AddSelectedVideosRequest(BaseModel):
    videos: list[dict]   # 각 dict: {video_id, url, title, view_count, ...} (preview 결과 그대로)
    note: str | None = None


@app.post("/api/dissections/{diss_id}/add-video-url")
async def add_video_url(diss_id: str, req: AddVideoUrlRequest,
                          current=Depends(auth.authenticate)):
    """URL 직접 입력으로 영상을 카테고리 후보풀에 추가.
    YouTube / TikTok / Instagram 다 지원.
    yt-dlp로 메타 받아서 candidate_videos에 INSERT. 중복은 자동 거절."""
    import re
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "카테고리 없음")
    job_id = diss.get("related_job_id")
    if not job_id:
        raise HTTPException(400, "카테고리에 작업 연결 없음 — 발굴 먼저")
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "이 카테고리 권한 없음")

    url = (req.url or "").strip()
    # URL 패턴 검증 — YouTube / TikTok / Instagram
    is_youtube = bool(re.search(
        r"(youtube\.com/(?:shorts/|watch\?v=)|youtu\.be/)[A-Za-z0-9_-]{11}", url
    ))
    is_tiktok = bool(re.search(
        r"(tiktok\.com/(@[\w.-]+/video/\d+|t/[\w]+|v/\d+)|vm\.tiktok\.com/[\w]+)", url
    ))
    is_instagram = bool(re.search(
        r"instagram\.com/(reel|reels|p)/[\w-]+", url
    ))
    if not (is_youtube or is_tiktok or is_instagram):
        raise HTTPException(
            400,
            "지원 안 되는 URL 형식 — YouTube (shorts/watch/youtu.be) / TikTok / Instagram (reel/p)만 지원",
        )

    # yt-dlp로 메타 받기 — URL 그대로 넘기면 platform 자동 검출
    from workers.youtube_client import _ytdlp_video_meta_by_url
    meta = await _ytdlp_video_meta_by_url(url, timeout=25)
    if not meta:
        raise HTTPException(400, "영상 메타 받기 실패 (URL 잘못됐거나 영상 차단)")

    # 이미 후보풀에 있나 확인 — platform + video_id 기준
    with db.get_db() as conn:
        existing = conn.execute(
            "SELECT id, title FROM candidate_videos "
            "WHERE job_id=? AND platform=? AND video_id=?",
            (job_id, meta["platform"], meta["video_id"]),
        ).fetchone()
    if existing:
        return {
            "ok": True, "duplicate": True,
            "id": existing["id"],
            "message": f"이미 후보풀에 있음 ({existing['title'][:40]})",
        }

    # candidate_videos INSERT
    note = (req.note or "").strip() or "수동 추가 (URL)"
    try:
        with db.get_db() as conn:
            cur = conn.execute(
                "INSERT INTO candidate_videos "
                "(job_id, platform, video_id, url, title, channel_name, channel_id, "
                "view_count, duration, published_at, thumbnail_url, caption, "
                "classification, notes, is_manual) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 1)",
                (job_id, meta["platform"], meta["video_id"], meta["url"],
                 meta["title"], meta["channel_name"], meta["channel_id"],
                 meta["view_count"], meta["duration"], meta["published_at"],
                 meta["thumbnail_url"], meta["caption"], note),
            )
            new_id = cur.lastrowid
    except Exception as e:
        raise HTTPException(500, f"자료 저장 실패: {e}")

    return {
        "ok": True, "duplicate": False, "id": new_id,
        "title": meta["title"], "view_count": meta["view_count"],
        "duration": meta["duration"], "channel_name": meta["channel_name"],
        "thumbnail_url": meta["thumbnail_url"],
    }


@app.post("/api/dissections/{diss_id}/preview-channel")
async def preview_channel(diss_id: str, req: PreviewChannelRequest,
                            current=Depends(auth.authenticate)):
    """채널 스캔 + 조건 만족 viral list 반환 (자료 추가 X). 화면에서 선택용."""
    from datetime import datetime, timezone, timedelta
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "카테고리 없음")
    job_id = diss.get("related_job_id")
    if not job_id:
        raise HTTPException(400, "카테고리 작업 연결 없음")
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "권한 없음")

    channel_url = (req.channel_url or "").strip()
    if "tiktok.com/@" not in channel_url and "youtube.com" not in channel_url \
       and "youtu.be" not in channel_url:
        raise HTTPException(400, "TikTok / YouTube 채널 URL만 지원")

    from workers.youtube_client import _ytdlp_channel_list, search_channel_videos_via_api
    # 1) yt-dlp로 채널 전체 영상 (최신순, 빠름)
    ytdlp_videos = await _ytdlp_channel_list(channel_url, max_videos=req.max_videos, timeout=120)
    # 2) YouTube Data API로 인기순 50개 (yt-dlp 못 잡는 옛 viral 잡기)
    api_videos = []
    if ytdlp_videos:
        sample_ch_id = next((v.get("channel_id") for v in ytdlp_videos if v.get("channel_id")), None)
        if sample_ch_id:
            try:
                api_videos = await search_channel_videos_via_api(sample_ch_id, max_results=50, order="viewCount")
            except Exception as _e:
                print(f"[preview-channel] API fallback fail: {_e}", flush=True)
                api_videos = []
    # 3) dedup (video_id 기준) — yt-dlp 우선 (full metadata)
    seen = {v.get("video_id") for v in ytdlp_videos if v.get("video_id")}
    for v in api_videos:
        vid = v.get("video_id") or v.get("id")
        if vid and vid not in seen:
            # normalize API result format → ytdlp format
            ytdlp_videos.append({
                "platform": "youtube",
                "video_id": vid,
                "url": v.get("url") or f"https://youtube.com/shorts/{vid}",
                "title": v.get("title", ""),
                "view_count": v.get("view_count") or 0,
                "duration": v.get("duration") or 0,
                "channel_name": v.get("channel_name", ""),
                "channel_id": v.get("channel_id", ""),
                "thumbnail_url": v.get("thumbnail_url"),
                "published_at": v.get("published_at"),
            })
            seen.add(vid)
    videos = ytdlp_videos
    if not videos:
        raise HTTPException(400, "채널 영상 못 받음")

    # 필터
    matched = []
    if req.since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=req.since_days)
    else:
        cutoff = None
    for v in videos:
        if v["view_count"] < req.min_views:
            continue
        if cutoff is not None:
            pub = v.get("published_at")
            if pub:
                try:
                    pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass
        matched.append(v)
    matched.sort(key=lambda v: v["view_count"], reverse=True)

    # 이미 후보풀에 있는 거 표시 (중복)
    existing_ids = set()
    if matched:
        with db.get_db() as conn:
            placeholders = ",".join("?" for _ in matched)
            rows = conn.execute(
                f"SELECT platform || ':' || video_id AS key FROM candidate_videos "
                f"WHERE job_id=? AND platform || ':' || video_id IN ({placeholders})",
                [job_id] + [f"{v['platform']}:{v['video_id']}" for v in matched],
            ).fetchall()
            existing_ids = {r["key"] for r in rows}
    for v in matched:
        v["already_added"] = f"{v['platform']}:{v['video_id']}" in existing_ids

    return {
        "scanned": len(videos),
        "matched": len(matched),
        "videos": matched,
    }


@app.post("/api/dissections/{diss_id}/add-selected-videos")
async def add_selected_videos(diss_id: str, req: AddSelectedVideosRequest,
                                current=Depends(auth.authenticate)):
    """미리보기에서 형님이 선택한 영상들을 자료에 추가."""
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "카테고리 없음")
    job_id = diss.get("related_job_id")
    if not job_id:
        raise HTTPException(400, "카테고리 작업 연결 없음")
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "권한 없음")

    note = (req.note or "").strip() or "채널에서 선택 추가"
    added = []
    duplicates = []
    with db.get_db() as conn:
        for v in (req.videos or []):
            if not isinstance(v, dict) or not v.get("video_id"):
                continue
            try:
                existing = conn.execute(
                    "SELECT id FROM candidate_videos "
                    "WHERE job_id=? AND platform=? AND video_id=?",
                    (job_id, v.get("platform", "youtube"), v["video_id"]),
                ).fetchone()
                if existing:
                    duplicates.append(v["video_id"])
                    continue
                conn.execute(
                    "INSERT INTO candidate_videos "
                    "(job_id, platform, video_id, url, title, channel_name, channel_id, "
                    "view_count, duration, published_at, thumbnail_url, "
                    "classification, notes, is_manual) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 1)",
                    (job_id, v.get("platform", "youtube"), v["video_id"],
                     v.get("url", ""), v.get("title", ""),
                     v.get("channel_name", ""), v.get("channel_id", ""),
                     int(v.get("view_count") or 0), int(v.get("duration") or 0),
                     v.get("published_at"), v.get("thumbnail_url"), note),
                )
                added.append(v["video_id"])
            except Exception as e:
                print(f"[add-selected] {v.get('video_id')} fail: {e}", flush=True)

    return {"ok": True, "added": len(added), "duplicates": len(duplicates)}


@app.post("/api/dissections/{diss_id}/add-from-channel")
async def add_from_channel(diss_id: str, req: AddFromChannelRequest,
                             current=Depends(auth.authenticate)):
    """채널 URL (TikTok / YouTube) 입력 → 기준 (조회수/날짜) 만족 영상만
    후보풀에 추가. 모두 is_manual=1 저장."""
    from datetime import datetime, timezone, timedelta
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "카테고리 없음")
    job_id = diss.get("related_job_id")
    if not job_id:
        raise HTTPException(400, "카테고리에 작업 연결 없음 — 발굴 먼저")
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "이 카테고리 권한 없음")

    channel_url = (req.channel_url or "").strip()
    if not channel_url:
        raise HTTPException(400, "채널 URL 필요")
    if "tiktok.com/@" not in channel_url and "youtube.com" not in channel_url \
       and "youtu.be" not in channel_url:
        raise HTTPException(
            400, "지원 채널 URL — TikTok (@user) / YouTube (@user/channel)"
        )

    from workers.youtube_client import _ytdlp_channel_list
    videos = await _ytdlp_channel_list(channel_url, max_videos=req.max_videos, timeout=90)
    if not videos:
        raise HTTPException(400, "채널 영상 못 받음 — URL 잘못됐거나 채널 비공개")

    # 필터: 조회수 + 날짜 (since_days=0이면 날짜 무시 — 전체 기간 viral 다 가져옴)
    matched = []
    below_views = 0
    too_old = 0
    no_date = 0
    if req.since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=req.since_days)
    else:
        cutoff = None
    for v in videos:
        if v["view_count"] < req.min_views:
            below_views += 1
            continue
        if cutoff is not None:
            pub = v.get("published_at")
            if not pub:
                no_date += 1
            else:
                try:
                    pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if pub_dt < cutoff:
                        too_old += 1
                        continue
                except Exception:
                    no_date += 1
        matched.append(v)
    # view_count 내림차순 정렬 (인기순)
    matched.sort(key=lambda v: v["view_count"], reverse=True)

    # DB INSERT (중복 자동 skip)
    note = (req.note or "").strip() or f"채널 추가 ({req.since_days}일 + {req.min_views//10000}만+)"
    added = []
    duplicates = []
    with db.get_db() as conn:
        for v in matched:
            try:
                existing = conn.execute(
                    "SELECT id FROM candidate_videos "
                    "WHERE job_id=? AND platform=? AND video_id=?",
                    (job_id, v["platform"], v["video_id"]),
                ).fetchone()
                if existing:
                    duplicates.append(v["video_id"])
                    continue
                conn.execute(
                    "INSERT INTO candidate_videos "
                    "(job_id, platform, video_id, url, title, channel_name, channel_id, "
                    "view_count, duration, published_at, thumbnail_url, "
                    "classification, notes, is_manual) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 1)",
                    (job_id, v["platform"], v["video_id"], v["url"],
                     v["title"], v["channel_name"], v["channel_id"],
                     v["view_count"], v["duration"], v["published_at"],
                     v["thumbnail_url"], note),
                )
                added.append({
                    "video_id": v["video_id"], "title": v["title"],
                    "view_count": v["view_count"], "url": v["url"],
                })
            except Exception as e:
                print(f"[add-from-channel] {v['video_id']} insert fail: {e}", flush=True)

    return {
        "ok": True,
        "scanned": len(videos),
        "matched": len(matched),
        "added": len(added),
        "duplicates": len(duplicates),
        "below_views": below_views,
        "too_old": too_old,
        "no_date": no_date,
        "added_videos": added[:30],   # 처음 30개만 반환
    }


@app.get("/api/candidates/{cand_id}/preview-mp4")
async def candidate_preview_mp4(cand_id: int,
                                  current=Depends(auth.authenticate)):
    """후보 영상 작은 미리보기 mp4 (360p, 작은 사이즈). cache 후 stream.
    yt-dlp로 작게 다운 (~5MB) → /data/preview_cache/에 저장 → mp4 stream.
    iframe embed가 안 되는 TikTok/Instagram도 작은 video tag로 재생 가능."""
    from fastapi.responses import FileResponse
    import hashlib, asyncio
    from workers.youtube_client import YT_DLP

    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (cand_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    cand = dict(row)
    url = cand.get("url", "")
    if not url:
        raise HTTPException(404, "candidate url 없음")

    cache_dir = (_BB_DATA / "preview_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{h}.mp4"

    if not cache_file.exists() or cache_file.stat().st_size < 1000:
        # yt-dlp 작은 사이즈 다운 (360p 이하 우선, 안 되면 best)
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, url,
            # 브라우저 호환성 우선 H.264 + 작은 사이즈 우선 (720p 이하)
            "-f", "best[height<=720][vcodec*=avc1][ext=mp4]/"
                  "best[vcodec*=avc1][ext=mp4]/"
                  "best[height<=720][ext=mp4]/"
                  "best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-warnings", "--no-part",
            "--retries", "3",
            "--socket-timeout", "20",
            "-o", str(cache_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=80)
        except asyncio.TimeoutError:
            try: proc.kill()
            except Exception: pass
            raise HTTPException(504, "다운로드 시간 초과")
        if not cache_file.exists() or cache_file.stat().st_size < 1000:
            err_msg = err.decode()[-300:] if err else "unknown"
            raise HTTPException(500, f"yt-dlp 다운 실패: {err_msg}")

    return FileResponse(
        str(cache_file), media_type="video/mp4",
        headers={"Cache-Control": "max-age=86400", "Accept-Ranges": "bytes"},
    )


@app.post("/api/jobs/{job_id}/upload-candidate")
async def upload_candidate_video(job_id: str,
                                   file: UploadFile = File(...),
                                   title: str = Form(""),
                                   source_url: str = Form(""),
                                   current=Depends(auth.admin_only)):
    """후보 풀에 mp4 직접 업로드 (yt-dlp 차단 영상 등을 형님이 외부에서 받아 올림).
    새 candidate row 만들고 data/originals/에 저장."""
    from workers import ai_remix as _ar
    import hashlib as _hl
    from datetime import datetime as _dt

    content = await file.read()
    if len(content) < 1000:
        raise HTTPException(400, f"file too small ({len(content)} bytes)")

    # Make a stable URL+id from file content hash
    fhash = _hl.md5(content).hexdigest()[:16]
    fake_url = source_url.strip() or f"manual://uploaded/{fhash}"
    video_id = (file.filename or fhash).replace("/", "_")[:80] or fhash

    # Save to originals cache
    cache = _ar._orig_cache_path(fake_url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(content)
    if not await _ar._validate_video_file(cache):
        cache.unlink(missing_ok=True)
        raise HTTPException(400, "video invalid (0 frames or corrupt)")

    # Probe duration via ffprobe
    duration = 0.0
    try:
        import asyncio as _aio
        ffprobe_bin = _ar._ffmpeg().replace("ffmpeg", "ffprobe")
        proc = await _aio.create_subprocess_exec(
            ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(cache),
            stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE,
        )
        out, _ = await _aio.wait_for(proc.communicate(), timeout=10)
        duration = float((out or b"0").decode().strip() or "0")
    except Exception:
        pass

    now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    title_use = title.strip() or (file.filename or "직접 업로드 영상")

    with db.get_db() as conn:
        cur = conn.execute("""
            INSERT INTO candidate_videos
            (job_id, platform, video_id, url, title, channel_name, channel_id,
             view_count, duration, classification, used, created_at, memo_kr)
            VALUES (?, 'manual', ?, ?, ?, '직접 업로드', '', 0, ?, '키핑', 0, ?, '[manual upload]')
        """, (job_id, video_id, fake_url, title_use, duration, now))
        new_id = cur.lastrowid
        conn.commit()
    return {"ok": True, "candidate_id": new_id, "size": len(content), "duration": duration}


class BulkCandidateRequest(BaseModel):
    candidate_ids: list[int]
    action: str  # "delete" or "exclude"


@app.get("/api/candidates/{cand_id}/download-original")
async def candidate_download_original(cand_id: int,
                                       current=Depends(auth.authenticate)):
    """원본 best quality 영상 다운로드 — yt-dlp로 best mp4 다운 후 stream.
    캐시 ~/data/download_cache/. 큰 사이즈라 비싸지만 진짜 원본.
    """
    from fastapi.responses import FileResponse
    import hashlib, asyncio
    from workers.youtube_client import YT_DLP

    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (cand_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    cand = dict(row)
    url = cand.get("url", "")
    if not url:
        raise HTTPException(404, "candidate url 없음")

    cache_dir = (_BB_DATA / "download_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{h}.mp4"

    if not cache_file.exists() or cache_file.stat().st_size < 10000:
        # yt-dlp best quality 다운 (H.264 + 1080p+)
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, url,
            "--remote-components", "ejs:github",
            "-f", "best[ext=mp4][vcodec*=avc1]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-warnings", "--no-part",
            "--retries", "3",
            "--socket-timeout", "30",
            "-o", str(cache_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            try: proc.kill()
            except Exception: pass
            raise HTTPException(504, "다운로드 시간 초과 (3분)")
        if not cache_file.exists() or cache_file.stat().st_size < 10000:
            err_msg = err.decode()[-300:] if err else "unknown"
            raise HTTPException(500, f"yt-dlp 실패: {err_msg}")

    # 파일명 — 채널_제목.mp4
    safe_ch = "".join(c if c.isalnum() or c in "_-가-힣" else "_" for c in (cand.get("channel_name") or "vid"))[:20]
    safe_title = "".join(c if c.isalnum() or c in "_-가-힣" else "_" for c in (cand.get("title") or ""))[:40]
    download_name = f"{safe_ch}_{safe_title}.mp4"

    # 색보정 강제 (crf 0 무손실 = 원본화질 보존 + 중복 감지 회피) — 캐시
    color_file = cache_dir / f"{h}_color.mp4"
    if not color_file.exists() or color_file.stat().st_size < 10000:
        import random
        hue = random.uniform(-5, 5)
        sat = random.uniform(1.02, 1.06)
        bri = random.uniform(0.008, 0.022)
        con = random.uniform(1.01, 1.035)
        color = f"hue=h={hue:.1f}:s={sat:.3f},eq=brightness={bri:.3f}:contrast={con:.3f}"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(cache_file), "-vf", color,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "copy", "-pix_fmt", "yuv420p", "-y", str(color_file),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, cerr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            try: proc.kill()
            except Exception: pass
            raise HTTPException(504, "색보정 시간 초과")
        if not color_file.exists() or color_file.stat().st_size < 10000:
            # 색보정 실패 시 원본 fallback
            color_file = cache_file

    return FileResponse(
        color_file, media_type="video/mp4",
        filename=download_name,
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/candidates/bulk")
async def bulk_candidate_action(req: BulkCandidateRequest,
                                  current=Depends(auth.admin_only)):
    """선택 candidates 일괄 처리. action='delete' 진짜 삭제 | 'exclude' 마킹."""
    if not req.candidate_ids:
        return {"ok": True, "affected": 0}
    if req.action not in ("delete", "exclude"):
        raise HTTPException(400, "action must be delete or exclude")
    placeholders = ",".join(["?"] * len(req.candidate_ids))
    with db.get_db() as conn:
        if req.action == "delete":
            # FK constraint 회피 — remixes 먼저 삭제 후 candidate 삭제
            conn.execute(
                f"DELETE FROM remixes WHERE candidate_id IN ({placeholders})",
                req.candidate_ids,
            )
            cur = conn.execute(
                f"DELETE FROM candidate_videos WHERE id IN ({placeholders})",
                req.candidate_ids,
            )
        else:
            cur = conn.execute(
                f"UPDATE candidate_videos SET classification='제외', "
                f"memo_kr = COALESCE(memo_kr || ' / ', '') || '[bulk] manual exclude' "
                f"WHERE id IN ({placeholders})",
                req.candidate_ids,
            )
        affected = cur.rowcount
    return {"ok": True, "affected": affected, "action": req.action}


class KoreanMatchRequest(BaseModel):
    korean_channel_url: str


_active_kor_match: dict[str, asyncio.Task] = {}


@app.post("/api/jobs/{job_id}/match-korean")
async def match_korean(job_id: str, req: KoreanMatchRequest,
                          current=Depends(auth.admin_only)):
    """한국 채널 URL을 받아 후보 풀과 1:1 매칭. 매칭된 후보는 자동 사용됨 마킹."""
    from workers import korean_match as _km
    existing = _active_kor_match.get(job_id)
    if existing and not existing.done():
        raise HTTPException(409, "이미 매칭 진행 중")
    async def progress_cb(pct, msg):
        await ws_manager.broadcast(f"kormatch:{job_id}", {
            "type": "progress", "progress": pct, "message": msg,
        })
    async def runner():
        try:
            result = await _km.match_korean_channel(
                job_id=job_id, korean_channel_url=req.korean_channel_url,
                progress_cb=progress_cb,
            )
            await ws_manager.broadcast(f"kormatch:{job_id}", {
                "type": "completed", "result": result,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws_manager.broadcast(f"kormatch:{job_id}", {
                "type": "error", "message": str(e),
            })
    task = asyncio.create_task(runner())
    _active_kor_match[job_id] = task
    return {"ok": True, "started": True}


@app.post("/api/jobs/{job_id}/match-korean-v4")
async def match_korean_v4(job_id: str, req: KoreanMatchRequest,
                          current=Depends(auth.admin_only)):
    """한국 매칭 v4 — CLIP 1차 + Pro Vision 2차 하이브리드."""
    from workers import korean_match_v4 as _km4
    existing = _active_kor_match.get(job_id)
    if existing and not existing.done():
        raise HTTPException(409, "이미 매칭 진행 중")
    async def progress_cb(pct, msg):
        await ws_manager.broadcast(f"kormatch:{job_id}", {
            "type": "progress", "progress": pct, "message": msg,
        })
    async def runner():
        try:
            result = await _km4.match_korean_v4(
                job_id=job_id, korean_channel_url=req.korean_channel_url,
                progress_cb=progress_cb,
            )
            await ws_manager.broadcast(f"kormatch:{job_id}", {
                "type": "completed", "result": result,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws_manager.broadcast(f"kormatch:{job_id}", {
                "type": "error", "message": str(e),
            })
    task = asyncio.create_task(runner())
    _active_kor_match[job_id] = task
    return {"ok": True, "started": True, "version": "v4"}


class RejectChannelRequest(BaseModel):
    channel_id: str | None = None
    channel_name: str | None = None
    reason: str | None = None


@app.post("/api/channels/reject")
async def reject_channel(req: RejectChannelRequest,
                         current=Depends(auth.admin_only)):
    """채널을 거부 list에 추가. 다음 매칭부터 자동 제외."""
    if not req.channel_id and not req.channel_name:
        raise HTTPException(400, "channel_id 또는 channel_name 필요")
    with db.get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO rejected_channels (channel_id, channel_name, reason, rejected_by) "
            "VALUES (?, ?, ?, ?)",
            (req.channel_id or "", req.channel_name or "", req.reason or "",
             current.get("id")),
        )
        conn.commit()
    return {"ok": True}


@app.get("/api/channels/rejected")
async def list_rejected_channels(current=Depends(auth.admin_only)):
    """거부 채널 list 반환."""
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id, channel_id, channel_name, reason, rejected_at FROM rejected_channels ORDER BY rejected_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/channels/rejected/{reject_id}")
async def unreject_channel(reject_id: int, current=Depends(auth.admin_only)):
    """거부 list에서 채널 제거."""
    with db.get_db() as conn:
        conn.execute("DELETE FROM rejected_channels WHERE id=?", (reject_id,))
        conn.commit()
    return {"ok": True}


@app.post("/api/candidates/{candidate_id}/reject-channel")
async def reject_candidate_channel(candidate_id: int,
                                    current=Depends(auth.admin_only)):
    """이 영상의 채널 통째로 거부 (form에 가짜 버튼 누르면)."""
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT channel_id, channel_name, job_id FROM candidate_videos WHERE id=?",
            (candidate_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    cand = dict(row)
    with db.get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO rejected_channels (channel_id, channel_name, reason, rejected_by) "
            "VALUES (?, ?, ?, ?)",
            (cand.get("channel_id") or "", cand.get("channel_name") or "",
             "수동 거부 (가짜 표시)", current.get("id")),
        )
        # 그 채널의 이 카테 안 모든 영상 자동 제외
        # ⚠️ channel_id 빈 string일 경우 channel_name만 사용 (빈 ID끼리 매치 방지)
        chan_id = cand.get("channel_id") or ""
        chan_name = cand.get("channel_name") or ""
        if chan_id.strip():
            # channel_id 있으면 ID 우선 매치 (id가 정확)
            conn.execute(
                "UPDATE candidate_videos SET classification='제외', used=1, "
                "memo_kr = COALESCE(memo_kr || ' / ', '') || '[거부 채널 자동 정리]' "
                "WHERE job_id=? AND channel_id=? AND used=0",
                (cand["job_id"], chan_id),
            )
        elif chan_name.strip():
            # channel_id 없으면 name으로만 매치
            conn.execute(
                "UPDATE candidate_videos SET classification='제외', used=1, "
                "memo_kr = COALESCE(memo_kr || ' / ', '') || '[거부 채널 자동 정리]' "
                "WHERE job_id=? AND channel_name=? AND used=0",
                (cand["job_id"], chan_name),
            )
        # 둘 다 빈 경우는 자동 제외 X (안전)
        affected = conn.execute(
            "SELECT changes() AS n"
        ).fetchone()["n"]
        conn.commit()
    return {"ok": True, "channel": cand.get("channel_name"), "auto_excluded": affected}


@app.websocket("/ws/kormatch/{job_id}")
async def ws_kormatch(ws: WebSocket, job_id: str, token: str | None = None):
    await ws_manager.connect(f"kormatch:{job_id}", ws)
    try:
        while True:
            await asyncio.sleep(30)
    except Exception:
        pass
    finally:
        ws_manager.disconnect(f"kormatch:{job_id}", ws)


@app.get("/api/jobs/{job_id}/candidate-stats")
async def get_candidate_stats(job_id: str, current=Depends(auth.authenticate)):
    """전체 / 사용 / 미사용 / 제외 stats — filter와 무관, job 전체 기준."""
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "이 작업에 대한 권한 없음")
    with db.get_db() as conn:
        row = conn.execute("""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN used=1 THEN 1 ELSE 0 END) AS used,
              SUM(CASE WHEN used=0 AND COALESCE(classification,'') != '제외' THEN 1 ELSE 0 END) AS unused,
              SUM(CASE WHEN COALESCE(classification,'') = '제외' THEN 1 ELSE 0 END) AS excluded
            FROM candidate_videos WHERE job_id=?
        """, (job_id,)).fetchone()
    return {
        "total": row["total"] or 0,
        "used": row["used"] or 0,
        "unused": row["unused"] or 0,
        "excluded": row["excluded"] or 0,
    }


@app.post("/api/admin/backfill-channel-names")
async def backfill_channel_names(job_id: str | None = None,
                                  current=Depends(auth.admin_only)):
    """빈 channel_name 영상 메타 일괄 보강. yt-dlp single fetch.
    job_id 주면 그 카테만, 안 주면 전체."""
    from workers import youtube_client
    with db.get_db() as conn:
        if job_id:
            rows = conn.execute(
                "SELECT id, video_id FROM candidate_videos "
                "WHERE (channel_name IS NULL OR channel_name='') AND platform='youtube' "
                "AND video_id != '' AND job_id=?", (job_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, video_id FROM candidate_videos "
                "WHERE (channel_name IS NULL OR channel_name='') AND platform='youtube' "
                "AND video_id != ''"
            ).fetchall()
    total = len(rows)
    if total == 0:
        return {"ok": True, "total": 0, "fixed": 0, "message": "빈 channel_name 영상 없음"}

    sem = asyncio.Semaphore(5)
    fixed = 0
    async def fix_one(rid, vid):
        nonlocal fixed
        async with sem:
            meta = await youtube_client._ytdlp_video_meta(vid)
            if not meta or not meta.get("channel_name"):
                return
            with db.get_db() as c:
                c.execute(
                    "UPDATE candidate_videos SET channel_name=?, channel_id=? WHERE id=?",
                    (meta.get("channel_name") or "", meta.get("channel_id") or "", rid),
                )
                c.commit()
            fixed += 1

    async def runner():
        await asyncio.gather(*[fix_one(r["id"], r["video_id"]) for r in rows])

    asyncio.create_task(runner())
    return {"ok": True, "total": total, "started": True,
            "message": f"{total}개 backfill 시작 — yt-dlp 병렬 5개, 분당 약 100개 처리"}


@app.get("/api/jobs/{job_id}/channel-stats")
async def get_channel_stats(job_id: str, current=Depends(auth.authenticate)):
    """채널별 영상 수 + 평균 view + 평균 길이. 점검용 (몇 개씩 가져왔는지 확인)."""
    if not auth.can_user_see_job(current, job_id):
        raise HTTPException(403, "이 작업에 대한 권한 없음")
    with db.get_db() as conn:
        rows = conn.execute("""
            SELECT
              COALESCE(NULLIF(channel_name, ''), '(unknown)') AS channel_name,
              channel_id,
              COUNT(*) AS count,
              AVG(view_count) AS avg_views,
              MAX(view_count) AS max_views,
              AVG(duration) AS avg_duration,
              SUM(CASE WHEN used=1 THEN 1 ELSE 0 END) AS used,
              SUM(CASE WHEN COALESCE(classification,'') = '제외' THEN 1 ELSE 0 END) AS excluded,
              SUM(CASE WHEN COALESCE(classification,'') = '키핑' THEN 1 ELSE 0 END) AS keep,
              SUM(CASE WHEN COALESCE(classification,'') = 'review' THEN 1 ELSE 0 END) AS review
            FROM candidate_videos WHERE job_id=?
            GROUP BY COALESCE(NULLIF(channel_name, ''), '(unknown)'), channel_id
            ORDER BY count DESC, max_views DESC
        """, (job_id,)).fetchall()
    return {
        "job_id": job_id,
        "channels": [dict(r) for r in rows],
        "total_channels": len(rows),
    }


@app.websocket("/ws/jobs/{job_id}/candidates")
async def ws_candidates(ws: WebSocket, job_id: str, token: str | None = None,
                        api_key: str | None = None):
    """Real-time updates room — broadcasts when any user marks/unmarks a
    candidate in this job. Multiple viewers (admin + freelancers) stay in sync."""
    user = None
    if token:
        try:
            payload = auth.decode_token(token)
            user = auth.get_user_by_id(int(payload.get("sub", 0)))
        except Exception:
            pass
    if not user:
        legacy = os.getenv("BACKEND_API_KEY", "")
        if not (api_key and legacy and api_key == legacy):
            await ws.close(code=4401)
            return
        user = {"id": 0, "role": "admin"}
    if not auth.can_user_see_job(user, job_id):
        await ws.close(code=4403)
        return
    room = f"candidates:{job_id}"
    await ws_manager.connect(room, ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; ignore content
    except WebSocketDisconnect:
        ws_manager.disconnect(room, ws)


@app.patch("/api/candidates/{candidate_id}")
async def update_candidate(candidate_id: int, classification: str | None = None,
                           notes: str | None = None, used: int | None = None,
                           memo_kr: str | None = None,
                           on_hold: int | None = None,
                           current=Depends(auth.authenticate)):
    """Update a candidate's classification, notes, used flag, memo_kr, or on_hold.

    used=1 → marks as 실제 업로드함, used=0 → unmark.
    on_hold=1 → 보류 (미사용 탭에 남아있되 색상 다름, 나중에 사용 예정).
    memo_kr → 한국어 메모.
    """
    updates: dict = {}
    if classification:
        updates["classification"] = classification
    if notes is not None:
        updates["notes"] = notes
    if memo_kr is not None:
        updates["memo_kr"] = memo_kr
    if on_hold is not None:
        updates["on_hold"] = int(bool(on_hold))
    if used is not None:
        flag = int(bool(used))
        updates["used"] = flag
        if flag:
            updates["used_at"] = datetime.utcnow().isoformat()
            updates["used_by_user_id"] = current.get("id")
            updates["used_by_username"] = current.get("full_name") or current.get("username")
        else:
            updates["used_at"] = None
            updates["used_by_user_id"] = None
            updates["used_by_username"] = None
    if updates:
        db.update_candidate(candidate_id, **updates)
        # Broadcast to anyone watching this job's candidate room
        cand = None
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_videos WHERE id=?", (candidate_id,)
            ).fetchone()
            if row:
                cand = dict(row)
        if cand:
            await ws_manager.broadcast(f"candidates:{cand['job_id']}", {
                "type": "candidate_updated",
                "candidate_id": candidate_id,
                "updates": updates,
                "by": current.get("full_name") or current.get("username"),
            })
    return {"ok": True}


# ============================================================
# Dissection (Stage 1: analyze) + start-search (Stage 2: trigger)
# ============================================================

class DissectRequest(BaseModel):
    name: str = Field(..., description="분석 작업 이름")
    reference_channels: list[str] = Field(default_factory=list,
                                          description="레퍼런스 채널 URL 0~5개")
    platforms: list[str] = Field(default=["youtube", "tiktok", "instagram"])
    min_views: int = Field(default=5_000_000)
    max_duration: int = Field(default=55)
    topic_hint: str = Field(default="",
                            description="주제 hint (콤마 구분 키워드). 채널이 여러 주제 다룰 때 그중 한 주제만 골라 분석. 예: '밀리터리, 군대, 총'")
    excluded_keywords: list[str] = Field(default_factory=lambda: [
        # Generic ranking/compilation
        "top", "ranking", "compilation", "worst", "best", "funniest",
        "girls vs boys", "girls vs guys", "vs",
        # Edit/troll/CGI patterns
        "troll", "trollface", "edit", "ai animation", "ai video",
        # Common spammy templates
        "tier list", "every", "fails of",
        # 인도 시그니처 — 형님 룰 인도 제외
        "hindi", "bollywood", "desi", "bhai", "bhaiya", "namaste",
        "हिन्दी", "भारत",
    ])
    excluded_channels: list[str] = Field(default_factory=list)
    notion_database_id: str | None = Field(None)


class DissectResponse(BaseModel):
    diss_id: str
    name: str
    status: str
    progress: int = 0
    progress_message: str = ""


@app.post("/api/dissect", response_model=DissectResponse)
async def start_dissection(req: DissectRequest):
    """Stage 1: Run dissection + keyword generation (no search yet).

    같은 이름 dissection이 이미 있으면 새로 만들지 않고 기존 거 업데이트 (재분석).
    → 형님 룰: 무조건 같은 이름은 합쳐짐. 드롭다운에 중복 안 생김.
    """
    if len(req.reference_channels) > 5:
        raise HTTPException(400, "Maximum 5 reference channels")

    # 같은 이름 dissection 검색 (대소문자 무시 + 앞뒤 공백 제거)
    target_name = (req.name or "").strip()
    existing = None
    for d in db.list_dissections(limit=500):
        if (d.get("name") or "").strip() == target_name:
            existing = d
            break

    if existing:
        diss_id = existing["id"]
        # 기존 dissection 업데이트 — 재분석 시작. 후보 영상은 보존 (재검색 시 dedup으로 추가).
        db.update_dissection(
            diss_id=diss_id,
            status="analyzing",
            progress=0,
            progress_message="재분석 시작 — 같은 이름 작업에 합쳐짐",
            reference_channels=json.dumps(req.reference_channels, ensure_ascii=False),
            platforms=json.dumps(req.platforms),
            min_views=req.min_views,
            max_duration=req.max_duration,
            excluded_keywords=json.dumps(req.excluded_keywords, ensure_ascii=False),
            excluded_channels=json.dumps(req.excluded_channels, ensure_ascii=False),
            notion_database_id=req.notion_database_id,
        )
    else:
        diss_id = f"diss_{uuid.uuid4().hex[:12]}"
        db.insert_dissection(
            diss_id=diss_id,
            name=req.name,
            reference_channels=json.dumps(req.reference_channels, ensure_ascii=False),
            platforms=json.dumps(req.platforms),
            min_views=req.min_views,
            max_duration=req.max_duration,
            excluded_keywords=json.dumps(req.excluded_keywords, ensure_ascii=False),
            excluded_channels=json.dumps(req.excluded_channels, ensure_ascii=False),
            notion_database_id=req.notion_database_id,
        )

    async def progress_cb(pct: int, msg: str):
        await ws_manager.broadcast(diss_id, {
            "type": "progress", "progress": pct, "message": msg,
        })

    async def runner():
        try:
            await pipeline.run_dissection_only(
                diss_id=diss_id,
                reference_channels=req.reference_channels,
                progress_cb=progress_cb,
                topic_hint=req.topic_hint or "",
            )
            await ws_manager.broadcast(diss_id, {"type": "ready"})
        except Exception as e:
            db.update_dissection(diss_id, status="failed", error=str(e))
            await ws_manager.broadcast(diss_id, {"type": "error", "message": str(e)})

    asyncio.create_task(runner())
    return DissectResponse(diss_id=diss_id, name=req.name,
                           status="analyzing", progress=0)


@app.get("/api/dissect/{diss_id}")
async def get_dissection(diss_id: str):
    row = db.get_dissection(diss_id)
    if not row:
        raise HTTPException(404, "Dissection not found")
    # Parse JSON fields for client convenience
    for key in ("dissection_result", "keywords_result",
                "reference_channels", "platforms",
                "excluded_keywords", "excluded_channels"):
        if row.get(key):
            try:
                row[key] = json.loads(row[key])
            except Exception:
                pass
    return row


class KeywordsUpdateRequest(BaseModel):
    keywords_result: dict


@app.get("/api/dissect/{diss_id}/references")
async def get_dissect_references(diss_id: str, current=Depends(auth.authenticate)):
    """카테에 사용된 레퍼런스 채널 list 반환 + 거부 여부 표시."""
    d = db.get_dissection(diss_id)
    if not d:
        raise HTTPException(404, "dissection not found")
    raw = d.get("reference_channels")
    try:
        ref_list = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except Exception:
        ref_list = []
    # ref_list는 URL string list 또는 [name, handle] 쌍 가능
    channels = []
    rejected_set = set()
    with db.get_db() as conn:
        for row in conn.execute("SELECT channel_id, channel_name FROM rejected_channels"):
            r = dict(row)
            if r.get("channel_id"): rejected_set.add(r["channel_id"])
            if r.get("channel_name"): rejected_set.add(r["channel_name"])
    for item in ref_list:
        if isinstance(item, list) and len(item) >= 2:
            name, handle = item[0], item[1]
        elif isinstance(item, str):
            name = item.split("/")[-1].lstrip("@")
            handle = item if item.startswith("@") or item.startswith("http") else "@" + item
        else:
            continue
        url = handle if handle.startswith("http") else f"https://www.youtube.com/{handle}"
        channels.append({
            "name": name,
            "handle": handle,
            "url": url,
            "rejected": (name in rejected_set) or (handle.lstrip("@") in rejected_set),
        })
    return {"diss_id": diss_id, "channels": channels}


@app.post("/api/jobs/{job_id}/refresh-candidates")
async def refresh_candidates(job_id: str, current=Depends(auth.admin_only)):
    """후보풀 재정리 — 거부 채널 영상 자동 제외 + 사용=1은 그대로 유지.

    동작: 거부 채널 list에 등록된 채널 영상 중 used=0인 거 자동으로
    classification='제외'+used=1 마킹 + 메모에 [거부 채널] 표시.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    with db.get_db() as conn:
        excluded = conn.execute(
            "UPDATE candidate_videos SET classification='제외', used=1, "
            "memo_kr = COALESCE(memo_kr || ' / ', '') || '[거부 채널 자동 정리]' "
            "WHERE job_id=? AND (used=0 OR used IS NULL) AND "
            "(channel_id IN (SELECT channel_id FROM rejected_channels WHERE channel_id != '') OR "
            " channel_name IN (SELECT channel_name FROM rejected_channels WHERE channel_name != ''))",
            (job_id,),
        ).rowcount
        conn.commit()
    return {"ok": True, "excluded": excluded}


@app.post("/api/candidates/{candidate_id}/unmark")
async def unmark_candidate(candidate_id: int, current=Depends(auth.admin_only)):
    """영상 1개 롤백 — used=0, classification='pending' 으로 reset.

    형님이 잘못 매칭된 거 발견 시 풀어주는 용도.
    """
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT memo_kr FROM candidate_videos WHERE id=?", (candidate_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    # 메모에서 [KOR... 부분 제거
    memo = (dict(row).get("memo_kr") or "")
    new_memo = re.sub(r"\s*/?\s*\[KOR[^\[]*", "", memo).strip(" /")
    with db.get_db() as conn:
        conn.execute(
            "UPDATE candidate_videos SET used=0, classification='pending', memo_kr=? WHERE id=?",
            (new_memo, candidate_id),
        )
        conn.commit()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/unmark-kor")
async def unmark_kor_all(job_id: str, current=Depends(auth.admin_only)):
    """카테 전체 한국 매칭 롤백 — [KOR... 메모 있는 영상 다 used=0 reset.

    위험. 형님 동의 후만.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    with db.get_db() as conn:
        affected = conn.execute(
            "UPDATE candidate_videos SET used=0, classification='pending' "
            "WHERE job_id=? AND memo_kr LIKE '%[KOR%'",
            (job_id,),
        ).rowcount
        # 메모에서 [KOR... 제거 (간단 처리 — 자세히는 위 unmark에서)
        for row in conn.execute(
            "SELECT id, memo_kr FROM candidate_videos WHERE job_id=? AND memo_kr LIKE '%[KOR%'",
            (job_id,),
        ).fetchall():
            r = dict(row)
            new_memo = re.sub(r"\s*/?\s*\[KOR[^\[]*", "", r["memo_kr"] or "").strip(" /")
            conn.execute("UPDATE candidate_videos SET memo_kr=? WHERE id=?",
                         (new_memo, r["id"]))
        conn.commit()
    return {"ok": True, "reset_count": affected}


@app.put("/api/dissect/{diss_id}/keywords")
async def update_dissect_keywords(diss_id: str, req: KeywordsUpdateRequest,
                                    current=Depends(auth.admin_only)):
    """형님이 PWA에서 키워드 수정/삭제/추가 시 호출."""
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    db.update_dissection(
        diss_id,
        keywords_result=json.dumps(req.keywords_result, ensure_ascii=False),
    )
    return {"ok": True}


@app.get("/api/dissections", response_model=list[dict])
async def list_dissections(current=Depends(auth.authenticate)):
    rows = db.list_dissections(limit=500)
    # 빈 dissection 자동 filter — 사용자 진행 안 한 거 list에서 X
    # (status='ready' + reference_channels 빈 거 + 이름이 자동 생성된 "해체 YYYY..." 형식)
    def _is_empty(r):
        ref = r.get("reference_channels") or "[]"
        try:
            import json as _j
            ref_list = _j.loads(ref) if isinstance(ref, str) else ref
        except Exception:
            ref_list = []
        name = (r.get("name") or "").strip()
        return (
            r.get("status") == "ready"
            and not ref_list
            and name.startswith("해체 ")
        )
    rows = [r for r in rows if not _is_empty(r)]
    if current.get("role") != "admin":
        # Freelancers only see dissections whose related_job is assigned to them
        assigned = set(auth.list_assigned_jobs(current["id"]))
        rows = [r for r in rows if r.get("related_job_id") in assigned]
    # 각 dissection에 영상 stats 추가 (총/사용/미사용)
    job_ids = [r.get("related_job_id") for r in rows if r.get("related_job_id")]
    stats: dict[str, dict] = {}
    if job_ids:
        with db.get_db() as conn:
            placeholders = ",".join("?" * len(job_ids))
            for row in conn.execute(
                f"SELECT job_id, COUNT(*) AS total, SUM(used) AS used "
                f"FROM candidate_videos WHERE job_id IN ({placeholders}) GROUP BY job_id",
                job_ids,
            ):
                total = row["total"] or 0
                used = row["used"] or 0
                stats[row["job_id"]] = {
                    "total": total,
                    "used": used,
                    "unused": total - used,
                }
    # 각 dissection의 assignees (프리랜서) list 추가 — 관리자 화면에서 필터링용
    assignees_by_job: dict[str, list[dict]] = {}
    if job_ids:
        with db.get_db() as conn:
            placeholders = ",".join("?" * len(job_ids))
            for row in conn.execute(
                f"SELECT ja.job_id, u.id AS user_id, u.username, "
                f"u.full_name, u.role_label "
                f"FROM job_assignments ja JOIN users u ON u.id = ja.user_id "
                f"WHERE ja.job_id IN ({placeholders})",
                job_ids,
            ):
                assignees_by_job.setdefault(row["job_id"], []).append({
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "full_name": row["full_name"],
                    "role_label": row["role_label"],
                })
    for r in rows:
        jid = r.get("related_job_id")
        r["stats"] = stats.get(jid) or {"total": 0, "used": 0, "unused": 0}
        r["assignees"] = assignees_by_job.get(jid) or []
    return rows


@app.get("/api/cost-summary")
async def cost_summary(current=Depends(auth.authenticate)):
    """전체 + 카테별 누적 cost (USD + KRW). KRW 환율 1380 hardcode."""
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, COALESCE(cost_usd, 0) AS cost_usd "
            "FROM dissection_analyses ORDER BY cost_usd DESC"
        ).fetchall()
        total_diss = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM dissection_analyses"
        ).fetchone()["s"]
        total_remix = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM remixes"
        ).fetchone()["s"]
    total = float(total_diss or 0) + float(total_remix or 0)
    KRW_RATE = 1380
    return {
        "total_usd": round(total, 4),
        "total_krw": round(total * KRW_RATE),
        "krw_rate": KRW_RATE,
        "by_dissection": [
            {"id": r["id"], "name": r["name"],
             "cost_usd": round(float(r["cost_usd"] or 0), 4),
             "cost_krw": round(float(r["cost_usd"] or 0) * KRW_RATE)}
            for r in rows
        ],
    }


class StartSearchRequest(BaseModel):
    extra_keywords: list[str] = Field(default_factory=list,
                                      description="추가 키워드 (선택)")
    enable_visual_match: bool = Field(default=False,
                                      description="후보 영상을 한국 풀과 CLIP 매칭 (풀 빌드 선행 필요)")


@app.post("/api/dissect/{diss_id}/start-search")
async def start_search_from_dissection(diss_id: str, req: StartSearchRequest):
    """Stage 2: Trigger the actual search using analyzed keywords."""
    row = db.get_dissection(diss_id)
    if not row:
        raise HTTPException(404, "Dissection not found")
    if row["status"] not in ("ready", "completed"):
        raise HTTPException(400, f"Not ready (status={row['status']})")

    async def progress_cb(pct: int, msg: str):
        await ws_manager.broadcast(diss_id, {
            "type": "progress", "progress": pct, "message": msg, "phase": "search",
        })

    async def runner():
        try:
            result = await pipeline.run_search_from_dissection(
                diss_id=diss_id,
                extra_keywords=req.extra_keywords,
                progress_cb=progress_cb,
                enable_visual_match=req.enable_visual_match,
            )
            await ws_manager.broadcast(diss_id, {"type": "search_completed",
                                                  "result": result})
        except Exception as e:
            db.update_dissection(diss_id, status="failed", error=str(e))
            await ws_manager.broadcast(diss_id, {"type": "error", "message": str(e)})

    asyncio.create_task(runner())
    return {"ok": True, "diss_id": diss_id, "status": "searching"}


@app.websocket("/ws/dissect/{diss_id}")
async def ws_dissect(websocket: WebSocket, diss_id: str):
    await ws_manager.connect(diss_id, websocket)
    try:
        row = db.get_dissection(diss_id)
        if row:
            await websocket.send_json({
                "type": "state",
                "progress": row.get("progress", 0),
                "message": row.get("progress_message", ""),
                "status": row.get("status", "pending"),
            })
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(diss_id, websocket)
    except Exception:
        ws_manager.disconnect(diss_id, websocket)


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str):
    """WebSocket for live progress updates."""
    await ws_manager.connect(job_id, websocket)
    try:
        # send current state immediately
        row = db.get_job(job_id)
        if row:
            await websocket.send_json({
                "type": "state",
                "progress": row.get("progress", 0),
                "message": row.get("progress_message", ""),
                "status": row.get("status", "pending"),
            })
        while True:
            # Keep alive
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
    except Exception:
        ws_manager.disconnect(job_id, websocket)


# ===== iOS Shortcut endpoint =====

class QuickAddRequest(BaseModel):
    url: str = Field(..., description="틱톡/인스타/유튜브 URL")
    note: str = ""


@app.post("/api/quick-add")
async def quick_add_from_url(req: QuickAddRequest):
    """Polish endpoint for iOS Shortcut: receive a URL → analyze → save."""
    # TODO: detect platform from URL, extract metadata, run light DNA, save
    return {"ok": True, "url": req.url, "note": "TODO: implement"}


# ===== Authentication endpoints =====

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "freelancer"  # admin or freelancer
    full_name: str | None = None


# 개인 키 등록 필수 사용자(이 사람들만 본인 키 넣어야 기능 열림 = 비용 분리).
# 그 외 프리랜서는 글로벌(대표님) 키로 작동.
PERSONAL_KEY_USERNAMES = {"parkyuchan", "dksgusxo"}  # 김지영, 안현태


class UserPublic(BaseModel):
    id: int
    username: str
    role: str
    full_name: str | None = None
    role_label: str | None = None
    created_at: str | None = None
    last_login_at: str | None = None
    features: list[str] = []
    has_api_key: bool = False
    has_typecast_key: bool = False
    requires_personal_key: bool = False


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Username/password login → JWT token (PUBLIC endpoint)."""
    user = auth.get_user_by_username(req.username)
    if not user or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "잘못된 아이디 또는 비밀번호")
    auth.update_last_login(user["id"])
    token = auth.create_token(user["id"], user["username"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "full_name": user.get("full_name"),
        },
    }


@app.get("/api/auth/me", response_model=UserPublic)
async def me(current=Depends(auth.authenticate)):
    """Current user info (any authenticated user)."""
    data = {k: current.get(k) for k in UserPublic.model_fields if k not in ("features", "has_api_key", "has_typecast_key", "requires_personal_key")}
    data["features"] = auth._user_features(current)
    data["has_api_key"] = bool(current.get("gemini_api_key"))
    data["has_typecast_key"] = bool(current.get("typecast_api_key"))
    data["requires_personal_key"] = current.get("username") in PERSONAL_KEY_USERNAMES
    return UserPublic(**data)


class MyApiKeyRequest(BaseModel):
    gemini_api_key: str = ""


@app.post("/api/auth/my-api-key")
async def set_my_api_key(req: MyApiKeyRequest, current=Depends(auth.authenticate)):
    """본인 Gemini API 키 등록 — 프리랜서 자가 등록 시 기능 활성화."""
    key = (req.gemini_api_key or "").strip() or None
    with db.get_db() as conn:
        conn.execute("UPDATE users SET gemini_api_key=? WHERE id=?", (key, current["id"]))
    return {"ok": True, "has_key": bool(key)}


class MyTypecastKeyRequest(BaseModel):
    typecast_api_key: str = ""


@app.post("/api/auth/my-typecast-key")
async def set_my_typecast_key(req: MyTypecastKeyRequest, current=Depends(auth.authenticate)):
    """본인 타입캐스트(TTS) API 키 등록 — 더빙 TTS를 본인 키로 (비용 분리)."""
    key = (req.typecast_api_key or "").strip() or None
    with db.get_db() as conn:
        conn.execute("UPDATE users SET typecast_api_key=? WHERE id=?", (key, current["id"]))
    return {"ok": True, "has_key": bool(key)}


@app.post("/api/auth/register", response_model=UserPublic)
async def register(req: RegisterRequest, current=Depends(auth.admin_only)):
    """Admin-only — create a new user (freelancer or admin)."""
    if auth.get_user_by_username(req.username):
        raise HTTPException(400, "이미 존재하는 아이디")
    if req.role not in ("admin", "freelancer"):
        raise HTTPException(400, "role은 admin 또는 freelancer만")
    user = auth.create_user(req.username, req.password,
                            role=req.role, full_name=req.full_name)
    data = {k: user.get(k) for k in UserPublic.model_fields if k not in ("features", "has_api_key", "has_typecast_key", "requires_personal_key")}
    data["features"] = auth._user_features(user)
    data["has_api_key"] = bool(user.get("gemini_api_key"))
    data["has_typecast_key"] = bool(user.get("typecast_api_key"))
    data["requires_personal_key"] = user.get("username") in PERSONAL_KEY_USERNAMES
    return UserPublic(**data)


@app.get("/api/users", response_model=list[UserPublic])
async def list_users(current=Depends(auth.admin_only)):
    """Admin-only — list all users."""
    out = []
    for u in auth.list_users():
        data = {k: u.get(k) for k in UserPublic.model_fields if k not in ("features", "has_api_key", "has_typecast_key", "requires_personal_key")}
        data["features"] = auth._user_features(u)
        data["has_api_key"] = bool(u.get("gemini_api_key"))
        data["has_typecast_key"] = bool(u.get("typecast_api_key"))
        data["requires_personal_key"] = u.get("username") in PERSONAL_KEY_USERNAMES
        out.append(UserPublic(**data))
    return out


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, current=Depends(auth.admin_only)):
    """Admin-only — delete a user. Cannot delete self."""
    if current.get("id") == user_id:
        raise HTTPException(400, "자기 자신은 삭제 불가")
    auth.delete_user(user_id)
    return {"ok": True}


class ResetPasswordRequest(BaseModel):
    new_password: str


@app.post("/api/users/{user_id}/reset-password")
async def reset_password(user_id: int, req: ResetPasswordRequest,
                          current=Depends(auth.admin_only)):
    """Admin-only — 사용자 비밀번호 재설정."""
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "사용자 없음")
    if len(req.new_password) < 2:
        raise HTTPException(400, "비밀번호가 너무 짧음")
    auth.update_password(user_id, req.new_password)
    return {"ok": True, "username": user["username"]}


class SetFeaturesRequest(BaseModel):
    features: list[str]


@app.post("/api/users/{user_id}/features")
async def set_features(user_id: int, req: SetFeaturesRequest,
                        current=Depends(auth.admin_only)):
    """Admin-only — 사용자 기능 권한 설정 (subtitle/japanese/clip)."""
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "사용자 없음")
    import json as _j
    valid = [f for f in req.features if f in ("shorts", "subtitle", "audiosub", "ttsdub", "clip", "japanese")]
    with db.get_db() as conn:
        conn.execute("UPDATE users SET features=? WHERE id=?",
                     (_j.dumps(valid), user_id))
    return {"ok": True, "username": user["username"], "features": valid}


class SetApiKeyRequest(BaseModel):
    gemini_api_key: str = ""


@app.post("/api/users/{user_id}/api-key")
async def set_api_key(user_id: int, req: SetApiKeyRequest,
                       current=Depends(auth.admin_only)):
    """Admin — 사용자 개인 Gemini API 키 저장. 그 사용자의 작업은 이 키로 Gemini 호출 (비용 분리)."""
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "사용자 없음")
    key = (req.gemini_api_key or "").strip() or None
    with db.get_db() as conn:
        conn.execute("UPDATE users SET gemini_api_key=? WHERE id=?", (key, user_id))
    return {"ok": True, "username": user["username"], "has_key": bool(key)}


class SetTypecastKeyRequest(BaseModel):
    typecast_api_key: str = ""


@app.post("/api/users/{user_id}/typecast-key")
async def set_typecast_key(user_id: int, req: SetTypecastKeyRequest,
                            current=Depends(auth.admin_only)):
    """Admin — 사용자 개인 타입캐스트(TTS) 키 저장. 그 사용자 더빙 TTS는 이 키로 (비용 분리)."""
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "사용자 없음")
    key = (req.typecast_api_key or "").strip() or None
    with db.get_db() as conn:
        conn.execute("UPDATE users SET typecast_api_key=? WHERE id=?", (key, user_id))
    return {"ok": True, "username": user["username"], "has_key": bool(key)}


# ===== Job assignments (admin assigns jobs/categories to freelancers) =====

class AssignRequest(BaseModel):
    job_id: str
    user_id: int


@app.post("/api/assignments")
async def assign(req: AssignRequest, current=Depends(auth.admin_only)):
    """Admin-only — assign a job (=category/후보풀) to a freelancer."""
    auth.assign_job(req.job_id, req.user_id, assigned_by=current["id"])
    return {"ok": True}


@app.delete("/api/assignments")
async def unassign(req: AssignRequest, current=Depends(auth.admin_only)):
    auth.unassign_job(req.job_id, req.user_id)
    return {"ok": True}


@app.get("/api/assignments/job/{job_id}")
async def list_job_assignees(job_id: str, current=Depends(auth.admin_only)):
    return auth.list_job_freelancers(job_id)


@app.get("/api/assignments/user/{user_id}")
async def list_user_assignments(user_id: int, current=Depends(auth.admin_only)):
    """List job_ids assigned to a specific user (admin only)."""
    return auth.list_assigned_jobs(user_id)


@app.get("/api/assignments/me")
async def my_assignments(current=Depends(auth.authenticate)):
    """List job_ids assigned to current user (admins see all)."""
    if current.get("role") == "admin":
        return [j["id"] for j in db.list_jobs(limit=200)]
    return auth.list_assigned_jobs(current["id"])


# ===== Search result cache management =====

@app.get("/api/cache/stats")
async def cache_stats(current=Depends(auth.authenticate)):
    """How many cached searches are active per source."""
    return db.cache_stats()


@app.post("/api/cache/purge")
async def cache_purge(current=Depends(auth.admin_only)):
    """Drop expired cache rows (lazy cleanup)."""
    n = db.cache_purge_expired()
    return {"purged_expired": n}


@app.delete("/api/cache")
async def cache_clear_all(current=Depends(auth.admin_only)):
    """Wipe the entire search cache. Next searches will hit live APIs."""
    with db.get_db() as conn:
        cur = conn.execute("DELETE FROM search_cache")
        n = cur.rowcount
    return {"deleted": n}


# ===== Mascot management (per-dissection 2D 캐릭터) =====

class MascotOptionsRequest(BaseModel):
    concept: str = Field(..., description="컨셉 키워드 (예: 양봉 저승사자 코믹)")
    count: int = Field(default=5, ge=1, le=10)


class MascotSelectRequest(BaseModel):
    name: str
    concept: str
    style_prompt: str
    seed: int
    reference_image_path: str  # 시안 중 선택한 것의 로컬 경로


@app.post("/api/mascot/{diss_id}/options")
async def mascot_options(diss_id: str, req: MascotOptionsRequest,
                         current=Depends(auth.admin_only)):
    """Generate N candidate mascot images. Returns options with seed+url."""
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    options = await mascot_worker.generate_mascot_options(
        diss_id, req.concept, count=req.count)
    return {"options": options}


@app.post("/api/mascot/{diss_id}/select")
async def mascot_select(diss_id: str, req: MascotSelectRequest,
                        current=Depends(auth.admin_only)):
    """Persist the user's chosen mascot for this dissection."""
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    try:
        m = mascot_worker.select_mascot(
            dissection_id=diss_id,
            name=req.name,
            concept=req.concept,
            style_prompt=req.style_prompt,
            seed=req.seed,
            reference_image_path=req.reference_image_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(400, f"image not found: {e}")
    return m


@app.get("/api/mascot/{diss_id}")
async def mascot_get(diss_id: str, current=Depends(auth.authenticate)):
    """Read the mascot for a dissection (any auth user can read)."""
    m = db.get_mascot(diss_id)
    if not m:
        raise HTTPException(404, "no mascot set")
    return m


@app.get("/api/mascots")
async def mascot_list(current=Depends(auth.admin_only)):
    """List all mascots (admin only)."""
    return db.list_mascots()


@app.post("/api/mascot/{diss_id}/pose")
async def mascot_pose(diss_id: str, action_prompt: str,
                       current=Depends(auth.admin_only)):
    """Test endpoint — generate a single pose image with the saved mascot."""
    try:
        return await mascot_worker.generate_pose_image(diss_id, action_prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ===== Webtoon mascot pair system (savior + victim) =====

@app.post("/api/mascot/{diss_id}/recommend-pair")
async def mascot_recommend_pair(diss_id: str,
                                 current=Depends(auth.admin_only)):
    """Gemini로 카테고리 narrative archetype 분석 → savior + victim 1쌍 컨셉 추천.
    형님이 confirm 전에 review/edit 가능."""
    try:
        return await mascot_worker.recommend_mascot_pair(diss_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"추천 실패: {e}")


class MascotBaselineGenRequest(BaseModel):
    role: str = Field(..., description="savior or victim")
    character_spec: str = Field(..., description="명시적 캐릭터 spec (lockdown용)")
    count: int = Field(default=3, ge=1, le=6)


@app.post("/api/mascot/{diss_id}/generate-baseline")
async def mascot_generate_baseline(diss_id: str,
                                    req: MascotBaselineGenRequest,
                                    current=Depends(auth.admin_only)):
    """GPT-image-2 t2i로 baseline 시안 N장 generate (병렬). 형님이 1장 선택."""
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    options = await mascot_worker.generate_baseline_for_role(
        dissection_id=diss_id, role=req.role,
        character_spec=req.character_spec, count=req.count,
    )
    return {"options": options}


class MascotBaselineSelectRequest(BaseModel):
    role: str
    chosen_image_path: str
    concept_kr: str
    concept_en: str
    character_spec: str


@app.post("/api/mascot/{diss_id}/select-baseline")
async def mascot_select_baseline(diss_id: str,
                                  req: MascotBaselineSelectRequest,
                                  current=Depends(auth.admin_only)):
    """선택된 시안 → webtoon_{role}_baseline.png 확정 + DB cache."""
    try:
        m = mascot_worker.select_baseline(
            dissection_id=diss_id, role=req.role,
            chosen_image_path=req.chosen_image_path,
            concept_kr=req.concept_kr, concept_en=req.concept_en,
            character_spec=req.character_spec,
        )
    except FileNotFoundError as e:
        raise HTTPException(400, f"image not found: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return m


@app.get("/api/mascot/{diss_id}/pair")
async def mascot_pair_get(diss_id: str,
                           current=Depends(auth.authenticate)):
    """카테고리의 savior + victim 1쌍 정보 + 디스크에 누적된 모든 시안 list."""
    pair = db.get_mascot_pair(diss_id) or {
        "dissection_id": diss_id,
        "savior": {"concept_kr": None, "concept_en": None,
                   "character_spec": None, "baseline_path": None,
                   "baseline_url": None},
        "victim": {"concept_kr": None, "concept_en": None,
                   "character_spec": None, "baseline_path": None,
                   "baseline_url": None},
    }
    # 디스크에 저장된 모든 시안 자동 list (계속 골라 쓸 수 있게)
    pair["savior"]["options"] = mascot_worker.list_baseline_options(diss_id, "savior")
    pair["victim"]["options"] = mascot_worker.list_baseline_options(diss_id, "victim")
    return pair


@app.get("/api/mascot/{diss_id}/options")
async def mascot_options_list(diss_id: str, role: str,
                                current=Depends(auth.admin_only)):
    """디스크에 누적된 모든 시안 list (단독 호출용)."""
    return {"options": mascot_worker.list_baseline_options(diss_id, role)}


@app.delete("/api/mascot/{diss_id}/options")
async def mascot_options_delete(diss_id: str, role: str, filename: str,
                                  current=Depends(auth.admin_only)):
    """단일 시안 삭제."""
    ok = mascot_worker.delete_baseline_option(diss_id, role, filename)
    if not ok:
        raise HTTPException(404, "option not found")
    return {"ok": True}


# ===== Dynamic N-roles endpoints (자유 archetype 1~3 마스코트) =====

@app.post("/api/mascot/{diss_id}/recommend-roles")
async def mascot_recommend_roles(diss_id: str,
                                  current=Depends(auth.admin_only)):
    """Gemini로 카테고리에 맞는 마스코트 1~3개 자유 추천 (savior/victim 강제 없음)."""
    try:
        roles = await mascot_worker.recommend_mascot_roles(diss_id)
        return {"roles": roles}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"추천 실패: {e}")


class RoleBaselineGenRequest(BaseModel):
    role_id: str
    character_spec: str
    count: int = Field(default=3, ge=1, le=6)
    # 자료에 role 자동 박을 때 같이 박을 메타 (Cloudflare 524 나도 화면 reload 후 보임)
    name_kr: str = ""
    role_label_kr: str = ""
    narrative_role: str = ""
    concept_kr: str = ""
    concept_en: str = ""


@app.post("/api/mascot/{diss_id}/generate-baseline-role")
async def mascot_generate_baseline_role(diss_id: str,
                                          req: RoleBaselineGenRequest,
                                          current=Depends(auth.admin_only)):
    """role_id별 GPT-image-2 시안 N장 (누적).

    Cloudflare 524 (100초 시간 초과) 떨어져도 자료에 role 남게 — 시작 시 즉시 자료에 박음.
    그래야 화면 reload 후 디스크에 만들어진 시안 1~N장이라도 표시됨.
    """
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    # 시작 시 즉시 자료에 role 박음 — 시간 초과로 응답 못 가도 안전
    db.upsert_mascot_role(
        dissection_id=diss_id, role_id=req.role_id,
        character_spec=req.character_spec,
        name_kr=req.name_kr or None,
        role_label_kr=req.role_label_kr or None,
        narrative_role=req.narrative_role or None,
        concept_kr=req.concept_kr or None,
        concept_en=req.concept_en or None,
    )
    options = await mascot_worker.generate_baseline_for_role_id(
        dissection_id=diss_id, role_id=req.role_id,
        character_spec=req.character_spec, count=req.count,
    )
    return {"options": options}


class RoleBaselineSelectRequest(BaseModel):
    role_id: str
    chosen_image_path: str
    name_kr: str = ""
    role_label_kr: str = ""
    narrative_role: str = ""
    concept_kr: str = ""
    concept_en: str = ""
    character_spec: str = ""


@app.post("/api/mascot/{diss_id}/select-baseline-role")
async def mascot_select_baseline_role(diss_id: str,
                                        req: RoleBaselineSelectRequest,
                                        current=Depends(auth.admin_only)):
    try:
        roles = await mascot_worker.select_baseline_for_role_id(
            dissection_id=diss_id, role_id=req.role_id,
            chosen_image_path=req.chosen_image_path,
            name_kr=req.name_kr, role_label_kr=req.role_label_kr,
            narrative_role=req.narrative_role,
            concept_kr=req.concept_kr, concept_en=req.concept_en,
            character_spec=req.character_spec,
        )
    except FileNotFoundError as e:
        raise HTTPException(400, f"image not found: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"roles": roles}


@app.post("/api/mascot/{diss_id}/auto-spec/{role_id}")
async def mascot_auto_spec(diss_id: str, role_id: str,
                             current=Depends(auth.admin_only)):
    """기존 confirmed baseline image를 Gemini Vision으로 분석 → character_spec 자동 보강."""
    roles = db.get_mascot_roles(diss_id) or []
    role = next((r for r in roles if r.get("role_id") == role_id), None)
    if not role or not role.get("baseline_path"):
        raise HTTPException(404, "role baseline not found")
    from pathlib import Path as _P
    try:
        new_spec = await mascot_worker.auto_describe_baseline(_P(role["baseline_path"]))
    except Exception as e:
        raise HTTPException(500, f"auto-spec 실패: {e}")
    db.upsert_mascot_role(dissection_id=diss_id, role_id=role_id,
                          character_spec=new_spec)
    return {"role_id": role_id, "character_spec": new_spec}


@app.get("/api/mascot/{diss_id}/roles")
async def mascot_roles_get(diss_id: str,
                            current=Depends(auth.authenticate)):
    """카테고리 마스코트 N roles + 디스크 누적 시안 list.

    turnaround_urls에 mtime cache-bust query 박음 (재생성한 이미지가 같은 파일명이라
    브라우저가 옛 이미지 그대로 표시하는 거 막음).
    """
    roles = db.get_mascot_roles(diss_id) or []
    for r in roles:
        rid = r.get("role_id") or "role"
        r["options"] = mascot_worker.list_baseline_options_for_role_id(diss_id, rid)
        # turnaround urls — mtime cache-bust 박음
        ta_paths = r.get("turnaround_paths") or []
        urls = []
        for p in ta_paths:
            pp = Path(p)
            if pp.exists():
                mt = int(pp.stat().st_mtime)
                urls.append(f"/mascots/{diss_id}/{pp.name}?v={mt}")
        r["turnaround_urls"] = urls
        # baseline_url도 cache-bust (재생성 시 같은 파일명 보존)
        bp = r.get("baseline_path")
        if bp:
            bpp = Path(bp)
            if bpp.exists():
                mt = int(bpp.stat().st_mtime)
                r["baseline_url"] = f"/mascots/{diss_id}/{bpp.name}?v={mt}"
    return {"dissection_id": diss_id, "roles": roles}


# ============================================================
# 8각 turnaround endpoints
# ============================================================
_active_turnarounds: dict[str, asyncio.Task] = {}


@app.post("/api/mascot/{diss_id}/generate-turnaround/{role_id}")
async def mascot_generate_turnaround(diss_id: str, role_id: str,
                                      current=Depends(auth.admin_only)):
    """role의 baseline → 8각 turnaround generate (~$0.24, ~4분 sequential).
    background에서 진행 + WebSocket으로 progress 전송.
    """
    roles = db.get_mascot_roles(diss_id) or []
    role = next((r for r in roles if r.get("role_id") == role_id), None)
    if not role:
        raise HTTPException(404, f"role {role_id} not found")
    if not role.get("baseline_path"):
        raise HTTPException(400, "baseline 확정 먼저 — 시안 ✅ 후 8각 generate 가능")

    key = f"{diss_id}:{role_id}"
    existing = _active_turnarounds.get(key)
    if existing and not existing.done():
        raise HTTPException(409, "이미 8각 generate 진행 중")

    async def progress_cb(pct: int, msg: str):
        await ws_manager.broadcast(f"turnaround:{key}", {
            "type": "turnaround_progress",
            "role_id": role_id, "progress": pct, "message": msg,
        })

    async def runner():
        try:
            result = await mascot_worker.generate_turnaround_for_role(
                diss_id, role_id, progress_cb=progress_cb,
            )
            await ws_manager.broadcast(f"turnaround:{key}", {
                "type": "turnaround_completed",
                "role_id": role_id, "result": result,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            db.upsert_mascot_role(diss_id, role_id, turnaround_status="failed")
            await ws_manager.broadcast(f"turnaround:{key}", {
                "type": "error", "message": f"8각 generate 실패: {e}",
            })

    task = asyncio.create_task(runner())
    _active_turnarounds[key] = task

    def _cleanup(t: asyncio.Task) -> None:
        if _active_turnarounds.get(key) is t:
            _active_turnarounds.pop(key, None)

    task.add_done_callback(_cleanup)
    return {"ok": True, "diss_id": diss_id, "role_id": role_id, "status": "generating"}


@app.post("/api/mascot/{diss_id}/regenerate-turnaround/{role_id}/{angle_id}")
async def mascot_regenerate_turnaround_angle(diss_id: str, role_id: str,
                                              angle_id: str,
                                              current=Depends(auth.admin_only)):
    """단일 각도 ↻ 재생성 (~$0.03, ~30초)."""
    try:
        result = await mascot_worker.regenerate_turnaround_angle(
            diss_id, role_id, angle_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"각도 재생성 실패: {e}")


@app.delete("/api/mascot/{diss_id}/turnaround/{role_id}/{angle_id}")
async def mascot_delete_turnaround_angle(diss_id: str, role_id: str,
                                          angle_id: str,
                                          current=Depends(auth.admin_only)):
    """단일 각도 ✕ 삭제 — 디스크 unlink + DB sync."""
    try:
        result = mascot_worker.delete_turnaround_angle(
            diss_id, role_id, angle_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/mascot/{diss_id}/roles/{role_id}")
async def mascot_role_delete(diss_id: str, role_id: str,
                               current=Depends(auth.admin_only)):
    """role 자체 삭제 (DB row + 디스크 시안 폴더는 그대로 보존)."""
    roles = db.delete_mascot_role(diss_id, role_id)
    return {"roles": roles}


class MascotRolesSaveRequest(BaseModel):
    roles: list[dict]


@app.post("/api/mascot/{diss_id}/save-roles")
async def mascot_roles_save(diss_id: str, req: MascotRolesSaveRequest,
                              current=Depends(auth.admin_only)):
    """frontend mascotRoles 전체를 backend에 한꺼번에 sync (저장 버튼).
    삭제된 role 제외 + 편집된 정보 (concept_kr/character_spec 등) 모두 저장.
    """
    if not db.get_dissection(diss_id):
        raise HTTPException(404, "dissection not found")
    # 기존 자료의 baseline/turnaround 보호:
    # 화면이 baseline_path 등을 안 보내고 저장하면 자료의 정상 값이 null로 덮여버리는 버그가 있었음.
    # 시안 확정 후 저장 누르면 baseline 사라져서 클립 만들기 실패. 이걸 막기 위해
    # 화면이 안 보낸 필드는 기존 자료 값을 유지함.
    existing_roles = db.get_mascot_roles(diss_id) or []
    existing_by_id = {r.get("role_id"): r for r in existing_roles}
    cleaned = []
    for r in (req.roles or []):
        if not isinstance(r, dict): continue
        rid = (r.get("role_id") or "").strip()
        if not rid: continue
        existing = existing_by_id.get(rid) or {}
        cleaned.append({
            "role_id": rid,
            "role_label_kr": r.get("role_label_kr") or "",
            "narrative_role": r.get("narrative_role") or "",
            "name_kr": r.get("name_kr") or "",
            "concept_kr": r.get("concept_kr") or "",
            "concept_en": r.get("concept_en") or "",
            "character_spec": r.get("character_spec") or "",
            # 화면이 보낸 값 우선, 비어있으면 기존 자료 값 유지 (덮어쓰기 X)
            "baseline_path": r.get("baseline_path") or existing.get("baseline_path"),
            "baseline_url": r.get("baseline_url") or existing.get("baseline_url"),
            "chosen_filename": r.get("chosen_filename") or existing.get("chosen_filename"),
            # turnaround 8각 정보도 같이 보호
            "turnaround_paths": r.get("turnaround_paths") or existing.get("turnaround_paths"),
            "turnaround_status": r.get("turnaround_status") or existing.get("turnaround_status"),
        })
    db.replace_mascot_roles(diss_id, cleaned)
    return {"ok": True, "roles_count": len(cleaned)}


class RoleOptionDeleteRequest(BaseModel):
    role_id: str
    filename: str


@app.delete("/api/mascot/{diss_id}/options-role")
async def mascot_options_role_delete(diss_id: str,
                                       role_id: str, filename: str,
                                       current=Depends(auth.admin_only)):
    """role_id별 단일 시안 삭제."""
    ok = mascot_worker.delete_baseline_option_for_role_id(diss_id, role_id, filename)
    if not ok:
        raise HTTPException(404, "option not found")
    return {"ok": True}


# ===== AI 변형 (마스코트 합본) =====

class RemixSpecUpdate(BaseModel):
    spec: dict


def _candidate_or_404(candidate_id: int) -> dict:
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (candidate_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    return dict(row)


def _mascot_for_job(job_id: str) -> dict | None:
    with db.get_db() as conn:
        diss = conn.execute(
            "SELECT id FROM dissection_analyses WHERE related_job_id=?",
            (job_id,),
        ).fetchone()
    if not diss:
        return None
    return db.get_mascot(diss["id"])


def _mascot_pair_for_job(job_id: str) -> dict | None:
    """legacy webtoon_static용 1쌍 (savior + victim)."""
    with db.get_db() as conn:
        diss = conn.execute(
            "SELECT id FROM dissection_analyses WHERE related_job_id=?",
            (job_id,),
        ).fetchone()
    if not diss:
        return None
    return db.get_mascot_pair(diss["id"])


def _mascot_roles_for_job(job_id: str) -> list[dict]:
    """동적 N roles list (자유 archetype) — webtoon_static 우선."""
    with db.get_db() as conn:
        diss = conn.execute(
            "SELECT id FROM dissection_analyses WHERE related_job_id=?",
            (job_id,),
        ).fetchone()
    if not diss:
        return []
    return db.get_mascot_roles(diss["id"]) or []


async def _fix_zero_duration(cand: dict) -> dict:
    """candidate의 duration이 0이면 yt-dlp로 실측 + 자료 갱신"""
    if cand.get("duration") and cand.get("duration") > 0:
        return cand
    url = cand.get("url")
    if not url:
        return cand
    try:
        import asyncio as _asyncio
        proc = await _asyncio.create_subprocess_exec(
            _bb_ytdlp(),
            "--get-duration", "--quiet", "--no-warnings", url,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        out, _ = await _asyncio.wait_for(proc.communicate(), timeout=20)
        dur_str = (out or b"").decode().strip().splitlines()[-1] if out else ""
        # yt-dlp는 "MM:SS" 또는 "HH:MM:SS" 또는 "SSS" 형식 반환
        parts = dur_str.split(":")
        if len(parts) == 1:
            dur = float(parts[0]) if parts[0] else 0
        elif len(parts) == 2:
            dur = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            dur = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            dur = 0
        if dur > 0:
            db.update_candidate(cand["id"], duration=dur)
            cand["duration"] = dur
    except Exception as e:
        print(f"[duration_fix] candidate {cand['id']} duration 실측 실패: {e}", flush=True)
    return cand


@app.post("/api/remix/analyze/{candidate_id}")
async def remix_analyze(candidate_id: int, force: bool = False,
                        current=Depends(auth.authenticate)):
    """Gemini가 영상 직접 보고 spec JSON 자동 추천.

    Cache: 이전 분석 결과 있으면 즉시 반환 (Gemini 호출 X).
    force=true 로 강제 재분석 가능.
    """
    cand = _candidate_or_404(candidate_id)
    # duration이 0이면 yt-dlp로 실측 + 자료 갱신
    cand = await _fix_zero_duration(cand)
    mascot = _mascot_for_job(cand["job_id"])
    mascot_pair = _mascot_pair_for_job(cand["job_id"])
    mascot_roles = _mascot_roles_for_job(cand["job_id"])
    has_baseline = (
        any(r.get("baseline_path") for r in mascot_roles) or
        bool(mascot_pair and (
            (mascot_pair.get("savior") or {}).get("baseline_path") or
            (mascot_pair.get("victim") or {}).get("baseline_path")
        ))
    )
    if not mascot and not has_baseline:
        raise HTTPException(
            400,
            "이 영상이 속한 카테고리에 마스코트가 설정되지 않았습니다. "
            "작업 탭에서 🎭 마스코트를 먼저 만들어주세요.",
        )

    if not force:
        existing = db.get_latest_remix_for_candidate(candidate_id)
        if existing and existing.get("spec"):
            try:
                spec = json.loads(existing["spec"])
            except Exception:
                spec = None
            if spec and (spec.get("clips") is not None or spec.get("error")):
                return {
                    "candidate": {k: cand.get(k) for k in
                                  ("id", "url", "title", "duration", "channel_name")},
                    "mascot": {k: (mascot or {}).get(k) for k in
                               ("id", "name", "concept", "reference_image_url",
                                "seed", "style_prompt")} if mascot else None,
                    "mascot_pair": mascot_pair,
                    "mascot_roles": mascot_roles,
                    "spec": spec,
                    "cached": True,
                    "cached_at": existing.get("created_at"),
                    "remix_id": existing["id"],
                }

    # graceful analyze fail — error in spec, but mascot dropdown still works
    try:
        # 영상 길이 전달 → clip start/end 영상 길이 초과 자동 clamp
        _cand_dur = None
        try:
            _cand_dur = float(cand.get("duration") or 0) or None
        except (TypeError, ValueError):
            _cand_dur = None
        spec = await ai_remix.analyze_for_remix(
            cand["url"], mascot=mascot, mascot_pair=mascot_pair,
            mascot_roles=mascot_roles, video_duration=_cand_dur,
        )
    except Exception as e:
        err_msg = str(e)[:500]
        spec = {
            "error": f"분석 실패: {err_msg}",
            "concept": f"❌ 분석 실패: {err_msg[:200]}",
            "pattern": "",
            "clips": [],
        }

    # Persist for cache + future render step
    mid = (mascot or {}).get("id")
    remix_id = db.insert_remix(candidate_id=candidate_id, mascot_id=mid)
    n_clips = len(spec.get("clips") or [])
    pattern = spec.get("pattern") or ""
    msg = (f"분석 완료 — {n_clips}개 클립" +
           (f" (pattern={pattern})" if pattern else ""))
    db.update_remix(
        remix_id,
        status="analyzed",
        spec=json.dumps(spec, ensure_ascii=False),
        progress_message=msg,
    )

    return {
        "candidate": {k: cand.get(k) for k in
                      ("id", "url", "title", "duration", "channel_name")},
        "mascot": {k: (mascot or {}).get(k) for k in
                   ("id", "name", "concept", "reference_image_url", "seed",
                    "style_prompt")} if mascot else None,
        "mascot_pair": mascot_pair,
        "mascot_roles": mascot_roles,
        "spec": spec,
        "cached": False,
        "remix_id": remix_id,
    }


def _strip_frontend_flags(spec: dict) -> dict:
    """frontend-only _* flag (_rerendering, _regenerating, _refining 등) 청소."""
    if not isinstance(spec, dict):
        return spec
    clips = spec.get("clips") or []
    for c in clips:
        if not isinstance(c, dict):
            continue
        for k in list(c.keys()):
            if k.startswith("_"):
                del c[k]
    return spec


@app.put("/api/remix/{remix_id}/spec")
async def remix_update_spec(remix_id: int, req: RemixSpecUpdate,
                            current=Depends(auth.authenticate)):
    """형님이 모달에서 inline 편집한 spec 저장. _* flag 자동 청소."""
    if not db.get_remix(remix_id):
        raise HTTPException(404, "remix not found")
    cleaned = _strip_frontend_flags(dict(req.spec))
    db.update_remix(
        remix_id,
        spec=json.dumps(cleaned, ensure_ascii=False),
        status="analyzed",
    )
    return {"ok": True}


class RefineClipRequest(BaseModel):
    purpose_kr: str = ""
    mascot_action_kr: str = ""
    character_role_id: str = "savior"


@app.post("/api/remix/{remix_id}/refine-clip")
async def remix_refine_clip(remix_id: int, req: RefineClipRequest,
                              current=Depends(auth.authenticate)):
    """한글 동작/목적 → Gemini 분석 → 영어 expression + hand_action 자동 생성."""
    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    # 카테고리 mascot_roles 가져오기
    cand = db.get_candidate(remix["candidate_id"]) if hasattr(db, "get_candidate") else None
    if not cand:
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
            ).fetchone()
        cand = dict(row) if row else None
    mascot_roles = _mascot_roles_for_job(cand["job_id"]) if cand else []
    try:
        result = await ai_remix.refine_clip_with_korean(
            purpose_kr=req.purpose_kr,
            mascot_action_kr=req.mascot_action_kr,
            character_role_id=req.character_role_id,
            mascot_roles=mascot_roles,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"refine 실패: {e}")


@app.delete("/api/remix/{remix_id}")
async def remix_delete(remix_id: int, current=Depends(auth.admin_only)):
    """분석 결과 폐기 (재분석 위해)."""
    if not db.get_remix(remix_id):
        raise HTTPException(404, "remix not found")
    db.delete_remix(remix_id)
    return {"ok": True}


# ===== Active render task tracking — 창 닫아도 진행 + 취소 =====
# 영상 전체 만들기 (render-all + concat) — 영상당 1개만 동시 가능
_active_full_render: dict[int, asyncio.Task] = {}
# 클립별 재생성 — 같은 영상의 다른 클립은 동시 OK, 같은 클립만 막음
_active_clip_renders: dict[tuple[int, int], asyncio.Task] = {}


@app.get("/api/remix/{remix_id}/state")
async def remix_state(remix_id: int, current=Depends(auth.authenticate)):
    """렌더링 진행 상태 + 결과 fetch — 모달 다시 열 때 호출.
    창을 닫아도 backend는 계속 진행, 이걸로 다시 동기화.
    """
    r = db.get_remix(remix_id)
    if not r:
        raise HTTPException(404, "remix not found")
    spec = None
    if r.get("spec"):
        try:
            spec = json.loads(r["spec"])
        except Exception:
            pass
    full_task = _active_full_render.get(remix_id)
    is_full_running = bool(full_task) and not full_task.done()
    running_clip_idxs = sorted([
        cidx for (rid, cidx), t in _active_clip_renders.items()
        if rid == remix_id and not t.done()
    ])
    is_running = is_full_running or len(running_clip_idxs) > 0
    return {
        "remix_id": remix_id,
        "status": r.get("status"),
        "progress": r.get("progress") or 0,
        "progress_message": r.get("progress_message") or "",
        "spec": spec,
        "output_url": r.get("output_url"),
        "cost_usd": r.get("cost_usd"),
        "error": r.get("error"),
        "completed_at": r.get("completed_at"),
        "is_running_now": is_running,
        "is_full_rendering": is_full_running,
        "running_clip_idxs": running_clip_idxs,
    }


class SingleClipRenderRequest(BaseModel):
    clip_idx: int


@app.post("/api/remix/{remix_id}/render-clip")
async def remix_render_single_clip(remix_id: int, req: SingleClipRenderRequest,
                                     current=Depends(auth.authenticate)):
    """spec.clips[idx] 1개만 다시 render. 만들어진 영상 ↻ 재생성용.
    background에서 진행 + WebSocket 같은 채널 사용.
    """
    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    spec = json.loads(remix.get("spec") or "{}")
    clips = spec.get("clips") or []
    if req.clip_idx < 0 or req.clip_idx >= len(clips):
        raise HTTPException(400, f"clip_idx {req.clip_idx} out of range (0~{len(clips)-1})")

    # 같은 클립만 중복 막음 — 다른 클립은 동시 OK
    key = (remix_id, req.clip_idx)
    existing = _active_clip_renders.get(key)
    if existing and not existing.done():
        raise HTTPException(
            409,
            f"클립 {req.clip_idx+1} 이미 작업 중 — 끝날 때까지 기다리거나 🛑 취소 후 다시 시도",
        )
    # 영상 전체 만들기 (또는 합본) 진행 중이면 클립 재생성 거부
    full_task = _active_full_render.get(remix_id)
    if full_task and not full_task.done():
        raise HTTPException(
            409,
            "영상 전체 작업 중 — 그 작업 끝나야 클립 재생성 가능",
        )

    cand = None
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate gone")
    cand = dict(row)

    with db.get_db() as conn:
        diss = conn.execute(
            "SELECT id FROM dissection_analyses WHERE related_job_id=?",
            (cand["job_id"],),
        ).fetchone()
    if not diss:
        raise HTTPException(400, "dissection not found")
    diss_id = diss["id"]

    # 단일 clip만 추출해서 새 spec으로 render
    target_clip = clips[req.clip_idx]
    single_spec = {**spec, "clips": [target_clip]}

    async def progress_cb(pct: int, msg: str):
        db.update_remix(remix_id, status="rendering",
                        progress_message=f"clip {req.clip_idx+1} 재생성: {msg}")
        await ws_manager.broadcast(f"remix:{remix_id}", {
            "type": "remix_progress", "progress": pct,
            "message": f"clip {req.clip_idx+1} 재생성: {msg}",
            "clip_idx": req.clip_idx,
        })

    async def runner():
        try:
            from workers.ai_remix import REMIX_OUT_DIR
            # 단일 clip render — clip_path만 새로 만들기 위해 sample render 호출
            # clip_path는 final_dir / clip_<idx>.mp4 — 같은 path로 overwrite
            final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
            final_dir.mkdir(parents=True, exist_ok=True)
            clip_path = final_dir / f"clip_{req.clip_idx}.mp4"
            # ai_remix 의 single clip render 흐름 직접 호출
            # 내부 _gen_one 로직 재사용 — 새 motion_mode (webtoon_static) + role 정보
            from workers import webtoon_static as wts
            from workers import mascot as mascot_worker
            from workers.ai_remix import get_or_download_original
            duration = max(2.5, float(target_clip["end"] - target_clip["start"]))
            ch_raw = (target_clip.get("character") or "savior").strip()
            ch_legacy = {"angel": "savior", "reaper": "victim"}
            ch = ch_legacy.get(ch_raw.lower(), ch_raw)
            roles = db.get_mascot_roles(diss_id) or []
            role_data = next((r for r in roles if r.get("role_id") == ch), None) or (roles[0] if roles else None)
            if not role_data:
                raise RuntimeError(
                    f"마스코트 role 없음 — 카테고리 마스코트 먼저 만들어주세요"
                )
            baseline = role_data.get("baseline_path")
            if not baseline:
                role_name = role_data.get("role_id") or "마스코트"
                raise RuntimeError(
                    f"⚠️ '{role_name}' baseline 시안 확정 안 됨 — "
                    f"마스코트 모달에서 시안 만들고 '✅ 이 시안으로 확정' 눌러주세요"
                )
            spec_str = role_data.get("character_spec") or "- chibi character"
            expression = target_clip.get("expression") or target_clip.get("actor_emotion") or "calm neutral"
            hand_action = target_clip.get("hand_action") or None
            expr_prompt = mascot_worker.expression_prompt_for(
                character_spec=spec_str, expression=expression, hand_action=hand_action,
            )
            # 마스코트 위치 — spec.mascot_placement.x_center / y_center (가운데 좌표)
            # webtoon_static.py 안에서 sprite 실제 크기로 ffmpeg overlay 좌표 변환
            xc = None
            yc = None
            mirror = False
            mascot_w = 480
            mp = target_clip.get("mascot_placement")
            if isinstance(mp, dict):
                sz = mp.get("size")
                if isinstance(sz, (int, float)) and 100 <= int(sz) <= 1080:
                    mascot_w = int(sz)
                _xc = mp.get("x_center")
                if isinstance(_xc, (int, float)):
                    xc = int(_xc)
                _yc = mp.get("y_center")
                if isinstance(_yc, (int, float)):
                    yc = int(_yc)
                if mp.get("mirror"):
                    mirror = True
            clip_work = final_dir / f"_work_webtoon_{req.clip_idx}_regen"
            orig_path = await get_or_download_original(cand["url"])
            await progress_cb(20, "Kontext bg + GPT-image-2 표정 generate 중")
            fr = await wts.make_webtoon_static_clip(
                orig_path, float(target_clip["start"]), duration,
                out_clip=clip_path, work_dir=clip_work,
                baseline_path=Path(baseline),
                expression_prompt=expr_prompt,
                x_center=xc,
                y_center=yc,
                mirror=mirror,
                mascot_w=mascot_w,
                progress_cb=progress_cb,
            )
            shutil.rmtree(clip_work, ignore_errors=True)
            # spec.clips[idx] update — cache-bust query: 같은 path 새 mp4면 브라우저가 옛 cached 사용
            import time as _t
            cb = int(_t.time())
            target_clip["output_url"] = f"/remixes/remix_{remix_id}/clip_{req.clip_idx}.mp4?v={cb}"
            target_clip["duration_sec"] = round(duration, 2)
            target_clip["motion_mode"] = "webtoon_static"
            spec["clips"][req.clip_idx] = target_clip
            db.update_remix(
                remix_id, status="completed", progress=100,
                spec=json.dumps(spec, ensure_ascii=False),
                progress_message=f"✅ clip {req.clip_idx+1} 재생성 완료",
            )
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "clip_completed",
                "clip_idx": req.clip_idx,
                "output_url": target_clip["output_url"],
                "duration_sec": target_clip["duration_sec"],
                "motion_mode": target_clip["motion_mode"],
                "cost_usd": round(float(fr.get("cost_usd") or 0), 4),
            })
            from workers.notify import notify_success
            await notify_success(f"클립 {req.clip_idx+1} 완료 (remix {remix_id})",
                                  f"비용 ${round(float(fr.get('cost_usd') or 0), 4)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            db.update_remix(remix_id, status="failed", error=str(e))
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "clip_failed",
                "clip_idx": req.clip_idx,
                "message": f"clip {req.clip_idx+1} 재생성 실패: {e}",
            })
            from workers.notify import notify_error
            await notify_error(f"클립 {req.clip_idx+1} 실패 (remix {remix_id})", e)

    import shutil
    task = asyncio.create_task(runner())
    _active_clip_renders[key] = task

    def _cleanup(t: asyncio.Task) -> None:
        if _active_clip_renders.get(key) is t:
            _active_clip_renders.pop(key, None)

    task.add_done_callback(_cleanup)
    return {"ok": True, "remix_id": remix_id, "clip_idx": req.clip_idx, "status": "rendering"}


@app.post("/api/remix/{remix_id}/cancel")
async def remix_cancel(remix_id: int, current=Depends(auth.authenticate)):
    """진행 중 모든 task 취소 + 임시 자료 삭제.

    1) 영상 전체 만들기 task cancel
    2) 그 영상의 클립별 task 다 cancel
    3) 임시 work 폴더 (_work_webtoon_* / _concat_work*) 삭제
    4) DB status=cancelled
    """
    if not db.get_remix(remix_id):
        raise HTTPException(404, "remix not found")

    cancelled = []

    # 1) 영상 전체 작업 cancel
    full_task = _active_full_render.get(remix_id)
    if full_task and not full_task.done():
        full_task.cancel()
        try:
            await full_task
        except (asyncio.CancelledError, Exception):
            pass
        cancelled.append("full")
    _active_full_render.pop(remix_id, None)

    # 2) 클립별 task 다 cancel
    clip_keys = [k for k in list(_active_clip_renders.keys()) if k[0] == remix_id]
    for key in clip_keys:
        t = _active_clip_renders.get(key)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
            cancelled.append(f"clip{key[1]+1}")
        _active_clip_renders.pop(key, None)

    # 3) 임시 폴더 삭제 — 진행 중이던 자료 정리
    import shutil as _sh
    try:
        from workers.ai_remix import REMIX_OUT_DIR
        final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
        if final_dir.exists():
            for sub in final_dir.glob("_work_webtoon_*"):
                _sh.rmtree(sub, ignore_errors=True)
            for sub in final_dir.glob("_concat_work*"):
                _sh.rmtree(sub, ignore_errors=True)
    except Exception:
        pass

    # 4) DB + WebSocket 알림
    if cancelled:
        sep = ", "
        db.update_remix(remix_id, status="cancelled",
                        progress_message=f"취소 ({sep.join(cancelled)})",
                        error="cancelled by user")
        await ws_manager.broadcast(f"remix:{remix_id}", {
            "type": "error", "message": f"cancelled by user ({sep.join(cancelled)})",
        })
        return {"ok": True, "cancelled": True, "items": cancelled}

    # 이미 끝났거나 task 없는 경우 — DB만 mark + 임시 폴더 정리 끝남
    db.update_remix(remix_id, status="cancelled",
                    progress_message="취소 (이미 종료)",
                    error="cancelled (no active task)")
    return {"ok": True, "cancelled": False, "note": "no active task to cancel"}


@app.post("/api/remix/{remix_id}/cancel-clip/{clip_idx}")
async def remix_cancel_clip(remix_id: int, clip_idx: int,
                              current=Depends(auth.authenticate)):
    """단일 클립만 취소 — 다른 클립 작업은 계속 진행.

    1) (remix_id, clip_idx) task만 cancel
    2) 그 클립의 임시 work 폴더만 삭제 (_work_webtoon_{idx}_*)
    3) DB는 그대로 — 다른 클립 작업 중일 수 있으니 status 안 바꿈
    """
    if not db.get_remix(remix_id):
        raise HTTPException(404, "remix not found")

    key = (remix_id, clip_idx)
    task = _active_clip_renders.get(key)
    cancelled = False
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        cancelled = True
    _active_clip_renders.pop(key, None)

    # 그 클립의 임시 폴더만 삭제
    import shutil as _sh
    try:
        from workers.ai_remix import REMIX_OUT_DIR
        final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
        if final_dir.exists():
            for sub in final_dir.glob(f"_work_webtoon_{clip_idx}_*"):
                _sh.rmtree(sub, ignore_errors=True)
    except Exception:
        pass

    # WebSocket으로 클립 실패 알림 (frontend가 _rerendering 리셋)
    await ws_manager.broadcast(f"remix:{remix_id}", {
        "type": "clip_failed",
        "clip_idx": clip_idx,
        "message": "형님이 취소함",
    })
    return {"ok": True, "cancelled": cancelled, "clip_idx": clip_idx}


_active_concat: dict[int, asyncio.Task] = {}


@app.post("/api/remix/{remix_id}/concat")
async def remix_concat(remix_id: int, current=Depends(auth.authenticate)):
    """모든 clip output_url 검증 + insert mode concat → combined.mp4.

    각 clip이 ▶ 영상 만들기로 검수 완료된 후 호출. 원본 다운 + cut + concat만 함.
    같은 remix_id concat 진행 중이면 새 trigger 차단 (ffmpeg 중복 실행 방지).
    """
    # 중복 trigger 차단
    existing = _active_concat.get(remix_id)
    if existing and not existing.done():
        raise HTTPException(409, "이 합본 작업이 이미 진행 중 — 끝날 때까지 기다려")

    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    spec = json.loads(remix.get("spec") or "{}")
    clips = spec.get("clips") or []
    if not clips:
        raise HTTPException(400, "spec.clips 비어있음 — 분석 먼저")

    from workers.ai_remix import REMIX_OUT_DIR
    final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
    missing = []
    for i, _c in enumerate(clips):
        clip_path = final_dir / f"clip_{i}.mp4"
        if not clip_path.exists():
            missing.append(i + 1)
    if missing:
        raise HTTPException(400, f"미완성 클립: {missing} — ▶ 영상 만들기 먼저")

    with db.get_db() as conn:
        row = conn.execute(
            "SELECT url FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate gone")
    candidate_url = row["url"]

    # 이미 영상 전체 작업 또는 클립 작업 진행 중이면 합본 거부 (race-safe)
    existing = _active_full_render.get(remix_id)
    if existing and not existing.done():
        raise HTTPException(409, "영상 전체 작업 중 — 먼저 취소하거나 기다려")
    running_clip_idxs = [
        cidx for (rid, cidx), t in _active_clip_renders.items()
        if rid == remix_id and not t.done()
    ]
    if running_clip_idxs:
        raise HTTPException(
            409,
            f"클립 {[i+1 for i in running_clip_idxs]} 작업 중 — 끝나야 합본 가능",
        )

    async def runner():
        from workers.ai_remix import (get_or_download_original, cut_segment,
                                       concat_segments, _probe_duration)
        try:
            db.update_remix(remix_id, status="rendering", progress=5,
                            progress_message="🎬 합본 시작 — 원본 다운로드")
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "remix_progress", "progress": 5,
                "message": "🎬 합본 시작 — 원본 다운로드",
            })
            orig_path = await get_or_download_original(candidate_url)

            db.update_remix(remix_id, progress=40,
                            progress_message="원본 cut + clip 삽입 sequence 만드는 중")
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "remix_progress", "progress": 40,
                "message": "원본 cut + clip 삽입",
            })

            work_dir = final_dir / "_concat_work"
            work_dir.mkdir(exist_ok=True)
            # 원본 영상 실제 길이 측정 (tail cut 정확하게 자르기 위해)
            orig_total_dur = await _probe_duration(orig_path)
            sequence: list[Path] = []
            cursor = 0.0
            for i, c in enumerate(clips):
                s = float(c["start"])
                if s > cursor + 0.05:
                    seg = work_dir / f"orig_{i}.mp4"
                    await cut_segment(orig_path, cursor, s, seg)
                    sequence.append(seg)
                sequence.append(final_dir / f"clip_{i}.mp4")
                cursor = s
            # tail: 마지막 clip 이후 ~ 영상 끝. 영상 실제 길이 기반 (옛 600초 hack 제거).
            # 진짜 실패면 except: pass 대신 raise — 짤린 합본 silent 생성 막음.
            if orig_total_dur > 0 and cursor < orig_total_dur - 0.1:
                tail = work_dir / "orig_tail.mp4"
                await cut_segment(orig_path, cursor, orig_total_dur, tail)
                sequence.append(tail)

            db.update_remix(remix_id, progress=80,
                            progress_message=f"ffmpeg concat ({len(sequence)} 세그먼트)")
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "remix_progress", "progress": 80,
                "message": f"ffmpeg concat ({len(sequence)} 세그먼트)",
            })

            combined_path = final_dir / "combined.mp4"

            async def _concat_progress(pct: int, msg: str):
                # base 50% + concat pct의 절반 (50~100%)
                final_pct = 50 + int(pct / 2)
                db.update_remix(remix_id, progress=final_pct, progress_message=msg)
                await ws_manager.broadcast(f"remix:{remix_id}", {
                    "type": "remix_progress", "progress": final_pct, "message": msg,
                })

            await concat_segments(sequence, combined_path, progress_cb=_concat_progress)
            import time as _t
            combined_url = f"/remixes/remix_{remix_id}/combined.mp4?v={int(_t.time())}"

            import shutil as _sh
            _sh.rmtree(work_dir, ignore_errors=True)

            db.update_remix(remix_id, status="completed", progress=100,
                            output_url=combined_url,
                            progress_message="✅ 합본 완료")
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "remix_completed",
                "result": {
                    "clip_count": len(clips),
                    "clips": clips,
                    "combined_url": combined_url,
                    "cost_usd": 0,
                },
            })
            from workers.notify import notify_success
            await notify_success(f"합본 완료 (remix {remix_id})",
                                  f"클립 {len(clips)}개. 영상 확인하세요.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            db.update_remix(remix_id, status="failed", error=str(e))
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "error", "message": f"합본 실패: {e}",
            })
            from workers.notify import notify_error
            await notify_error(f"합본 실패 (remix {remix_id})", e)

    task = asyncio.create_task(runner())
    _active_full_render[remix_id] = task
    _active_concat[remix_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        if _active_concat.get(remix_id) is t:
            _active_concat.pop(remix_id, None)
        if _active_full_render.get(remix_id) is t:
            _active_full_render.pop(remix_id, None)

    task.add_done_callback(_cleanup)
    return {"ok": True, "remix_id": remix_id, "status": "concatenating",
            "clip_count": len(clips)}


@app.delete("/api/remix/{remix_id}/clip/{clip_idx}/output")
async def remix_delete_clip_output(remix_id: int, clip_idx: int,
                                    current=Depends(auth.admin_only)):
    """단일 clip의 만들어진 영상 삭제 — 디스크 파일 + spec.output_url + combined.mp4 모두 정리.

    합본은 stale이라 같이 무효화. 형님이 다시 ▶ 영상 만들기 + 🎬 합본 만들기 흐름.
    """
    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    spec = json.loads(remix.get("spec") or "{}")
    clips = spec.get("clips") or []
    if clip_idx < 0 or clip_idx >= len(clips):
        raise HTTPException(400, f"clip_idx {clip_idx} out of range (0~{len(clips)-1})")

    from workers.ai_remix import REMIX_OUT_DIR
    final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
    clip_path = final_dir / f"clip_{clip_idx}.mp4"
    clip_unlinked = False
    if clip_path.exists():
        clip_path.unlink()
        clip_unlinked = True

    combined_path = final_dir / "combined.mp4"
    combined_unlinked = False
    if combined_path.exists():
        combined_path.unlink()
        combined_unlinked = True

    clips[clip_idx]["output_url"] = None
    spec["clips"] = clips
    db.update_remix(
        remix_id,
        spec=json.dumps(spec, ensure_ascii=False),
        output_url=None,
    )
    return {
        "ok": True,
        "clip_idx": clip_idx,
        "clip_unlinked": clip_unlinked,
        "combined_unlinked": combined_unlinked,
    }


@app.post("/api/candidates/{candidate_id}/upload-original")
async def upload_original_video(candidate_id: int,
                                  file: UploadFile = File(...),
                                  current=Depends(auth.admin_only)):
    """Manually upload an mp4 for a candidate when yt-dlp fails (e.g., TikTok video stream blocked).
    Saves to data/originals/<md5(url)>.mp4 — same path get_or_download_original would use."""
    from workers import ai_remix as _ar
    cand = _candidate_or_404(candidate_id)
    cache = _ar._orig_cache_path(cand["url"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    if len(content) < 1000:
        raise HTTPException(400, f"file too small ({len(content)} bytes)")
    cache.write_bytes(content)
    # Validate via ffprobe (must have actual video frames)
    ok = await _ar._validate_video_file(cache)
    if not ok:
        cache.unlink(missing_ok=True)
        raise HTTPException(400, "video invalid (0 frames or corrupt)")
    return {"ok": True, "size": len(content), "path": str(cache)}


@app.get("/api/remix/{remix_id}/source-video")
async def remix_source_video(remix_id: int,
                             current=Depends(auth.authenticate)):
    """Serve the cached source mp4 for in-PWA player (Phase 2)."""
    from workers import ai_remix as _ar
    from fastapi.responses import FileResponse
    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate gone")
    cand = dict(row)
    try:
        orig = await _ar.get_or_download_original(cand["url"])
    except Exception as e:
        raise HTTPException(500, f"original fetch failed: {e}")
    if not orig.exists():
        raise HTTPException(404, "original mp4 not cached")
    return FileResponse(str(orig), media_type="video/mp4",
                        headers={"Cache-Control": "max-age=600",
                                 "Accept-Ranges": "bytes"})


@app.get("/api/remix/{remix_id}/preview-frame")
async def remix_preview_frame(remix_id: int, t: float = 0.0,
                              current=Depends(auth.authenticate)):
    """후보 영상의 t초 시점 frame을 1080x1920로 추출 → PNG 반환.
    디스크 cache 적용 — 같은 영상 같은 시점은 즉시 stream.
    """
    from workers import ai_remix as _ar
    from fastapi.responses import FileResponse
    import hashlib

    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate gone")
    cand = dict(row)

    # 디스크 cache — url + sec 조합으로 hash
    t_sec = max(0.0, float(t))
    cache_dir = (_BB_DATA / "frame_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{cand['url']}|{t_sec:.2f}".encode()).hexdigest()[:20]
    cache_file = cache_dir / f"{key}.png"
    if cache_file.exists() and cache_file.stat().st_size > 500:
        return FileResponse(
            str(cache_file), media_type="image/png",
            headers={"Cache-Control": "max-age=3600"},
        )

    # cache miss — 원본 다운 + ffmpeg
    try:
        orig = await _ar.get_or_download_original(cand["url"])
    except Exception as e:
        raise HTTPException(500, f"original fetch failed: {e}")

    # 영상 길이 확인 → t가 초과면 clamp (no packets 에러 방지)
    try:
        rc_d, out_d, _ = await _ar._run(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(orig), timeout=10.0,
        )
        dur = float(out_d.strip()) if rc_d == 0 and out_d.strip() else 0.0
    except Exception:
        dur = 0.0
    if dur > 0 and t_sec >= dur:
        t_sec = max(0.0, dur - 0.2)  # 영상 끝에서 0.2초 전
        key = hashlib.md5(f"{cand['url']}|{t_sec:.2f}".encode()).hexdigest()[:20]
        cache_file = cache_dir / f"{key}.png"
        if cache_file.exists() and cache_file.stat().st_size > 500:
            return FileResponse(
                str(cache_file), media_type="image/png",
                headers={"Cache-Control": "max-age=3600"},
            )

    # -ss를 -i 앞에 두면 fast seek (keyframe 기반, 빠름)
    rc, _, err = await _ar._run(
        _ar._ffmpeg(), "-y",
        "-ss", str(t_sec),
        "-i", str(orig),
        "-frames:v", "1",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
        "-q:v", "3",
        str(cache_file),
        timeout=30.0,
    )
    if rc != 0 or not cache_file.exists():
        raise HTTPException(500, f"frame extract failed (t={t_sec:.1f}, dur={dur:.1f}): {err[:200]}")
    return FileResponse(
        str(cache_file), media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.get("/api/remix/by-candidate/{candidate_id}")
async def remix_by_candidate(candidate_id: int,
                             current=Depends(auth.admin_only)):
    """가장 최근 remix row 조회 (cache hit 여부 + spec)."""
    row = db.get_latest_remix_for_candidate(candidate_id)
    if not row:
        return {"exists": False}
    spec = None
    if row.get("spec"):
        try:
            spec = json.loads(row["spec"])
        except Exception:
            pass
    return {"exists": True, "remix_id": row["id"], "status": row["status"],
            "output_url": row.get("output_url"),
            "spec": spec, "created_at": row.get("created_at")}


class RemixRenderRequest(BaseModel):
    motion_mode: str = Field(default="yangbong_v14")
    make_combined: bool = Field(default=True, description="원본+클립 합본 mp4도 만들지")


@app.get("/api/comfy/health")
async def comfy_health(current=Depends(auth.admin_only)):
    """Quick reachability check for local ComfyUI. Frontend uses to enable/disable comfy radio."""
    available = await comfy_client.healthcheck()
    return {"available": available, "url": comfy_client.COMFY_URL}


@app.get("/api/fal/health")
async def fal_health(current=Depends(auth.admin_only)):
    """Whether FAL_API_KEY is configured (cheap check, no API call)."""
    available = await fal_client.healthcheck()
    return {"available": available, "endpoint": fal_client.FAL_ENDPOINT}


@app.get("/api/kie/health")
async def kie_health(current=Depends(auth.admin_only)):
    """Whether KIE_API_KEY is configured."""
    available = await kie_client.healthcheck()
    return {"available": available, "base": kie_client.KIE_BASE}


@app.post("/api/remix/{remix_id}/render")
async def remix_render(remix_id: int, req: RemixRenderRequest = RemixRenderRequest(),
                       current=Depends(auth.admin_only)):
    """Phase 2: 마스코트 이미지 → sprite 클립 → 원본 다운 → ffmpeg 합본.
    Background task. 진행 상황은 /ws/remix/{remix_id} 로 push.
    """
    remix = db.get_remix(remix_id)
    if not remix:
        raise HTTPException(404, "remix not found")
    spec = json.loads(remix.get("spec") or "{}")
    if not spec.get("clips"):
        raise HTTPException(400, "spec has no clips — analyze first")

    cand = db.get_candidate(remix["candidate_id"]) if hasattr(db, "get_candidate") else None
    if not cand:
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_videos WHERE id=?", (remix["candidate_id"],)
            ).fetchone()
        cand = dict(row) if row else None
    if not cand:
        raise HTTPException(404, "candidate gone")

    # Find dissection_id from job → dissection
    with db.get_db() as conn:
        diss = conn.execute(
            "SELECT id FROM dissection_analyses WHERE related_job_id=?",
            (cand["job_id"],),
        ).fetchone()
    if not diss:
        raise HTTPException(400, "dissection not found for this job")
    diss_id = diss["id"]

    async def progress_cb(pct: int, msg: str):
        db.update_remix(remix_id, status="rendering", progress=pct,
                        progress_message=msg)
        await ws_manager.broadcast(f"remix:{remix_id}", {
            "type": "remix_progress", "progress": pct, "message": msg,
        })

    async def runner():
        try:
            db.update_remix(remix_id, status="rendering", progress=0)
            result = await ai_remix.render_remix(
                remix_id=remix_id,
                dissection_id=diss_id,
                candidate_url=cand["url"],
                spec=spec,
                progress_cb=progress_cb,
                motion_mode=req.motion_mode,
                make_combined=req.make_combined,
            )
            # Persist enriched spec (with per-clip output_url) back to DB
            spec["clips"] = result["clips"]
            db.update_remix(
                remix_id, status="completed",
                progress=100,
                output_url=result.get("combined_url"),
                spec=json.dumps(spec, ensure_ascii=False),
                cost_usd=result["cost_usd"],
                completed_at=datetime.utcnow().isoformat(),
            )
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "remix_completed", "result": result,
            })
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"❌ render_remix failed (remix_id={remix_id}):\n{tb}")
            # Persist short error + 짧은 traceback (first 2000 chars) for debugging
            db.update_remix(remix_id, status="failed",
                            error=f"{e}\n---\n{tb[:1900]}")
            await ws_manager.broadcast(f"remix:{remix_id}", {
                "type": "error", "message": str(e),
            })

    task = asyncio.create_task(runner())
    _active_full_render[remix_id] = task
    def _cleanup(t):
        # task 완료 시 active 목록에서 제거
        try:
            if _active_full_render.get(remix_id) is t:
                _active_full_render.pop(remix_id, None)
        except Exception:
            pass
    task.add_done_callback(_cleanup)
    return {"ok": True, "remix_id": remix_id, "status": "rendering"}


@app.websocket("/ws/remix/{remix_id}")
async def ws_remix(ws: WebSocket, remix_id: int, token: str | None = None,
                   api_key: str | None = None):
    """Real-time render progress room."""
    if token:
        try:
            auth.decode_token(token)
        except Exception:
            await ws.close(code=4401); return
    elif not (api_key and api_key == os.getenv("BACKEND_API_KEY", "")):
        await ws.close(code=4401); return
    room = f"remix:{remix_id}"
    await ws_manager.connect(room, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(room, ws)


@app.websocket("/ws/turnaround/{key}")
async def ws_turnaround(ws: WebSocket, key: str, token: str | None = None,
                          api_key: str | None = None):
    """8각 turnaround generate progress room. key = '{diss_id}:{role_id}'."""
    if token:
        try:
            auth.decode_token(token)
        except Exception:
            await ws.close(code=4401); return
    elif not (api_key and api_key == os.getenv("BACKEND_API_KEY", "")):
        await ws.close(code=4401); return
    room = f"turnaround:{key}"
    await ws_manager.connect(room, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(room, ws)


# ===== Visual matching: Korean pool endpoints =====

@app.post("/api/pool/build/{diss_id}")
async def build_korean_pool(diss_id: str):
    """Trigger a Korean visual-match pool build for a given dissection.

    Pulls the dissection's DNA, then runs:
      KR keyword search → channel discovery → DNA classify → CLIP index.
    Runs in background; progress flows to the existing /ws/dissect/{diss_id}.
    """
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "dissection not found")

    raw = diss.get("dissection_result") or "{}"
    try:
        dna = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        dna = {}
    if not dna:
        raise HTTPException(400, "dissection has no dissection_result yet")
    # If dissection has multi-channel structure, use the common DNA
    target_dna = dna.get("common_dna") or dna

    async def progress_cb(pct: int, msg: str):
        db.update_dissection(diss_id, progress=pct, progress_message=msg)
        await ws_manager.broadcast(diss_id, {
            "type": "pool_progress", "progress": pct, "message": msg,
        })

    async def runner():
        try:
            db.update_dissection(diss_id, status="pool_building", progress=0,
                                 progress_message="풀 빌드 시작")
            result = await kr_pool.build_korean_pool(
                dissection_id=diss_id,
                target_dna=target_dna,
                progress_cb=progress_cb,
            )
            db.update_dissection(diss_id, status="ready",
                                 progress_message=f"풀: {result['indexed_videos']}영상")
            await ws_manager.broadcast(diss_id, {"type": "pool_completed",
                                                  "result": result})
        except Exception as e:
            db.update_dissection(diss_id, status="failed", error=str(e))
            await ws_manager.broadcast(diss_id, {"type": "error",
                                                  "message": str(e)})

    asyncio.create_task(runner())
    return {"ok": True, "dissection_id": diss_id, "status": "pool_building"}


@app.get("/api/pool/stats")
async def pool_stats():
    """Aggregate stats across the Korean visual-match pool."""
    db_stats = db.pool_stats()
    qd_stats = kr_qdrant.stats()
    return {**db_stats, "qdrant": qd_stats}


@app.get("/api/pool/channels/{diss_id}")
async def pool_channels(diss_id: str, only_reference: bool = True):
    """List classified KR channels for a dissection."""
    if only_reference:
        return db.list_pool_reference_channels(diss_id)
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM korean_pool_channels WHERE dissection_id=? "
            "ORDER BY matching_count DESC",
            (diss_id,),
        ).fetchall()
        return [dict(r) for r in rows]


class VisualMatchRequest(BaseModel):
    url: str = Field(..., description="후보 영상 URL")
    max_frames: int = 40


@app.post("/api/pool/match")
async def visual_match_one(req: VisualMatchRequest):
    """Run visual match for a single candidate URL — debug/inspection."""
    return await kr_visual.match_candidate(req.url, max_frames=req.max_frames)


@app.post("/api/candidates/{candidate_id}/visual-match")
async def candidate_visual_match(candidate_id: int,
                                 current=Depends(auth.authenticate)):
    """Run CLIP visual match for one candidate against the Korean pool.
    Updates the candidate row + broadcasts the new verdict over WebSocket.
    """
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (candidate_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    cand = dict(row)

    try:
        res = await kr_visual.match_candidate(cand["url"])
    except Exception as e:
        raise HTTPException(500, f"visual match failed: {e}")

    updates = {
        "visual_match_verdict": res.get("verdict"),
        "visual_match_score": res.get("score"),
        "visual_match_video_id": res.get("best_match_video"),
        "visual_match_url": res.get("matched_url"),
        "visual_match_channel": res.get("matched_channel"),
    }
    db.update_candidate(candidate_id, **updates)

    await ws_manager.broadcast(f"candidates:{cand['job_id']}", {
        "type": "candidate_updated",
        "candidate_id": candidate_id,
        "updates": updates,
        "by": current.get("full_name") or current.get("username"),
    })
    return res


# ===== Phase 2: Delete dissection / job (admin) =====

class DissectionRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


@app.patch("/api/dissect/{diss_id}")
async def rename_dissection_route(diss_id: str, req: DissectionRenameRequest,
                                    current=Depends(auth.admin_only)):
    """작업(dissection) 이름 변경. 형님이 카드에서 inline edit."""
    diss = db.get_dissection(diss_id)
    if not diss:
        raise HTTPException(404, "dissection not found")
    new_name = req.name.strip()
    db.update_dissection(diss_id, name=new_name)
    # Sync linked discovery_jobs.name (freelancer 모달 등에서 보이게)
    job_id = diss.get("related_job_id")
    if job_id:
        try:
            with db.get_db() as conn:
                conn.execute("UPDATE discovery_jobs SET name=? WHERE id=?", (new_name, job_id))
        except Exception:
            pass
    return {"ok": True, "id": diss_id, "name": new_name}


class ManualPoolRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    channels: list[str] = Field(..., min_length=1, max_length=50)
    min_views: int = Field(default=5_000_000, ge=0)
    max_duration: int = Field(default=55, ge=10, le=180)
    skip_channel_ids: list[str] = Field(default_factory=list)


_active_pools: dict[str, asyncio.Task] = {}


@app.post("/api/dissect/manual-pool")
async def create_manual_pool(req: ManualPoolRequest,
                              current=Depends(auth.admin_only)):
    """수동 작업: 채널 list만 받아서 우리 조건 (5M+, 55s, 랭킹/컴필 제외)으로 발굴.
    background로 yt-dlp + filter + DB import 진행. ws progress.
    """
    import hashlib
    import time as _t
    h = hashlib.md5(f"{req.name}_{int(_t.time())}".encode()).hexdigest()
    diss_id = "diss_" + h[:12]
    job_id = "job_" + h[12:24]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. dissection + job 생성
    with db.get_db() as conn:
        conn.execute("""
            INSERT INTO discovery_jobs (id, name, status, platforms, created_at)
            VALUES (?, ?, 'running', '["youtube"]', ?)
        """, (job_id, req.name, now))
        conn.execute("""
            INSERT INTO dissection_analyses (id, name, status, related_job_id,
                progress, progress_message, created_at)
            VALUES (?, ?, 'pool_building', ?, 0, ?, ?)
        """, (diss_id, req.name, job_id, "🚀 수동 풀 시작", now))

    # 2. background discover
    async def progress_cb(pct: int, msg: str):
        db.update_dissection(diss_id, progress=pct, progress_message=msg)
        await ws_manager.broadcast(f"dissect:{diss_id}", {
            "type": "progress", "progress": pct, "message": msg,
        })

    async def runner():
        from workers.channel_pool import discover_pool
        try:
            result = await discover_pool(
                db_path=str(db.DB_PATH),
                job_id=job_id,
                channels=req.channels,
                min_views=req.min_views,
                max_duration=req.max_duration,
                skip_channel_ids=set(req.skip_channel_ids),
                progress_cb=progress_cb,
            )
            db.update_dissection(diss_id, status="ready", progress=100,
                                 progress_message=f"✅ 총 {result['final_total']}개")
            with db.get_db() as conn:
                conn.execute("UPDATE discovery_jobs SET status='completed', completed_at=? WHERE id=?",
                             (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), job_id))
            await ws_manager.broadcast(f"dissect:{diss_id}", {
                "type": "completed", "result": result,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            db.update_dissection(diss_id, status="failed", error=str(e))
            await ws_manager.broadcast(f"dissect:{diss_id}", {
                "type": "error", "message": str(e),
            })

    task = asyncio.create_task(runner())
    _active_pools[diss_id] = task

    def _cleanup(t):
        if _active_pools.get(diss_id) is t:
            _active_pools.pop(diss_id, None)
    task.add_done_callback(_cleanup)
    return {"ok": True, "diss_id": diss_id, "job_id": job_id, "status": "pool_building"}


@app.delete("/api/dissect/{diss_id}")
async def delete_dissection_route(diss_id: str,
                                  current=Depends(auth.admin_only)):
    """Cascade delete: dissection + related job + candidates + remixes + pool rows.
    형님이 작업 카드에서 ✕ 누를 때 호출.
    """
    n = db.delete_dissection(diss_id)
    if n == 0:
        raise HTTPException(404, "dissection not found")
    return {"ok": True, "deleted": n}


@app.delete("/api/jobs/{job_id}")
async def delete_job_route(job_id: str, current=Depends(auth.admin_only)):
    """Cascade delete: job + candidates + remixes + assignments."""
    n = db.delete_job(job_id)
    if n == 0:
        raise HTTPException(404, "job not found")
    return {"ok": True, "deleted": n}


# ===== Phase 5: Channel auto-collect =====

class ChannelAutoCollectRequest(BaseModel):
    channel_url: str = Field(..., description="YouTube 채널 URL or @handle")
    job_id: str = Field(..., description="후보를 추가할 작업 id")
    max_videos: int = Field(default=200, ge=1, le=1000)
    min_views: int = Field(default=0, ge=0, description="최소 조회수")
    max_duration: float = Field(default=0.0, ge=0, description="최대 영상 길이(초), 0=제한 없음")
    auto_memo: bool = Field(default=True, description="Gemini로 한 줄 메모 자동 작성")
    auto_dna_match: bool = Field(default=True, description="카테 결(DNA)과 영상 자동 분류")
    order: str = Field(default="viewCount", description="viewCount=인기순(옛 viral까지), date=최신순")


@app.post("/api/channels/auto-collect")
async def channels_auto_collect(req: ChannelAutoCollectRequest,
                                current=Depends(auth.admin_only)):
    """Phase 5 — 채널 URL → 영상 list → 필터(min_views, max_duration, AI/RC/어린이/편집본 배제)
    → candidate_videos INSERT (중복 SKIP) → auto_memo=True면 Gemini로 한 줄 요약 채움
    + 카테 결(DNA)이 있으면 영상마다 결 매칭 → classification(키핑/review/제외) 자동 채움.
    """
    job = db.get_job(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")

    from workers import youtube_client
    handle = youtube_client.extract_channel_handle(req.channel_url) \
        if hasattr(youtube_client, "extract_channel_handle") else req.channel_url

    EXCLUDED_KEYWORDS_LOWER = [
        "ai animation", "ai video", "ai shorts", "trollface", "troll",
        "rc car", "rc plane", "rc heli",
        "kids", "child", "어린이", "유아", "키즈",
        "edit", "editing", "compilation", "compil",
        "tier list", "ranking", "fails of", "best of",
        # 인도 시그니처 — 형님 룰 인도 제외
        "hindi", "bollywood", "desi", "bhai", "bhaiya", "namaste",
        "हिन्दी", "भारत",
    ]

    async def runner():
        progress_room = f"auto_collect:{req.job_id}"
        try:
            await ws_manager.broadcast(progress_room, {
                "type": "auto_collect_progress",
                "message": f"📡 {handle} 영상 {req.max_videos}개 가져오는 중 ({'인기순' if req.order == 'viewCount' else '최신순'}, min {req.min_views:,} views, " + (f"≤{int(req.max_duration)}s" if req.max_duration > 0 else "길이 무제한") + ")…",
            })
            videos = await youtube_client.get_channel_videos(
                req.channel_url, max_results=req.max_videos, order=req.order
            )
            # 디버그 — 채널 URL + 가져온 영상 view_count 분포
            print(f"[debug-autoCollect] url={req.channel_url} order={req.order} min_views={req.min_views} max_duration={req.max_duration}", flush=True)
            print(f"[debug-autoCollect] fetched={len(videos or [])}개, view_counts top5: " +
                  str([v.get('view_count') for v in (videos or [])[:5]]), flush=True)
            if videos:
                vc_max = max((v.get('view_count') or 0) for v in videos)
                vc_min = min((v.get('view_count') or 0) for v in videos)
                vc_pass = sum(1 for v in videos if (v.get('view_count') or 0) >= req.min_views)
                print(f"[debug-autoCollect] vc range [{vc_min:,} ~ {vc_max:,}], min_views {req.min_views:,} 통과 {vc_pass}개", flush=True)
            inserted_ids: list[int] = []
            skipped_dur = 0
            skipped_views = 0
            skipped_keyword = 0
            skipped_dup = 0
            for v in videos or []:
                dur = v.get("duration") or 0
                if req.max_duration > 0 and dur > req.max_duration:
                    skipped_dur += 1
                    continue
                vc = v.get("view_count") or 0
                if vc < req.min_views:
                    skipped_views += 1
                    continue
                blob = " ".join([
                    str(v.get("title") or ""),
                    str(v.get("caption") or ""),
                    str(v.get("channel_name") or ""),
                ]).lower()
                if any(k in blob for k in EXCLUDED_KEYWORDS_LOWER):
                    skipped_keyword += 1
                    continue
                rid = db.insert_candidate(
                    job_id=req.job_id,
                    platform="youtube",
                    video_id=v["video_id"],
                    url=v["url"],
                    title=v.get("title"),
                    caption=v.get("caption"),
                    channel_name=v.get("channel_name"),
                    channel_id=v.get("channel_id"),
                    view_count=vc,
                    like_count=v.get("like_count"),
                    duration=dur,
                    published_at=v.get("published_at"),
                    thumbnail_url=v.get("thumbnail_url"),
                    classification="pending",
                )
                if rid:
                    inserted_ids.append(rid)
                else:
                    skipped_dup += 1

            await ws_manager.broadcast(progress_room, {
                "type": "auto_collect_progress",
                "message": f"✅ {len(inserted_ids)}건 추가됨"
                           + (f" — Gemini로 메모 자동 작성 시작…" if req.auto_memo and inserted_ids else ""),
            })

            # Gemini 메모 자동 작성 (병렬, semaphore로 동시 5개)
            memo_done = 0
            if req.auto_memo and inserted_ids:
                from workers import llm
                sem = asyncio.Semaphore(5)
                async def fill_memo(cid: int):
                    nonlocal memo_done
                    async with sem:
                        try:
                            with db.get_db() as conn:
                                row = conn.execute(
                                    "SELECT url, title, caption, channel_name, duration "
                                    "FROM candidate_videos WHERE id=?", (cid,)
                                ).fetchone()
                            if not row:
                                return
                            cand = dict(row)
                            memo = await _gen_video_memo(cand)
                            if memo:
                                db.update_candidate(cid, memo_kr=memo)
                                memo_done += 1
                                await ws_manager.broadcast(f"candidates:{req.job_id}", {
                                    "type": "candidate_updated",
                                    "candidate_id": cid,
                                    "updates": {"memo_kr": memo},
                                    "by": "Gemini auto-memo",
                                })
                        except Exception as e:
                            print(f"⚠️ memo fail {cid}: {e}")
                await asyncio.gather(*[fill_memo(c) for c in inserted_ids])

            # 카테 결(DNA) 자동 분류 — DNA 있고 옵션 켜져있을 때만
            dna_done = 0
            dna_keep = 0
            dna_review = 0
            dna_exclude = 0
            if req.auto_dna_match and inserted_ids:
                dna_summary = _extract_dna_summary(req.job_id)
                if dna_summary:
                    await ws_manager.broadcast(progress_room, {
                        "type": "auto_collect_progress",
                        "message": f"🧬 결 자동 분류 시작 — {len(inserted_ids)}개 영상 vs 카테 결...",
                    })
                    sem_dna = asyncio.Semaphore(5)
                    async def classify_one(cid: int):
                        nonlocal dna_done, dna_keep, dna_review, dna_exclude
                        async with sem_dna:
                            try:
                                with db.get_db() as conn:
                                    row = conn.execute(
                                        "SELECT title, caption, channel_name, duration "
                                        "FROM candidate_videos WHERE id=?", (cid,)
                                    ).fetchone()
                                if not row:
                                    return
                                cand = dict(row)
                                verdict, reason = await _dna_classify(cand, dna_summary)
                                note = f"[결 {verdict}] {reason}"
                                with db.get_db() as conn:
                                    conn.execute(
                                        "UPDATE candidate_videos SET classification=?, "
                                        "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                                        (verdict, note, cid),
                                    )
                                    conn.commit()
                                dna_done += 1
                                if verdict == "키핑":
                                    dna_keep += 1
                                elif verdict == "제외":
                                    dna_exclude += 1
                                else:
                                    dna_review += 1
                                await ws_manager.broadcast(f"candidates:{req.job_id}", {
                                    "type": "candidate_updated",
                                    "candidate_id": cid,
                                    "updates": {"classification": verdict},
                                    "by": "Gemini 결 매칭",
                                })
                            except Exception as e:
                                print(f"⚠️ dna {cid}: {e}")
                    await asyncio.gather(*[classify_one(c) for c in inserted_ids])
                else:
                    await ws_manager.broadcast(progress_room, {
                        "type": "auto_collect_progress",
                        "message": "⚠️ 카테 결(DNA) 비어있음 — 결 분류 생략 (dissection 분석 먼저 돌려야 함)",
                    })

            await ws_manager.broadcast(progress_room, {
                "type": "auto_collect_completed",
                "inserted": len(inserted_ids),
                "skipped_duration": skipped_dur,
                "skipped_views": skipped_views,
                "skipped_keyword": skipped_keyword,
                "skipped_duplicate": skipped_dup,
                "total_fetched": len(videos or []),
                "memo_done": memo_done,
                "dna_done": dna_done,
                "dna_keep": dna_keep,
                "dna_review": dna_review,
                "dna_exclude": dna_exclude,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws_manager.broadcast(progress_room, {
                "type": "error", "message": str(e),
            })
            from workers.notify import notify_error
            await notify_error(f"채널 추가 실패: {handle}", e)

    task = asyncio.create_task(runner())
    # job_id별 task list 관리 — 중지 버튼에서 cancel 용도
    _active_auto_collect.setdefault(req.job_id, []).append(task)
    def _cleanup(t: asyncio.Task) -> None:
        lst = _active_auto_collect.get(req.job_id, [])
        if t in lst:
            lst.remove(t)
        if not lst:
            _active_auto_collect.pop(req.job_id, None)
    task.add_done_callback(_cleanup)
    return {"ok": True, "job_id": req.job_id, "channel": handle,
            "status": "collecting"}


_active_auto_collect: dict[str, list[asyncio.Task]] = {}


@app.post("/api/jobs/{job_id}/cancel-auto-collect")
async def cancel_auto_collect(job_id: str,
                              current=Depends(auth.admin_only)):
    """진행 중인 채널 추가 작업 전부 중지. job_id 안의 모든 채널 task cancel."""
    tasks = _active_auto_collect.get(job_id, [])
    if not tasks:
        return {"ok": True, "cancelled": 0, "message": "진행 중인 작업 없음"}
    n = 0
    for t in list(tasks):
        if not t.done():
            t.cancel()
            n += 1
    await ws_manager.broadcast(f"auto_collect:{job_id}", {
        "type": "auto_collect_progress",
        "message": f"🛑 사용자가 중지함 — {n}개 작업 취소",
    })
    return {"ok": True, "cancelled": n}


def _extract_dna_summary(job_id: str) -> str | None:
    """job의 카테 결(DNA) 요약 텍스트 추출. dissection_result 비었으면 None."""
    try:
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT dissection_result FROM dissection_analyses "
                "WHERE related_job_id=? AND dissection_result IS NOT NULL "
                "AND length(dissection_result) > 100 LIMIT 1",
                (job_id,),
            ).fetchone()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    try:
        d = json.loads(row[0])
    except Exception:
        return None
    parts: list[str] = []
    common = d.get("common_dna") or {}
    items = common.get("items") or {}
    for k in ("1_narrative_structure", "3_writing_formula",
              "4_view_drivers", "5_viral_reasons", "6_topics_list"):
        it = items.get(k) or {}
        v = it.get("value") or ""
        if v:
            label = it.get("label") or k
            parts.append(f"[{label}] {v[:300]}")
    if not parts:
        chans = d.get("channels") or []
        if chans:
            parts.append((chans[0].get("summary_kr") or "")[:500])
    text = chr(10).join(parts).strip()
    return text if text else None


async def _dna_classify(cand: dict, dna_summary: str) -> tuple[str, str]:
    """영상 1개가 카테 결(DNA)과 맞는지 Gemini로 판단."""
    from workers import llm as _llm
    if not _llm.GEMINI_API_KEY:
        return "review", "GEMINI_API_KEY 없음"
    title = cand.get("title") or ""
    channel = cand.get("channel_name") or ""
    caption = (cand.get("caption") or "")[:600]
    duration = cand.get("duration") or 0
    prompt = chr(10).join([
        "카테 결(DNA 요약):",
        dna_summary,
        "",
        "영상 정보:",
        f"- 제목: {title}",
        f"- 채널: {channel}",
        f"- 자막/설명: {caption}",
        f"- 길이: {duration}초",
        "",
        "이 영상이 위 카테 결과 맞는지 판단해.",
        "- 완전 매칭 = \"키핑\"",
        "- 일부만 매칭 / 애매 = \"review\"",
        "- 결 완전 다름 / 카테와 무관 = \"제외\"",
        "",
        "JSON으로만 답: {\"verdict\": \"키핑\"|\"review\"|\"제외\", \"reason\": \"한 줄 설명\"}",
    ])
    last_err = ""
    for attempt in range(3):
        try:
            resp = await _llm.gemini_chat(
                prompt=prompt, model="gemini-3.5-flash",
                max_tokens=800, json_mode=True,
            )
            text = (resp.text or "").strip()
            if not text:
                last_err = "빈 응답"
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                break
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # JSON 안 깨끗할 때 부분 추출
                m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text)
                if m:
                    try:
                        data = json.loads(m.group())
                    except Exception:
                        data = {}
                else:
                    data = {}
            verdict = (data.get("verdict") or "").strip()
            if verdict not in ("키핑", "review", "제외"):
                if attempt < 2:
                    last_err = f"verdict 없음 ({text[:50]})"
                    await asyncio.sleep(2)
                    continue
                verdict = "review"
            reason = (data.get("reason") or "")[:200]
            return verdict, reason
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
                continue
    return "review", f"[ERROR] {last_err}"


async def _gen_video_memo(cand: dict) -> str | None:
    """Gemini로 영상 짧게 분석 → 한 줄 한국어 요약 (어떤 영상인지).
    YouTube/TikTok/Instagram URL 모두 OK — llm.gemini_video_chat이 yt-dlp+Files API로
    자동 처리.
    """
    from workers import llm as _llm
    if not _llm.GEMINI_API_KEY:
        print("⚠️ memo: GEMINI_API_KEY missing")
        return None
    url = cand.get("url") or ""
    if not url:
        return None
    system = (
        "너는 한국 쇼츠 발굴 전문가야. 영상이 어떤 내용인지 한국어 한 줄(50자 이내)로 요약해. "
        "구체적인 비주얼/상황/사건만 적어. 추측 금지, 본 것만. 이모지 X, 따옴표 X, 줄바꿈 X, "
        "한 줄만 출력."
    )
    prompt = (
        f"제목: {cand.get('title') or ''}\n"
        f"채널: {cand.get('channel_name') or ''}\n"
        f"길이: {cand.get('duration') or 0}초\n\n"
        "이 영상의 핵심 비주얼/사건을 한 줄(50자 이내)로 요약."
    )
    try:
        resp = await _llm.gemini_video_chat(
            youtube_url=url, prompt=prompt, system=system,
            model="gemini-3.5-flash", max_tokens=200,
        )
    except Exception as e:
        print(f"⚠️ memo gemini fail ({url[:50]}): {e}")
        return None
    txt = (resp.text or "").strip()
    txt = txt.replace("\n", " ").replace('"', "").strip()
    return txt[:120] if txt else None


class ChannelDiscoverRequest(BaseModel):
    keyword: str = Field(..., description="발굴 시드 키워드. 파생 키워드 자동 생성")
    job_id: str = Field(..., description="중복 검사용 (이 카테에 이미 있는 채널 제외)")
    max_channels: int = Field(default=30, ge=5, le=100)
    generate_related: bool = Field(default=True, description="Gemini로 파생 키워드 12개 자동 생성")


@app.post("/api/channels/discover-by-keyword")
async def channels_discover_by_keyword(req: ChannelDiscoverRequest,
                                       current=Depends(auth.admin_only)):
    """키워드 → 파생 키워드 자동 생성 → YouTube search → 채널별 ranking → 신규 채널 list 반환.
    형님이 카테 후보풀에 추가할 채널 발굴 용도. 이미 있는 채널/거부 채널 자동 제외.
    """
    from workers import youtube_client, llm

    # 1. 파생 키워드 생성 (Gemini Flash)
    keywords = [req.keyword.strip()]
    if req.generate_related:
        try:
            prompt = (
                f"'{req.keyword}'와 관련된 YouTube 쇼츠 검색 키워드 12개 추천. "
                f"영어 6개 + 한국어 6개. 콤마로 구분만 해서 답해. 다른 설명 X.\n"
                f"형식: keyword1, keyword2, keyword3, ..."
            )
            resp = await llm.gemini_chat(prompt=prompt, model="gemini-3.5-flash",
                                          max_tokens=500)
            text = (resp.text or "").strip()
            related = [k.strip() for k in text.split(",") if k.strip() and len(k.strip()) < 60]
            keywords = list(dict.fromkeys([req.keyword.strip()] + related))[:13]
        except Exception as e:
            print(f"[discover] 파생 키워드 실패: {e}", flush=True)

    # 2. 각 키워드 YouTube search (병렬 5개)
    sem = asyncio.Semaphore(5)
    async def search_one(kw):
        async with sem:
            try:
                return await youtube_client.search_youtube(kw, max_results=20, region="US")
            except Exception:
                return []
    results = await asyncio.gather(*[search_one(kw) for kw in keywords])

    # 3. 채널별 group
    channel_stats: dict[str, dict] = {}
    for kw, videos in zip(keywords, results):
        for v in videos:
            cid = v.get("channel_id")
            if not cid:
                continue
            if cid not in channel_stats:
                channel_stats[cid] = {
                    "channel_id": cid,
                    "channel_name": v.get("channel_name") or "",
                    "video_count": 0,
                    "total_views": 0,
                    "sample_videos": [],
                    "matched_keywords": [],
                }
            stat = channel_stats[cid]
            stat["video_count"] += 1
            stat["total_views"] += v.get("view_count") or 0
            if kw not in stat["matched_keywords"]:
                stat["matched_keywords"].append(kw)
            if len(stat["sample_videos"]) < 3:
                stat["sample_videos"].append({
                    "video_id": v.get("video_id"),
                    "title": (v.get("title") or "")[:80],
                    "view_count": v.get("view_count") or 0,
                    "duration": v.get("duration") or 0,
                    "thumbnail_url": v.get("thumbnail_url"),
                })

    # 4. 이미 후보풀 있는 채널 + 거부 채널 제외
    existing_channels: set[str] = set()
    rejected: set[str] = set()
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT channel_id FROM candidate_videos "
            "WHERE job_id=? AND channel_id IS NOT NULL AND channel_id != ''",
            (req.job_id,)
        ).fetchall()
        existing_channels = {r[0] for r in rows if r[0]}
        rej_rows = conn.execute("SELECT channel_id FROM rejected_channels").fetchall()
        rejected = {r[0] for r in rej_rows if r[0]}

    # 5. 신규 채널만, video_count + total_views 내림차순
    new_channels = []
    for cid, stat in channel_stats.items():
        if cid in existing_channels or cid in rejected:
            continue
        new_channels.append(stat)
    new_channels.sort(key=lambda x: (-x["video_count"], -x["total_views"]))
    new_channels = new_channels[:req.max_channels]

    # 6. 각 채널 메타 보강 (병렬)
    async def fetch_meta(stat):
        try:
            meta = await youtube_client.get_channel_meta(stat["channel_id"])
            if meta:
                stat["subscriber_count"] = int(meta.get("statistics", {}).get("subscriberCount", 0))
                stat["total_video_count"] = int(meta.get("statistics", {}).get("videoCount", 0))
                stat["country"] = meta.get("snippet", {}).get("country", "")
                handle = meta.get("snippet", {}).get("customUrl", "")
                stat["channel_url"] = (f"https://www.youtube.com/{handle}"
                                        if handle else f"https://www.youtube.com/channel/{stat['channel_id']}")
            else:
                stat.setdefault("subscriber_count", 0)
                stat.setdefault("total_video_count", 0)
                stat.setdefault("country", "")
                stat["channel_url"] = f"https://www.youtube.com/channel/{stat['channel_id']}"
        except Exception:
            stat.setdefault("subscriber_count", 0)
            stat.setdefault("total_video_count", 0)
            stat.setdefault("country", "")
            stat["channel_url"] = f"https://www.youtube.com/channel/{stat['channel_id']}"
        return stat

    new_channels = list(await asyncio.gather(*[fetch_meta(c) for c in new_channels]))

    return {
        "ok": True,
        "keywords_used": keywords,
        "channels": new_channels,
        "total_found": len(channel_stats),
        "new": len(new_channels),
        "excluded_existing": len(existing_channels & set(channel_stats.keys())),
        "excluded_rejected": len(rejected & set(channel_stats.keys())),
    }


class DnaSearchRequest(BaseModel):
    job_id: str
    per_keyword: int = Field(default=30, ge=5, le=100)
    min_views: int = Field(default=100000, ge=0)
    max_duration: int = Field(default=90, ge=10, le=600)
    skip_pro: bool = Field(default=False, description="True면 Pro 검증 skip (메타만)")


@app.post("/api/jobs/{job_id}/dna-search")
async def jobs_dna_search(job_id: str, req: DnaSearchRequest,
                           current=Depends(auth.admin_only)):
    """Phase 2 핵심 — 카테 키워드로 YouTube 검색 + reference 결 매칭 + 통과만 후보풀.

    흐름:
    1. 카테의 keywords_result + DNA 가져옴
    2. 키워드별 YouTube 검색 → 영상 list (대량)
    3. 메타 1차 (시그니처/거부/조회수/길이)
    4. Pro 2차 (의심만, 결 검증)
    5. 통과한 영상 → 후보풀 insert
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    # 카테 + DNA + 키워드
    with db.get_db() as conn:
        diss_row = conn.execute(
            "SELECT id, name, dissection_result, keywords_result "
            "FROM dissection_analyses WHERE related_job_id=? LIMIT 1",
            (job_id,),
        ).fetchone()
    if not diss_row:
        raise HTTPException(400, "이 카테에 dissection 없음 — 분석 먼저")
    diss = dict(diss_row)
    cate_name = diss.get("name") or ""

    # DNA 요약
    dna_summary = f"카테 = {cate_name}"
    try:
        d = json.loads(diss.get("dissection_result") or "{}")
        common = d.get("common_dna") or {}
        items = common.get("items") or {}
        for k in ("4_view_drivers", "6_topics_list"):
            it = items.get(k) or {}
            v = it.get("value") or ""
            if v:
                dna_summary += "\n" + (it.get("label", k) or k) + ": " + v[:200]
        if not items:
            chans = d.get("channels") or []
            if chans:
                dna_summary += "\n" + (chans[0].get("summary_kr") or "")[:300]
    except Exception:
        pass

    # 키워드 list — keywords_result.english 우선
    keywords = []
    try:
        kw = json.loads(diss.get("keywords_result") or "{}")
        for lang in ("english", "chinese", "japanese"):
            for item in kw.get(lang, []):
                k = item.get("keyword") if isinstance(item, dict) else str(item)
                if k and k not in keywords:
                    keywords.append(k)
    except Exception:
        pass
    if not keywords:
        raise HTTPException(400, "키워드 없음 — 분석 다시")

    from workers import video_dna_filter as _vdf
    signature_words = _vdf.get_cate_signature_words(cate_name)

    async def runner():
        room = f"dna_search:{job_id}"
        try:
            async def progress_cb(pct, msg):
                await ws_manager.broadcast(room, {
                    "type": "dna_search_progress", "progress": pct, "message": msg,
                })

            result = await _vdf.filter_search_videos(
                keywords=keywords,
                cate_name=cate_name,
                dna_summary=dna_summary,
                signature_words=signature_words,
                per_keyword=req.per_keyword,
                min_views=req.min_views,
                max_duration=req.max_duration,
                skip_pro=req.skip_pro,
                progress_cb=progress_cb,
            )

            if not result.get("ok"):
                await ws_manager.broadcast(room, {
                    "type": "error", "message": result.get("error", "fail"),
                })
                return

            # 통과 영상 후보풀에 insert
            inserted = 0
            skipped_dup = 0
            for v in result.get("passed", []):
                note_parts = []
                if v.get("meta_verdict") == "pass":
                    note_parts.append("[메타통과] " + (v.get("meta_reason") or ""))
                if v.get("pro_verdict") == "yes":
                    note_parts.append("[결" + str(v.get("pro_confidence", 0)) + "/통과] " + (v.get("pro_reason") or ""))
                memo = " / ".join(note_parts)[:300]

                rid = db.insert_candidate(
                    job_id=job_id,
                    platform="youtube",
                    video_id=v["video_id"],
                    url="https://youtu.be/" + v["video_id"],
                    title=v.get("title"),
                    caption=None,
                    channel_name=v.get("channel_name") or "",
                    channel_id=v.get("channel_id") or "",
                    view_count=int(v.get("view_count") or 0),
                    like_count=None,
                    duration=float(v.get("duration") or 0),
                    published_at=None,
                    thumbnail_url=None,
                    classification="키핑",
                )
                if rid:
                    inserted += 1
                    if memo:
                        with db.get_db() as conn:
                            conn.execute(
                                "UPDATE candidate_videos SET memo_kr=? WHERE id=?",
                                (memo, rid),
                            )
                            conn.commit()
                else:
                    skipped_dup += 1

            await ws_manager.broadcast(room, {
                "type": "completed",
                "total_searched": result.get("total_searched"),
                "passed_count": len(result.get("passed", [])),
                "inserted": inserted,
                "skipped_dup": skipped_dup,
                "excluded_count": result.get("excluded_count", 0),
                "suspect_pro_checked": result.get("suspect_pro_checked", 0),
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws_manager.broadcast(room, {"type": "error", "message": str(e)})

    asyncio.create_task(runner())
    return {"ok": True, "started": True, "diss_id": diss.get("id"),
            "keywords_count": len(keywords), "signature_words": signature_words}


@app.websocket("/ws/dna_search/{job_id}")
async def ws_dna_search(ws: WebSocket, job_id: str, token: str | None = None):
    await ws_manager.connect("dna_search:" + job_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("dna_search:" + job_id, ws)


class DnaCollectRequest(BaseModel):
    channel_url: str
    job_id: str
    min_views: int = Field(default=100000, ge=0)
    max_duration: int = Field(default=90, ge=10, le=600)
    skip_pro: bool = Field(default=False, description="True면 Pro 검증 skip (메타만)")


@app.post("/api/channels/dna-collect")
async def channels_dna_collect(req: DnaCollectRequest,
                                current=Depends(auth.admin_only)):
    """Phase 2 — 채널 모든 영상 → 메타 + Pro 결 검증 → 통과한 영상만 후보풀 추가.

    기존 autoCollect와 차이:
    - limit 없음 (채널 전체 영상)
    - 카테 결 검증 (메타 + Pro)
    - 통과한 영상만 후보풀에
    """
    job = db.get_job(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")

    # 카테 정보 + DNA + 시그니처 키워드 가져옴
    with db.get_db() as conn:
        diss_row = conn.execute(
            "SELECT id, name, dissection_result FROM dissection_analyses "
            "WHERE related_job_id=? LIMIT 1",
            (req.job_id,),
        ).fetchone()
    if not diss_row:
        raise HTTPException(400, "이 카테에 dissection 없음 — 분석 먼저 진행")
    diss = dict(diss_row)
    cate_name = diss.get("name") or ""

    # DNA 요약
    dna_summary = f"카테 = {cate_name}"
    try:
        d = json.loads(diss.get("dissection_result") or "{}")
        common = d.get("common_dna") or {}
        items = common.get("items") or {}
        for k in ("4_view_drivers", "6_topics_list"):
            it = items.get(k) or {}
            v = it.get("value") or ""
            if v:
                dna_summary += f"\n{it.get('label', k)}: {v[:200]}"
        if not items:
            chans = d.get("channels") or []
            if chans:
                dna_summary += "\n" + (chans[0].get("summary_kr") or "")[:300]
    except Exception:
        pass

    # 시그니처 키워드 추출
    from workers import video_dna_filter as _vdf
    signature_words = _vdf.get_cate_signature_words(cate_name)

    async def runner():
        room = f"dna_collect:{req.job_id}"
        try:
            async def progress_cb(pct, msg):
                await ws_manager.broadcast(room, {
                    "type": "dna_collect_progress", "progress": pct, "message": msg,
                })

            result = await _vdf.filter_channel_videos(
                channel_url=req.channel_url,
                cate_name=cate_name,
                dna_summary=dna_summary,
                signature_words=signature_words,
                min_views=req.min_views,
                max_duration=req.max_duration,
                skip_pro=req.skip_pro,
                progress_cb=progress_cb,
            )

            if not result.get("ok"):
                await ws_manager.broadcast(room, {
                    "type": "error", "message": result.get("error", "fail"),
                })
                return

            # 통과한 영상 → 후보풀 추가
            inserted = 0
            for v in result.get("passed", []):
                # 영상 메타 더 가져옴 (yt-dlp single)
                try:
                    pmeta = await asyncio.create_subprocess_exec(
                        _bb_ytdlp(),
                        "--skip-download", "--print",
                        "%(view_count)s|%(duration)s|%(channel)s|%(channel_id)s|%(thumbnail)s|%(upload_date)s|%(description)s",
                        f"https://www.youtube.com/watch?v={v['video_id']}",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    )
                    out, _ = await asyncio.wait_for(pmeta.communicate(), timeout=30)
                    parts = out.decode().strip().split("|", 6)
                except Exception:
                    parts = []
                vc = int(parts[0]) if len(parts) > 0 and parts[0] not in ("NA", "") else v.get("view_count", 0)
                dur = float(parts[1]) if len(parts) > 1 and parts[1] not in ("NA", "") else v.get("duration", 0)
                ch_name = parts[2] if len(parts) > 2 and parts[2] != "NA" else ""
                ch_id = parts[3] if len(parts) > 3 and parts[3] != "NA" else ""
                thumb = parts[4] if len(parts) > 4 and parts[4] != "NA" else None
                pub = parts[5] if len(parts) > 5 and parts[5] != "NA" else None
                desc = parts[6] if len(parts) > 6 else ""

                note_parts = []
                if v.get("meta_verdict") == "pass":
                    note_parts.append(f"[메타통과] {v.get('meta_reason', '')}")
                if v.get("pro_verdict") == "yes":
                    note_parts.append(f"[결{v.get('pro_confidence', 0)}/통과] {v.get('pro_reason', '')}")
                memo = " / ".join(note_parts)[:300]

                rid = db.insert_candidate(
                    job_id=req.job_id,
                    platform="youtube",
                    video_id=v["video_id"],
                    url=f"https://youtu.be/{v['video_id']}",
                    title=v.get("title"),
                    caption=desc[:500] if desc else None,
                    channel_name=ch_name,
                    channel_id=ch_id,
                    view_count=vc,
                    like_count=None,
                    duration=dur,
                    published_at=pub,
                    thumbnail_url=thumb,
                    classification="키핑",
                )
                if rid:
                    inserted += 1
                    if memo:
                        with db.get_db() as conn:
                            conn.execute(
                                "UPDATE candidate_videos SET memo_kr=? WHERE id=?",
                                (memo, rid),
                            )
                            conn.commit()

            await ws_manager.broadcast(room, {
                "type": "completed",
                "total_videos": result.get("total_videos"),
                "passed": len(result.get("passed", [])),
                "excluded": len(result.get("excluded", [])),
                "inserted": inserted,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            await ws_manager.broadcast(room, {"type": "error", "message": str(e)})

    asyncio.create_task(runner())
    return {"ok": True, "started": True, "diss_id": diss.get("id"),
            "signature_words": signature_words}


@app.websocket("/ws/dna_collect/{job_id}")
async def ws_dna_collect(ws: WebSocket, job_id: str, token: str | None = None):
    await ws_manager.connect(f"dna_collect:{job_id}", ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(f"dna_collect:{job_id}", ws)


@app.post("/api/candidates/{candidate_id}/auto-memo")
async def candidate_auto_memo(candidate_id: int,
                               current=Depends(auth.admin_only)):
    """단일 후보에 대해 Gemini 메모 즉시 생성 (백필용)."""
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM candidate_videos WHERE id=?", (candidate_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "candidate not found")
    cand = dict(row)
    memo = await _gen_video_memo(cand)
    if memo:
        db.update_candidate(candidate_id, memo_kr=memo)
        await ws_manager.broadcast(f"candidates:{cand['job_id']}", {
            "type": "candidate_updated",
            "candidate_id": candidate_id,
            "updates": {"memo_kr": memo},
            "by": "Gemini auto-memo",
        })
    return {"ok": True, "memo_kr": memo}


@app.post("/api/jobs/{job_id}/backfill-memos")
async def backfill_memos(job_id: str, limit: int = 200,
                         current=Depends(auth.admin_only)):
    """job 내 memo_kr 비어있는 후보들에 Gemini 메모 일괄 백필."""
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id FROM candidate_videos WHERE job_id=? "
            "AND (memo_kr IS NULL OR memo_kr='') ORDER BY view_count DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    cand_ids = [r[0] for r in rows]
    if not cand_ids:
        return {"ok": True, "backfilled": 0, "remaining": 0}

    sem = asyncio.Semaphore(5)
    done = 0
    async def fill(cid: int):
        nonlocal done
        async with sem:
            try:
                with db.get_db() as conn:
                    row = conn.execute(
                        "SELECT url, title, caption, channel_name, duration "
                        "FROM candidate_videos WHERE id=?", (cid,)
                    ).fetchone()
                if not row: return
                memo = await _gen_video_memo(dict(row))
                if memo:
                    db.update_candidate(cid, memo_kr=memo)
                    done += 1
                    await ws_manager.broadcast(f"candidates:{job_id}", {
                        "type": "candidate_updated",
                        "candidate_id": cid,
                        "updates": {"memo_kr": memo},
                        "by": "Gemini backfill",
                    })
            except Exception as e:
                print(f"⚠️ backfill {cid}: {e}")

    async def _backfill_runner():
        await asyncio.gather(*[fill(c) for c in cand_ids])
    asyncio.create_task(_backfill_runner())
    return {"ok": True, "queued": len(cand_ids), "running_in_background": True}


@app.websocket("/ws/auto-collect/{job_id}")
async def ws_auto_collect(ws: WebSocket, job_id: str,
                          token: str | None = None):
    if token:
        try:
            auth.decode_token(token)
        except Exception:
            await ws.close(code=4401); return
    else:
        await ws.close(code=4401); return
    room = f"auto_collect:{job_id}"
    await ws_manager.connect(room, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(room, ws)


# ===== Phase 10: Freelancer report (admin only) =====

@app.get("/api/admin/freelancer-report")
async def freelancer_report(current=Depends(auth.admin_only)):
    """user별 작업 수 / 사용함 수 / 마지막 활동 시각."""
    return {"users": db.freelancer_report()}


@app.get("/api/admin/freelancer-report/{user_id}")
async def freelancer_report_detail(user_id: int,
                                    current=Depends(auth.admin_only)):
    """drill-down: 그 사용자가 한 작업 + 사용한 영상들 상세."""
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "user not found")
    with db.get_db() as conn:
        # Assigned jobs (admin은 전체)
        if user.get("role") == "admin":
            jobs = [dict(r) for r in conn.execute(
                "SELECT id, name, status, created_at FROM discovery_jobs ORDER BY created_at DESC"
            ).fetchall()]
        else:
            jobs = [dict(r) for r in conn.execute(
                """SELECT j.id, j.name, j.status, j.created_at, ja.assigned_at
                   FROM discovery_jobs j
                   JOIN job_assignments ja ON ja.job_id = j.id
                   WHERE ja.user_id=?
                   ORDER BY ja.assigned_at DESC""",
                (user_id,),
            ).fetchall()]
        # Used candidates
        used = [dict(r) for r in conn.execute(
            """SELECT id, job_id, platform, video_id, url, title, channel_name,
                      view_count, duration, used_at, memo_kr, thumbnail_url
               FROM candidate_videos
               WHERE used=1 AND used_by_user_id=?
               ORDER BY used_at DESC""",
            (user_id,),
        ).fetchall()]
        # Per-job candidate count for assigned jobs
        per_job_counts = {}
        if jobs:
            ids_sql = ",".join("?" * len(jobs))
            for r in conn.execute(
                f"SELECT job_id, COUNT(*) as n, "
                f"SUM(CASE WHEN used=1 AND used_by_user_id=? THEN 1 ELSE 0 END) as used_n "
                f"FROM candidate_videos WHERE job_id IN ({ids_sql}) GROUP BY job_id",
                [user_id] + [j["id"] for j in jobs],
            ).fetchall():
                per_job_counts[r["job_id"]] = {
                    "candidate_count": r["n"],
                    "used_count": r["used_n"] or 0,
                }
        for j in jobs:
            stats = per_job_counts.get(j["id"], {})
            j["candidate_count"] = stats.get("candidate_count", 0)
            j["used_count"] = stats.get("used_count", 0)
    return {
        "user": {k: user.get(k) for k in
                 ("id", "username", "full_name", "role", "created_at",
                  "last_login_at")},
        "jobs": jobs,
        "used_candidates": used,
    }


# ============================================================
# 자막 자동 생성 (외부 영상 업로드 → Gemini 5중 검증 → srt + 제목)
# ============================================================
from fastapi import UploadFile, File, Form

SUBTITLES_DIR = Path(__file__).parent.parent / "data" / "subtitles"
SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/subtitle/upload")
async def subtitle_upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    original_urls: str = Form(default=""),  # \n 또는 , 구분
    style: str = Form(default="shorts"),  # shorts / emotion
    song_title: str = Form(default=""),  # 감성 스타일 — 사용자 입력 노래 제목
    current=Depends(auth.require_feature("subtitle")),
):
    """외부 영상 업로드 → 5중 검증 + 자막 자동 생성 시작."""
    # URL parse
    urls = []
    if original_urls.strip():
        for u in original_urls.replace(",", "\n").split("\n"):
            u = u.strip()
            if u and (u.startswith("http://") or u.startswith("https://")):
                urls.append(u)

    # 영상 파일 저장
    safe_name = Path(video.filename or "upload.mp4").name
    job_id = db.insert_subtitle_job(
        video_filename=safe_name, video_path="",
        original_urls=urls, user_id=current.get("id"),
    )
    save_path = SUBTITLES_DIR / f"job_{job_id}" / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = await video.read()
    save_path.write_bytes(data)
    db.update_subtitle_job(job_id, video_path=str(save_path),
                            style=style, progress_message="업로드 받음")

    # 백그라운드에서 5중 분석 + srt 생성 (style + 노래제목 적용)
    from workers.auto_subtitle import run_auto_subtitle
    background_tasks.add_task(
        run_auto_subtitle, job_id, save_path, urls, None, style, song_title,
    )
    return {"job_id": job_id, "status": "pending",
            "message": "분석 시작. /api/subtitle/{job_id}/status로 진행률 확인"}


SUBTITLE_CHUNK_DIR = Path(__file__).parent.parent / "data" / "subtitle_chunks"


@app.post("/api/subtitle/upload-chunk")
async def subtitle_upload_chunk(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    chunk: UploadFile = File(...),
    original_urls: str = Form(default=""),
    style: str = Form(default="shorts"),
    song_title: str = Form(default=""),
    current=Depends(auth.require_feature("subtitle")),
):
    """대용량 영상 청크 업로드 (Cloudflare 100MB 요청 제한 회피).
    프론트가 80MB씩 잘라 순차 전송 → 서버가 이어붙임 → 마지막 청크에서 잡 생성."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", upload_id)[:64]
    if not safe_id:
        raise HTTPException(400, "bad upload_id")
    SUBTITLE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    part = SUBTITLE_CHUNK_DIR / f"{safe_id}.part"
    data = await chunk.read()
    with open(part, "wb" if chunk_index == 0 else "ab") as f:
        f.write(data)
    if chunk_index + 1 < total_chunks:
        return {"ok": True, "received": chunk_index}

    # 마지막 청크 → 합본을 잡 폴더로 옮기고 분석 시작
    urls = []
    if original_urls.strip():
        for u in original_urls.replace(",", "\n").split("\n"):
            u = u.strip()
            if u and (u.startswith("http://") or u.startswith("https://")):
                urls.append(u)
    safe_name = Path(filename or "upload.mp4").name
    job_id = db.insert_subtitle_job(
        video_filename=safe_name, video_path="",
        original_urls=urls, user_id=current.get("id"),
    )
    save_path = SUBTITLES_DIR / f"job_{job_id}" / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        part.replace(save_path)
    except Exception:
        import shutil
        shutil.move(str(part), str(save_path))
    db.update_subtitle_job(job_id, video_path=str(save_path),
                            style=style, progress_message="업로드 받음(청크)")
    from workers.auto_subtitle import run_auto_subtitle
    background_tasks.add_task(run_auto_subtitle, job_id, save_path, urls, None, style, song_title)
    return {"job_id": job_id, "status": "pending",
            "message": "분석 시작 (청크 업로드 완료)"}


@app.get("/api/subtitle/{job_id}/status")
async def subtitle_status(job_id: int,
                            current=Depends(auth.require_feature("subtitle"))):
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "progress_message": job.get("progress_message", ""),
        "needs_review": bool(job.get("needs_review", 0)),
        "error": job.get("error"),
    }


@app.get("/api/subtitle/{job_id}/result")
async def subtitle_result(job_id: int,
                            current=Depends(auth.require_feature("subtitle"))):
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def _parse_json(s):
        if not s:
            return None
        try:
            return json.loads(s) if isinstance(s, str) else s
        except Exception:
            return None

    subtitle_paths = _parse_json(job.get("subtitle_paths")) or {}
    title_candidates = _parse_json(job.get("title_candidates")) or []
    gemini_results = _parse_json(job.get("gemini_results")) or {}
    validation = _parse_json(job.get("cross_validation")) or {}

    # srt 파일을 web 다운로드 URL로 변환
    srt_urls = {}
    for kind, path in subtitle_paths.items():
        if path and Path(path).exists():
            filename = Path(path).name
            srt_urls[kind] = f"/api/subtitle/{job_id}/download/{filename}"

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "video_filename": job.get("video_filename"),
        "duration_sec": job.get("duration_sec"),
        "subtitle_urls": srt_urls,
        "title_candidates": title_candidates,
        "primary_analysis": gemini_results.get("primary", {}),
        "cross_validation": validation,
        "needs_review": bool(job.get("needs_review", 0)),
        "review_note": job.get("review_note"),
    }


@app.get("/api/subtitle/{job_id}/download/{filename}")
async def subtitle_download(job_id: int, filename: str,
                              current=Depends(auth.require_feature("subtitle"))):
    """srt/mp3/txt 파일 다운로드."""
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    file_path = SUBTITLES_DIR / f"job_{job_id}" / filename
    if not file_path.exists():
        raise HTTPException(404, "file not found")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = {
        "srt": "application/x-subrip",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "txt": "text/plain",
        "json": "application/json",
    }.get(ext, "application/octet-stream")
    return FileResponse(file_path, media_type=mime, filename=filename)


# ===== URL로 영상 다운로드 (여러 개) =====
VIDEO_DL_DIR = Path(__file__).parent.parent / "data" / "video_downloads"
VIDEO_DL_DIR.mkdir(parents=True, exist_ok=True)


class DownloadUrlsRequest(BaseModel):
    urls: str  # 여러 줄 또는 콤마 구분


@app.post("/api/subtitle/download-from-urls")
async def download_from_urls(req: DownloadUrlsRequest,
                              current=Depends(auth.require_feature("subtitle"))):
    """URL 여러 개 → yt-dlp로 영상 다운로드. 최대 20개."""
    import uuid as _uuid
    urls = []
    for u in req.urls.replace(",", "\n").split("\n"):
        u = u.strip()
        if u and (u.startswith("http://") or u.startswith("https://")):
            urls.append(u)
    if not urls:
        raise HTTPException(400, "URL 없음")

    ytdlp = _bb_ytdlp()
    results = []
    for url in urls[:20]:
        fid = _uuid.uuid4().hex[:10]
        out_tmpl = str(VIDEO_DL_DIR / f"{fid}.%(ext)s")
        title = url
        try:
            # 제목 먼저 (실패해도 무시)
            try:
                tproc = await asyncio.create_subprocess_exec(
                    ytdlp, "--skip-download", "--print", "%(title)s", url,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                tout, _ = await asyncio.wait_for(tproc.communicate(), timeout=30)
                t = tout.decode().strip()
                if t:
                    title = t
            except Exception:
                pass
            # 다운로드
            proc = await asyncio.create_subprocess_exec(
                ytdlp, "-f", "best[ext=mp4][vcodec*=avc1]/best[ext=mp4]/best",
                "-o", out_tmpl, url,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            files = list(VIDEO_DL_DIR.glob(f"{fid}.*"))
            if files:
                fname = files[0].name
                size_mb = round(files[0].stat().st_size / 1024 / 1024, 1)
                results.append({
                    "url": url, "ok": True, "title": title,
                    "filename": fname, "size_mb": size_mb,
                    "download_url": f"/api/subtitle/downloaded/{fname}",
                })
            else:
                err = stderr.decode()[-200:] if stderr else "다운 실패"
                results.append({"url": url, "ok": False, "error": err})
        except asyncio.TimeoutError:
            results.append({"url": url, "ok": False, "error": "시간 초과 (180초)"})
        except Exception as e:
            results.append({"url": url, "ok": False, "error": str(e)[:200]})
    return {"results": results}


@app.get("/api/subtitle/downloaded/{filename}")
async def serve_downloaded_video(filename: str,
                                   current=Depends(auth.require_feature("subtitle"))):
    """다운로드한 영상 파일 serve."""
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    fpath = VIDEO_DL_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, "file not found")
    from fastapi.responses import FileResponse
    return FileResponse(fpath, media_type="video/mp4", filename=filename)




@app.post("/api/subtitle/{job_id}/youtube-backfill")
async def subtitle_youtube_backfill(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    """옛 job에 YouTube 업로드용 메타 (제목+설명+해쉬태그) 추가 생성.

    **영상 직접 보고 만듦** (Gemini Files API 업로드). 정확성 우선.
    비용: ~$0.05 ≈ ₩70/회.
    """
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")

    gr = job.get("gemini_results") or "{}"
    try:
        gr = json.loads(gr) if isinstance(gr, str) else gr
    except Exception:
        gr = {}
    primary = gr.get("primary") or {}

    # 이미 박혀 있으면 skip
    if primary.get("youtube_upload_title") and primary.get("youtube_description"):
        return {"job_id": job_id, "message": "이미 YouTube 자료 있음 (skip)",
                "youtube_upload_title": primary.get("youtube_upload_title")}

    video_path_str = job.get("video_path")
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(400, "영상 파일 없음 — backfill 불가")

    # 영상에 박혀있는 자막을 보고 YouTube 메타 생성 (영상 + 댓글 + 자료 자막 다 inject)
    from workers.auto_subtitle import (
        upload_video_to_gemini, call_gemini, GEMINI_FLASH_MODEL,
        fetch_youtube_comments_for_urls, format_comments_for_prompt,
    )

    # 원본 URL 댓글 (있으면 inject)
    original_urls = []
    try:
        original_urls = json.loads(job.get("original_urls") or "[]")
    except Exception:
        pass
    comments_dict = await fetch_youtube_comments_for_urls(original_urls or [], per_video=5) if original_urls else {}
    comments_text = format_comments_for_prompt(comments_dict) if comments_dict else ""

    # primary 자료 인용 (제목/줄거리 등)
    existing_title = primary.get("title", "")
    existing_summary = primary.get("summary", "")
    title_cands = primary.get("title_candidates", [])
    cands_str = "\n".join([f"  - {t}" for t in title_cands[:3]])

    yt_prompt = f"""이 영상을 면밀히 보고 YouTube 업로드용 메타 (제목+설명+해쉬태그)를 만들어줘.

[이미 분석된 자료 — 참고용]
- 영상 상단 타이틀 (영상 안 자막): {existing_title}
- 줄거리: {existing_summary}
- 영상 안 자막용 제목 후보 (참고만, 따라하지 X):
{cands_str}
{comments_text}

[YouTube 업로드용 제목·설명·해쉬태그 룰]
- **YouTube 업로드용 제목**과 **영상 안 자막용 제목**은 다름!
- 영상 안 자막용: 12~16자, 영상 위에 박는 짧은 후크
- **YouTube 업로드용: 35~60자**, SEO + 호기심 + 인기 키워드 + #shorts inline
- 영상 직접 보고 시청자가 클릭할만한 제목 만들기
- 설명: 200~500자, 줄거리 1~2줄 + 호기심 한 줄 + 마지막 inline 해쉬태그 5개
- 해쉬태그: 8~12개, 한국어 + 영어 mix, 인기 태그 + 영상 특화 태그

[출력 — JSON만]
```json
{{
  "youtube_upload_title": "YouTube 업로드용 메인 제목 (35~60자) #shorts",
  "youtube_upload_title_candidates": [
    "(SEO+호기심) YouTube 후보1 (35~60자)",
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
        file_uri = await upload_video_to_gemini(Path(video_path_str))
        yt_meta = await call_gemini(GEMINI_FLASH_MODEL, file_uri, yt_prompt,
                                      temperature=0.3, max_retries=3)
    except Exception as e:
        raise HTTPException(500, f"Gemini 영상 분석 실패: {e}")

    if not isinstance(yt_meta, dict):
        raise HTTPException(500, "Gemini 응답이 dict 아님")

    # primary에 YouTube fields 추가 + 자료 갱신
    primary["youtube_upload_title"] = yt_meta.get("youtube_upload_title", "")
    primary["youtube_upload_title_candidates"] = yt_meta.get("youtube_upload_title_candidates", [])
    primary["youtube_description"] = yt_meta.get("youtube_description", "")
    primary["hashtags"] = yt_meta.get("hashtags", [])
    gr["primary"] = primary

    db.update_subtitle_job(job_id, gemini_results=gr,
                            cost_usd=(job.get("cost_usd") or 0) + 0.05)

    return {
        "job_id": job_id,
        "youtube_upload_title": primary["youtube_upload_title"],
        "youtube_upload_title_candidates": primary["youtube_upload_title_candidates"],
        "youtube_description": primary["youtube_description"],
        "hashtags": primary["hashtags"],
        "cost_added_usd": 0.05,
    }


class SubtitleReviewRequest(BaseModel):
    review_note: str  # 대표님이 입력하는 한 줄 정정


@app.post("/api/subtitle/{job_id}/review")
async def subtitle_review(job_id: int, req: SubtitleReviewRequest,
                            background_tasks: BackgroundTasks,
                            current=Depends(auth.require_feature("subtitle"))):
    """사람 검수 정정 받아서 자막 재생성."""
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    video_path_str = job.get("video_path")
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(400, "영상 파일 없음 — 재분석 불가")

    # 자료 reset + review_note 저장
    db.update_subtitle_job(
        job_id, review_note=req.review_note,
        status="analyzing", progress=10,
        progress_message="검수 정정 반영해 재분석 시작..",
        needs_review=0,
        error=None,
    )

    # 원본 URL list 복원
    original_urls_str = job.get("original_urls") or "[]"
    try:
        original_urls = json.loads(original_urls_str) if isinstance(original_urls_str, str) else original_urls_str
    except Exception:
        original_urls = []

    # 백그라운드 재분석 시작 (review_note prefix prompt 사용)
    from workers.auto_subtitle import run_auto_subtitle
    background_tasks.add_task(
        run_auto_subtitle, job_id, Path(video_path_str), original_urls,
        req.review_note, job.get("style") or "shorts",
    )

    return {"job_id": job_id, "status": "analyzing",
            "message": "검수 메모 받음. 재분석 시작 (시간 좀 걸림)"}


@app.get("/api/subtitle/list")
async def subtitle_list(current=Depends(auth.require_feature("subtitle")),
                          limit: int = 30):
    """내 자막 작업 list (최근순). 쇼츠 메이커/더빙이 내부적으로 만든 subtitle_jobs는 제외."""
    user_id = current.get("id")
    # admin은 다 봄, 다른 사람은 자기 거만 — 다른 메뉴가 만든 잡은 listing에서 제외하려고 limit*4 가져와서 필터
    fetch_limit = max(limit * 4, 200)
    if current.get("role") == "admin":
        jobs = db.list_subtitle_jobs(user_id=None, limit=fetch_limit)
    else:
        jobs = db.list_subtitle_jobs(user_id=user_id, limit=fetch_limit)
    # 쇼츠 메이커/더빙이 만든 잡 제외 (video_path가 다른 메뉴 폴더면 제외)
    _own = []
    for j in jobs:
        vp = (j.get("video_path") or "").lower()
        # 자막 메뉴가 만든 잡 = data/subtitles/_uploads/ 또는 다른 메뉴 폴더가 아닌 것
        if "/shorts/" in vp or "/shorts_test/" in vp or "/tts_dubs/" in vp:
            continue
        _own.append(j)
        if len(_own) >= limit:
            break
    jobs = _own
    # JSON parse
    for j in jobs:
        for k in ("original_urls", "subtitle_paths", "title_candidates",
                   "cross_validation"):
            if j.get(k) and isinstance(j[k], str):
                try:
                    j[k] = json.loads(j[k])
                except Exception:
                    pass
    return {"jobs": jobs}


@app.delete("/api/subtitle/{job_id}")
async def subtitle_delete(job_id: int,
                            current=Depends(auth.require_feature("subtitle"))):
    job = db.get_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    # admin 또는 본인만
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")
    # 파일 삭제
    import shutil
    folder = SUBTITLES_DIR / f"job_{job_id}"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    db.delete_subtitle_job(job_id)
    return {"deleted": job_id}




# ===== 자막 학습 시스템 (2026-05-17) =====

@app.post("/api/subtitle/learn/start")
async def subtitle_learn_start(
    background_tasks: BackgroundTasks,
    parallel: int = 5,
    notify_every: int = 50,
    current=Depends(auth.admin_only),
):
    """297영상 학습 시작 (백그라운드, comments_done 영상 다 분석)"""
    if current.get("role") != "admin":
        raise HTTPException(403, "admin만 학습 시작 가능")

    # 현재 진행 중 세션 있는지 점검
    import sqlite3
    conn = sqlite3.connect(str(Path(__file__).parent.parent / "db" / "discover.db"))
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM learning_progress WHERE status='running'")
    running = cur.fetchone()
    if running:
        conn.close()
        raise HTTPException(409, f"이미 진행 중 — session_id={running[0]}")

    # 학습 대상 영상 수 점검
    cur.execute("SELECT COUNT(*) FROM subtitle_learning_queue WHERE status='comments_done'")
    target_count = cur.fetchone()[0]
    conn.close()

    if target_count == 0:
        raise HTTPException(400, "학습 대상 영상 0개 (comments_done 영상 없음)")

    # 세션 ID
    import uuid, time as _time
    session_id = f"learn_{int(_time.time())}_{uuid.uuid4().hex[:6]}"

    # 백그라운드 학습 시작
    from workers.subtitle_learning import learn_all

    async def _run():
        try:
            await learn_all(session_id=session_id, parallel=parallel, notify_every=notify_every)
        except Exception as e:
            import sqlite3
            conn2 = sqlite3.connect(str(Path(__file__).parent.parent / "db" / "discover.db"))
            conn2.execute(
                "UPDATE learning_progress SET status='failed', summary=? WHERE session_id=?",
                (f"실패: {str(e)[:500]}", session_id),
            )
            conn2.commit()
            conn2.close()
            from workers import notify
            notify.send_telegram(f"❌ 자막 학습 실패: {str(e)[:300]}")

    background_tasks.add_task(_run)

    return {
        "session_id": session_id,
        "target_videos": target_count,
        "parallel": parallel,
        "notify_every": notify_every,
        "message": f"{target_count}영상 학습 시작. 매 {notify_every}마다 텔레그램 진행률"
    }


@app.get("/api/subtitle/learn/progress")
async def subtitle_learn_progress(
    session_id: str | None = None,
    current=Depends(auth.admin_only),
):
    """학습 진행률 점검 (session_id 없으면 가장 최근 세션)"""
    import sqlite3
    conn = sqlite3.connect(str(Path(__file__).parent.parent / "db" / "discover.db"))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if session_id:
        cur.execute("SELECT * FROM learning_progress WHERE session_id=?", (session_id,))
    else:
        cur.execute("SELECT * FROM learning_progress ORDER BY id DESC LIMIT 1")

    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "session not found")

    progress = dict(row)

    # 자료 통계 추가
    cur.execute("SELECT COUNT(*) FROM subtitle_situations")
    progress["situation_count"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subtitle_expression_pool")
    progress["expression_count"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subtitle_learnings")
    progress["learning_count"] = cur.fetchone()[0]

    # top 표현 (빈도)
    cur.execute("SELECT phrase, use_count FROM subtitle_phrase_frequency ORDER BY use_count DESC LIMIT 20")
    progress["top_phrases"] = [{"phrase": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()

    # summary JSON 풀기
    if progress.get("summary"):
        try:
            import json
            progress["summary"] = json.loads(progress["summary"])
        except Exception:
            pass

    return progress


@app.get("/api/subtitle/learn/queue/status")
async def subtitle_learn_queue_status(current=Depends(auth.admin_only)):
    """queue 영상 상태 별 통계"""
    import sqlite3
    conn = sqlite3.connect(str(Path(__file__).parent.parent / "db" / "discover.db"))
    cur = conn.cursor()

    cur.execute("""
        SELECT channel_name, status, COUNT(*) FROM subtitle_learning_queue
        GROUP BY channel_name, status
        ORDER BY channel_name, status
    """)
    rows = cur.fetchall()

    by_channel: dict = {}
    by_status: dict = {}
    total = 0
    for ch, st, cnt in rows:
        by_channel.setdefault(ch, {})[st] = cnt
        by_status[st] = by_status.get(st, 0) + cnt
        total += cnt

    conn.close()
    return {"total": total, "by_status": by_status, "by_channel": by_channel}




# ===== 일본어 멀티유즈 (2026-05-17) =====

JAPANESE_DIR = Path(__file__).parent.parent / "data" / "japanese_multiuse"


@app.post("/api/japanese/upload")
async def japanese_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current=Depends(auth.require_feature("japanese")),
):
    """영상 업로드 + 일본어 번역 시작"""
    JAPANESE_DIR.mkdir(parents=True, exist_ok=True)

    # 영상 저장
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    job_id = db.insert_japanese_multiuse_job(
        video_filename=safe_name,
        video_path="",  # 박은 후 갱신
        user_id=current.get("id"),
    )
    save_dir = JAPANESE_DIR / f"job_{job_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / safe_name

    content_bytes = await file.read()
    save_path.write_bytes(content_bytes)
    db.update_japanese_multiuse_job(job_id, video_path=str(save_path))

    # 백그라운드 시작
    from workers.japanese_multiuse import run_japanese_multiuse
    background_tasks.add_task(run_japanese_multiuse, job_id, save_path)

    return {"job_id": job_id,
            "message": "일본어 번역 시작. /api/japanese/{job_id}/status로 진행률"}


@app.get("/api/japanese/{job_id}/status")
async def japanese_status(job_id: int, current=Depends(auth.require_feature("japanese"))):
    job = db.get_japanese_multiuse_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")
    return {
        "id": job["id"],
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "progress_message": job.get("progress_message"),
        "error": job.get("error"),
    }


@app.get("/api/japanese/{job_id}/result")
async def japanese_result(job_id: int, current=Depends(auth.require_feature("japanese"))):
    job = db.get_japanese_multiuse_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")

    def _parse_json(v):
        if not v: return None
        try: return json.loads(v) if isinstance(v, str) else v
        except: return v

    srt_urls = {}
    for kind, key in [("situation", "japanese_situation_srt_path"),
                       ("jjap_jjap_i", "japanese_jjap_jjap_i_srt_path"),
                       ("dialogue", "japanese_dialogue_srt_path")]:
        path = job.get(key)
        if path:
            filename = Path(path).name
            srt_urls[kind] = f"/api/japanese/{job_id}/download/{filename}"

    gr = _parse_json(job.get("gemini_result")) or {}
    return {
        "id": job["id"],
        "status": job.get("status"),
        "japanese_title": job.get("japanese_title"),
        "title_candidates_jp": _parse_json(job.get("title_candidates_jp")) or [],
        "korean_subtitles_extracted": _parse_json(job.get("korean_subtitles_extracted")) or [],
        "srt_urls": srt_urls,
        "cost_usd": job.get("cost_usd", 0),
        # 일본어 YouTube 업로드 메타 (gemini_result에서 추출)
        "youtube_upload_title_jp": gr.get("youtube_upload_title_jp", "") if isinstance(gr, dict) else "",
        "youtube_upload_title_candidates_jp": gr.get("youtube_upload_title_candidates_jp", []) if isinstance(gr, dict) else [],
        "youtube_description_jp": gr.get("youtube_description_jp", "") if isinstance(gr, dict) else "",
        "hashtags_jp": gr.get("hashtags_jp", []) if isinstance(gr, dict) else [],
        "gemini_result": gr,
    }


@app.get("/api/japanese/{job_id}/download/{filename}")
async def japanese_download(job_id: int, filename: str,
                              current=Depends(auth.require_feature("japanese"))):
    job = db.get_japanese_multiuse_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    file_path = JAPANESE_DIR / f"job_{job_id}" / filename
    if not file_path.exists():
        raise HTTPException(404, "파일 없음")
    from fastapi.responses import FileResponse
    return FileResponse(file_path, media_type="application/x-subrip", filename=filename)


@app.get("/api/japanese/list")
async def japanese_list(current=Depends(auth.require_feature("japanese")),
                          limit: int = 50):
    user_id = current.get("id")
    if current.get("role") == "admin":
        jobs = db.list_japanese_multiuse_jobs(user_id=None, limit=limit)
    else:
        jobs = db.list_japanese_multiuse_jobs(user_id=user_id, limit=limit)
    # JSON 필드 파싱
    for j in jobs:
        for k in ("title_candidates_jp", "korean_subtitles_extracted"):
            v = j.get(k)
            if v:
                try: j[k] = json.loads(v) if isinstance(v, str) else v
                except: pass
    return {"jobs": jobs}


@app.delete("/api/japanese/{job_id}")
async def japanese_delete(job_id: int, current=Depends(auth.require_feature("japanese"))):
    job = db.get_japanese_multiuse_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if current.get("role") != "admin" and job.get("user_id") != current.get("id"):
        raise HTTPException(403, "권한 없음")
    import shutil
    folder = JAPANESE_DIR / f"job_{job_id}"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    db.delete_japanese_multiuse_job(job_id)
    return {"deleted": job_id}


# ===== 클립 편집 (여러 영상 → 노래 분위기 구간 → 컷 편집) =====
CLIP_EDIT_DIR = Path(__file__).parent.parent / "data" / "clip_edits"
CLIP_EDIT_DIR.mkdir(parents=True, exist_ok=True)


class ClipEditRequest(BaseModel):
    urls: list[str]
    topic: str
    target_duration: int = 50


class ClipSuggestRequest(BaseModel):
    topic: str


@app.post("/api/clip-edit/suggest")
async def clip_edit_suggest(req: ClipSuggestRequest,
                            current=Depends(auth.require_feature("clip"))):
    """주제(인물/사건) → 시기별 추천 검색어 + 대본 미리보기. (검색은 사용자가 직접 = 정확도↑)"""
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(400, "주제(인물/역사 등) 필요")
    from workers.clip_editor import _career_timeline
    timeline = await _career_timeline(topic)
    if not timeline:
        raise HTTPException(502, "추천 생성 실패 (잠시 후 다시 시도)")
    return {"topic": topic, "timeline": timeline}


@app.post("/api/clip-edit/create")
async def clip_edit_create(req: ClipEditRequest, background_tasks: BackgroundTasks,
                            current=Depends(auth.require_feature("clip"))):
    """여러 영상 URL + 주제(인물/역사 등) → 그 대상 나오는 구간 + 스토리 자막 컷 편집. (TTS 제외)"""
    urls = [u.strip() for u in (req.urls or []) if u.strip().startswith("http")]
    if not (req.topic or "").strip():
        raise HTTPException(400, "주제(인물/역사 등) 필요")
    topic = req.topic.strip()
    # 총 길이 10~900초 (롱폼 지원 — 쇼츠 짧게, 롱폼 최대 15분)
    target = max(10, min(900, int(req.target_duration or 50)))
    job_id = db.insert_clip_edit_job(urls, topic, target, current.get("id"))
    # 독립 프로세스로 spawn — 긴 작업(영상 다운+ArcFace+편집)이 uvicorn 안 막고 재시작에 안 죽게
    _spawn_shorts_worker("clip_edit", job_id)
    return {"job_id": job_id, "status": "pending",
            "message": "클립 편집 시작 (다운+주제 구간분석+자막+편집, 수분 소요)"}


@app.get("/api/clip-edit/{job_id}/status")
async def clip_edit_status(job_id: int, current=Depends(auth.require_feature("clip"))):
    job = db.get_clip_edit_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {"id": job["id"], "status": job.get("status"),
            "progress": job.get("progress", 0),
            "progress_message": job.get("progress_message", "")}


@app.get("/api/clip-edit/{job_id}/result")
async def clip_edit_result(job_id: int, current=Depends(auth.require_feature("clip"))):
    job = db.get_clip_edit_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def _pj(v):
        if not v: return None
        try: return json.loads(v) if isinstance(v, str) else v
        except: return v

    download_url = None
    if job.get("result_path") and Path(job["result_path"]).exists():
        download_url = f"/api/clip-edit/{job_id}/download"
    srt_url = None
    if job.get("srt_path") and Path(job["srt_path"]).exists():
        srt_url = f"/api/clip-edit/{job_id}/srt"
    # 유튜브 메타 (상단제목/유튜브제목/설명/태그) — meta.json에서 (대표님 0613)
    meta = None
    for base in (job.get("srt_path"), job.get("result_path")):
        if base:
            mp = Path(base).parent / "meta.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = None
                break
    return {
        "id": job["id"],
        "status": job.get("status"),
        "topic": job.get("song_title"),
        "song_title": job.get("song_title"),
        "urls": _pj(job.get("urls")) or [],
        "segments": _pj(job.get("segments_json")) or [],
        "download_url": download_url,
        "srt_url": srt_url,
        "meta": meta,
        "cost_usd": job.get("cost_usd", 0),
    }


@app.get("/api/clip-edit/{job_id}/download")
async def clip_edit_download(job_id: int, current=Depends(auth.authenticate)):
    """편집 영상 다운로드 (token query 지원 — <a download>용)."""
    job = db.get_clip_edit_job(job_id)
    if not job or not job.get("result_path"):
        raise HTTPException(404, "결과 없음")
    fp = Path(job["result_path"])
    if not fp.exists():
        raise HTTPException(404, "file not found")
    from fastapi.responses import FileResponse
    safe = (job.get("song_title") or "clip").replace("/", "_")
    return FileResponse(fp, media_type="video/mp4",
                         filename=f"{safe}_클립편집.mp4")


@app.get("/api/clip-edit/{job_id}/srt")
async def clip_edit_srt(job_id: int, current=Depends(auth.authenticate)):
    """스토리 내레이션 자막 SRT 다운로드 (캡컷 후처리용, token query 지원)."""
    job = db.get_clip_edit_job(job_id)
    if not job or not job.get("srt_path"):
        raise HTTPException(404, "자막 없음")
    fp = Path(job["srt_path"])
    if not fp.exists():
        raise HTTPException(404, "file not found")
    from fastapi.responses import FileResponse
    safe = (job.get("song_title") or "clip").replace("/", "_")
    return FileResponse(fp, media_type="text/plain",
                         filename=f"{safe}_자막.srt")


@app.get("/api/clip-edit/list")
async def clip_edit_list(current=Depends(auth.require_feature("clip"))):
    # admin은 전체, 프리랜서는 본인이 만든 것만
    if current.get("role") == "admin":
        jobs = db.list_clip_edit_jobs(limit=30)
    else:
        jobs = db.list_clip_edit_jobs(user_id=current.get("id"), limit=30)
    return {"jobs": jobs}


@app.delete("/api/clip-edit/{job_id}")
async def clip_edit_delete(job_id: int, current=Depends(auth.require_feature("clip"))):
    import shutil
    folder = CLIP_EDIT_DIR / f"job_{job_id}"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    db.delete_clip_edit_job(job_id)
    return {"deleted": job_id}


# ===== 음성 자막 (Whisper large-v3 + Gemini 어투보존 교정 + 사람 검수) =====
import shutil as _shutil
import subprocess as _subp

AUDIO_SUB_DIR = Path(__file__).resolve().parent.parent / "data" / "audio_subtitles"


class AudioSubSaveRequest(BaseModel):
    segments: list


@app.post("/api/audio-subtitle/upload")
async def audio_sub_upload(file: UploadFile = File(...),
                           current=Depends(auth.require_feature("subtitle"))):
    job_id = db.insert_audio_subtitle_job(file.filename or "audio.mp3", "", 0, current["id"])
    jdir = AUDIO_SUB_DIR / f"job_{job_id}"
    jdir.mkdir(parents=True, exist_ok=True)
    ext = ((file.filename or "audio.mp3").rsplit(".", 1)[-1] or "mp3").lower()[:5]
    apath = jdir / f"audio.{ext}"
    with open(apath, "wb") as f:
        _shutil.copyfileobj(file.file, f)
    dur = 0.0
    try:
        out = _subp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(apath)],
                        capture_output=True, text=True, timeout=30)
        dur = float((out.stdout or "0").strip() or 0)
    except Exception:
        pass
    db.update_audio_subtitle_job(job_id, audio_path=str(apath), duration_sec=dur)
    from workers.audio_subtitle import run_audio_subtitle
    asyncio.create_task(asyncio.to_thread(run_audio_subtitle, job_id))
    return {"id": job_id, "filename": file.filename, "duration_sec": dur}


@app.get("/api/audio-subtitle/list")
async def audio_sub_list(current=Depends(auth.require_feature("subtitle"))):
    uid = None if current.get("role") == "admin" else current["id"]
    jobs = db.list_audio_subtitle_jobs(user_id=uid, limit=100)
    for j in jobs:
        j.pop("segments_json", None)
    return jobs


@app.get("/api/audio-subtitle/{job_id}")
async def audio_sub_get(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    job = db.get_audio_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "없는 작업")
    if current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    if isinstance(job.get("segments_json"), str):
        try:
            job["segments_json"] = json.loads(job["segments_json"])
        except Exception:
            job["segments_json"] = []
    return job


@app.post("/api/audio-subtitle/{job_id}/save")
async def audio_sub_save(job_id: int, req: AudioSubSaveRequest,
                         current=Depends(auth.require_feature("subtitle"))):
    job = db.get_audio_subtitle_job(job_id)
    if not job:
        raise HTTPException(404, "없는 작업")
    if current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    db.update_audio_subtitle_job(job_id, segments_json=req.segments, status="done")
    from workers.audio_subtitle import rebuild_srt
    rebuild_srt(job_id)
    return {"ok": True}


@app.get("/api/audio-subtitle/{job_id}/download")
async def audio_sub_download(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    from fastapi.responses import FileResponse
    job = db.get_audio_subtitle_job(job_id)
    if not job or not job.get("srt_path"):
        raise HTTPException(404, "SRT 없음")
    if current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    base = (job.get("audio_filename") or f"job_{job_id}")
    name = base.rsplit(".", 1)[0] + ".srt"
    return FileResponse(job["srt_path"], filename=name, media_type="application/x-subrip")


@app.delete("/api/audio-subtitle/{job_id}")
async def audio_sub_delete(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    job = db.get_audio_subtitle_job(job_id)
    if job and current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    db.delete_audio_subtitle_job(job_id)
    return {"ok": True}


# ===== 대본+더빙 (영상→Gemini 대본/메타 + 타입캐스트 TTS + SRT) =====

TTS_DUB_DIR = Path(__file__).parent.parent / "data" / "tts_dub"


@app.post("/api/tts-dub/upload")
async def tts_dub_upload(file: UploadFile = File(...),
                         voice_id: str = Form(default=""),
                         make_tts: str = Form(default="1"),
                         current=Depends(auth.require_feature("subtitle"))):
    job_id = db.insert_tts_dub_job(file.filename or "video.mp4", "",
                                   current["id"], voice_id.strip() or None)
    jdir = TTS_DUB_DIR / f"job_{job_id}"
    jdir.mkdir(parents=True, exist_ok=True)
    ext = ((file.filename or "video.mp4").rsplit(".", 1)[-1] or "mp4").lower()[:5]
    vpath = jdir / f"input.{ext}"
    with open(vpath, "wb") as f:
        _shutil.copyfileobj(file.file, f)
    db.update_tts_dub_job(job_id, video_path=str(vpath))
    from workers.tts_dub import run_tts_dub
    asyncio.create_task(run_tts_dub(job_id, make_tts != "0"))
    return {"id": job_id, "filename": file.filename}


@app.get("/api/tts-dub/list")
async def tts_dub_list(current=Depends(auth.require_feature("subtitle"))):
    uid = None if current.get("role") == "admin" else current["id"]
    return db.list_tts_dub_jobs(user_id=uid, limit=100)


@app.get("/api/tts-dub/{job_id}")
async def tts_dub_get(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    job = db.get_tts_dub_job(job_id)
    if not job:
        raise HTTPException(404, "없는 작업")
    if current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    if isinstance(job.get("result_json"), str):
        try:
            job["result_json"] = json.loads(job["result_json"])
        except Exception:
            job["result_json"] = None
    return job


@app.get("/api/tts-dub/{job_id}/download/{kind}")
async def tts_dub_download(job_id: int, kind: str,
                          current=Depends(auth.require_feature("subtitle"))):
    from fastapi.responses import FileResponse
    job = db.get_tts_dub_job(job_id)
    if not job:
        raise HTTPException(404, "없는 작업")
    if current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    base = (job.get("video_filename") or f"job_{job_id}").rsplit(".", 1)[0]
    if kind == "srt" and job.get("srt_path"):
        return FileResponse(job["srt_path"], filename=base + ".srt",
                            media_type="application/x-subrip")
    if kind == "mp3" and job.get("tts_path"):
        return FileResponse(job["tts_path"], filename=base + "_TTS.mp3",
                            media_type="audio/mpeg")
    if kind == "bgm" and job.get("tts_path"):
        # convention: bgm_mix.mp3는 tts.mp3와 같은 폴더
        bgm_path = Path(job["tts_path"]).parent / "bgm_mix.mp3"
        if bgm_path.exists():
            return FileResponse(str(bgm_path), filename=base + "_BGM.mp3",
                                media_type="audio/mpeg")
        raise HTTPException(404, "BGM 믹스 없음 (옛 잡일 수 있음)")
    if kind == "cctv" and job.get("tts_path"):
        # convention: cctv_last.png는 tts.mp3와 같은 폴더
        cctv_path = Path(job["tts_path"]).parent / "cctv_last.png"
        if cctv_path.exists():
            return FileResponse(str(cctv_path), filename=base + "_CCTV.png",
                                media_type="image/png")
        raise HTTPException(404, "CCTV 변환 없음 (옛 잡일 수 있음)")
    raise HTTPException(404, "파일 없음 (kind=srt|mp3|bgm|cctv)")


@app.delete("/api/tts-dub/{job_id}")
async def tts_dub_delete(job_id: int, current=Depends(auth.require_feature("subtitle"))):
    job = db.get_tts_dub_job(job_id)
    if job and current.get("role") != "admin" and job.get("user_id") != current["id"]:
        raise HTTPException(403, "권한 없음")
    db.delete_tts_dub_job(job_id)
    return {"ok": True}


# ===== 쇼츠 메이커 (긴 URL → 하이라이트 N개 → ≤59초 쇼츠 양산) =====

SHORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "shorts"
SHORTS_DIR.mkdir(parents=True, exist_ok=True)


def _spawn_shorts_worker(func: str, job_id: int, extra=None):
    """발굴/제작을 uvicorn과 분리된 독립 프로세스로 실행 (workers/job_runner.py).
    대표님 2026-05-29 근본 해결: 작업이 uvicorn 이벤트 루프에서 돌면 재시작 시 죽고
    무거운 작업이 웹서버를 막음. 독립 프로세스면 재시작/부하 무관 + 작업 보존."""
    import subprocess as _subp, sys as _sys, os as _os, tempfile as _tf
    repo = str(Path(__file__).resolve().parent.parent)
    # 🔴 윈도우 임베디드 파이썬은 cwd를 sys.path에 안 넣어 `-m workers.job_runner`가
    #    ModuleNotFoundError(workers)로 즉사 → 클립편집 0% 멈춤. 절대경로 스크립트로 실행
    #    (job_runner 상단이 repo를 sys.path에 추가하므로 from workers/api 가 잡힘). 2026-06-16 수정.
    _runner = str(Path(__file__).resolve().parent.parent / "workers" / "job_runner.py")
    args = [_sys.executable, _runner, func, str(job_id)]
    if extra is not None:
        args.append(str(extra))
    try:
        # 🔴 윈도우엔 /tmp 가 없어 open 실패→except→워커 미실행으로 클립편집이 0%에 멈췄음.
        #    tempfile.gettempdir()(맥=/tmp·윈도우=%TEMP%)로 크로스플랫폼. 로그 못 열어도 워커는 뜨게 DEVNULL.
        #    (2026-06-16 클립편집 윈도우 0% 멈춤 근본수정)
        try:
            logf = open(_os.path.join(_tf.gettempdir(), f"shorts_worker_{job_id}.log"), "ab")
        except Exception:
            logf = _subp.DEVNULL
        _popen_kw = {} if _sys.platform == "win32" else {"start_new_session": True}
        _subp.Popen(args, cwd=repo, env=dict(_os.environ),
                    stdout=logf, stderr=logf, **_popen_kw)
        print(f"🚀 독립 워커 spawn: {func} job={job_id}", flush=True)
    except Exception as e:
        print(f"⚠️ 워커 spawn 실패 {func} {job_id}: {e}", flush=True)


class ShortsCreateReq(BaseModel):
    url: str
    name: str | None = None
    # 타입 — "highlight" (긴 영상 N개 추출, 기본) / "drama" (짧은 드라마 흐름 보존 1~2편)
    type: str | None = "highlight"


def _shorts_check_owner(job: dict, current: dict):
    if not job:
        raise HTTPException(404, "없는 작업")
    if (current.get("role") != "admin"
            and job.get("assigned_user_id") != current["id"]
            and job.get("user_id") != current["id"]):
        raise HTTPException(403, "권한 없음")


import io as _io
import zipfile as _zipfile
from urllib.parse import quote as _quote
from fastapi.responses import Response as _Resp


def _shorts_parse_results(job: dict) -> dict:
    """results_json 파싱. 두 가지 형식 호환:
    - 신: {"highlights": [...], "characters": [...], "characters_dir": ...}
    - 구: [...] (배열만, 캐리커처 없던 버전)
    반환은 항상 dict로 통일.
    """
    raw = job.get("results_json")
    if not raw:
        return {"highlights": [], "characters": []}
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            return {"highlights": d, "characters": []}
        return {
            "highlights": d.get("highlights", []),
            "characters": d.get("characters", []),
            "characters_dir": d.get("characters_dir"),
            "candidates": d.get("candidates", []),  # v3 발굴 후보
        }
    except Exception:
        return {"highlights": [], "characters": []}


def _shorts_parse_pass1(job: dict) -> dict:
    raw = job.get("pass1_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


@app.post("/api/shorts/create")
async def shorts_create(req: ShortsCreateReq,
                        current=Depends(auth.require_feature("subtitle"))):
    url = (req.url or "").strip()
    if not url.startswith("http"):
        raise HTTPException(400, "URL 형식 오류")
    ptype = (req.type or "highlight").strip().lower()
    if ptype not in ("highlight", "drama", "movie", "anime", "folktale"):
        ptype = "highlight"
    # 잡 이름 — 사용자 입력 우선, 없으면 YouTube 영상 제목 (yt-dlp --get-title)
    name = (req.name or "").strip()
    if not name:
        try:
            ytdlp = _bb_ytdlp()
            import subprocess as _sp
            r = _sp.run([ytdlp, "--get-title", "--no-warnings", url],
                        capture_output=True, text=True, timeout=15)
            t = (r.stdout or "").strip().split("\n")[0]
            if t:
                # 파일 안전 + 60자 cap
                name = t.replace("/", "·").replace("\\", "·")[:60]
        except Exception:
            pass
    if not name:
        name = f"job_{datetime.now().strftime('%m%d_%H%M%S')}"
    job_id = db.insert_shorts_job(url, name, user_id=current["id"])
    out_dir = str(SHORTS_DIR / f"job_{job_id}")
    db.update_shorts_job(job_id, out_dir=out_dir, type=ptype)
    # v3: 전 타입 발굴→선택→제작 통일. create는 발굴만 (자동 제작 X).
    _spawn_shorts_worker("shorts_discover", job_id)
    return {"id": job_id, "name": name, "out_dir": out_dir, "type": ptype}


@app.post("/api/shorts/upload")
async def shorts_upload(
    video: UploadFile = File(...),
    name: str = Form(default=""),
    type: str = Form(default="highlight"),
    current=Depends(auth.require_feature("subtitle")),
):
    """영상 파일 직접 업로드 → 발굴 (URL 없이도 제작 가능). 로컬 파일 흐름.
    대표님 요청 2026-05-29: 유튜브 URL 외에 내 영상 올려서도 쇼츠 제작.
    url에 로컬 경로를 넣으면 discover가 yt-dlp 다운 건너뛰고 그 파일을 직접 처리."""
    ptype = (type or "highlight").strip().lower()
    if ptype not in ("highlight", "drama", "movie", "anime", "folktale"):
        ptype = "highlight"
    safe_name = Path(video.filename or "upload.mp4").name
    nm = (name or "").strip() or Path(safe_name).stem[:60] \
        or f"job_{datetime.now().strftime('%m%d_%H%M%S')}"
    # 업로드 영상 저장 — 외장 SSD uploads(대용량 대비), 없으면 data/uploads 폴백.
    # 스트리밍 저장(1MB씩)으로 대용량(수백 MB)도 메모리 안 터지게.
    up_dir = Path("/Volumes/BanbajiMedia/uploads")
    try:
        up_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        up_dir = SHORTS_DIR.parent / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
    save_path = up_dir / f"shorts_{datetime.now().strftime('%m%d_%H%M%S')}_{safe_name}"
    with save_path.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    job_id = db.insert_shorts_job(str(save_path), nm, user_id=current["id"])
    out_dir = str(SHORTS_DIR / f"job_{job_id}")
    db.update_shorts_job(job_id, out_dir=out_dir, type=ptype)
    _spawn_shorts_worker("shorts_discover", job_id)
    return {"id": job_id, "name": nm, "out_dir": out_dir, "type": ptype}


SHORTS_CHUNK_DIR = Path(__file__).resolve().parent.parent / "data" / "shorts_chunks"


@app.post("/api/shorts/upload-chunk")
async def shorts_upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    chunk: UploadFile = File(...),
    name: str = Form(default=""),
    type: str = Form(default="highlight"),
    current=Depends(auth.require_feature("subtitle")),
):
    """대용량 영상 청크 업로드 (Cloudflare 100MB 요청 제한 회피).
    프론트가 80MB씩 잘라 순차 전송 → 서버가 이어붙임 → 마지막 청크에서 발굴 잡 생성.
    대표님 요청 2026-05-29: 631MB 영상이 cloudflare 터널 100MB 제한에 막혀 업로드 실패."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", upload_id)[:64]
    if not safe_id:
        raise HTTPException(400, "bad upload_id")
    # 청크 임시 = uploads와 같은 디스크(.chunks) → 마지막 합치기가 rename(즉시).
    # 로컬→외장 move(복사)면 631MB에 수십 초 걸려 cloudflare 100초 timeout("실패") 유발.
    up_dir = Path("/Volumes/BanbajiMedia/uploads")
    try:
        up_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        up_dir = SHORTS_DIR.parent / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = up_dir / ".chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    part = chunk_dir / f"{safe_id}.part"
    data = await chunk.read()
    with open(part, "wb" if chunk_index == 0 else "ab") as f:
        f.write(data)
    if chunk_index + 1 < total_chunks:
        return {"ok": True, "received": chunk_index}
    # 마지막 청크 — 같은 디스크라 move=rename(즉시). 발굴 잡 생성.
    ptype = (type or "highlight").strip().lower()
    if ptype not in ("highlight", "drama", "movie", "anime", "folktale"):
        ptype = "highlight"
    safe_name = Path(filename or "upload.mp4").name
    nm = (name or "").strip() or Path(safe_name).stem[:60] \
        or f"job_{datetime.now().strftime('%m%d_%H%M%S')}"
    save_path = up_dir / f"shorts_{datetime.now().strftime('%m%d_%H%M%S')}_{safe_name}"
    _shutil.move(str(part), str(save_path))
    job_id = db.insert_shorts_job(str(save_path), nm, user_id=current["id"])
    out_dir = str(SHORTS_DIR / f"job_{job_id}")
    db.update_shorts_job(job_id, out_dir=out_dir, type=ptype)
    _spawn_shorts_worker("shorts_discover", job_id)
    return {"id": job_id, "name": nm, "out_dir": out_dir, "type": ptype, "done": True}


class ShortsRenderSelectedReq(BaseModel):
    idxs: list[int]  # 제작할 후보 idx 리스트


@app.post("/api/shorts/{job_id}/render-selected")
async def shorts_render_selected(job_id: int, req: ShortsRenderSelectedReq,
                                  current=Depends(auth.require_feature("subtitle"))):
    """v3 — 발굴된 후보 중 선택한 것만 제작."""
    job = db.get_shorts_job(job_id)
    _shorts_check_owner(job, current)
    if not req.idxs:
        raise HTTPException(400, "선택된 후보 없음")
    _spawn_shorts_worker("shorts_render", job_id, ",".join(str(i) for i in req.idxs))
    return {"id": job_id, "rendering": req.idxs}


@app.get("/api/shorts/list")
async def shorts_list(current=Depends(auth.require_feature("shorts"))):
    if current.get("role") == "admin":
        jobs = db.list_shorts_jobs(user_id=None, limit=100)
    else:
        jobs = db.list_shorts_jobs_assigned(current["id"], limit=100)
    # 응답 슬림: results_json / pass1_json 제외
    slim = []
    for j in jobs:
        d = {k: v for k, v in j.items() if k not in ("results_json", "pass1_json")}
        d["highlights_count"] = j.get("highlights_count") or 0
        slim.append(d)
    return slim


class ShortsAssignReq(BaseModel):
    user_id: int | None = None


@app.post("/api/shorts/{job_id}/assign")
async def shorts_assign(job_id: int, req: ShortsAssignReq,
                        current=Depends(auth.admin_only)):
    """대표님 전용 — 완성된 쇼츠 작업을 특정 프리랜서에게 지정(그 프리만 보고 다운로드)."""
    job = db.get_shorts_job(job_id)
    if not job:
        raise HTTPException(404, "없는 작업")
    db.update_shorts_job(job_id, assigned_user_id=req.user_id)
    return {"ok": True, "assigned_user_id": req.user_id}


@app.get("/api/shorts/{job_id}")
async def shorts_get(job_id: int,
                     current=Depends(auth.require_feature("shorts"))):
    job = db.get_shorts_job(job_id)
    _shorts_check_owner(job, current)
    d = dict(job)
    parsed = _shorts_parse_results(job)
    highlights = parsed["highlights"]
    # 각 hl의 meta.json 로드 (UI에서 제목/설명/태그 박스 표시용)
    for hl in highlights:
        hd = Path(hl.get("dir", ""))
        meta_path = hd / "meta.json"
        if meta_path.exists():
            try:
                hl["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                hl["meta"] = None
    d["results"] = highlights
    d["candidates"] = parsed.get("candidates", [])  # v3 발굴 후보 (discovered 상태 UI 표시 — 누락 버그 fix)
    d["characters"] = parsed["characters"]
    d["characters_dir"] = parsed.get("characters_dir")
    d["pass1"] = _shorts_parse_pass1(job)
    d.pop("results_json", None)
    d.pop("pass1_json", None)
    return d


@app.get("/api/shorts/{job_id}/file")
async def shorts_file(job_id: int, path: str,
                      current=Depends(auth.require_feature("shorts"))):
    job = db.get_shorts_job(job_id)
    _shorts_check_owner(job, current)
    base = Path(job.get("out_dir") or "").resolve()
    if not base.exists():
        raise HTTPException(404, "잡 폴더 없음")
    target = (base / path).resolve()
    # path traversal 방지: target은 반드시 base 아래
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(400, "잘못된 경로")
    if not target.is_file():
        raise HTTPException(404, "파일 없음")
    ext = target.suffix.lower()
    mime = {
        ".mp4": "video/mp4", ".srt": "application/x-subrip",
        ".txt": "text/plain", ".json": "application/json",
        ".vtt": "text/vtt",
    }.get(ext, "application/octet-stream")
    return FileResponse(str(target), filename=target.name, media_type=mime)


@app.get("/api/shorts/{job_id}/zip/{hl_idx}")
async def shorts_zip(job_id: int, hl_idx: int,
                     current=Depends(auth.require_feature("shorts"))):
    """hl 1개 ZIP — final.mp4 + SRT 3종 + 04_제목후보.txt + meta.json + characters/*.png (잡 공유)."""
    job = db.get_shorts_job(job_id)
    _shorts_check_owner(job, current)
    base = Path(job.get("out_dir") or "").resolve()
    if not base.exists():
        raise HTTPException(404, "잡 폴더 없음")
    hl_dir = (base / f"hl_{hl_idx:02d}").resolve()
    try:
        hl_dir.relative_to(base)
    except ValueError:
        raise HTTPException(400, "잘못된 hl 경로")
    if not hl_dir.exists():
        raise HTTPException(404, f"hl_{hl_idx:02d} 폴더 없음")
    chars_dir = base / "characters"

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # hl 폴더 — 최종 파일만 (중간 작업 파일 제외)
        for f in sorted(hl_dir.iterdir()):
            if not f.is_file():
                continue
            n = f.name
            if n.startswith("_") or n.startswith("highlight_"):
                continue
            zf.write(f, arcname=n)
        # 잡 공유 캐리커처들
        if chars_dir.exists() and chars_dir.is_dir():
            for f in sorted(chars_dir.glob("*.png")):
                zf.write(f, arcname=f"characters/{f.name}")
    buf.seek(0)
    job_name = (job.get("name") or f"job_{job_id}")
    fname = f"{job_name}_hl_{hl_idx:02d}.zip"
    return _Resp(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{_quote(fname)}"},
    )


@app.post("/api/shorts/{job_id}/restart")
async def shorts_restart(job_id: int,
                         current=Depends(auth.require_feature("subtitle"))):
    job = db.get_shorts_job(job_id)
    _shorts_check_owner(job, current)
    if job.get("status") in ("pending", "downloading", "picking", "processing"):
        raise HTTPException(409, "이미 실행 중입니다")
    db.update_shorts_job(job_id, status="pending", progress=0,
                          progress_message="재시작 대기", error=None,
                          results_json=None, completed_at=None)
    from workers.shorts_maker import run_shorts_maker
    asyncio.create_task(run_shorts_maker(job_id))
    return {"ok": True, "id": job_id}


@app.delete("/api/shorts/{job_id}")
async def shorts_delete(job_id: int,
                        current=Depends(auth.require_feature("subtitle"))):
    job = db.get_shorts_job(job_id)
    if job:
        _shorts_check_owner(job, current)
        # 잡 폴더 삭제 (있으면)
        od = job.get("out_dir")
        if od:
            p = Path(od)
            if p.exists() and p.is_dir() and str(p.resolve()).startswith(str(SHORTS_DIR.resolve())):
                try:
                    _shutil.rmtree(p)
                except Exception as e:
                    print(f"⚠️ shorts 폴더 삭제 실패 {p}: {e}")
    db.delete_shorts_job(job_id)
    return {"ok": True}


# ===== Static frontend (PWA) =====

frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
