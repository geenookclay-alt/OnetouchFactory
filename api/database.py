"""SQLite database access layer."""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Optional


DB_PATH = Path(__file__).parent.parent / "db" / "discover.db"


@contextmanager
def get_db():
    """Context manager for SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)  # 30s busy_timeout
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")  # reader/writer 분리
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_CANDIDATE_EXTRA_COLUMNS = [
    # Visual matching (added 2026-05-07)
    ("visual_match_verdict", "TEXT"),
    ("visual_match_score", "REAL"),
    ("visual_match_video_id", "TEXT"),
    ("visual_match_url", "TEXT"),
    ("visual_match_channel", "TEXT"),
    # Usage tracking — 누가 실제로 채널에 업로드했는지 + 언제
    ("used", "INTEGER DEFAULT 0"),
    ("used_at", "TEXT"),
    ("used_by_user_id", "INTEGER"),
    ("used_by_username", "TEXT"),  # denormalized for fast list display
    # 양봉여리 한국어 메모 (Phase 8) — 형님이 영상 보고 짧게 적는 메모
    ("memo_kr", "TEXT"),
    # 수동 추가 (URL로 직접 입력) 표시 — 정렬 시 위에 + 화면에서 다른 색상
    ("is_manual", "INTEGER DEFAULT 0"),
    # 보류 — 미사용 탭에 남아있되 노란 색상, 나중에 사용 예정 의미
    ("on_hold", "INTEGER DEFAULT 0"),
]

# 카테고리(dissection)별 마스코트 시스템 — webtoon_static motion_mode용.
# 동적 N개 role 지원 (1~3개, 카테고리에 맞게 자유 archetype). roles_json 우선,
# 기존 savior/victim prefix 컬럼은 backward compat용으로 보존.
_MASCOT_EXTRA_COLUMNS = [
    # 동적 N roles JSON list (각 dict: role_id, role_label_kr, narrative_role,
    #   name_kr, concept_kr, concept_en, character_spec, baseline_path, baseline_url)
    ("roles_json", "TEXT"),
    # Legacy: savior + victim 1쌍 prefix (backward compat — 기존 데이터 보존)
    ("savior_concept_kr", "TEXT"),
    ("savior_concept_en", "TEXT"),
    ("savior_character_spec", "TEXT"),
    ("savior_baseline_path", "TEXT"),
    ("savior_baseline_url", "TEXT"),
    ("victim_concept_kr", "TEXT"),
    ("victim_concept_en", "TEXT"),
    ("victim_character_spec", "TEXT"),
    ("victim_baseline_path", "TEXT"),
    ("victim_baseline_url", "TEXT"),
]


def _ensure_visual_match_columns(conn) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(candidate_videos)")}
    for name, typ in _CANDIDATE_EXTRA_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE candidate_videos ADD COLUMN {name} {typ}")
    # mascot pair columns
    existing_m = {r[1] for r in conn.execute("PRAGMA table_info(mascots)")}
    for name, typ in _MASCOT_EXTRA_COLUMNS:
        if name not in existing_m:
            conn.execute(f"ALTER TABLE mascots ADD COLUMN {name} {typ}")


def init_db():
    """Initialize DB from schema.sql if not exists."""
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    schema_v2_path = Path(__file__).parent.parent / "db" / "schema_v2.sql"
    schema_v3_path = Path(__file__).parent.parent / "db" / "schema_v3.sql"
    schema_v4_path = Path(__file__).parent.parent / "db" / "schema_v4.sql"
    schema_v5_path = Path(__file__).parent.parent / "db" / "schema_v5.sql"
    schema_v6_path = Path(__file__).parent.parent / "db" / "schema_v6.sql"
    schema_v7_path = Path(__file__).parent.parent / "db" / "schema_v7.sql"
    with get_db() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        if schema_v2_path.exists():
            conn.executescript(schema_v2_path.read_text(encoding="utf-8"))
        if schema_v3_path.exists():
            conn.executescript(schema_v3_path.read_text(encoding="utf-8"))
        if schema_v4_path.exists():
            conn.executescript(schema_v4_path.read_text(encoding="utf-8"))
        if schema_v5_path.exists():
            conn.executescript(schema_v5_path.read_text(encoding="utf-8"))
        if schema_v6_path.exists():
            conn.executescript(schema_v6_path.read_text(encoding="utf-8"))
        if schema_v7_path.exists():
            conn.executescript(schema_v7_path.read_text(encoding="utf-8"))
        _ensure_visual_match_columns(conn)
        # users 개인 API 키 컬럼 (프리랜서 비용 분리: Gemini=대본, Typecast=TTS)
        try:
            ucols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
            if "gemini_api_key" not in ucols:
                conn.execute("ALTER TABLE users ADD COLUMN gemini_api_key TEXT")
            if "typecast_api_key" not in ucols:
                conn.execute("ALTER TABLE users ADD COLUMN typecast_api_key TEXT")
        except Exception:
            pass
        # subtitle_jobs.style — 스키마 파일엔 없고 운영 DB에만 있던 컬럼. 빈 DB 보강.
        try:
            scols = {r[1] for r in conn.execute("PRAGMA table_info(subtitle_jobs)")}
            if "style" not in scols:
                conn.execute("ALTER TABLE subtitle_jobs ADD COLUMN style TEXT DEFAULT 'shorts'")
        except Exception:
            pass
        # 클립편집 잡 테이블 (스키마 파일엔 없음 — 빈 DB에서 자동 생성, 대표님 0614 배포 호환)
        conn.execute("""CREATE TABLE IF NOT EXISTS clip_edit_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, urls TEXT, song_title TEXT,
            target_duration INTEGER DEFAULT 50, status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0, progress_message TEXT, result_path TEXT,
            segments_json TEXT, cost_usd REAL DEFAULT 0, user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, completed_at TEXT, error TEXT,
            srt_path TEXT, make_tts INTEGER DEFAULT 0)""")
        # 음성 자막 잡 테이블 (Whisper large-v3 + Gemini 어투보존 교정 + 검수)
        conn.execute("""CREATE TABLE IF NOT EXISTS audio_subtitle_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_filename TEXT,
            audio_path TEXT,
            duration_sec REAL,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            progress_message TEXT,
            segments_json TEXT,
            srt_path TEXT,
            user_id INTEGER,
            cost_usd REAL DEFAULT 0,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
        )""")

        # 대본+더빙 (영상→Gemini 대본/메타 + 타입캐스트 TTS + SRT)
        conn.execute("""CREATE TABLE IF NOT EXISTS tts_dub_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_filename TEXT,
            video_path TEXT,
            duration_sec REAL,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            progress_message TEXT,
            result_json TEXT,
            srt_path TEXT,
            tts_path TEXT,
            voice_id TEXT,
            user_id INTEGER,
            cost_usd REAL DEFAULT 0,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
        )""")

        # 쇼츠 메이커 (긴 URL → 하이라이트 N개 → 각 ≤59초 쇼츠+자막+메타)
        conn.execute("""CREATE TABLE IF NOT EXISTS shorts_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            name TEXT,
            out_dir TEXT,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            progress_message TEXT,
            highlights_count INTEGER DEFAULT 0,
            results_json TEXT,
            pass1_json TEXT,
            source_duration REAL,
            source_size_mb INTEGER,
            user_id INTEGER,
            cost_usd REAL DEFAULT 0,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
        )""")


# ===== Dissection analysis CRUD =====

def insert_dissection(diss_id: str, name: str, **kwargs) -> None:
    fields = ["id", "name"] + list(kwargs.keys())
    placeholders = ",".join("?" for _ in fields)
    values = [diss_id, name] + [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ]
    with get_db() as conn:
        conn.execute(
            f"INSERT INTO dissection_analyses ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )


def update_dissection(diss_id: str, **kwargs) -> None:
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ] + [diss_id]
    with get_db() as conn:
        conn.execute(
            f"UPDATE dissection_analyses SET {sets} WHERE id=?", values
        )


def get_dissection(diss_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM dissection_analyses WHERE id=?", (diss_id,)
        ).fetchone()
        return dict(row) if row else None


def list_dissections(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM dissection_analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_job(job_id: str, name: str, **kwargs) -> None:
    """Insert a new discovery job."""
    fields = ["id", "name"] + list(kwargs.keys())
    placeholders = ",".join("?" for _ in fields)
    values = [job_id, name] + [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ]
    with get_db() as conn:
        conn.execute(
            f"INSERT INTO discovery_jobs ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )


def update_job(job_id: str, **kwargs) -> None:
    """Update job fields."""
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = list(kwargs.values()) + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE discovery_jobs SET {sets} WHERE id=?", values)


def get_job(job_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM discovery_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM discovery_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_candidate(job_id: str, **kwargs) -> Optional[int]:
    """Insert candidate video, ignore if duplicate."""
    fields = ["job_id"] + list(kwargs.keys())
    placeholders = ",".join("?" for _ in fields)
    values = [job_id] + [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ]
    with get_db() as conn:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO candidate_videos ({','.join(fields)}) "
            f"VALUES ({placeholders})",
            values,
        )
        return cur.lastrowid


def list_candidates(
    job_id: Optional[str] = None,
    platform: Optional[str] = None,
    classification: Optional[str] = None,
    min_views: Optional[int] = None,
    used: Optional[int] = None,
    limit: int = 200,
) -> list[dict]:
    """List candidates with filters. used=1 → 사용된 것만, used=0 → 미사용만."""
    where = []
    params: list[Any] = []
    if job_id:
        where.append("job_id=?")
        params.append(job_id)
    if platform:
        where.append("platform=?")
        params.append(platform)
    if classification:
        where.append("classification=?")
        params.append(classification)
    if min_views:
        where.append("view_count>=?")
        params.append(min_views)
    if used is not None:
        where.append("COALESCE(used, 0)=?")
        params.append(int(bool(used)))
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM candidate_videos {where_sql} "
            f"ORDER BY COALESCE(is_manual,0) DESC, view_count DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def update_candidate(candidate_id: int, **kwargs) -> None:
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = list(kwargs.values()) + [candidate_id]
    with get_db() as conn:
        conn.execute(f"UPDATE candidate_videos SET {sets} WHERE id=?", values)


def delete_dissection(diss_id: str) -> int:
    """Cascade delete — DB rows + 디스크 파일 (마스코트 폴더, remix 폴더) 모조리.
    형님 룰: 작업 카드 ✕ 누르면 흔적 0.
    Returns # of dissections deleted (0 or 1).
    """
    import shutil
    from pathlib import Path

    with get_db() as conn:
        diss = conn.execute(
            "SELECT related_job_id FROM dissection_analyses WHERE id=?", (diss_id,)
        ).fetchone()
        if not diss:
            return 0
        related_job = diss["related_job_id"]

        # 1. remix files + DB rows (candidate별 — 디스크 폴더 frontend/dist/remixes/remix_<N>/)
        remix_ids: list[int] = []
        if related_job:
            cand_ids = [r[0] for r in conn.execute(
                "SELECT id FROM candidate_videos WHERE job_id=?", (related_job,)
            ).fetchall()]
            if cand_ids:
                ids_sql = ",".join("?" * len(cand_ids))
                remix_ids = [r[0] for r in conn.execute(
                    f"SELECT id FROM remixes WHERE candidate_id IN ({ids_sql})",
                    cand_ids,
                ).fetchall()]
                conn.execute(
                    f"DELETE FROM remixes WHERE candidate_id IN ({ids_sql})",
                    cand_ids,
                )
            conn.execute("DELETE FROM candidate_videos WHERE job_id=?", (related_job,))
            conn.execute("DELETE FROM job_assignments WHERE job_id=?", (related_job,))
            conn.execute("DELETE FROM discovery_jobs WHERE id=?", (related_job,))

        # 2. mascots DB + 디스크 폴더
        conn.execute("DELETE FROM mascots WHERE dissection_id=?", (diss_id,))
        conn.execute("DELETE FROM korean_pool_indexed WHERE dissection_id=?", (diss_id,))
        conn.execute("DELETE FROM korean_pool_channels WHERE dissection_id=?", (diss_id,))

        # 3. dissection 자체
        cur = conn.execute("DELETE FROM dissection_analyses WHERE id=?", (diss_id,))
        n = cur.rowcount
        # commit (with block 끝에서)

    # === 디스크 파일 정리 (DB lock 풀린 후) ===
    repo_root = Path(__file__).parent.parent
    # 마스코트 폴더 — frontend/dist/mascots/<diss_id>/ (baseline + turnaround + options)
    mascot_dir = repo_root / "frontend" / "dist" / "mascots" / diss_id
    if mascot_dir.exists():
        shutil.rmtree(mascot_dir, ignore_errors=True)
    # remix 폴더들 — frontend/dist/remixes/remix_<N>/
    remix_root = repo_root / "frontend" / "dist" / "remixes"
    for rid in remix_ids:
        rdir = remix_root / f"remix_{rid}"
        if rdir.exists():
            shutil.rmtree(rdir, ignore_errors=True)
    return n


def delete_job(job_id: str) -> int:
    """Delete a discovery job + cascade candidate_videos + remixes."""
    with get_db() as conn:
        cand_ids = [r[0] for r in conn.execute(
            "SELECT id FROM candidate_videos WHERE job_id=?", (job_id,)
        ).fetchall()]
        if cand_ids:
            ids_sql = ",".join("?" * len(cand_ids))
            conn.execute(f"DELETE FROM remixes WHERE candidate_id IN ({ids_sql})",
                         cand_ids)
        conn.execute("DELETE FROM candidate_videos WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_assignments WHERE job_id=?", (job_id,))
        cur = conn.execute("DELETE FROM discovery_jobs WHERE id=?", (job_id,))
        return cur.rowcount


def freelancer_report() -> list[dict]:
    """Per-user work summary: assigned jobs, marked-used count, total candidates.
    Used by /api/admin/freelancer-report (Phase 10)."""
    with get_db() as conn:
        users = [dict(r) for r in conn.execute(
            "SELECT id, username, full_name, role FROM users ORDER BY role, created_at"
        ).fetchall()]
        out = []
        for u in users:
            uid = u["id"]
            assigned_jobs = [r["job_id"] for r in conn.execute(
                "SELECT job_id FROM job_assignments WHERE user_id=?", (uid,)
            ).fetchall()]
            # Admin sees all jobs
            if u["role"] == "admin":
                assigned_jobs = [r["id"] for r in conn.execute(
                    "SELECT id FROM discovery_jobs"
                ).fetchall()]
            # Aggregate counts
            used_count = conn.execute(
                "SELECT COUNT(*) FROM candidate_videos WHERE used=1 AND used_by_user_id=?",
                (uid,)
            ).fetchone()[0]
            assigned_total = 0
            if assigned_jobs:
                ids_sql = ",".join("?" * len(assigned_jobs))
                assigned_total = conn.execute(
                    f"SELECT COUNT(*) FROM candidate_videos WHERE job_id IN ({ids_sql})",
                    assigned_jobs,
                ).fetchone()[0]
            # Latest activity
            last_used = conn.execute(
                "SELECT MAX(used_at) FROM candidate_videos WHERE used_by_user_id=?",
                (uid,),
            ).fetchone()[0]
            out.append({
                **u,
                "assigned_jobs": len(assigned_jobs),
                "assigned_candidates": assigned_total,
                "used_count": used_count,
                "last_used_at": last_used,
            })
        return out


def is_video_in_reference(platform: str, video_id: str) -> bool:
    """Check if a video is already in the reference channel's used videos."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM reference_videos WHERE platform=? AND video_id=?",
            (platform, video_id),
        ).fetchone()
        return row is not None


def is_video_reposted_in_korea(platform: str, video_id: str) -> Optional[str]:
    """Check if a video is reposted in Korean channel pool. Returns Korean channel name or None."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT k.channel_id FROM korean_channel_videos kv
               JOIN korean_repost_channels k ON kv.channel_id = k.channel_id
               WHERE kv.original_video_id = ?""",
            (video_id,),
        ).fetchone()
        return row["channel_id"] if row else None


def add_reference_video(channel_id: str, channel_name: str, platform: str,
                        video_id: str, **kwargs) -> None:
    """Add a video to the reference channel blacklist."""
    fields = ["channel_id", "channel_name", "platform", "video_id"] + list(kwargs.keys())
    placeholders = ",".join("?" for _ in fields)
    values = [channel_id, channel_name, platform, video_id] + list(kwargs.values())
    with get_db() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO reference_videos ({','.join(fields)}) "
            f"VALUES ({placeholders})",
            values,
        )


# ===== Visual matching: Korean pool tracking =====

def add_korean_pool_channel(dissection_id: str, channel_id: str,
                            channel_name: str, is_reference: bool,
                            matching_count: int, total_sampled: int,
                            matching_ratio: float) -> None:
    """Record a channel's classification result for a given dissection."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO korean_pool_channels
               (dissection_id, channel_id, channel_name, is_reference,
                matching_count, total_sampled, matching_ratio)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (dissection_id, channel_id, channel_name, 1 if is_reference else 0,
             matching_count, total_sampled, matching_ratio),
        )


def list_pool_reference_channels(dissection_id: str) -> list[dict]:
    """All reference channels for a dissection."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM korean_pool_channels
               WHERE dissection_id=? AND is_reference=1
               ORDER BY matching_count DESC""",
            (dissection_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_korean_pool_video(dissection_id: str | None, channel_id: str,
                          channel_handle: str, channel_name: str,
                          video_id: str, video_url: str, title: str | None,
                          frames_count: int) -> None:
    """Record an indexed video in the Korean pool (mirrors Qdrant)."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO korean_pool_indexed
               (dissection_id, channel_id, channel_handle, channel_name,
                video_id, video_url, title, frames_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (dissection_id, channel_id, channel_handle, channel_name,
             video_id, video_url, title, frames_count),
        )


def is_video_in_pool(video_id: str) -> bool:
    """Cheap pre-check before running CLIP — already indexed?"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM korean_pool_indexed WHERE video_id=?", (video_id,)
        ).fetchone()
        return row is not None


def pool_stats() -> dict:
    """Aggregate stats for the Korean pool."""
    with get_db() as conn:
        videos = conn.execute(
            "SELECT COUNT(*) FROM korean_pool_indexed"
        ).fetchone()[0]
        channels = conn.execute(
            "SELECT COUNT(DISTINCT channel_id) FROM korean_pool_indexed"
        ).fetchone()[0]
        ref_channels = conn.execute(
            "SELECT COUNT(*) FROM korean_pool_channels WHERE is_reference=1"
        ).fetchone()[0]
    return {
        "indexed_videos": videos,
        "indexed_channels": channels,
        "reference_channels": ref_channels,
    }


def get_pool_video(video_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM korean_pool_indexed WHERE video_id=?", (video_id,)
        ).fetchone()
        return dict(row) if row else None


# ===== Search result cache (1-hour TTL by default) =====
# Used by pipeline.py to deduplicate identical keyword searches across runs.
# Same key within TTL → cache hit → quota 0 + deterministic result. Empty
# results are NOT cached so transient failures retry on next call.

from datetime import datetime, timedelta
import hashlib


def _cache_key(source: str, query: str, params: dict | None = None) -> str:
    """Stable cache key from source label + query + sorted params."""
    base = f"{source}::{query}"
    if params:
        items = sorted((str(k), str(v)) for k, v in params.items())
        base += "::" + "&".join(f"{k}={v}" for k, v in items)
    # Hash for compact key
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def cache_get(source: str, query: str, params: dict | None = None) -> Optional[list]:
    """Return cached results if not expired, else None."""
    key = _cache_key(source, query, params)
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT results FROM search_cache WHERE cache_key=? AND expires_at > ?",
            (key, now),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["results"])
    except Exception:
        return None


def cache_set(source: str, query: str, results: list,
              ttl_seconds: int = 3600,
              params: dict | None = None) -> None:
    """Store search results. No-op for empty/falsy results."""
    if not results:
        return
    key = _cache_key(source, query, params)
    now = datetime.utcnow()
    expires = now + timedelta(seconds=ttl_seconds)
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO search_cache
               (cache_key, source, query, results, n_results, cached_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key, source, query[:300],
             json.dumps(results, ensure_ascii=False),
             len(results),
             now.isoformat(), expires.isoformat()),
        )


def cache_purge_expired() -> int:
    """Remove expired cache rows. Call periodically (or on-demand)."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.execute("DELETE FROM search_cache WHERE expires_at <= ?", (now,))
        return cur.rowcount


def cache_stats() -> dict:
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM search_cache WHERE expires_at > ?",
            (datetime.utcnow().isoformat(),),
        ).fetchone()[0]
        by_source = [
            dict(r) for r in conn.execute(
                "SELECT source, COUNT(*) as n FROM search_cache GROUP BY source"
            ).fetchall()
        ]
    return {"total": total, "active": active, "by_source": by_source}


# ===== Mascots (per-dissection 2D 캐릭터) =====

def upsert_mascot(dissection_id: str, **fields) -> dict:
    """Insert or replace mascot for a dissection. Returns the row."""
    fields["dissection_id"] = dissection_id
    fields["updated_at"] = datetime.utcnow().isoformat()
    cols = list(fields.keys())
    placeholders = ",".join("?" for _ in cols)
    values = list(fields.values())
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM mascots WHERE dissection_id=?", (dissection_id,)
        ).fetchone()
        if existing:
            sets = ",".join(f"{c}=?" for c in cols if c != "dissection_id")
            update_vals = [v for c, v in zip(cols, values) if c != "dissection_id"] + [dissection_id]
            conn.execute(f"UPDATE mascots SET {sets} WHERE dissection_id=?", update_vals)
        else:
            conn.execute(
                f"INSERT INTO mascots ({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
    return get_mascot(dissection_id)


def get_mascot(dissection_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM mascots WHERE dissection_id=?", (dissection_id,)
        ).fetchone()
        return dict(row) if row else None


def list_mascots() -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM mascots ORDER BY created_at DESC"
        ).fetchall()]


def upsert_mascot_pair_role(dissection_id: str, role: str, **fields) -> dict:
    """카테고리(dissection)에 savior/victim 1쌍 마스코트 정보 upsert.
    role은 'savior' 또는 'victim'. fields는 prefix 빠진 키 (concept_kr, concept_en,
    character_spec, baseline_path, baseline_url 중 일부).

    legacy mascots 테이블은 style_prompt + seed가 NOT NULL — 새 row INSERT 시
    placeholder 값으로 채워서 webtoon pair만 사용해도 작동.
    """
    if role not in ("savior", "victim"):
        raise ValueError(f"role must be savior or victim, got {role}")
    prefixed = {f"{role}_{k}": v for k, v in fields.items()}
    prefixed["updated_at"] = datetime.utcnow().isoformat()
    cols = list(prefixed.keys())
    values = list(prefixed.values())
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM mascots WHERE dissection_id=?", (dissection_id,)
        ).fetchone()
        if existing:
            sets = ",".join(f"{c}=?" for c in cols)
            conn.execute(f"UPDATE mascots SET {sets} WHERE dissection_id=?",
                         values + [dissection_id])
        else:
            # legacy NOT NULL 컬럼 (style_prompt, seed) placeholder로 채움
            legacy_defaults = {
                "style_prompt": "(webtoon_pair — legacy unused)",
                "seed": 0,
            }
            all_fields = {**legacy_defaults, **prefixed}
            cols_all = ["dissection_id"] + list(all_fields.keys())
            vals_all = [dissection_id] + list(all_fields.values())
            placeholders = ",".join("?" for _ in cols_all)
            conn.execute(
                f"INSERT INTO mascots ({','.join(cols_all)}) VALUES ({placeholders})",
                vals_all,
            )
    return get_mascot(dissection_id)


def get_mascot_pair(dissection_id: str) -> Optional[dict]:
    """savior + victim 1쌍 정보 조회 (legacy — backward compat). 새 시스템은 get_mascot_roles."""
    m = get_mascot(dissection_id)
    if not m:
        return None
    return {
        "dissection_id": dissection_id,
        "savior": {
            "concept_kr": m.get("savior_concept_kr"),
            "concept_en": m.get("savior_concept_en"),
            "character_spec": m.get("savior_character_spec"),
            "baseline_path": m.get("savior_baseline_path"),
            "baseline_url": m.get("savior_baseline_url"),
        },
        "victim": {
            "concept_kr": m.get("victim_concept_kr"),
            "concept_en": m.get("victim_concept_en"),
            "character_spec": m.get("victim_character_spec"),
            "baseline_path": m.get("victim_baseline_path"),
            "baseline_url": m.get("victim_baseline_url"),
        },
    }


# ===== Dynamic N-roles system (자유 archetype, 1~3개) =====

def get_mascot_roles(dissection_id: str) -> list[dict]:
    """동적 N roles list. 없으면 legacy savior/victim 자동 마이그 (read-only)."""
    m = get_mascot(dissection_id)
    if not m:
        return []
    raw = m.get("roles_json")
    if raw:
        try:
            roles = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(roles, list):
                return roles
        except Exception:
            pass
    # legacy → roles list로 가상 변환 (read 시점)
    legacy = []
    for role_id, prefix, label in (("savior", "savior_", "🛡 SAVIOR"),
                                     ("victim", "victim_", "⚠️ VICTIM")):
        if m.get(f"{prefix}baseline_path") or m.get(f"{prefix}concept_kr"):
            legacy.append({
                "role_id": role_id,
                "role_label_kr": label,
                "name_kr": "",
                "concept_kr": m.get(f"{prefix}concept_kr") or "",
                "concept_en": m.get(f"{prefix}concept_en") or "",
                "character_spec": m.get(f"{prefix}character_spec") or "",
                "baseline_path": m.get(f"{prefix}baseline_path"),
                "baseline_url": m.get(f"{prefix}baseline_url"),
            })
    return legacy


def replace_mascot_roles(dissection_id: str, roles: list[dict]) -> dict:
    """전체 roles list 한번에 set (Gemini 추천 결과 적용용)."""
    payload = json.dumps(roles or [], ensure_ascii=False)
    legacy_defaults = {
        "style_prompt": "(webtoon_pair — legacy unused)",
        "seed": 0,
    }
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM mascots WHERE dissection_id=?", (dissection_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE mascots SET roles_json=?, updated_at=? WHERE dissection_id=?",
                (payload, datetime.utcnow().isoformat(), dissection_id),
            )
        else:
            cols = ["dissection_id", "roles_json", "updated_at"] + list(legacy_defaults.keys())
            vals = [dissection_id, payload, datetime.utcnow().isoformat()] + list(legacy_defaults.values())
            placeholders = ",".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO mascots ({','.join(cols)}) VALUES ({placeholders})", vals,
            )
    return {"roles": roles}


def upsert_mascot_role(dissection_id: str, role_id: str, **fields) -> list[dict]:
    """N roles list에서 특정 role_id의 fields만 update (또는 새 role 추가).
    Returns updated roles list.
    """
    if not role_id:
        raise ValueError("role_id required")
    roles = get_mascot_roles(dissection_id) or []
    found = False
    for r in roles:
        if r.get("role_id") == role_id:
            r.update({k: v for k, v in fields.items() if v is not None})
            found = True
            break
    if not found:
        roles.append({"role_id": role_id, **{k: v for k, v in fields.items() if v is not None}})
    replace_mascot_roles(dissection_id, roles)
    return roles


def delete_mascot_role(dissection_id: str, role_id: str) -> list[dict]:
    """특정 role_id 삭제. Returns updated roles list."""
    roles = get_mascot_roles(dissection_id) or []
    roles = [r for r in roles if r.get("role_id") != role_id]
    replace_mascot_roles(dissection_id, roles)
    return roles


# ===== Remixes (AI 변형본 작업 추적) =====

def insert_remix(candidate_id: int, mascot_id: int | None = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO remixes (candidate_id, mascot_id, status) VALUES (?, ?, 'pending')",
            (candidate_id, mascot_id),
        )
        return cur.lastrowid


def update_remix(remix_id: int, **fields) -> None:
    sets = ",".join(f"{k}=?" for k in fields.keys())
    values = list(fields.values()) + [remix_id]
    with get_db() as conn:
        conn.execute(f"UPDATE remixes SET {sets} WHERE id=?", values)


def get_remix(remix_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM remixes WHERE id=?", (remix_id,)).fetchone()
        return dict(row) if row else None


def list_remixes_for_candidate(candidate_id: int) -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM remixes WHERE candidate_id=? ORDER BY created_at DESC",
            (candidate_id,),
        ).fetchall()]


def get_latest_remix_for_candidate(candidate_id: int) -> Optional[dict]:
    """Most recent remix row (used as analysis cache)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM remixes WHERE candidate_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return dict(row) if row else None


def delete_remix(remix_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM remixes WHERE id=?", (remix_id,))


# ===== Subtitle Jobs (자막 자동 생성 작업) =====

def insert_subtitle_job(video_filename: str, video_path: str,
                        original_urls: list = None, user_id: int = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO subtitle_jobs (video_filename, video_path, original_urls, user_id, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (video_filename, video_path,
             json.dumps(original_urls or []), user_id),
        )
        return cur.lastrowid


def update_subtitle_job(job_id: int, **kwargs) -> None:
    if not kwargs:
        return
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ] + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE subtitle_jobs SET {sets} WHERE id=?", values)


def get_subtitle_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM subtitle_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_subtitle_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM subtitle_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM subtitle_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_subtitle_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM subtitle_jobs WHERE id=?", (job_id,))


# ===== 음성 자막 잡 (Whisper large-v3 + Gemini 어투보존 교정 + 검수) =====

def insert_audio_subtitle_job(audio_filename: str, audio_path: str,
                              duration_sec: float = 0, user_id: int = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO audio_subtitle_jobs (audio_filename, audio_path, duration_sec, user_id, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (audio_filename, audio_path, duration_sec, user_id),
        )
        return cur.lastrowid


def update_audio_subtitle_job(job_id: int, **kwargs) -> None:
    if not kwargs:
        return
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = [json.dumps(v) if isinstance(v, (list, dict)) else v
              for v in kwargs.values()] + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE audio_subtitle_jobs SET {sets} WHERE id=?", values)


def get_audio_subtitle_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM audio_subtitle_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_audio_subtitle_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM audio_subtitle_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audio_subtitle_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_audio_subtitle_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM audio_subtitle_jobs WHERE id=?", (job_id,))


# ===== 대본+더빙 (tts_dub) CRUD =====

def insert_tts_dub_job(video_filename: str, video_path: str,
                       user_id: int = None, voice_id: str = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO tts_dub_jobs (video_filename, video_path, user_id, voice_id, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (video_filename, video_path, user_id, voice_id),
        )
        return cur.lastrowid


def update_tts_dub_job(job_id: int, **kwargs) -> None:
    if not kwargs:
        return
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = [json.dumps(v) if isinstance(v, (list, dict)) else v
              for v in kwargs.values()] + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE tts_dub_jobs SET {sets} WHERE id=?", values)


def get_tts_dub_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tts_dub_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_tts_dub_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM tts_dub_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tts_dub_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_tts_dub_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM tts_dub_jobs WHERE id=?", (job_id,))


# ===== 쇼츠 메이커 (긴 URL → 하이라이트 N개) CRUD =====

def insert_shorts_job(url: str, name: str, out_dir: str = None,
                      user_id: int = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO shorts_jobs (url, name, out_dir, user_id, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (url, name, out_dir, user_id),
        )
        return cur.lastrowid


def update_shorts_job(job_id: int, **kwargs) -> None:
    if not kwargs:
        return
    sets = ",".join(f"{k}=?" for k in kwargs.keys())
    values = [
        json.dumps(v) if isinstance(v, (list, dict)) else v
        for v in kwargs.values()
    ] + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE shorts_jobs SET {sets} WHERE id=?", values)


def get_shorts_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM shorts_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_shorts_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id is not None:
            rows = conn.execute(
                "SELECT * FROM shorts_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM shorts_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def list_shorts_jobs_assigned(user_id: int, limit: int = 50) -> list[dict]:
    """프리랜서용 — 자기에게 지정(assigned)됐거나 자기가 만든 쇼츠 작업만."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM shorts_jobs WHERE assigned_user_id=? OR user_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_shorts_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM shorts_jobs WHERE id=?", (job_id,))


# ===== 클립 편집 (여러 영상 → 노래 분위기 구간 추출 → 컷 편집) =====
def insert_clip_edit_job(urls: list, song_title: str,
                          target_duration: int = 50, user_id: int = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO clip_edit_jobs (urls, song_title, target_duration, user_id, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (json.dumps(urls or []), song_title, target_duration, user_id),
        )
        return cur.lastrowid


def update_clip_edit_job(job_id: int, completed_at_now: bool = False, **kwargs) -> None:
    if not kwargs and not completed_at_now:
        return
    sets_parts = []
    values = []
    for k, v in kwargs.items():
        sets_parts.append(f"{k}=?")
        values.append(json.dumps(v) if isinstance(v, (list, dict)) else v)
    if completed_at_now:
        sets_parts.append("completed_at=CURRENT_TIMESTAMP")
    values.append(job_id)
    with get_db() as conn:
        conn.execute(f"UPDATE clip_edit_jobs SET {','.join(sets_parts)} WHERE id=?", values)


def get_clip_edit_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM clip_edit_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_clip_edit_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM clip_edit_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?", (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM clip_edit_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_clip_edit_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM clip_edit_jobs WHERE id=?", (job_id,))



def insert_japanese_multiuse_job(video_filename: str, video_path: str,
                                   user_id: int = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO japanese_multiuse_jobs (video_filename, video_path, user_id, status) "
            "VALUES (?, ?, ?, 'pending')",
            (video_filename, video_path, user_id),
        )
        return cur.lastrowid


def update_japanese_multiuse_job(job_id: int, completed_at_now: bool = False, **kwargs) -> None:
    if not kwargs and not completed_at_now:
        return
    if completed_at_now:
        kwargs["completed_at"] = None  # CURRENT_TIMESTAMP은 직접 set
    sets_parts = []
    values = []
    for k, v in kwargs.items():
        if k == "completed_at" and v is None and completed_at_now:
            sets_parts.append("completed_at=CURRENT_TIMESTAMP")
            continue
        sets_parts.append(f"{k}=?")
        values.append(json.dumps(v) if isinstance(v, (list, dict)) else v)
    sets = ",".join(sets_parts)
    values.append(job_id)
    with get_db() as conn:
        conn.execute(f"UPDATE japanese_multiuse_jobs SET {sets} WHERE id=?", values)


def get_japanese_multiuse_job(job_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM japanese_multiuse_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def list_japanese_multiuse_jobs(user_id: int = None, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM japanese_multiuse_jobs WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM japanese_multiuse_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_japanese_multiuse_job(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM japanese_multiuse_jobs WHERE id=?", (job_id,))
