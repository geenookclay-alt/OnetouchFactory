"""Authentication: bcrypt password hashing + JWT tokens.

Two-tier auth model:
- admin (양봉여리/형님): full access — manages categories and freelancers
- freelancer: sees only the discovery jobs assigned to them, can mark
  candidates as 사용함

Backward compat: legacy X-API-Key (BACKEND_API_KEY env) still authenticates
as a synthetic admin. The frontend should migrate to JWT login.
"""
import os
import secrets
from datetime import datetime, timezone, timedelta

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Query

from . import database as db


JWT_ALG = "HS256"
TOKEN_TTL_HOURS = 24 * 30  # 30일 — PWA 운영 편의

_JWT_SECRET = os.getenv("JWT_SECRET")
if not _JWT_SECRET:
    # .env 파일에서 직접 읽기 — uvicorn이 .env를 환경변수로 안 불러와도 동작
    # (이게 없어서 restart마다 랜덤 시크릿 → 전원 로그아웃 버그였음)
    try:
        import os.path as _op
        _env = _op.join(_op.dirname(_op.dirname(os.path.abspath(__file__))), ".env")
        if _op.exists(_env):
            with open(_env, encoding="utf-8") as _f:
                for _line in _f:
                    if _line.startswith("JWT_SECRET="):
                        _JWT_SECRET = _line.split("=", 1)[1].strip()
                        break
    except Exception:
        pass
if not _JWT_SECRET:
    _JWT_SECRET = secrets.token_urlsafe(64)
    print("⚠️  JWT_SECRET 못 찾음 — 랜덤 시크릿 (restart마다 로그아웃됨). .env에 JWT_SECRET 넣으세요.")
else:
    print("🔐 JWT_SECRET 로드됨 (로그인 유지)", flush=True)


# ===== Password hashing =====

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode("utf-8"), bcrypt.gensalt(rounds=12)
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ===== JWT =====

def create_token(user_id: int, username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=TOKEN_TTL_HOURS)).timestamp()),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "토큰 만료 — 다시 로그인")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "유효하지 않은 토큰")


# ===== User CRUD =====

def create_user(username: str, password: str, role: str = "freelancer",
                full_name: str | None = None) -> dict:
    pwh = hash_password(password)
    with db.get_db() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) "
            "VALUES (?, ?, ?, ?)",
            (username, pwh, role, full_name),
        )
    user = get_user_by_username(username)
    assert user is not None
    return user


def get_user_by_username(username: str) -> dict | None:
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def update_last_login(user_id: int) -> None:
    import sqlite3
    try:
        with db.get_db() as conn:
            conn.execute(
                "UPDATE users SET last_login_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), user_id),
            )
    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            return
        raise


def update_password(user_id: int, new_password: str) -> None:
    pwh = hash_password(new_password)
    with db.get_db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (pwh, user_id)
        )


def list_users() -> list[dict]:
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, full_name, role_label, created_at, last_login_at, gemini_api_key, typecast_api_key "
            "FROM users ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user(user_id: int) -> None:
    # Delete dependent rows FIRST (job_assignments has FK → users.id with no
    # ON DELETE CASCADE), otherwise PRAGMA foreign_keys=ON raises IntegrityError.
    with db.get_db() as conn:
        conn.execute("DELETE FROM job_assignments WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))


# ===== Job assignments (which freelancers see which jobs) =====

def assign_job(job_id: str, user_id: int, assigned_by: int) -> None:
    with db.get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO job_assignments (job_id, user_id, assigned_by) "
            "VALUES (?, ?, ?)",
            (job_id, user_id, assigned_by),
        )


def unassign_job(job_id: str, user_id: int) -> None:
    with db.get_db() as conn:
        conn.execute(
            "DELETE FROM job_assignments WHERE job_id=? AND user_id=?",
            (job_id, user_id),
        )


def list_assigned_jobs(user_id: int) -> list[str]:
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT job_id FROM job_assignments WHERE user_id=?", (user_id,)
        ).fetchall()
        return [r["job_id"] for r in rows]


def list_job_freelancers(job_id: str) -> list[dict]:
    with db.get_db() as conn:
        rows = conn.execute(
            """SELECT u.id, u.username, u.full_name, u.role, ja.assigned_at
               FROM job_assignments ja JOIN users u ON ja.user_id = u.id
               WHERE ja.job_id=?""",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def can_user_see_job(user: dict, job_id: str) -> bool:
    """Admins see everything. Freelancers only see assigned jobs."""
    if user.get("role") == "admin":
        return True
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM job_assignments WHERE job_id=? AND user_id=?",
            (job_id, user.get("id")),
        ).fetchone()
        return row is not None


# ===== Bootstrap admin from env =====

def bootstrap_admin() -> None:
    """Create initial admin from env vars on startup if no admin exists."""
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        return
    admin_username = os.getenv("ADMIN_USERNAME") or "admin"
    full_name = os.getenv("ADMIN_FULL_NAME") or "양봉여리"

    with db.get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE role='admin' LIMIT 1"
        ).fetchone()
    if existing:
        return
    create_user(admin_username, admin_password, role="admin",
                full_name=full_name)
    print(f"✅ Admin created: {admin_username}")


# ===== FastAPI dependencies =====

async def authenticate(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    token: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
) -> dict:
    """Resolve current user from Bearer JWT, legacy X-API-Key, or query token.
    Query (?token= / ?api_key=) is needed for <img>/<video> tags that can't
    send headers. Use only for read-only endpoints (preview thumbnails etc.).
    """
    # [배포판] SOLO_MODE: 개인용 — 로그인 없이 관리자 권한
    if os.getenv("SOLO_MODE", "") == "1":
        return {"id": 1, "username": "owner", "role": "admin",
                "full_name": "사장님", "features": ["subtitle", "ttsdub", "clip"],
                "requires_personal_key": False, "has_api_key": False,
                "has_typecast_key": False}

    backend_key = os.getenv("BACKEND_API_KEY", "")

    if authorization and authorization.lower().startswith("bearer "):
        token_str = authorization.split(" ", 1)[1].strip()
    elif token:
        token_str = token
    else:
        token_str = None

    if token_str:
        payload = decode_token(token_str)
        try:
            uid = int(payload.get("sub") or 0)
        except ValueError:
            raise HTTPException(401, "토큰 형식 오류")
        user = get_user_by_id(uid)
        if not user:
            raise HTTPException(401, "사용자를 찾을 수 없음")
        return user

    key_str = x_api_key or api_key
    if key_str and backend_key and key_str == backend_key:
        return {"id": 0, "username": "_legacy_api_key_", "role": "admin",
                "full_name": "Legacy"}

    raise HTTPException(401, "인증 필요 (Bearer JWT 또는 X-API-Key)")


async def admin_only(user: dict = Depends(authenticate)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "관리자 전용")
    return user


def _user_features(user: dict) -> list:
    """user dict의 features (JSON 문자열 또는 list) → list."""
    feats = user.get("features")
    if isinstance(feats, str):
        try:
            import json as _j
            feats = _j.loads(feats)
        except Exception:
            feats = []
    return feats or []


def require_feature(feature: str):
    """admin 또는 해당 기능 권한 가진 사용자만 통과.
    사용: Depends(require_feature("subtitle"))
    """
    async def checker(user: dict = Depends(authenticate)) -> dict:
        if user.get("role") == "admin":
            return user
        if feature in _user_features(user):
            return user
        raise HTTPException(403, f"{feature} 기능 권한 없음")
    return checker
