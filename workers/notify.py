"""Telegram 알림 — 서버 에러/완료 시 등록된 모든 사용자에게 메시지.

수신자:
  - .env의 TELEGRAM_CHAT_ID (대표님)
  - 자료의 telegram_users 테이블에 등록된 모든 사용자 (중간관리자)
"""
import os
import sqlite3
import httpx
import asyncio
from pathlib import Path

DB_PATH = Path("/Users/shortsking/banbaji-discover/db/discover.db")


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


def _ensure_telegram_users_table() -> None:
    """telegram_users 테이블이 없으면 만듦."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telegram_users (
                chat_id TEXT PRIMARY KEY,
                name TEXT,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                added_by TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_all_chat_ids() -> list[str]:
    """알림 받을 모든 텔레그램 ID — .env의 대표님 + 자료의 추가 사용자."""
    _ensure_telegram_users_table()
    ids: set[str] = set()
    root = _chat_id()
    if root:
        ids.add(str(root))
    try:
        conn = sqlite3.connect(str(DB_PATH))
        for row in conn.execute("SELECT chat_id FROM telegram_users"):
            ids.add(str(row[0]))
        conn.close()
    except Exception:
        pass
    return sorted(ids)


async def send_telegram(text: str, silent: bool = False) -> bool:
    """등록된 모든 사용자에게 메시지 보냄. 한 명이라도 성공하면 True."""
    token = _token()
    if not token:
        return False
    chat_ids = get_all_chat_ids()
    if not chat_ids:
        return False
    success_any = False
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            for cid in chat_ids:
                try:
                    r = await c.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={
                            "chat_id": cid,
                            "text": text[:4000],
                            "disable_notification": silent,
                            "parse_mode": "HTML",
                        },
                    )
                    if r.status_code == 200:
                        success_any = True
                except Exception as e:
                    print(f"[telegram] send to {cid} failed: {e}", flush=True)
    except Exception as e:
        print(f"[telegram] send failed: {e}", flush=True)
    return success_any


def send_telegram_sync(text: str, silent: bool = False) -> bool:
    """sync wrapper — async 안 쓰는 곳에서 호출."""
    try:
        return asyncio.run(send_telegram(text, silent))
    except RuntimeError:
        # 이미 event loop 안에 있으면 task 생성
        loop = asyncio.get_event_loop()
        asyncio.create_task(send_telegram(text, silent))
        return True


# 편의 함수
async def notify_error(context: str, error: Exception | str):
    """에러 발생 시. ⚠️ 아이콘. 에러 빈도 추적용 카운터 자동 증가."""
    # 에러 빈도 추적 — 5분에 5개 이상이면 자가 진단이 시스템 알림 발송
    try:
        from . import disk_janitor
        disk_janitor.record_error(context)
    except Exception:
        pass
    msg = f"⚠️ <b>{context}</b>\n\n<code>{str(error)[:1000]}</code>"
    await send_telegram(msg)


async def notify_success(context: str, detail: str = ""):
    """성공 시. ✅ 아이콘. silent=True (배지 알림만)."""
    msg = f"✅ <b>{context}</b>"
    if detail:
        msg += f"\n{detail[:500]}"
    await send_telegram(msg, silent=True)


async def notify_progress(context: str, detail: str = ""):
    """진행 중 알림. 🔄 아이콘. silent."""
    msg = f"🔄 {context}"
    if detail:
        msg += f"\n{detail[:300]}"
    await send_telegram(msg, silent=True)
