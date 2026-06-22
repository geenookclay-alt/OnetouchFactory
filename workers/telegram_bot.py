"""Telegram 양방향 봇 — 등록된 사용자들이 명령 보내면 시스템 실행 + 답장.

수신 사용자:
  - 대표님: .env의 TELEGRAM_CHAT_ID — 모든 명령 + 사용자 관리 (/adduser 등)
  - 중간관리자: 자료에 등록된 사용자 — 모든 명령 (사용자 관리 빼고)
  - 미등록 사용자: /whoami로 자기 ID 확인만 가능, 다른 명령은 차단

지원 명령:
  /help      — 명령 list
  /status    — 진행 작업 + 최근 에러
  /logs      — 최근 에러 로그 20줄
  /disk      — 디스크 사용량
  /restart   — 서버 강제 재시작
  /cancel    — 진행 중 ffmpeg + yt-dlp 작업 종료
  /whoami    — 자기 텔레그램 ID + 권한 확인 (누구나)
  /users     — 등록된 사용자 목록 (대표님만)
  /adduser   — 사용자 등록 (대표님만)
  /removeuser — 사용자 삭제 (대표님만)
"""
import os
import asyncio
import httpx
import subprocess
import sqlite3
from pathlib import Path

_LAST_UPDATE_ID = 0
DB_PATH = Path("/Users/shortsking/banbaji-discover/db/discover.db")


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


def _is_root(chat_id) -> bool:
    """.env의 TELEGRAM_CHAT_ID와 일치하면 대표님."""
    root = _chat_id()
    return bool(root) and str(chat_id) == str(root)


def _is_authorized(chat_id) -> bool:
    """등록된 사용자 여부 — 대표님 OR 자료에 있는 사용자."""
    if _is_root(chat_id):
        return True
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT 1 FROM telegram_users WHERE chat_id=?", (str(chat_id),)
        ).fetchone()
        conn.close()
        return bool(row)
    except Exception:
        return False


def _get_user_name(chat_id) -> str:
    """등록된 사용자 이름 — 대표님은 '대표님', 추가 사용자는 자료에서."""
    if _is_root(chat_id):
        return "대표님"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT name FROM telegram_users WHERE chat_id=?", (str(chat_id),)
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


# ===== 명령 핸들러 =====

async def cmd_help(args: list[str], chat_id=None) -> str:
    is_root = _is_root(chat_id) if chat_id else False
    msg = (
        "🤖 <b>Banbaji 알림 봇 — 사용 가능 명령</b>\n\n"
        "<b>/status</b> — 진행 작업 + 최근 에러 요약\n"
        "<b>/logs</b> — 최근 에러 로그 20줄\n"
        "<b>/disk</b> — 디스크 사용량\n"
        "<b>/restart</b> — 서버 강제 재시작\n"
        "<b>/cancel</b> — 진행 중 ffmpeg + yt-dlp 작업 강제 종료\n"
        "<b>/whoami</b> — 자기 텔레그램 ID + 권한 확인\n"
        "<b>/help</b> — 이 메시지\n"
    )
    if is_root:
        msg += (
            "\n<b>👑 대표님 전용 명령</b>\n"
            "<b>/users</b> — 등록된 사용자 목록\n"
            "<b>/adduser ID 이름</b> — 새 사용자 등록\n"
            "<b>/removeuser ID</b> — 사용자 삭제\n"
        )
    msg += "\n💡 50~70%의 에러는 /restart로 해결됨"
    return msg


async def cmd_whoami(args: list[str], chat_id=None) -> str:
    if _is_root(chat_id):
        badge = "👑 대표님 (root)"
    elif _is_authorized(chat_id):
        name = _get_user_name(chat_id) or "이름 없음"
        badge = f"👤 {name} (중간관리자)"
    else:
        return (
            f"❌ <b>미등록 사용자</b>\n\n"
            f"<b>당신의 텔레그램 ID</b>: <code>{chat_id}</code>\n\n"
            f"이 ID를 대표님께 알려주세요. 대표님이 등록해주면 알림 받기 + 명령 사용 가능."
        )
    return (
        f"{badge}\n"
        f"<b>텔레그램 ID</b>: <code>{chat_id}</code>"
    )


async def cmd_adduser(args: list[str], chat_id=None) -> str:
    if not _is_root(chat_id):
        return "⚠️ 대표님만 사용 가능"
    if len(args) < 2:
        return (
            "사용법: <code>/adduser 텔레그램ID 이름</code>\n"
            "예시: <code>/adduser 123456789 홍길동</code>\n\n"
            "💡 대상자가 먼저 봇과 대화 시작 (/start 또는 /whoami) → 자기 ID 확인 → 대표님께 알림"
        )
    target_id = args[0].strip()
    name = " ".join(args[1:]).strip()
    if not target_id.lstrip("-").isdigit():
        return f"⚠️ 텔레그램 ID는 숫자여야 함: {target_id}"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telegram_users (
                chat_id TEXT PRIMARY KEY, name TEXT,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP, added_by TEXT
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO telegram_users (chat_id, name, added_by) VALUES (?, ?, ?)",
            (target_id, name, str(chat_id)),
        )
        conn.commit()
        conn.close()
        # 대상자에게 환영 메시지 (가능하면)
        token = _token()
        if token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    await c.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={
                            "chat_id": target_id,
                            "text": (
                                f"✅ <b>등록 완료 — {name}</b>\n\n"
                                "이제 알림 받기 + 모든 명령 사용 가능합니다.\n"
                                "/help 입력하면 명령 목록 보임."
                            ),
                            "parse_mode": "HTML",
                        },
                    )
            except Exception:
                pass
        return f"✅ <b>{name}</b> (ID=<code>{target_id}</code>) 등록됨\n알림 받기 + 모든 명령 사용 가능"
    except Exception as e:
        return f"⚠️ 등록 실패: {e}"


async def cmd_removeuser(args: list[str], chat_id=None) -> str:
    if not _is_root(chat_id):
        return "⚠️ 대표님만 사용 가능"
    if not args:
        return "사용법: <code>/removeuser 텔레그램ID</code>"
    target_id = args[0].strip()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.execute(
            "DELETE FROM telegram_users WHERE chat_id=?", (target_id,)
        )
        conn.commit()
        conn.close()
        if cur.rowcount > 0:
            return f"✅ ID <code>{target_id}</code> 삭제됨"
        return f"⚠️ ID <code>{target_id}</code> 등록 안 돼있음"
    except Exception as e:
        return f"⚠️ 삭제 실패: {e}"


async def cmd_users(args: list[str], chat_id=None) -> str:
    if not _is_root(chat_id):
        return "⚠️ 대표님만 사용 가능"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telegram_users (
                chat_id TEXT PRIMARY KEY, name TEXT,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP, added_by TEXT
            )
        """)
        rows = conn.execute(
            "SELECT chat_id, name, added_at FROM telegram_users ORDER BY added_at"
        ).fetchall()
        conn.close()
        root = _chat_id()
        lines = [f"📋 <b>등록 사용자</b>"]
        lines.append(f"👑 대표님 <code>{root}</code> (root)")
        if not rows:
            lines.append("\n중간관리자: 없음")
        else:
            lines.append(f"\n중간관리자 ({len(rows)}명):")
            for r in rows:
                cid, name, added = r
                date_str = (added or "")[:10]
                lines.append(f"👤 {name} <code>{cid}</code> (등록 {date_str})")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 조회 실패: {e}"


async def cmd_status(args: list[str]) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "ps aux | grep -E 'ffmpeg|yt-dlp|python.*workers' | grep -v grep | wc -l",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        active = int(out.decode().strip() or 0)
    except Exception:
        active = 0

    try:
        proc = await asyncio.create_subprocess_shell(
            "tail -300 /Users/shortsking/banbaji-discover/logs/server.log "
            "| grep -iE 'error|fail|⚠️|exception|traceback' | tail -3",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        errors = out.decode().strip()
    except Exception:
        errors = ""

    # uvicorn 살아있는지
    try:
        proc = await asyncio.create_subprocess_shell(
            "curl -s -m 3 http://localhost:8000/health || echo 'DOWN'",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        server_status = "✅ 정상" if "ok" in out.decode() else "❌ 죽음 (/restart 필요)"
    except Exception:
        server_status = "❓ 확인 불가"

    msg = f"📊 <b>현재 상태</b>\n\n서버: {server_status}\n활성 작업: {active}개"
    if errors:
        msg += f"\n\n<b>최근 에러:</b>\n<code>{errors[:1500]}</code>"
    else:
        msg += "\n\n✅ 최근 에러 없음"
    return msg


async def cmd_logs(args: list[str]) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "tail -500 /Users/shortsking/banbaji-discover/logs/server.log "
            "| grep -iE 'error|fail|⚠️|exception|traceback|raise' | tail -15",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        logs = out.decode().strip() or "에러 없음"
    except Exception as e:
        logs = f"확인 실패: {e}"
    return f"📋 <b>최근 에러 로그</b>\n\n<code>{logs[:3500]}</code>"


async def cmd_disk(args: list[str]) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "df -h /Users/shortsking/banbaji-discover | tail -1 "
            "| awk '{print $3 \" / \" $2 \" (\" $5 \" 사용)\"}'",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        disk = out.decode().strip()

        proc = await asyncio.create_subprocess_shell(
            "du -sh /Users/shortsking/banbaji-discover/data/originals/ 2>/dev/null "
            "| awk '{print $1}'",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        cache = out.decode().strip() or "0"

        proc = await asyncio.create_subprocess_shell(
            "du -sh /Users/shortsking/banbaji-discover/frontend/dist/remixes/ 2>/dev/null "
            "| awk '{print $1}'",
            stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        remixes = out.decode().strip() or "0"
    except Exception as e:
        return f"확인 실패: {e}"
    return (f"💾 <b>디스크 사용</b>\n\n"
            f"전체: {disk}\n"
            f"영상 cache: {cache}\n"
            f"합본/클립 결과: {remixes}")


async def cmd_restart(args: list[str]) -> str:
    asyncio.create_task(_do_restart())
    return "🔄 재시작 시작됨 — 10초 후 /status로 확인"


async def _do_restart():
    """uvicorn 재시작. port 8000 완전히 release될 때까지 대기 후 새 process 시작.
    옛 흐름은 pkill만 했는데 자식 process 또는 TIME_WAIT 으로 port 잡혀있는 케이스
    있어서 [Errno 48] Address already in use 떴음 → lsof + port 비대기 추가."""
    await asyncio.sleep(2)
    subprocess.Popen(
        ["bash", "-c",
         # 1. uvicorn + 자식 process 다 죽이기 (여러 패턴)
         "pkill -9 -f 'uvicorn.*api.main' 2>/dev/null; "
         "pkill -9 -f 'multiprocessing' 2>/dev/null; "
         "pkill -9 -f 'api.main:app' 2>/dev/null; "
         # 2. port 8000 잡은 process 강제 kill (lsof 기반 — TIME_WAIT 회피)
         "lsof -ti :8000 2>/dev/null | xargs -r kill -9 2>/dev/null; "
         # 3. port 빌 때까지 대기 (최대 15초)
         "for i in $(seq 1 15); do "
         "  if ! lsof -i:8000 -t > /dev/null 2>&1; then break; fi; "
         "  sleep 1; "
         "done; "
         "sleep 2; "
         # 4. 새 uvicorn 시작 (--reload 금지 — reload hang이 사이트 먹통 원인)
         "cd /Users/shortsking/banbaji-discover && "
         "nohup venv/bin/uvicorn api.main:app "
         "--host 0.0.0.0 --port 8000 "
         "> logs/server.log 2>&1 < /dev/null &"],
        start_new_session=True
    )


async def cmd_cancel(args: list[str]) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            "pkill -f 'ffmpeg' 2>/dev/null; "
            "pkill -f 'yt-dlp' 2>/dev/null; "
            "echo done"
        )
        await proc.wait()
    except Exception:
        pass
    return "🛑 진행 중 ffmpeg + yt-dlp 작업 종료됨"


COMMANDS = {
    "/help": cmd_help,
    "/start": cmd_help,
    "/status": cmd_status,
    "/logs": cmd_logs,
    "/disk": cmd_disk,
    "/restart": cmd_restart,
    "/cancel": cmd_cancel,
    "/whoami": cmd_whoami,
    "/help": cmd_help,
    "/users": cmd_users,
    "/adduser": cmd_adduser,
    "/removeuser": cmd_removeuser,
}

# 등록된 사용자만 사용 가능 (미등록은 /whoami만 가능)
ANYONE_COMMANDS = {"/whoami", "/help", "/start"}


async def handle_message(text: str, chat_id) -> str:
    if not text:
        return "/help 입력하면 명령 목록 보임"
    parts = text.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    # 미등록 사용자 — /whoami만 허용
    if not _is_authorized(chat_id) and cmd not in ANYONE_COMMANDS:
        # 대표님께 가입 요청 자동 알림 (대상이 대표님 본인이 아닌 경우)
        try:
            from . import notify
            await notify.send_telegram(
                f"📥 <b>봇 가입 요청</b>\n"
                f"ID: <code>{chat_id}</code>\n"
                f"메시지: <code>{text[:200]}</code>\n\n"
                f"등록: <code>/adduser {chat_id} 이름</code>"
            )
        except Exception:
            pass
        return (
            f"❌ <b>미등록 사용자</b>\n"
            f"당신 ID: <code>{chat_id}</code>\n\n"
            f"대표님께 이 ID 알려주세요. 등록 후 명령 사용 가능합니다.\n"
            f"(대표님께 가입 요청 알림 전송됨)"
        )

    handler = COMMANDS.get(cmd)
    if handler:
        try:
            # chat_id를 받는 핸들러인지 체크
            import inspect
            sig = inspect.signature(handler)
            if "chat_id" in sig.parameters:
                return await handler(args, chat_id=chat_id)
            return await handler(args)
        except Exception as e:
            return f"⚠️ 명령 실행 실패: {e}"
    return f"❓ 알 수 없는 명령: <code>{cmd}</code>\n/help 입력하면 명령 목록 보임"


async def poll_loop():
    """Telegram long polling — 백엔드 startup 시 백그라운드로 시작."""
    global _LAST_UPDATE_ID
    token = _token()
    chat_id_filter = _chat_id()
    if not token:
        print("[telegram-bot] TELEGRAM_BOT_TOKEN 없음 — 양방향 안 함", flush=True)
        return
    print(f"[telegram-bot] polling 시작 (chat_id={chat_id_filter})", flush=True)

    # 시작 시 startup 메시지 (silent)
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id_filter,
                    "text": "🤖 봇 polling 시작됨. /help 입력하면 명령 list",
                    "disable_notification": True,
                    "parse_mode": "HTML",
                },
            )
    except Exception:
        pass

    while True:
        try:
            async with httpx.AsyncClient(timeout=35.0) as c:
                params = {"timeout": 30}
                if _LAST_UPDATE_ID > 0:
                    params["offset"] = _LAST_UPDATE_ID + 1
                r = await c.get(f"https://api.telegram.org/bot{token}/getUpdates",
                                params=params)
                if r.status_code != 200:
                    await asyncio.sleep(5)
                    continue
                data = r.json()
                for u in data.get("result", []):
                    _LAST_UPDATE_ID = max(_LAST_UPDATE_ID, u.get("update_id", 0))
                    msg = u.get("message") or {}
                    chat = msg.get("chat") or {}
                    text = msg.get("text", "")
                    if not text:
                        continue
                    cid = chat.get("id")
                    auth_badge = "👑" if _is_root(cid) else (
                        "👤" if _is_authorized(cid) else "❌"
                    )
                    print(f"[telegram-bot] {auth_badge} {cid}: {text}", flush=True)
                    # 대표님(root) 자유 텍스트(명령 아님) = 의견 → inbox 기록(Claude 세션이 읽어 반영) + ack
                    if _is_root(cid) and not text.strip().startswith("/"):
                        try:
                            import json as _json
                            with open("/tmp/tg_inbox.jsonl", "a", encoding="utf-8") as _ib:
                                _ib.write(_json.dumps({"update_id": u.get("update_id"), "text": text}, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        reply = "✅ 의견 받았습니다 — 반영해서 새 영상 보내드릴게요"
                    else:
                        reply = await handle_message(text, cid)
                    try:
                        await c.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            json={"chat_id": cid, "text": reply,
                                  "parse_mode": "HTML"},
                        )
                    except Exception as e:
                        print(f"[telegram-bot] send reply failed: {e}", flush=True)
        except httpx.ReadTimeout:
            pass  # long polling timeout = 정상
        except Exception as e:
            print(f"[telegram-bot] poll error: {e}", flush=True)
            await asyncio.sleep(5)
