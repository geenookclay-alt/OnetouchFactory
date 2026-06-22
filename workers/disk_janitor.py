"""디스크 자동 정리 + 모니터링.

매 시간:
- 영상 cache (data/originals/) 7일 이상 안 쓴 파일 삭제
- 디스크 사용량 80% 넘으면 텔레그램 알림
- 90% 넘으면 옛 영상 강제 정리

백엔드 startup 시 백그라운드 task로 시작.
"""
import os
import asyncio
import shutil
from pathlib import Path
import time

ORIGINALS_DIR = Path("/Users/shortsking/banbaji-discover/data/originals")
REMIXES_DIR = Path("/Users/shortsking/banbaji-discover/frontend/dist/remixes")
DISK_PATH = Path("/Users/shortsking/banbaji-discover")


async def _disk_usage_pct() -> int:
    """디스크 사용률 (%)."""
    try:
        usage = shutil.disk_usage(str(DISK_PATH))
        return int(usage.used / usage.total * 100)
    except Exception:
        return 0


async def _disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage(str(DISK_PATH))
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


async def _cleanup_old_originals(max_age_days: int = 7) -> tuple[int, int]:
    """7일 이상 안 쓴 영상 cache 삭제. (삭제 개수, 절약 MB) 반환."""
    if not ORIGINALS_DIR.exists():
        return 0, 0
    cutoff = time.time() - max_age_days * 86400
    deleted = 0
    saved_bytes = 0
    for f in ORIGINALS_DIR.glob("*.mp4"):
        try:
            stat = f.stat()
            atime = stat.st_atime  # access time
            if atime < cutoff:
                saved_bytes += stat.st_size
                f.unlink()
                deleted += 1
        except Exception:
            pass
    return deleted, saved_bytes // (1024 * 1024)


async def _aggressive_cleanup() -> tuple[int, int]:
    """디스크 90%+ 시 — 3일 이상 안 쓴 영상 다 삭제."""
    return await _cleanup_old_originals(max_age_days=3)


async def _scan_corrupt_videos(root: Path, pattern: str = "*.mp4") -> list[tuple[Path, str]]:
    """디렉터리 안 mp4들 중 video stream이 format 길이의 절반 미만인 거 찾아서 list 반환.
    partial 다운/cut 짤림 검출용. 5초 이상 영상만 검사 (짧은 거는 오차 큼)."""
    import json
    bad = []
    if not root.exists():
        return bad
    for f in root.rglob(pattern):
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,duration", "-show_entries",
                "format=duration", "-of", "json", str(f),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            d = json.loads(out.decode() or "{}")
            vid_dur = 0.0
            for s in d.get("streams", []):
                if s.get("codec_type") == "video":
                    vid_dur = float(s.get("duration") or 0)
                    break
            fmt_dur = float(d.get("format", {}).get("duration") or 0)
            if fmt_dur >= 5.0 and vid_dur > 0 and vid_dur < fmt_dur * 0.5:
                bad.append((f, f"video={vid_dur:.1f}s fmt={fmt_dur:.1f}s"))
        except Exception:
            pass
    return bad


async def _scan_and_delete_corrupt() -> tuple[int, list[str]]:
    """망가진 cache + 합본 결과물 자동 삭제. (삭제 개수, 파일명 list) 반환."""
    bad_files: list[tuple[Path, str]] = []
    bad_files += await _scan_corrupt_videos(ORIGINALS_DIR)
    if REMIXES_DIR.exists():
        bad_files += await _scan_corrupt_videos(REMIXES_DIR, "combined.mp4")
    # spec 대비 길이가 짧은 합본 검출 (segment 빠짐)
    bad_files += await _scan_truncated_combined()
    names = []
    for f, reason in bad_files:
        try:
            sidecar = f.with_name(f.stem + ".f140.m4a")
            if sidecar.exists():
                sidecar.unlink()
            f.unlink()
            names.append(f"{f.parent.name}/{f.name} ({reason})")
        except Exception:
            pass
    return len(names), names


async def _scan_truncated_combined() -> list[tuple[Path, str]]:
    """spec.clips + 원본 영상 길이 대비 합본이 짧은 경우 검출 (tail/segment 빠진 합본).
    DB의 spec과 candidate duration 기준 expected vs 실제 비교."""
    import sqlite3, json, subprocess
    db_path = Path("/Users/shortsking/banbaji-discover/db/discover.db")
    if not db_path.exists() or not REMIXES_DIR.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return []
    bad = []
    try:
        for combined in REMIXES_DIR.glob("remix_*/combined.mp4"):
            try:
                remix_id = int(combined.parent.name.replace("remix_", ""))
            except Exception:
                continue
            try:
                row = conn.execute(
                    "SELECT r.spec, c.duration FROM remixes r "
                    "LEFT JOIN candidate_videos c ON c.id=r.candidate_id "
                    "WHERE r.id=?", (remix_id,)
                ).fetchone()
            except Exception:
                continue
            if not row or not row["spec"]:
                continue
            try:
                spec = json.loads(row["spec"])
                clips = spec.get("clips") or []
                orig_dur = float(row["duration"] or 0)
                if not clips or orig_dur <= 0:
                    continue
                clip_total = sum(float(c.get("duration_sec", 2.5)) for c in clips)
                expected = orig_dur + clip_total
                proc = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=nw=1:nk=1", str(combined),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                actual = float(out.decode().strip() or 0)
                if actual > 0 and expected - actual > 2.0:
                    bad.append((combined,
                                f"segment 빠짐 (기대 {expected:.1f}s, 실제 {actual:.1f}s)"))
            except Exception:
                continue
    finally:
        conn.close()
    return bad


async def backup_mascot_data() -> str | None:
    """매 1시간 마스코트 자료 통째로 백업. 25시간 이상 된 옛 백업은 자동 정리."""
    return await _backup_table("mascots")


async def backup_all_critical_data() -> dict[str, str]:
    """핵심 자료 테이블 전체 백업 — 마스코트 + 작업 + 영상 후보 + 합본.
    혹시 자료 손실되면 백업 파일들로 복원 가능."""
    tables = ["mascots", "dissection_analyses", "remixes", "candidate_videos"]
    results = {}
    for t in tables:
        path = await _backup_table(t)
        if path:
            results[t] = path
    return results


async def _backup_table(table_name: str) -> str | None:
    """특정 테이블 통째로 json 백업. 25시간 이상 된 옛 백업은 자동 정리."""
    import sqlite3, json, time
    BACKUP_DIR = Path("/Users/shortsking/banbaji-discover/data/backups") / table_name
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    db_path = Path("/Users/shortsking/banbaji-discover/db/discover.db")
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        ts = time.strftime("%Y%m%d_%H%M")
        backup_file = BACKUP_DIR / f"{table_name}_{ts}.json"
        backup_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cutoff = time.time() - 25 * 3600
        for old in BACKUP_DIR.glob(f"{table_name}_*.json"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except Exception:
                pass
        return str(backup_file)
    except Exception as e:
        print(f"[backup] {table_name} 백업 실패: {e}", flush=True)
        return None


# 에러 빈도 추적용 (메모리 내 카운터)
_recent_errors: list[float] = []


def record_error(context: str = ""):
    """에러 발생 시 호출 — 5분 이내 5개 이상이면 텔레그램 알림."""
    import time
    now = time.time()
    cutoff = now - 300  # 5분
    # 오래된 거 제거
    _recent_errors[:] = [t for t in _recent_errors if t > cutoff]
    _recent_errors.append(now)


async def check_error_rate() -> int:
    """최근 5분 에러 개수 반환. 5개 이상이면 알림."""
    import time
    cutoff = time.time() - 300
    _recent_errors[:] = [t for t in _recent_errors if t > cutoff]
    return len(_recent_errors)


async def health_check() -> dict:
    """시스템 상태 점검 — 디스크, 봇 프로세스, 자료 접근, 외부 API.
    문제 발견 시 텔레그램 알림."""
    import sqlite3
    status = {}
    # 디스크
    try:
        pct = await _disk_usage_pct()
        free = await _disk_free_gb()
        status["disk"] = {"ok": pct < 90, "pct": pct, "free_gb": round(free, 1)}
    except Exception as e:
        status["disk"] = {"ok": False, "error": str(e)}
    # 자료 접근
    try:
        db_path = Path("/Users/shortsking/banbaji-discover/db/discover.db")
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        status["db"] = {"ok": True}
    except Exception as e:
        status["db"] = {"ok": False, "error": str(e)}
    # 에러 빈도
    err_count = await check_error_rate()
    status["error_rate_5min"] = err_count
    return status


async def cleanup_stuck_statuses(force: bool = False) -> tuple[int, int]:
    """uvicorn reload 시 task 죽으면 DB status 'generating'/'rendering' 그대로 stuck.
    여기서 자동 finalize:
    - mascot turnaround: status='generating' + 마지막 디스크 파일 mtime이 5분 이전 → ready/partial
    - remix concat: status='rendering' + 30분 이상 활동 없으면 → failed

    force=True (startup 호출): 디스크 파일 0개 + mtime 없어도 무조건 정리.
    이전 process가 죽었으니 어떻게든 다시 안 돎.
    정리한 마스코트 turnaround는 텔레그램 알림으로 알려줌.

    (마스코트 카운트, 합본 카운트) 반환.
    """
    import sqlite3, json, time
    db_path = Path("/Users/shortsking/banbaji-discover/db/discover.db")
    if not db_path.exists():
        return 0, 0
    now = time.time()
    n_mascot = 0
    n_remix = 0
    stuck_mascot_info: list[dict] = []   # 텔레그램 알림용 — diss_id + role_id + 디스크 파일 갯수
    stuck_remix_info: list[int] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return 0, 0
    try:
        # 마스코트 turnaround stuck 정리
        try:
            rows = conn.execute(
                "SELECT dissection_id, roles_json FROM mascots WHERE roles_json IS NOT NULL"
            ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            try:
                roles = json.loads(row["roles_json"] or "[]")
            except Exception:
                continue
            changed = False
            for r in roles:
                if r.get("turnaround_status") != "generating":
                    continue
                paths = r.get("turnaround_paths") or []
                # 마지막 파일 mtime 확인 — 5분 이상 변화 없으면 stuck
                latest_mtime = 0.0
                for p in paths:
                    try:
                        pp = Path(p)
                        if pp.exists():
                            latest_mtime = max(latest_mtime, pp.stat().st_mtime)
                    except Exception:
                        pass
                # 디스크 file이 다 있으면 paths 갱신 후 finalize
                existing = [p for p in paths if Path(p).exists()]
                # 정리 조건:
                # 1. 디스크에 파일 있고 5분 이상 변화 없음 (정상 흐름 — janitor_loop도 이 룰)
                # 2. force=True (startup 호출 — process 죽었으니 무조건 정리)
                should_cleanup = (latest_mtime > 0 and now - latest_mtime > 300) or force
                if should_cleanup:
                    new_status = "ready" if len(existing) == 8 else (
                        "partial" if existing else "failed"
                    )
                    r["turnaround_status"] = new_status
                    r["turnaround_paths"] = existing
                    changed = True
                    n_mascot += 1
                    stuck_mascot_info.append({
                        "diss_id": row["dissection_id"],
                        "role_id": r.get("role_id") or "?",
                        "files": len(existing),
                        "status": new_status,
                    })
            if changed:
                try:
                    conn.execute(
                        "UPDATE mascots SET roles_json=? WHERE dissection_id=?",
                        (json.dumps(roles), row["dissection_id"]),
                    )
                except Exception:
                    pass
        # 합본 rendering stuck 정리 (30분 이상 활동 없으면 failed)
        try:
            cur = conn.execute(
                "SELECT id, status, progress_message FROM remixes "
                "WHERE status IN ('rendering', 'pending')"
            )
            stuck_remix_ids = []
            for rr in cur.fetchall():
                rid = rr["id"]
                final_dir = REMIXES_DIR / f"remix_{rid}"
                if not final_dir.exists():
                    continue
                # 폴더 내 최근 파일 mtime 검사
                latest = 0.0
                for p in final_dir.rglob("*"):
                    try:
                        latest = max(latest, p.stat().st_mtime)
                    except Exception:
                        pass
                if latest == 0:
                    continue
                if now - latest > 1800:  # 30분
                    stuck_remix_ids.append(rid)
            for rid in stuck_remix_ids:
                try:
                    conn.execute(
                        "UPDATE remixes SET status='failed', "
                        "progress_message='서버 reload로 task 종료됨 — 다시 시도하세요' "
                        "WHERE id=? AND status IN ('rendering','pending')",
                        (rid,),
                    )
                    n_remix += 1
                    stuck_remix_info.append(rid)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()
    # 텔레그램 알림 — 정리된 게 있으면 알려줌
    if stuck_mascot_info or stuck_remix_info:
        try:
            from . import notify
            lines = []
            if stuck_mascot_info:
                lines.append(f"마스코트 turnaround {len(stuck_mascot_info)}건 정리:")
                for info in stuck_mascot_info[:10]:
                    lines.append(
                        f"  - {info['diss_id'][:16]}/{info['role_id']}: "
                        f"{info['files']}/8 → {info['status']}"
                    )
            if stuck_remix_info:
                lines.append(f"합본 {len(stuck_remix_info)}건 실패 처리: {stuck_remix_info[:10]}")
            await notify.notify_error(
                "⚠️ 서버 reload — 진행 중이던 작업 자동 정리",
                "\n".join(lines),
            )
        except Exception as _e:
            print(f"[cleanup] 텔레그램 알림 실패: {_e}", flush=True)
    return n_mascot, n_remix


async def _health_check_loop():
    """5분마다 자가 진단 + 에러 빈도 점검. 문제 발견 시 텔레그램 알림.
    중복 알림 방지: 같은 문제는 30분 안에 한 번만 알림."""
    from . import notify
    last_alert: dict[str, float] = {}

    async def _alert_once(key: str, msg: str, silent: bool = False):
        import time
        now = time.time()
        if now - last_alert.get(key, 0) < 1800:  # 30분
            return
        last_alert[key] = now
        try:
            await notify.send_telegram(msg, silent=silent)
        except Exception:
            pass

    while True:
        try:
            status = await health_check()
            # 디스크 90%+ 위험
            disk = status.get("disk", {})
            if not disk.get("ok"):
                await _alert_once(
                    "disk",
                    f"⚠️ <b>디스크 위험</b>\n"
                    f"사용 {disk.get('pct')}%, 여유 {disk.get('free_gb')}GB",
                )
            # 자료 접근 안 됨
            if not status.get("db", {}).get("ok"):
                await _alert_once(
                    "db",
                    f"⚠️ <b>자료 접근 실패</b>\n{status['db'].get('error')}",
                )
            # 에러 5개 이상 in 5분
            err_count = status.get("error_rate_5min", 0)
            if err_count >= 5:
                await _alert_once(
                    "error_rate",
                    f"⚠️ <b>에러 빈도 높음</b>\n5분 안에 {err_count}개 발생 — 시스템 점검 필요",
                )
        except Exception as e:
            print(f"[health-check] error: {e}", flush=True)
        await asyncio.sleep(300)  # 5분


async def janitor_loop():
    """1시간마다 한 번 — 정리, 백업, stuck 정리.
    추가로 5분마다 자가 진단 + 에러 빈도 점검 (별도 task)."""
    from . import notify
    # 자가 진단 task 따로 실행 (5분 주기)
    asyncio.create_task(_health_check_loop())
    while True:
        try:
            # 핵심 자료 4개 테이블 백업 (매 시간) — 자료 손실 사고 대비
            try:
                backups = await backup_all_critical_data()
                if backups:
                    names = ", ".join(Path(p).name for p in backups.values())
                    print(f"[disk-janitor] 자료 백업: {len(backups)}개 ({names})", flush=True)
            except Exception as e:
                print(f"[disk-janitor] 백업 실패: {e}", flush=True)

            # stuck task 정리 (매 시간) — uvicorn reload로 죽은 task DB만 'generating' 그대로 stuck
            try:
                n_m, n_r = await cleanup_stuck_statuses()
                if n_m or n_r:
                    print(f"[disk-janitor] stuck cleanup: 마스코트 {n_m} + 합본 {n_r}", flush=True)
                    if n_m or n_r:
                        try:
                            await notify.send_telegram(
                                f"🩹 <b>stuck status 정리</b>\n"
                                f"마스코트: {n_m}건, 합본: {n_r}건\n"
                                f"(서버 reload로 죽은 task 자동 finalize)",
                                silent=True,
                            )
                        except Exception:
                            pass
            except Exception as e:
                print(f"[disk-janitor] stuck cleanup err: {e}", flush=True)
            pct = await _disk_usage_pct()
            free_gb = await _disk_free_gb()
            print(f"[disk-janitor] 사용 {pct}%, 여유 {free_gb:.1f}GB", flush=True)

            # 매 1시간 — 망가진 영상 cache + 합본 검출 (partial 다운/cut 짤림 자동 정리)
            corrupt_count, corrupt_names = await _scan_and_delete_corrupt()
            if corrupt_count > 0:
                print(f"[disk-janitor] 망가진 영상 {corrupt_count}개 삭제됨", flush=True)
                detail = "\n".join(f"  · {n}" for n in corrupt_names[:5])
                if len(corrupt_names) > 5:
                    detail += f"\n  · 외 {len(corrupt_names) - 5}개"
                try:
                    await notify.send_telegram(
                        f"🧹 <b>망가진 영상 자동 정리</b>\n"
                        f"{corrupt_count}개 삭제 (다음 요청 시 자동 재다운)\n"
                        f"{detail}",
                        silent=True
                    )
                except Exception:
                    pass

            if pct >= 90:
                # 위험 — 7일 안 쓴 영상 강제 정리
                deleted, saved_mb = await _cleanup_old_originals(7)
                await notify.send_telegram(
                    f"⚠️ <b>디스크 위험 ({pct}%)</b>\n"
                    f"여유 {free_gb:.1f}GB\n"
                    f"강제 정리: {deleted}개 영상 ({saved_mb}MB) 삭제\n"
                    f"필요 시 외장 SSD 추가 검토"
                )
            elif pct >= 80:
                # 경고 — 30일 이상 안 쓴 영상만 정리 (보수적)
                deleted, saved_mb = await _cleanup_old_originals(30)
                if deleted > 0:
                    await notify.send_telegram(
                        f"💾 디스크 {pct}% — 옛 영상 {deleted}개 ({saved_mb}MB) 정리됨",
                        silent=True
                    )
            # 80% 미만은 정리 안 함 — 형님이 다시 작업할 수 있게 영상 그대로 둠
        except Exception as e:
            print(f"[disk-janitor] error: {e}", flush=True)

        # 1시간 대기
        await asyncio.sleep(3600)
