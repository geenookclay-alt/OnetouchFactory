"""채널 풀 발굴 — yt-dlp + 우리 조건 필터 + DB import.

수동 작업 풀 / 자동 batch 둘 다 사용. discover_batch.py 로직 함수화.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

YT_DLP = "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp"

SKIP_TITLE_PATTERNS = [
    r"\btop\s*\d+", r"ranking", r"compilation", r"worst", r"best\s+of",
    r"funniest", r"몰아보기", r"모음", r"베스트", r"순위", r"\bvs\.?\b",
]


def _is_skip_title(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(p, t) for p in SKIP_TITLE_PATTERNS)


def _shorts_url(channel: str) -> str:
    """채널 식별자 → shorts URL. UC..., @handle, 또는 풀 URL 다 처리."""
    c = channel.strip()
    if not c:
        raise ValueError("empty channel")
    # 풀 URL이면 그대로 + /shorts 붙이기
    if c.startswith("http"):
        u = c.rstrip("/")
        if not u.endswith("/shorts"):
            u = u + "/shorts"
        return u
    # UC로 시작하면 channel ID
    if c.startswith("UC") and len(c) >= 22:
        return f"https://www.youtube.com/channel/{c}/shorts"
    # @로 시작하면 handle
    if c.startswith("@"):
        return f"https://www.youtube.com/{c}/shorts"
    # 그 외는 handle로 가정
    return f"https://www.youtube.com/@{c}/shorts"


async def _run_subprocess(cmd: list[str], timeout: float = 300) -> str:
    """async subprocess wrapper."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill(); await proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(err.decode()[:500])
    return out.decode()


async def fast_shorts_list(channel: str) -> list[dict]:
    """yt-dlp --print 빠른 list (id, title, view_count)."""
    url = _shorts_url(channel)
    cmd = [YT_DLP, "--flat-playlist", "--no-warnings", "-q",
           "--print", "%(id)s\t%(title)s\t%(view_count)s\t%(channel_id)s\t%(channel)s",
           url]
    try:
        out = await _run_subprocess(cmd, timeout=240)
    except Exception:
        return []
    rows = []
    for line in out.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t", 4)
        if len(parts) < 3:
            continue
        vid, title, views = parts[0], parts[1], parts[2]
        ch_id = parts[3] if len(parts) > 3 and parts[3] != "NA" else None
        ch_name = parts[4] if len(parts) > 4 and parts[4] != "NA" else channel
        try:
            views = int(views) if views and views != "NA" else 0
        except Exception:
            views = 0
        rows.append({"id": vid, "title": title, "view_count": views,
                     "channel_id": ch_id, "channel_name": ch_name})
    return rows


async def fetch_duration(vid: str) -> float | None:
    cmd = [YT_DLP, "--no-warnings", "-q", "--print", "%(duration)s",
           f"https://youtu.be/{vid}"]
    try:
        out = await _run_subprocess(cmd, timeout=30)
        out = out.strip()
        if out and out != "NA":
            return float(out)
    except Exception:
        pass
    return None


async def discover_pool(
    db_path: str,
    job_id: str,
    channels: Sequence[str],
    min_views: int = 5_000_000,
    max_duration: int = 55,
    skip_channel_ids: set[str] | None = None,
    progress_cb=None,
) -> dict:
    """채널 list → 5M+ filter → DB insert → duration fix → 55s 초과 DELETE.

    Returns {channels_processed, candidates_added, deleted_too_long, final_total}.
    """
    skip_set = set(skip_channel_ids or [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def conn_open():
        c = sqlite3.connect(db_path, timeout=30.0)
        c.execute("PRAGMA busy_timeout = 30000")
        return c

    # 1. fetch + filter + insert
    conn = conn_open()
    cur = conn.cursor()
    existing_vids = {r[0] for r in cur.execute(
        "SELECT video_id FROM candidate_videos WHERE job_id=?", (job_id,)
    ).fetchall()}
    conn.close()

    added = 0
    skipped_view = skipped_pat = skipped_dup = 0
    for i, ch in enumerate(channels, 1):
        if progress_cb:
            await progress_cb(int(i / len(channels) * 50),
                              f"📺 ({i}/{len(channels)}) 채널 fetch: {ch}")
        rows = await fast_shorts_list(ch)
        # skip channel by ID
        if rows and rows[0].get("channel_id") in skip_set:
            continue
        # batch insert
        conn = conn_open()
        cur = conn.cursor()
        for e in rows:
            if e["view_count"] < min_views:
                skipped_view += 1; continue
            if _is_skip_title(e["title"]):
                skipped_pat += 1; continue
            if e["id"] in existing_vids:
                skipped_dup += 1; continue
            cur.execute("""
                INSERT INTO candidate_videos (
                    job_id, platform, video_id, url, title, channel_name, channel_id,
                    view_count, duration, classification, used, created_at
                ) VALUES (?, 'youtube', ?, ?, ?, ?, ?, ?, 0, '키핑', 0, ?)
            """, (job_id, e["id"], f"https://youtu.be/{e['id']}", e["title"],
                  e["channel_name"], e["channel_id"] or "", e["view_count"], now))
            existing_vids.add(e["id"])
            added += 1
        conn.commit()
        conn.close()

    # 2. duration fix + 55s 초과 DELETE
    conn = conn_open()
    cur = conn.cursor()
    need_fix = cur.execute(
        "SELECT video_id FROM candidate_videos WHERE job_id=? AND (duration IS NULL OR duration=0)",
        (job_id,),
    ).fetchall()
    conn.close()

    deleted = 0
    fixed = 0
    for i, (vid,) in enumerate(need_fix, 1):
        if progress_cb and i % 20 == 0:
            pct = 50 + int(i / max(1, len(need_fix)) * 50)
            await progress_cb(pct, f"⏱ duration ({i}/{len(need_fix)}) fixed={fixed} deleted={deleted}")
        d = await fetch_duration(vid)
        if d is None:
            continue
        conn = conn_open()
        cur = conn.cursor()
        if d > max_duration:
            cur.execute("DELETE FROM candidate_videos WHERE video_id=? AND job_id=?",
                        (vid, job_id))
            deleted += 1
        else:
            cur.execute("UPDATE candidate_videos SET duration=? WHERE video_id=? AND job_id=?",
                        (d, vid, job_id))
            fixed += 1
        conn.commit()
        conn.close()

    # 3. final stats
    conn = conn_open()
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM candidate_videos WHERE job_id=?",
                        (job_id,)).fetchone()[0]
    conn.close()
    if progress_cb:
        await progress_cb(100, f"✅ 완료 — 총 {total}개")
    return {
        "channels_processed": len(channels),
        "candidates_added": added,
        "deleted_too_long": deleted,
        "final_total": total,
    }
