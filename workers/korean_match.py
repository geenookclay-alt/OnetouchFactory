"""한국 채널 vs 후보 풀 1:1 매칭. 후보 1개 vs 한국 1개씩 따로 Gemini 호출."""
import asyncio
import json
import sqlite3
import subprocess
import base64
import hashlib
from pathlib import Path
import httpx
import os

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"
ORIGINALS_DIR = Path("/Users/shortsking/banbaji-discover/data/originals")
KOR_DIR = Path("/Users/shortsking/banbaji-discover/data/korean")
KOR_DIR.mkdir(parents=True, exist_ok=True)
YT_DLP = "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp"
FFMPEG = "/opt/homebrew/bin/ffmpeg"


async def _run(cmd, timeout=120):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, out, err
    except asyncio.TimeoutError:
        proc.kill()
        return -1, b"", b"timeout"


async def fetch_kor_video_list(channel_url: str, limit: int = 50) -> list[dict]:
    if "/shorts" not in channel_url and not channel_url.endswith("/videos"):
        if not channel_url.endswith("/"):
            channel_url = channel_url + "/"
        channel_url = channel_url + "shorts"
    cmd = [
        YT_DLP, "--flat-playlist", "--no-warnings", "-q",
        "--print", "%(id)s\t%(title)s\t%(view_count)s",
        "--playlist-end", str(limit),
        channel_url,
    ]
    rc, out, err = await _run(cmd, timeout=180)
    rows = []
    for line in (out or b"").decode().strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        rows.append({
            "id": parts[0],
            "title": parts[1],
            "views": int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0,
        })
    return rows


async def download_video(video_id: str, out_path: Path, timeout: float = 120) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
    cmd = [
        YT_DLP, "-f", "18/best[ext=mp4][height<=360]/best",
        "--no-warnings", "--quiet", "--force-overwrites",
        "-o", str(out_path),
        f"https://youtube.com/shorts/{video_id}",
    ]
    rc, _, err = await _run(cmd, timeout=timeout)
    return rc == 0 and out_path.exists() and out_path.stat().st_size > 1000


async def extract_3_frames(video_path: Path, out_dir: Path, prefix: str) -> list[Path]:
    if not video_path.exists():
        return []
    rc, dur_out, _ = await _run([
        FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ], timeout=10)
    try:
        dur = float((dur_out or b"0").decode().strip() or "0")
    except Exception:
        dur = 0
    if dur < 2:
        dur = 5
    times = [max(0.5, dur * 0.2), max(1.0, dur * 0.5), max(1.5, dur * 0.85)]
    paths = []
    for i, t in enumerate(times, 1):
        out = out_dir / f"{prefix}_{i}.jpg"
        rc, _, _ = await _run([
            FFMPEG, "-y", "-i", str(video_path), "-ss", str(t),
            "-frames:v", "1", "-q:v", "3", "-vf", "scale=480:-2",
            str(out),
        ], timeout=15)
        if rc == 0 and out.exists() and out.stat().st_size > 100:
            paths.append(out)
    return paths


def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"


async def compare_one_to_one(kor: dict, cand: dict) -> dict:
    """후보 1개 vs 한국 1개 — 두 영상 frame 6장 비교. 같은 영상인지 판단."""
    parts = [{"text": f"한국 영상 (제목: {kor['title'][:80]}) — 시작/중간/끝 3 frame:"}]
    for fp in kor["frame_paths"]:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_b64(fp)}})
    parts.append({"text": f"\n후보 영상 (제목: {cand['title'][:80]}) — 시작/중간/끝 3 frame:"})
    for fp in cand["frame_paths"]:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_b64(fp)}})
    parts.append({"text": """\n
두 영상이 같은 영상(reupload)인가? 매우 엄격하게 판단:
- 같은 사람 얼굴 + 같은 옷 + 같은 배경/장소 모두 일치할 때만 같음
- 비슷한 컨셉/소품/방식만 같은 경우는 다른 영상 (다른 가족 다른 영상)
- 의심스러우면 다름

JSON: {"same": true/false, "score": 0~100, "reason": "<한 줄>"}
score 95+ = 거의 확실 같은 영상
"""})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                GEMINI_URL,
                params={"key": os.getenv("GEMINI_API_KEY", "")},
                json=body,
            )
            if r.status_code != 200:
                return {"same": False, "score": 0, "reason": f"API {r.status_code}"}
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
    except Exception as e:
        return {"same": False, "score": 0, "reason": f"ERR {e}"}


async def match_korean_channel(
    job_id: str, korean_channel_url: str, progress_cb=None,
) -> dict:
    """1:1 매칭 — 후보 1개 × 한국 1개씩 따로 Gemini 호출."""
    if progress_cb:
        await progress_cb(2, "한국 채널 영상 list 가져오는 중...")
    kor_list = await fetch_kor_video_list(korean_channel_url, limit=50)
    if not kor_list:
        return {"ok": False, "error": "한국 영상 못 가져옴"}

    if progress_cb:
        await progress_cb(5, f"한국 영상 {len(kor_list)}개 — 다운 + frame 추출 중...")

    # 한국 영상 다운 + frame (병렬, max 5)
    sem = asyncio.Semaphore(5)
    async def prep_kor(kv):
        async with sem:
            mp4 = KOR_DIR / f"{kv['id']}.mp4"
            existing = sorted(KOR_DIR.glob(f"{kv['id']}_*.jpg"))
            if not existing:
                ok = await download_video(kv["id"], mp4)
                if not ok:
                    return None
                existing = await extract_3_frames(mp4, KOR_DIR, kv["id"])
                if not existing:
                    return None
            return {"id": kv["id"], "title": kv["title"], "frame_paths": list(existing)}
    results = await asyncio.gather(*[prep_kor(kv) for kv in kor_list])
    kor_frames = [r for r in results if r]
    if not kor_frames:
        return {"ok": False, "error": "한국 frame 추출 실패"}
    if progress_cb:
        await progress_cb(15, f"한국 {len(kor_frames)}개 frame 준비 — 후보 비교 시작")

    # 후보 list
    with sqlite3.connect(DB_PATH, timeout=30) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT id, video_id, url, title FROM candidate_videos "
            "WHERE job_id = ? AND COALESCE(used, 0) = 0 "
            "AND (classification IS NULL OR classification IN ('키핑', 'review'))",
            (job_id,),
        ).fetchall()
    candidates = [dict(r) for r in rows]
    if not candidates:
        return {"ok": True, "korean_count": len(kor_frames),
                "candidate_count": 0, "matched": 0}

    if progress_cb:
        await progress_cb(20, f"후보 {len(candidates)}개 × 한국 {len(kor_frames)}개 = 1:1 비교 시작")

    affected = 0
    err_count = 0
    sem2 = asyncio.Semaphore(5)

    async def process_cand(cand, idx):
        nonlocal affected, err_count
        async with sem2:
            video_id = cand["video_id"]
            existing = sorted(KOR_DIR.glob(f"cand_{video_id}_*.jpg"))
            if not existing:
                cache_mp4 = ORIGINALS_DIR / f"{hashlib.md5(cand['url'].encode()).hexdigest()[:16]}.mp4"
                if not cache_mp4.exists():
                    ok = await download_video(video_id, cache_mp4)
                    if not ok:
                        err_count += 1
                        return
                existing = await extract_3_frames(cache_mp4, KOR_DIR, f"cand_{video_id}")
                if not existing:
                    err_count += 1
                    return
            cand_frame = {"id": video_id, "title": cand["title"], "frame_paths": list(existing)}

            # 1:1 비교 — 한국 frame 한 개씩 따로
            best_match = None
            for kor in kor_frames:
                result = await compare_one_to_one(kor, cand_frame)
                if result.get("same") and int(result.get("score", 0) or 0) >= 95:
                    best_match = (kor, result)
                    break
            if best_match:
                kor, result = best_match
                score = int(result.get("score", 0))
                reason = (result.get("reason") or "")[:200]
                note = f"[KOR{score}/used] 한국({kor['id']}): {reason}"
                with sqlite3.connect(DB_PATH, timeout=30) as c2:
                    c2.execute("PRAGMA busy_timeout = 30000")
                    c2.execute(
                        "UPDATE candidate_videos SET used=1, classification='제외', "
                        "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                        (note, cand["id"]),
                    )
                affected += 1
            if (idx + 1) % 5 == 0 and progress_cb:
                pct = 20 + int((idx + 1) / len(candidates) * 75)
                await progress_cb(pct, f"후보 {idx+1}/{len(candidates)} (매칭 {affected}, 에러 {err_count})")

    await asyncio.gather(*[process_cand(c, i) for i, c in enumerate(candidates)])

    if progress_cb:
        await progress_cb(100, f"✅ 완료 — {affected}개 매칭 마킹 (에러 {err_count})")

    return {
        "ok": True,
        "korean_count": len(kor_frames),
        "candidate_count": len(candidates),
        "matched": affected,
        "errors": err_count,
    }
