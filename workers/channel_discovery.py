"""채널 자동 발굴 (Phase 5).

흐름:
1. 카테 시그니처 키워드 추출 (keywords_result + 채널 영상 title 빈도)
2. yt-dlp로 YouTube 검색 → 채널 list 수집
3. 이미 있는 채널 / 거부 채널 제외
4. 각 새 채널의 영상 sample (5장) → Pro Vision으로 "카테 결 맞나?" 검증
5. 통과한 채널 list 반환

비용: Pro 호출 = 새 채널 수 (예 30개) × ₩30 = ₩900
시간: 검색 5분 + Pro 검증 5~10분 = 약 15분
"""
import asyncio
import base64
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"
TMP_DIR = Path("/tmp/discover_channels")
TMP_DIR.mkdir(exist_ok=True)
YT_DLP = "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp"


def get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return open("/Users/shortsking/banbaji-discover/.env").read().split("GEMINI_API_KEY=")[1].split("\n")[0].strip()
    except Exception:
        return ""


def get_keywords(diss_id: str) -> list[str]:
    """카테 시그니처 키워드 추출 — keywords_result의 영어 키워드."""
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT keywords_result FROM dissection_analyses WHERE id=?",
            (diss_id,),
        ).fetchone()
    if not row or not row[0]:
        return []
    try:
        d = json.loads(row[0])
    except Exception:
        return []
    out = []
    for lang in ("english",):  # 영어만 (YouTube 검색에서 가장 효과적)
        for item in d.get(lang, []):
            kw = item.get("keyword") if isinstance(item, dict) else str(item)
            if kw and kw not in out:
                out.append(kw)
    return out


def yt_search_channels(keyword: str, n: int = 20) -> list[dict]:
    """yt-dlp로 키워드 검색 → 채널 list (영상에서 channel_id 추출)."""
    try:
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist",
             "--print", "%(channel_id)s|%(channel)s|%(uploader_url)s|%(title)s",
             f"ytsearch{n}:{keyword}"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        print(f"  검색 실패 ({keyword}): {e}", flush=True)
        return []
    seen = set()
    result = []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split("|", 3)
        if len(parts) < 3 or not parts[0] or parts[0] == "NA":
            continue
        channel_id = parts[0]
        if channel_id in seen:
            continue
        seen.add(channel_id)
        result.append({
            "channel_id": channel_id,
            "channel_name": parts[1] or "",
            "channel_url": parts[2] or "",
            "sample_title": parts[3] if len(parts) > 3 else "",
        })
    return result


def get_existing_channels(job_id: str) -> set:
    """이미 후보풀에 있는 채널 ID 모음."""
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT DISTINCT channel_id FROM candidate_videos WHERE job_id=?",
            (job_id,),
        ).fetchall()
    return {r[0] for r in rows if r[0]}


def get_rejected_channels() -> tuple[set, set]:
    """거부 채널 ID + name."""
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT channel_id, channel_name FROM rejected_channels").fetchall()
    ids = {r[0] for r in rows if r[0]}
    names = {r[1] for r in rows if r[1]}
    return ids, names


def channel_short_frames(channel_url: str, n: int = 3) -> list[Path]:
    """채널의 shorts 영상 첫 N개 다운 + frame 추출 (각 영상 1 frame씩 = 채널 sample)."""
    shorts_url = channel_url.rstrip("/") + "/shorts"
    try:
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist", "--print", "%(id)s",
             "--playlist-end", str(n), shorts_url],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return []
    ids = [x for x in proc.stdout.strip().split("\n") if x]
    if not ids:
        return []
    frame_paths = []
    for vid_id in ids[:n]:
        out_mp4 = TMP_DIR / f"{vid_id}.mp4"
        if not out_mp4.exists():
            try:
                subprocess.run(
                    [YT_DLP, "-f", "best[height<=480]", "-o", str(out_mp4),
                     f"https://www.youtube.com/watch?v={vid_id}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=90,
                )
            except Exception:
                continue
        if not out_mp4.exists():
            continue
        out_jpg = TMP_DIR / f"{vid_id}.jpg"
        if not out_jpg.exists():
            try:
                # 영상 가운데 frame
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "5", "-i", str(out_mp4),
                     "-frames:v", "1", "-q:v", "5", str(out_jpg)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=20,
                )
            except Exception:
                continue
        if out_jpg.exists():
            frame_paths.append(out_jpg)
    return frame_paths


async def verify_channel_with_pro(channel: dict, dna_summary: str, frames: list[Path]) -> dict:
    """Pro Vision으로 채널 결 검증."""
    key = get_gemini_key()
    if not key or not frames:
        return {"verdict": "uncertain", "confidence": 0, "reason": "데이터 부족"}

    def load_b64(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((384, 384))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()

    parts = [{"text":
        f"카테 결(DNA): {dna_summary}\n\n"
        f"채널 이름: {channel.get('channel_name', '')}\n"
        f"샘플 영상 제목: {channel.get('sample_title', '')[:100]}\n\n"
        f"아래는 이 채널의 영상 frame {len(frames)}장이야. 이 채널이 위 카테 결과 맞는 채널인가?\n"
        f"같은 카테 영상을 메인으로 만드는 채널이면 'yes', 카테와 무관하거나 가끔만 다루면 'no'.\n\n"
        f'JSON: {{"verdict": "yes"|"no", "confidence": 0~10, "reason": "한 줄 한국어"}}\n\n'
        f"채널 영상 frame {len(frames)}장:"
    }]
    for fp in frames:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": load_b64(fp)}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 4000,
        },
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120.0) as cli:
                r = await cli.post(url, json=body, headers={"x-goog-api-key": key})
            if r.status_code == 429:
                if attempt < 2:
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                return {"verdict": "uncertain", "confidence": 0, "reason": "rate limit"}
            r.raise_for_status()
            data = r.json()
            cand = data.get("candidates", [{}])[0]
            txt = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            result = json.loads(txt)
            v = result.get("verdict", "uncertain")
            return {
                "verdict": v if v in ("yes", "no") else "uncertain",
                "confidence": int(result.get("confidence", 5)),
                "reason": (result.get("reason") or "")[:200],
            }
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(5)
                continue
            return {"verdict": "uncertain", "confidence": 0, "reason": f"{type(e).__name__}: {str(e)[:100]}"}
    return {"verdict": "uncertain", "confidence": 0, "reason": "fail"}


async def discover(job_id: str, diss_id: str, target_new: int = 20) -> dict:
    print(f"=== 채널 자동 발굴 시작 (job={job_id}, diss={diss_id}) ===", flush=True)

    # 1. 키워드
    keywords = get_keywords(diss_id)
    if not keywords:
        return {"ok": False, "error": "키워드 없음"}
    print(f"키워드 {len(keywords)}개: {keywords[:5]}...", flush=True)

    # 2. DNA 요약 (검증 용)
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT dissection_result, name FROM dissection_analyses WHERE id=?",
            (diss_id,),
        ).fetchone()
    dna_text = ""
    cate_name = ""
    if row:
        cate_name = row[1] or ""
        try:
            d = json.loads(row[0] or "{}")
            common = d.get("common_dna") or {}
            items = common.get("items") or {}
            # 핵심 항목만 짧게
            for k in ("4_view_drivers", "6_topics_list"):
                it = items.get(k) or {}
                v = it.get("value") or ""
                if v:
                    dna_text += f"{it.get('label', k)}: {v[:200]}\n"
            if not dna_text:
                chans = d.get("channels") or []
                if chans:
                    dna_text = (chans[0].get("summary_kr") or "")[:300]
        except Exception:
            pass
    if not dna_text:
        dna_text = f"카테 = {cate_name}"
    print(f"DNA: {dna_text[:200]}", flush=True)

    # 3. 모든 키워드 검색
    print("YouTube 검색 중...", flush=True)
    all_channels = {}
    for kw in keywords[:6]:  # 너무 많이 검색 X
        print(f"  검색: {kw}", flush=True)
        results = yt_search_channels(kw, n=15)
        for ch in results:
            if ch["channel_id"] not in all_channels:
                all_channels[ch["channel_id"]] = ch
        await asyncio.sleep(2)
    print(f"검색 결과: {len(all_channels)} 유니크 채널", flush=True)

    # 4. 이미 있는 채널 + 거부 채널 제외
    existing = get_existing_channels(job_id)
    rejected_ids, rejected_names = get_rejected_channels()
    print(f"기존: {len(existing)}, 거부: {len(rejected_ids) + len(rejected_names)}", flush=True)

    new_channels = []
    for cid, ch in all_channels.items():
        if cid in existing:
            continue
        if cid in rejected_ids:
            continue
        if ch["channel_name"] in rejected_names:
            continue
        if not ch["channel_url"]:
            continue
        new_channels.append(ch)
    print(f"새 채널 후보: {len(new_channels)}", flush=True)

    # 5. Pro Vision 검증 (sample frame 가져와서)
    print("\n=== Pro 검증 ===", flush=True)
    passed = []
    failed = []
    for idx, ch in enumerate(new_channels[:target_new * 2]):  # 통과율 50% 가정해서 2배
        print(f"[{idx+1}/{min(len(new_channels), target_new*2)}] {ch['channel_name']}", flush=True)
        frames = channel_short_frames(ch["channel_url"], n=3)
        if not frames:
            failed.append({**ch, "verdict": "no_frames"})
            continue
        result = await verify_channel_with_pro(ch, dna_text, frames)
        ch.update(result)
        if result["verdict"] == "yes" and result["confidence"] >= 7:
            passed.append(ch)
            print(f"  ✓ {result['reason']}", flush=True)
        else:
            failed.append(ch)
            print(f"  ✗ ({result['verdict']}) {result['reason']}", flush=True)
        if len(passed) >= target_new:
            print(f"  목표 {target_new}개 달성. 멈춤", flush=True)
            break
        await asyncio.sleep(8)  # Pro rate limit

    result = {
        "ok": True,
        "job_id": job_id,
        "diss_id": diss_id,
        "keywords_used": keywords[:6],
        "total_search_results": len(all_channels),
        "new_channels": len(new_channels),
        "passed": len(passed),
        "failed": len(failed),
        "passed_channels": passed,
        "failed_channels": failed[:20],  # 너무 많이 X
    }
    print(f"\n=== 끝 — 통과 {len(passed)} / 실패 {len(failed)} ===", flush=True)
    print(f"통과 채널: {[c['channel_name'] for c in passed]}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python channel_discovery.py <job_id> <diss_id> [target_new=20]")
        sys.exit(1)
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    result = asyncio.run(discover(sys.argv[1], sys.argv[2], target_new=target))
    Path("/tmp/channel_discovery_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print("\n저장 끝: /tmp/channel_discovery_result.json")
