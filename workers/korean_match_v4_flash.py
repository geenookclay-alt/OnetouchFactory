"""한국 매칭 v4 — CLIP 1차 + Pro Vision 2차 하이브리드.

흐름:
1. 한국 영상 다운 + frame 15장 추출
2. 후보 영상 frame 추출 (이미 있으면 skip)
3. CLIP 임베딩 (모든 영상)
4. 양방향 매칭 + uniqueness + z-score 필터링
5. 의심 페어만 Gemini Pro Vision으로 정밀 판단
6. Pro가 "same"이라고 한 거만 used=1 마킹

비용: 한 카테 50 한국 × 280 후보 → CLIP 무료 + Pro 30~50회 ≈ ₩1,500
시간: CLIP 5분 + Pro 5~10분 = 약 15분
"""
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sqlite3
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Awaitable

import httpx
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"
KOR_MP4_DIR = Path("/Users/shortsking/banbaji-discover/data/korean")
ORIG_DIR = Path("/Users/shortsking/banbaji-discover/data/originals")
FRAME_DIR = Path("/Users/shortsking/banbaji-discover/data/korean_v2")
FRAME_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_PER_VIDEO = 15

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PRO_MODEL = "gemini-3.5-flash"
FLASH_MODEL = "gemini-3.5-flash"
DEFAULT_MODEL = FLASH_MODEL  # Flash 시험
DRY_RUN = True  # DB 마킹 안 함

# CLIP 임계값 (이거 통과해야 Pro 2차로)
CLIP_MAX_SIM_MIN = 0.80
CLIP_Z_SCORE_MIN = 2.0


_clip_model = None
def get_clip():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer("clip-ViT-B-32")
    return _clip_model


async def fetch_korean_videos(channel_url: str, limit: int = 50):
    """한국 채널에서 영상 list 가져오기."""
    proc = await asyncio.create_subprocess_exec(
        "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp",
        "--flat-playlist", "--print", "%(id)s|%(title)s|%(duration)s",
        f"{channel_url.rstrip('/')}/shorts",
        "--playlist-end", str(limit),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    videos = []
    for line in out.decode().strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 1 and parts[0]:
            videos.append({"id": parts[0], "title": parts[1] if len(parts) > 1 else ""})
    return videos[:limit]


async def download_video_if_needed(video_id: str, mp4_path: Path) -> bool:
    if mp4_path.exists() and mp4_path.stat().st_size > 1000:
        return True
    proc = await asyncio.create_subprocess_exec(
        "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp",
        "-f", "best[height<=720]", "-o", str(mp4_path),
        f"https://www.youtube.com/watch?v={video_id}",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return mp4_path.exists() and mp4_path.stat().st_size > 1000


async def extract_frames(mp4: Path, prefix: str, n: int = FRAMES_PER_VIDEO) -> list[Path]:
    out_paths = [FRAME_DIR / f"{prefix}_{i:02d}.jpg" for i in range(n)]
    if all(p.exists() for p in out_paths):
        return out_paths
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(mp4),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        dur = float(out.decode().strip())
    except Exception:
        return []
    if dur < 1:
        return []
    paths: list[Path] = []
    for i in range(n):
        ts = dur * (i + 0.5) / n
        out_p = FRAME_DIR / f"{prefix}_{i:02d}.jpg"
        if out_p.exists():
            paths.append(out_p)
            continue
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(mp4),
                "-frames:v", "1", "-q:v", "5", str(out_p),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if out_p.exists():
                paths.append(out_p)
        except Exception:
            continue
    return paths


def embed_frames(frames: list[Path]) -> np.ndarray:
    model = get_clip()
    imgs = []
    for p in frames:
        with Image.open(p) as im:
            im.load()
            imgs.append(im.convert("RGB"))
    embs = model.encode(imgs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    for im in imgs:
        im.close()
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


async def gemini_compare(kor_frames: list[Path], cand_frames: list[Path],
                          model: str = PRO_MODEL, retries: int = 2) -> dict:
    """두 영상 frame 비교 → Gemini Vision으로 판정. 반환: {verdict, confidence, reason}."""
    def load_b64(paths: list[Path], n: int = 3) -> list[str]:
        if not paths:
            return []
        idxs = [len(paths) // 4, len(paths) // 2, 3 * len(paths) // 4][:n]
        result = []
        for i in idxs:
            with Image.open(paths[i]) as im:
                im = im.convert("RGB")
                im.thumbnail((384, 384))
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=75)
                result.append(base64.b64encode(buf.getvalue()).decode())
        return result

    kor_b64 = load_b64(kor_frames)
    cand_b64 = load_b64(cand_frames)
    if not kor_b64 or not cand_b64:
        return {"verdict": "uncertain", "confidence": 0, "reason": "frame 부족"}

    PROMPT = """두 영상이 동일한 원본 영상인지 판단해.
영상1은 한국 채널이 외국 영상을 카피해서 자막/줌인/크롭 변형했을 수 있어.
영상2는 외국 원본이야.

판단 기준:
- 같은 사람 + 같은 사건 + 같은 장소/객체가 보이면 → "same" (한국이 카피)
- 비슷한 카테 안의 다른 사람/다른 이벤트면 → "different"
- 자막/줌인/크롭 변형은 무시하고 본질 비교

JSON으로만 답:
{"verdict": "same"|"different", "confidence": 0~10, "reason": "한 줄 한국어 설명"}"""

    parts = [{"text": PROMPT + "\n\n영상1 (한국):"}]
    for b in kor_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b}})
    parts.append({"text": "\n영상2 (외국 원본):"})
    for b in cand_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 4000,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY})
            if r.status_code == 429:
                if attempt < retries:
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                return {"verdict": "uncertain", "confidence": 0, "reason": "rate limit"}
            r.raise_for_status()
            data = r.json()
            cand = data.get("candidates", [{}])[0]
            parts_resp = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts_resp)
            result = json.loads(text)
            v = result.get("verdict", "uncertain")
            if v not in ("same", "different"):
                v = "uncertain"
            return {
                "verdict": v,
                "confidence": int(result.get("confidence", 5)),
                "reason": (result.get("reason") or "")[:200],
            }
        except json.JSONDecodeError:
            if attempt < retries:
                await asyncio.sleep(5)
                continue
            return {"verdict": "uncertain", "confidence": 0, "reason": "JSON 파싱 실패"}
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(5)
                continue
            return {"verdict": "uncertain", "confidence": 0, "reason": f"에러: {str(e)[:100]}"}

    return {"verdict": "uncertain", "confidence": 0, "reason": "최대 retry 초과"}


async def is_channel_rejected(channel_id: str | None, channel_name: str | None) -> bool:
    """거부 채널 DB 룩업 — 형님이 한 번 '가짜'로 표시한 채널은 영구 제외."""
    if not channel_id and not channel_name:
        return False
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as c:
            row = c.execute(
                "SELECT 1 FROM rejected_channels WHERE channel_id=? OR channel_name=? LIMIT 1",
                (channel_id or "", channel_name or ""),
            ).fetchone()
            return row is not None
    except Exception:
        return False


async def match_korean_v4(
    job_id: str,
    korean_channel_url: str,
    progress_cb: Callable[[int, str], Awaitable[None]] | None = None,
) -> dict:
    """한국 매칭 v4 — CLIP 1차 + Pro Vision 2차."""
    async def _emit(pct: int, msg: str):
        if progress_cb:
            await progress_cb(pct, msg)
        print(f"[{pct}%] {msg}", flush=True)

    await _emit(0, "한국 매칭 v4 시작 (CLIP + Pro)")

    # 1. 한국 영상 list
    kor_list = await fetch_korean_videos(korean_channel_url, limit=50)
    if not kor_list:
        return {"ok": False, "error": "한국 영상 못 가져옴"}
    await _emit(5, f"한국 영상 {len(kor_list)}개 다운 + frame 추출 중...")

    # 2. 한국 영상 다운 + frame
    sem_dl = asyncio.Semaphore(3)
    async def prep_kor(kv):
        async with sem_dl:
            mp4 = KOR_MP4_DIR / f"{kv['id']}.mp4"
            ok = await download_video_if_needed(kv["id"], mp4)
            if not ok:
                return None
            frames = await extract_frames(mp4, kv["id"])
            if not frames:
                return None
            return {"id": kv["id"], "title": kv["title"], "frames": frames}

    kor_results = await asyncio.gather(*[prep_kor(kv) for kv in kor_list])
    kor_ready = [r for r in kor_results if r]
    if not kor_ready:
        return {"ok": False, "error": "한국 frame 추출 실패"}
    await _emit(20, f"한국 {len(kor_ready)}개 준비됨")

    # 3. 후보 list (used=0)
    with sqlite3.connect(DB_PATH, timeout=30) as c:
        c.row_factory = sqlite3.Row
        if DRY_RUN:
            rows = c.execute(
                "SELECT id, video_id, url, title, channel_name, channel_id "
                "FROM candidate_videos WHERE job_id=?",
                (job_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, video_id, url, title, channel_name, channel_id "
                "FROM candidate_videos WHERE job_id=? AND COALESCE(used, 0)=0",
                (job_id,),
            ).fetchall()
    candidates = [dict(r) for r in rows]
    if not candidates:
        return {"ok": True, "matched": 0, "candidate_count": 0,
                "korean_count": len(kor_ready), "rejected_skipped": 0}

    # 거부 채널 자동 제외
    filtered = []
    rejected_count = 0
    for c in candidates:
        if await is_channel_rejected(c.get("channel_id"), c.get("channel_name")):
            rejected_count += 1
            continue
        filtered.append(c)
    candidates = filtered
    await _emit(25, f"후보 {len(candidates)}개 (거부 채널 {rejected_count}개 자동 제외)")

    # 4. 후보 frame 추출 (mp4 hash 매핑)
    async def prep_cand(cand):
        async with sem_dl:
            mp4 = ORIG_DIR / f"{hashlib.md5(cand['url'].encode()).hexdigest()[:16]}.mp4"
            if not mp4.exists():
                ok = await download_video_if_needed(cand["video_id"], mp4)
                if not ok:
                    return None
            frames = await extract_frames(mp4, f"cand_{cand['video_id']}")
            if not frames:
                return None
            return {**cand, "frames": frames}

    cand_results = await asyncio.gather(*[prep_cand(c) for c in candidates])
    cand_ready = [c for c in cand_results if c]
    await _emit(40, f"후보 {len(cand_ready)}개 frame 준비됨")

    # 5. CLIP 임베딩 (모든 frame)
    await _emit(45, "CLIP 임베딩 중...")
    kor_embs = {}
    for k in kor_ready:
        kor_embs[k["id"]] = embed_frames(k["frames"]).mean(axis=0)
    cand_embs = {}
    for c in cand_ready:
        cand_embs[c["video_id"]] = embed_frames(c["frames"]).mean(axis=0)

    kor_ids = list(kor_embs.keys())
    cand_ids = list(cand_embs.keys())
    K = np.stack([kor_embs[k] for k in kor_ids])
    C = np.stack([cand_embs[c] for c in cand_ids])
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    sims = K @ C.T  # (kor, cand)

    # 6. 각 한국에 대해 최고 매칭 + 양방향 + z-score
    kor_best = {ki: int(np.argmax(sims[ki])) for ki in range(len(kor_ids))}
    cand_best = {ci: int(np.argmax(sims[:, ci])) for ci in range(len(cand_ids))}

    suspects = []  # Pro로 검증할 페어
    for ki, kid in enumerate(kor_ids):
        ci = kor_best[ki]
        cid = cand_ids[ci]
        max_sim = float(sims[ki, ci])
        mutual = cand_best[ci] == ki
        # z-score: 이 후보가 다른 한국과의 sim 분포 대비
        cand_sims = sims[:, ci]
        z = float((max_sim - cand_sims.mean()) / (cand_sims.std() + 1e-6))
        if max_sim < CLIP_MAX_SIM_MIN:
            continue
        if not mutual:
            continue
        if z < CLIP_Z_SCORE_MIN:
            continue
        suspects.append({
            "kor_id": kid, "cand_id": cid, "cand_video_id": cid,
            "max_sim": max_sim, "z_score": z,
            "kor_frames": [k for k in kor_ready if k["id"] == kid][0]["frames"],
            "cand_frames": [c for c in cand_ready if c["video_id"] == cid][0]["frames"],
            "cand_row": [c for c in cand_ready if c["video_id"] == cid][0],
        })

    await _emit(60, f"CLIP 1차 필터 끝 — Pro 2차 후보 {len(suspects)}개")

    # 7. Pro Vision 2차 (병렬 1, rate limit 회피)
    matched = 0
    pro_results = []
    for idx, sus in enumerate(suspects):
        pct = 60 + int((idx + 1) / max(len(suspects), 1) * 35)
        await _emit(pct, f"Pro 2차 검증 {idx+1}/{len(suspects)}")
        gem = await gemini_compare(sus["kor_frames"], sus["cand_frames"], model=DEFAULT_MODEL)
        sus["gemini"] = gem
        pro_results.append(sus)
        if gem["verdict"] == "same" and gem["confidence"] >= 7:
            # used=1 마킹 + memo에 한국 영상 링크 저장
            note = (
                f"[KOR{gem['confidence']}/매칭] "
                f"https://youtube.com/shorts/{sus['kor_id']} | "
                f"{gem['reason']}"
            )
            with sqlite3.connect(DB_PATH, timeout=30) as c2:
                c2.execute("PRAGMA busy_timeout = 30000")
                c2.execute(
                    "UPDATE candidate_videos SET used=1, classification='제외', "
                    "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                    (note, sus["cand_row"]["id"]),
                )
            matched += 1
        # rate limit (Pro 무료 tier 분당 약 5회)
        if idx + 1 < len(suspects):
            await asyncio.sleep(8)

    await _emit(100, f"✅ 완료 — {matched}개 매칭 마킹 (Pro 검증 {len(suspects)}개)")

    return {
        "ok": True,
        "korean_count": len(kor_ready),
        "candidate_count": len(cand_ready),
        "rejected_skipped": rejected_count,
        "clip_suspects": len(suspects),
        "matched": matched,
        "details": [
            {
                "kor_id": s["kor_id"],
                "cand_video_id": s["cand_video_id"],
                "cand_title": s["cand_row"].get("title"),
                "max_sim": s["max_sim"],
                "verdict": s["gemini"]["verdict"],
                "confidence": s["gemini"]["confidence"],
                "reason": s["gemini"]["reason"],
            } for s in pro_results
        ],
    }


# Standalone 실행 (테스트용)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python korean_match_v4.py <job_id> <korean_channel_url>")
        sys.exit(1)

    # GEMINI_API_KEY env에 없으면 .env에서
    if not GEMINI_API_KEY:
        try:
            GEMINI_API_KEY = open("/Users/shortsking/banbaji-discover/.env").read().split("GEMINI_API_KEY=")[1].split("\n")[0].strip()
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        except Exception:
            print("GEMINI_API_KEY missing")
            sys.exit(1)

    result = asyncio.run(match_korean_v4(sys.argv[1], sys.argv[2]))
    print("")
    print("===== 최종 결과 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    Path("/tmp/korean_match_v4_flash_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
