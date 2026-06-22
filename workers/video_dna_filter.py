"""영상 1개씩 카테 결 매칭 — 메타 1차 + (선택) CLIP 2차 + Pro 3차.

채널의 모든 영상 (limit 없음) → 결에 맞는 영상만 후보풀에 추가.

비용:
- 메타 1차: 무료
- CLIP 2차: 무료 (frame 추출 + embedding)
- Pro 3차: 의심만 ~10~20개 ≈ ₩300~600/채널

흐름:
1. 채널 URL → yt-dlp로 모든 영상 list (limit 없음)
2. 각 영상의 메타 (title + caption + tags)
3. [1차 메타] 시그니처 키워드 매칭
   - 시그니처 키워드 ≥ 1개 → 1차 통과
   - 명백한 거부 키워드 (game, cooking 등) → 자동 제외
   - 둘 다 없음 → 2차로
4. [2차 Pro Vision] 의심만 frame 3장 + 카테 DNA → Pro 판단
5. 통과한 영상 → 후보풀 추가
"""
import asyncio
import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Callable, Awaitable

import httpx
from PIL import Image

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"
ORIG_DIR = Path("/Users/shortsking/banbaji-discover/data/originals")
ORIG_DIR.mkdir(parents=True, exist_ok=True)
TMP_FRAME_DIR = Path("/tmp/dna_filter_frames")
TMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)
YT_DLP = "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp"

# 명백한 거부 키워드 (시그니처와 무관한 카테)
GENERIC_EXCLUDE = [
    "ai animation", "ai video", "rc car", "rc plane",
    "kids", "child", "어린이", "유아", "키즈",
    "compilation", "tier list", "ranking", "best of",
]


def get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return open("/Users/shortsking/banbaji-discover/.env").read().split("GEMINI_API_KEY=")[1].split("\n")[0].strip()
    except Exception:
        return ""


def get_cate_signature_words(cate_name: str, dna: dict | None = None) -> list[str]:
    """카테 이름 + DNA에서 시그니처 단어 추출."""
    words = []
    # 카테 이름 → 영어 변환
    KR_TO_EN = {
        "레고": ["lego", "brick"], "브릭": ["brick", "lego"],
        "강아지 미용": ["dog", "grooming", "groomer", "puppy"],
        "미용": ["grooming", "groomer"],
        "볼링": ["bowling", "strike", "pin"],
        "곤충": ["insect", "bug", "beetle"],
        "도자기": ["pottery", "ceramic", "clay"],
        "클라이밍": ["climbing", "bouldering"],
        "말": ["horse"], "복원": ["restoration", "restore"],
        "비트박스": ["beatbox"], "마술": ["magic", "trick"],
        "모델": ["model"], "배드민턴": ["badminton"],
        "젤더 리빌": ["gender", "reveal", "boy", "girl"],
        "젤더리빌": ["gender", "reveal", "boy", "girl"],
        "젠더 리빌": ["gender", "reveal", "boy", "girl"],
        "3d": ["3d print"],
    }
    cate_lower = (cate_name or "").lower()
    for kr, en_list in KR_TO_EN.items():
        if kr in cate_lower or kr.replace(" ", "") in cate_lower.replace(" ", ""):
            words.extend(en_list)
    # 영문 단어 직접
    for m in re.finditer(r"[a-zA-Z]{4,}", cate_name or ""):
        w = m.group().lower()
        if w not in ("shorts", "video", "channel"):
            words.append(w)
    return list(dict.fromkeys(words))  # unique 순서 유지


def meta_match(title: str, caption: str, channel_name: str,
               signature_words: list[str]) -> tuple[bool, str]:
    """메타데이터 1차 판단 — 강화 버전.

    반환:
      - True: 자동 통과 (시그니처 + 채널명 일치)
      - False: 자동 제외 (거부 키워드)
      - None: 의심 (Pro 검증 필요)

    강화 룰:
      - 시그니처 키워드만 매칭 = 의심 (#lego 태그 박은 비-레고 가능)
      - 채널명에도 시그니처 단어 있으면 = 자동 통과
      - 채널명에 시그니처 없으면 = Pro 검증으로 (의심)
    """
    title_caption = " ".join([str(title or ""), str(caption or "")]).lower()
    ch_lower = (channel_name or "").lower()
    blob = title_caption + " " + ch_lower

    # 거부 키워드 우선 — 자동 제외
    for kw in GENERIC_EXCLUDE:
        if kw in blob:
            return (False, f"거부 키워드: {kw}")

    # 시그니처 매칭
    title_matched = [w for w in signature_words if w in title_caption]
    ch_matched = [w for w in signature_words if w in ch_lower]

    if ch_matched:
        # 채널명에 시그니처 단어 있음 = 강한 시그널 = 자동 통과
        return (True, f"채널명+시그니처: {', '.join(ch_matched[:2])}")

    if title_matched:
        # title만 매칭 + 채널명 없음 = 의심 (Pro 검증)
        return (None, f"title 시그니처({', '.join(title_matched[:2])}) 있지만 채널명 일치 X → Pro 검증 필요")

    return (None, "메타 시그니처 없음 — Pro 판단")


async def download_video(video_id: str, mp4: Path, max_height: int = 480) -> bool:
    if mp4.exists() and mp4.stat().st_size > 1000:
        return True
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "-f", f"best[height<={max_height}]", "-o", str(mp4),
            f"https://www.youtube.com/watch?v={video_id}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
    except Exception:
        return False
    return mp4.exists() and mp4.stat().st_size > 1000


async def extract_frames(mp4: Path, prefix: str, n: int = 3) -> list[Path]:
    out_paths = [TMP_FRAME_DIR / f"{prefix}_{i}.jpg" for i in range(n)]
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
    paths = []
    for i in range(n):
        ts = dur * (i + 0.5) / n
        p = TMP_FRAME_DIR / f"{prefix}_{i}.jpg"
        if not p.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(mp4),
                    "-frames:v", "1", "-q:v", "5", str(p),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.communicate(), timeout=20)
            except Exception:
                continue
        if p.exists():
            paths.append(p)
    return paths


async def pro_check_video(video: dict, dna_summary: str, frames: list[Path]) -> dict:
    """Pro Vision으로 영상 1개 결 검증."""
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

    title = video.get("title", "")[:100]
    channel = video.get("channel_name", "")

    parts = [{"text":
        f"카테 결(DNA): {dna_summary}\n\n"
        f"영상 제목: {title}\n"
        f"채널: {channel}\n\n"
        f"이 영상이 카테 결과 맞는지 판단. frame {len(frames)}장 보고.\n"
        f"  - 카테 메인 콘텐츠와 일치 = 'yes'\n"
        f"  - 카테와 무관 / 다른 카테 = 'no'\n\n"
        f'JSON: {{"verdict": "yes"|"no", "confidence": 0~10, "reason": "한 줄 한국어"}}\n\n'
        f"frame {len(frames)}장:"
    }]
    for p in frames:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": load_b64(p)}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 3000,
        },
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=90.0) as c:
                r = await c.post(url, json=body, headers={"x-goog-api-key": key})
            if r.status_code == 429:
                if attempt < 2:
                    await asyncio.sleep(15 * (attempt + 1))
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
            return {"verdict": "uncertain", "confidence": 0, "reason": f"{type(e).__name__}: {str(e)[:80]}"}
    return {"verdict": "uncertain", "confidence": 0, "reason": "fail"}


async def filter_channel_videos(
    channel_url: str,
    cate_name: str,
    dna_summary: str,
    signature_words: list[str],
    min_views: int = 100000,
    max_duration: int = 90,
    skip_pro: bool = False,
    progress_cb: Callable[[int, str], Awaitable[None]] | None = None,
) -> dict:
    """채널 모든 영상 → 메타 + Pro 검증 → 통과한 영상 list 반환."""
    async def _emit(pct, msg):
        if progress_cb:
            await progress_cb(pct, msg)
        print(f"[{pct}%] {msg}", flush=True)

    await _emit(0, f"채널 영상 수집 시작: {channel_url}")

    # 1. 전체 영상 list (limit 없음)
    shorts_url = channel_url.rstrip("/") + "/shorts"
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "--flat-playlist",
            "--print", "%(id)s|%(title)s|%(view_count)s|%(duration)s",
            shorts_url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
    except Exception as e:
        return {"ok": False, "error": f"채널 영상 list 실패: {e}"}

    videos = []
    for line in out.decode().strip().split("\n"):
        parts = line.split("|", 3)
        if len(parts) < 1 or not parts[0]:
            continue
        try:
            vc = int(parts[2]) if len(parts) > 2 and parts[2] not in ("NA", "") else 0
        except Exception:
            vc = 0
        try:
            dur = float(parts[3]) if len(parts) > 3 and parts[3] not in ("NA", "") else 0
        except Exception:
            dur = 0
        videos.append({
            "video_id": parts[0],
            "title": parts[1] if len(parts) > 1 else "",
            "view_count": vc,
            "duration": dur,
        })
    await _emit(10, f"채널 영상 {len(videos)}개 수집")

    # 2. 메타 1차 — 조회수/길이 + 거부 키워드
    pass_meta = []
    suspect_meta = []
    excluded_meta = []
    for v in videos:
        if min_views > 0 and v["view_count"] < min_views:
            excluded_meta.append({**v, "reason": f"조회수 {v['view_count']:,} < {min_views:,}"})
            continue
        if max_duration > 0 and v["duration"] > max_duration:
            excluded_meta.append({**v, "reason": f"길이 {v['duration']}s > {max_duration}s"})
            continue
        verdict, reason = meta_match(v["title"], "", "", signature_words)
        if verdict is True:
            pass_meta.append({**v, "meta_verdict": "pass", "meta_reason": reason})
        elif verdict is False:
            excluded_meta.append({**v, "reason": reason})
        else:
            suspect_meta.append({**v, "meta_verdict": "suspect", "meta_reason": reason})

    await _emit(30, f"메타 1차: 통과 {len(pass_meta)} / 의심 {len(suspect_meta)} / 제외 {len(excluded_meta)}")

    # 3. Pro 2차 — 의심만 (skip_pro=True면 의심 다 통과)
    pro_results = []
    if skip_pro:
        pass_meta.extend(suspect_meta)
    else:
        for idx, v in enumerate(suspect_meta):
            pct = 30 + int((idx + 1) / max(len(suspect_meta), 1) * 60)
            await _emit(pct, f"Pro 검증 {idx+1}/{len(suspect_meta)}: {v['title'][:40]}")
            mp4 = ORIG_DIR / f"{hashlib.md5(v['video_id'].encode()).hexdigest()[:16]}.mp4"
            ok = await download_video(v["video_id"], mp4)
            if not ok:
                v["pro_verdict"] = "no_download"
                pro_results.append(v)
                continue
            frames = await extract_frames(mp4, v["video_id"])
            if not frames:
                v["pro_verdict"] = "no_frames"
                pro_results.append(v)
                continue
            result = await pro_check_video(v, dna_summary, frames)
            v.update({"pro_verdict": result["verdict"],
                      "pro_confidence": result["confidence"],
                      "pro_reason": result["reason"]})
            pro_results.append(v)
            if result["verdict"] == "yes" and result["confidence"] >= 5:
                pass_meta.append(v)
            if idx + 1 < len(suspect_meta):
                await asyncio.sleep(8)

    await _emit(100, f"끝 — 통과 {len(pass_meta)}개 / 제외 {len(excluded_meta) + len([p for p in pro_results if p.get('pro_verdict') not in ('yes',)])}개")

    return {
        "ok": True,
        "channel_url": channel_url,
        "total_videos": len(videos),
        "passed": pass_meta,
        "excluded": excluded_meta,
        "pro_results": pro_results,
    }



async def filter_search_videos(
    keywords: list[str],
    cate_name: str,
    dna_summary: str,
    signature_words: list[str],
    per_keyword: int = 30,
    min_views: int = 100000,
    max_duration: int = 90,
    skip_pro: bool = False,
    progress_cb=None,
) -> dict:
    """키워드 list로 YouTube 검색 → 영상 list → 메타 + Pro 결 매칭 → 통과 영상.

    동작:
    1. 각 키워드로 ytsearch{N} → 영상 list (중복 제거)
    2. 각 영상 메타 1차 (조회수/길이/시그니처/거부 키워드)
    3. 의심 영상은 Pro 2차 (frame 다운 + Pro 결 판단)
    4. 통과한 영상 list 반환
    """
    async def _emit(pct, msg):
        if progress_cb:
            await progress_cb(pct, msg)
        print(f"[{pct}%] {msg}", flush=True)

    # 1. 검색
    await _emit(5, f"키워드 {len(keywords)}개 검색 시작")
    all_videos = {}  # video_id -> meta
    for kw_idx, kw in enumerate(keywords[:10]):
        try:
            proc = await asyncio.create_subprocess_exec(
                YT_DLP, "--flat-playlist",
                "--print", "%(id)s|%(title)s|%(view_count)s|%(duration)s|%(channel)s|%(channel_id)s|%(uploader_url)s",
                f"ytsearch{per_keyword}:{kw}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        except Exception:
            continue
        for line in out.decode().strip().split("\n"):
            parts = line.split("|", 6)
            if len(parts) < 1 or not parts[0] or parts[0] == "NA":
                continue
            vid = parts[0]
            if vid in all_videos:
                continue
            try:
                vc = int(parts[2]) if len(parts) > 2 and parts[2] not in ("NA", "") else 0
            except Exception:
                vc = 0
            try:
                dur = float(parts[3]) if len(parts) > 3 and parts[3] not in ("NA", "") else 0
            except Exception:
                dur = 0
            all_videos[vid] = {
                "video_id": vid,
                "title": parts[1] if len(parts) > 1 else "",
                "view_count": vc,
                "duration": dur,
                "channel_name": parts[4] if len(parts) > 4 else "",
                "channel_id": parts[5] if len(parts) > 5 else "",
                "uploader_url": parts[6] if len(parts) > 6 else "",
                "matched_keyword": kw,
            }
        pct = 5 + int((kw_idx + 1) / max(len(keywords[:10]), 1) * 20)
        await _emit(pct, f"검색 {kw_idx+1}/{min(len(keywords),10)} '{kw}' — 누적 {len(all_videos)}개")

    videos = list(all_videos.values())
    await _emit(25, f"검색 결과: {len(videos)}개 유니크 영상")

    # 2. 메타 1차 — 조회수, 길이, 거부 키워드, 시그니처
    pass_meta = []
    suspect_meta = []
    excluded = []
    for v in videos:
        if min_views > 0 and v["view_count"] < min_views:
            excluded.append({**v, "reason": f"조회수 {v['view_count']:,} < {min_views:,}"})
            continue
        if max_duration > 0 and v["duration"] > 0 and v["duration"] > max_duration:
            excluded.append({**v, "reason": f"길이 {v['duration']:.0f}s > {max_duration}s"})
            continue
        verdict, reason = meta_match(v["title"], "", v.get("channel_name", ""), signature_words)
        if verdict is True:
            pass_meta.append({**v, "meta_verdict": "pass", "meta_reason": reason})
        elif verdict is False:
            excluded.append({**v, "reason": f"메타 거부: {reason}"})
        else:
            suspect_meta.append({**v, "meta_verdict": "suspect", "meta_reason": reason})

    await _emit(40, f"메타 1차: 통과 {len(pass_meta)} / 의심 {len(suspect_meta)} / 제외 {len(excluded)}")

    # 3. Pro 2차 — 의심만
    pro_results = []
    if skip_pro:
        pass_meta.extend(suspect_meta)
    else:
        # Pro 호출 비용 관리 — 의심 50개 이상이면 상위 view_count로 cap
        suspect_meta.sort(key=lambda x: -x.get("view_count", 0))
        suspect_to_check = suspect_meta[:50]
        if len(suspect_meta) > 50:
            await _emit(45, f"의심 {len(suspect_meta)}개 중 상위 50개만 Pro 검증 (조회수 순)")

        for idx, v in enumerate(suspect_to_check):
            pct = 45 + int((idx + 1) / max(len(suspect_to_check), 1) * 50)
            await _emit(pct, f"Pro 검증 {idx+1}/{len(suspect_to_check)}: {v['title'][:40]}")
            mp4 = ORIG_DIR / f"{hashlib.md5(v['video_id'].encode()).hexdigest()[:16]}.mp4"
            ok = await download_video(v["video_id"], mp4)
            if not ok:
                v["pro_verdict"] = "no_download"
                pro_results.append(v)
                continue
            frames = await extract_frames(mp4, v["video_id"])
            if not frames:
                v["pro_verdict"] = "no_frames"
                pro_results.append(v)
                continue
            result = await pro_check_video(v, dna_summary, frames)
            v.update({
                "pro_verdict": result["verdict"],
                "pro_confidence": result["confidence"],
                "pro_reason": result["reason"],
            })
            pro_results.append(v)
            if result["verdict"] == "yes" and result["confidence"] >= 5:
                pass_meta.append(v)
            if idx + 1 < len(suspect_to_check):
                await asyncio.sleep(8)

    await _emit(100, f"끝 — 통과 {len(pass_meta)} (메타+Pro), 제외 {len(excluded)} + Pro 부적합")

    return {
        "ok": True,
        "keywords": keywords[:10],
        "total_searched": len(videos),
        "passed": pass_meta,
        "excluded_count": len(excluded),
        "suspect_pro_checked": len(pro_results),
    }


import hashlib as _hashlib


async def recheck_candidate_pool(
    job_id: str,
    diss_id: str,
    cate_name: str,
    dna_summary: str,
    signature_words: list,
    max_pro_checks: int = 50,
    progress_cb=None,
) -> dict:
    """후보풀의 used=0 영상에 결 매칭 후처리. 결 안 맞는 영상 자동 제외."""
    async def _emit(pct, msg):
        if progress_cb:
            await progress_cb(pct, msg)
        print(f"[recheck {pct}%] {msg}", flush=True)

    with sqlite3.connect(DB_PATH, timeout=30) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT id, video_id, url, title, caption, channel_name FROM candidate_videos "
            "WHERE job_id=? AND (used=0 OR used IS NULL)",
            (job_id,),
        ).fetchall()
    cands = [dict(r) for r in rows]
    await _emit(0, f"후보풀 used=0 영상 {len(cands)}개")

    meta_excluded = 0
    suspect = []
    for c in cands:
        v, r = meta_match(
            c.get("title") or "",
            c.get("caption") or "",
            c.get("channel_name") or "",
            signature_words,
        )
        if v is False:
            note = "[메타거부] " + r
            with sqlite3.connect(DB_PATH, timeout=30) as c2:
                c2.execute(
                    "UPDATE candidate_videos SET used=1, classification='제외', "
                    "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                    (note, c["id"]),
                )
                c2.commit()
            meta_excluded += 1
        elif v is None:
            suspect.append(c)
    await _emit(20, f"메타 거부 {meta_excluded} / Pro 검증 대기 {len(suspect)}")

    # Pro 의심만 (max_pro_checks)
    suspect = suspect[:max_pro_checks]
    pro_excluded = 0
    for idx, c in enumerate(suspect):
        pct = 20 + int((idx + 1) / max(len(suspect), 1) * 75)
        await _emit(pct, f"Pro 검증 {idx+1}/{len(suspect)}")
        mp4 = ORIG_DIR / (_hashlib.md5(c["url"].encode()).hexdigest()[:16] + ".mp4")
        ok = await download_video(c["video_id"], mp4)
        if not ok:
            # 다운 실패 — title + 채널명에 시그니처 다 없으면 자동 제외 (명백 비-카테)
            title_l = (c.get("title") or "").lower()
            ch_l = (c.get("channel_name") or "").lower()
            both_empty_sig = (not any(w in title_l for w in signature_words)
                              and not any(w in ch_l for w in signature_words))
            if both_empty_sig:
                note = "[다운실패+시그니처전무] 자동제외"
                with sqlite3.connect(DB_PATH, timeout=30) as c2:
                    c2.execute(
                        "UPDATE candidate_videos SET used=1, classification='제외', "
                        "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                        (note, c["id"]),
                    )
                    c2.commit()
                pro_excluded += 1
            continue
        frames = await extract_frames(mp4, c["video_id"])
        if not frames:
            continue
        result = await pro_check_video(
            {"title": c.get("title") or "", "channel_name": c.get("channel_name") or ""},
            dna_summary, frames,
        )
        if result["verdict"] == "no" and result["confidence"] >= 7:
            note = "[Pro결거부] " + (result.get("reason") or "")
            with sqlite3.connect(DB_PATH, timeout=30) as c2:
                c2.execute(
                    "UPDATE candidate_videos SET used=1, classification='제외', "
                    "memo_kr = COALESCE(memo_kr || ' / ', '') || ? WHERE id=?",
                    (note, c["id"]),
                )
                c2.commit()
            pro_excluded += 1
        if idx + 1 < len(suspect):
            await asyncio.sleep(8)

    await _emit(100, f"끝 - 메타 {meta_excluded} + Pro {pro_excluded} 제외")
    return {
        "ok": True,
        "meta_excluded": meta_excluded,
        "pro_excluded": pro_excluded,
        "pro_checked": len(suspect),
    }
