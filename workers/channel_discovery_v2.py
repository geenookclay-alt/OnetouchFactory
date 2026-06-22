"""채널 자동 발굴 v2 — 키워드 강화 + 검증된 채널 description mention 추출.

핵심 개선:
1. 키워드 다양화 (lego shorts, lego moc, minifigure 등 시그니처 위주)
2. 검증된 채널 (c_lego.json의 영상이 후보풀에 있고 used=1된 채널 + 시드)
   의 영상 description에서 mention된 다른 채널 추출
3. Pro Vision 검증
4. 통과한 채널 → 후보풀에 자동 추가 (autoCollect 호출)
"""
import asyncio
import base64
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"
TMP_DIR = Path("/tmp/discover_channels_v2")
TMP_DIR.mkdir(exist_ok=True)
YT_DLP = "/Users/shortsking/banbaji-discover/venv/bin/yt-dlp"

# 키워드 카테별 시그니처 (기본 강화)
SIGNATURE_KEYWORDS = {
    "lego": ["lego shorts", "lego moc", "lego speed build", "lego minifigure", "lego stop motion", "lego technic", "afol", "brickfilm", "lego custom", "lego bricks"],
    "레고": ["lego shorts", "lego moc", "lego speed build", "lego minifigure", "lego stop motion", "lego technic", "afol", "brickfilm"],
    "브릭": ["lego bricks", "lego speed build"],
    "강아지 미용": ["dog grooming", "poodle grooming", "shih tzu grooming", "dog haircut", "groomer"],
    "미용": ["dog grooming", "poodle grooming"],
    "볼링": ["bowling strike", "bowling shorts", "bowling trick shot"],
    "곤충": ["insect close up", "beetle", "praying mantis", "bug catching"],
    "도자기": ["pottery", "ceramic wheel", "clay throwing"],
    "클라이밍": ["climbing", "bouldering", "rock climbing"],
    "말": ["horse riding", "horse training"],
    "3d": ["3d printing shorts", "3d print timelapse"],
    "복원": ["restoration", "rust removal", "refurbish"],
    "비트박스": ["beatbox", "beatboxing"],
    "마술": ["magic trick", "magician shorts", "sleight of hand"],
    "모델": ["model walk", "fashion model shorts"],
    "배드민턴": ["badminton smash", "badminton trick"],
    "젤더 리빌": ["gender reveal", "baby gender reveal"],
    "젤더리빌": ["gender reveal", "baby gender reveal"],
    "제더 리빌": ["gender reveal", "baby gender reveal"],
    "제더리빌": ["gender reveal", "baby gender reveal"],
    "default": [],
}


def get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return open("/Users/shortsking/banbaji-discover/.env").read().split("GEMINI_API_KEY=")[1].split("\n")[0].strip()
    except Exception:
        return ""


def get_carte_signature_keywords(diss_id: str, cate_hint: str = "") -> list[str]:
    """카테 시그니처 키워드 — 시그니처 + keywords_result 합집합."""
    keywords = []
    # 카테 힌트로 시그니처 찾기
    cate_lower = cate_hint.lower()
    for key, sigs in SIGNATURE_KEYWORDS.items():
        if key in cate_lower:
            keywords.extend(sigs)
            break

    # keywords_result에서 영어 키워드 추가
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT keywords_result FROM dissection_analyses WHERE id=?",
            (diss_id,),
        ).fetchone()
    if row and row[0]:
        try:
            d = json.loads(row[0])
            for item in d.get("english", []):
                kw = item.get("keyword") if isinstance(item, dict) else str(item)
                if kw and kw not in keywords:
                    keywords.append(kw)
        except Exception:
            pass
    return keywords


def get_seed_channels(job_id: str, diss_id: str = None) -> list[dict]:
    """후보풀 안에서 used=1 마킹된 영상의 채널 list = 한국이 카피한 카테 채널 = 검증된 시드."""
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT DISTINCT channel_id, channel_name, url FROM candidate_videos "
            "WHERE job_id=? AND used=1 AND channel_id IS NOT NULL AND channel_id != ''",
            (job_id,),
        ).fetchall()
    out = []
    for cid, name, url in rows:
        if cid and cid not in [o["id"] for o in out]:
            out.append({"id": cid, "name": name or "", "sample_url": url or "", "source": "used=1"})
    # reference_channels에서도 시드 추가
    if diss_id:
        try:
            with sqlite3.connect(DB_PATH) as c:
                row = c.execute(
                    "SELECT reference_channels FROM dissection_analyses WHERE id=?",
                    (diss_id,),
                ).fetchone()
            if row and row[0]:
                refs = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
                for item in refs[:20]:
                    if isinstance(item, list) and len(item) >= 2:
                        handle = str(item[1])
                    elif isinstance(item, str):
                        handle = item
                    else:
                        continue
                    handle = handle.replace("https://www.youtube.com/", "").lstrip("@").strip("/")
                    if len(handle) >= 3:
                        out.append({"id": "@" + handle, "name": handle, "source": "reference"})
        except Exception as e:
            print(f"  reference load fail: {e}", flush=True)
    return out


# 채널 URL/mention 추출 패턴
CHANNEL_PATTERNS = [
    re.compile(r"https?://(?:www\.)?youtube\.com/(?:c/|channel/|@|user/)([A-Za-z0-9_\-]+)"),
    re.compile(r"(?<![\w@])@([A-Za-z0-9_\-]{3,40})"),  # @handle
]


def extract_channels_from_description(text: str) -> set:
    """description에서 채널 handle/URL 추출."""
    found = set()
    if not text:
        return found
    for pat in CHANNEL_PATTERNS:
        for m in pat.finditer(text):
            handle = m.group(1)
            if len(handle) < 3:
                continue
            # 너무 일반적인 단어 제외
            if handle.lower() in ("subscribe", "comment", "like", "share", "music", "video"):
                continue
            found.add(handle.lower())
    return found


def fetch_video_descriptions(channel_url: str, n: int = 30) -> list[str]:
    """채널의 최근 N개 영상 description 모음."""
    descs = []
    try:
        # flat-playlist로 영상 ID
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist", "--print", "%(id)s",
             "--playlist-end", str(n), channel_url.rstrip("/") + "/shorts"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return descs
    ids = [x for x in proc.stdout.strip().split("\n") if x][:n]
    for vid in ids[:n]:
        try:
            p2 = subprocess.run(
                [YT_DLP, "--skip-download", "--print", "%(description)s",
                 f"https://www.youtube.com/watch?v={vid}"],
                capture_output=True, text=True, timeout=30,
            )
            if p2.stdout.strip():
                descs.append(p2.stdout.strip())
        except Exception:
            continue
    return descs


def yt_search_channels(keyword: str, n: int = 20) -> list[dict]:
    """yt-dlp 키워드 검색 → 채널 list."""
    try:
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist",
             "--print", "%(channel_id)s|%(channel)s|%(uploader_url)s|%(title)s",
             f"ytsearch{n}:{keyword}"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return []
    seen = set()
    out = []
    for line in proc.stdout.strip().split("\n"):
        parts = line.split("|", 3)
        if len(parts) < 3 or not parts[0] or parts[0] == "NA":
            continue
        if parts[0] in seen:
            continue
        seen.add(parts[0])
        # channel_url NA면 channel_id로 만듦
        ch_url = parts[2] if (parts[2] and parts[2] != "NA") else ""
        if not ch_url and parts[0]:
            ch_url = f"https://www.youtube.com/channel/{parts[0]}"
        out.append({
            "channel_id": parts[0],
            "channel_name": parts[1] or "",
            "channel_url": ch_url,
            "sample_title": parts[3] if len(parts) > 3 else "",
        })
    return out


def fetch_channel_meta(handle: str) -> dict | None:
    """handle로 채널 URL 정확히 찾기."""
    url = f"https://www.youtube.com/@{handle}" if not handle.startswith("@") else f"https://www.youtube.com/{handle}"
    try:
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist", "--print", "%(channel_id)s|%(channel)s|%(uploader_url)s",
             "--playlist-end", "1",
             url + "/shorts"],
            capture_output=True, text=True, timeout=30,
        )
        line = proc.stdout.strip().split("\n")[0] if proc.stdout.strip() else ""
        if not line or line == "NA":
            return None
        parts = line.split("|", 2)
        if len(parts) < 2 or not parts[0] or parts[0] == "NA":
            return None
        return {
            "channel_id": parts[0],
            "channel_name": parts[1] or handle,
            "channel_url": (parts[2] or url) if len(parts) > 2 else url,
            "sample_title": "",
        }
    except Exception:
        return None


def channel_short_frames(channel_url: str, n: int = 3) -> list[Path]:
    shorts_url = channel_url.rstrip("/") + "/shorts"
    try:
        proc = subprocess.run(
            [YT_DLP, "--flat-playlist", "--print", "%(id)s",
             "--playlist-end", str(n), shorts_url],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return []
    ids = [x for x in proc.stdout.strip().split("\n") if x][:n]
    if not ids:
        return []
    paths = []
    for vid in ids:
        mp4 = TMP_DIR / f"{vid}.mp4"
        if not mp4.exists():
            try:
                subprocess.run(
                    [YT_DLP, "-f", "best[height<=480]", "-o", str(mp4),
                     f"https://www.youtube.com/watch?v={vid}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=90,
                )
            except Exception:
                continue
        jpg = TMP_DIR / f"{vid}.jpg"
        if not jpg.exists():
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "5", "-i", str(mp4),
                     "-frames:v", "1", "-q:v", "5", str(jpg)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=20,
                )
            except Exception:
                continue
        if jpg.exists():
            paths.append(jpg)
    return paths


async def verify_channel_with_pro(channel: dict, dna_summary: str, frames: list[Path]) -> dict:
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
        f"채널 이름: {channel.get('channel_name', '')}\n\n"
        f"이 채널의 영상 frame {len(frames)}장 보고 2단계 판단:\n\n"
        f"1단계: 이 채널이 카테(예: 레고)의 메인 콘텐츠를 다루는 채널인가?\n"
        f"   - 카테 영상 (예: 레고 빌드/실험/리뷰) 80%+ = 'yes'\n"
        f"   - 일반 영상/다양한 주제 = 'no'\n\n"
        f"2단계 (1단계 yes일 때만): 카테 DNA의 세부 결(예: 실험/테스트 위주)에 얼마나 맞나? 0~10점\n"
        f"   - 완벽히 일치 = 10\n"
        f"   - 큰 카테는 맞지만 세부 결 다름 = 5~7\n"
        f"   - 큰 카테 다름 = 0~3\n\n"
        f'JSON: {{"verdict": "yes"|"no", "confidence": 0~10, "reason": "한 줄 한국어 (왜)"}}\n\n'
        f"frame {len(frames)}장:"
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


def get_existing_channels(job_id: str) -> set:
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT DISTINCT channel_id FROM candidate_videos WHERE job_id=?",
            (job_id,),
        ).fetchall()
    return {r[0] for r in rows if r[0]}


def get_rejected_channels() -> tuple[set, set]:
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT channel_id, channel_name FROM rejected_channels").fetchall()
    ids = {r[0] for r in rows if r[0]}
    names = {r[1] for r in rows if r[1]}
    return ids, names


async def discover_v2(job_id: str, diss_id: str, target_new: int = 20) -> dict:
    print(f"=== 채널 자동 발굴 v2 (job={job_id}, diss={diss_id}) ===", flush=True)

    # 1. 카테 이름 + DNA + 키워드
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT name, dissection_result, reference_channels FROM dissection_analyses WHERE id=?",
            (diss_id,),
        ).fetchone()
    if not row:
        return {"ok": False, "error": "dissection not found"}
    cate_name = row[0] or ""
    dna_text = f"카테 = {cate_name}"
    try:
        d = json.loads(row[1] or "{}")
        common = d.get("common_dna") or {}
        items = common.get("items") or {}
        for k in ("4_view_drivers", "6_topics_list"):
            it = items.get(k) or {}
            v = it.get("value") or ""
            if v:
                dna_text += f"\n{it.get('label', k)}: {v[:200]}"
        if "common_dna" not in d:
            chans = d.get("channels") or []
            if chans:
                dna_text += "\n" + (chans[0].get("summary_kr") or "")[:300]
    except Exception:
        pass

    keywords = get_carte_signature_keywords(diss_id, cate_name)
    print(f"카테: {cate_name}", flush=True)
    print(f"DNA: {dna_text[:200]}", flush=True)
    print(f"키워드 {len(keywords)}개: {keywords[:8]}...", flush=True)

    # 2. 시드 채널 (used=1 마킹된 채널) + reference 채널
    seed_channels = get_seed_channels(job_id, diss_id)
    print(f"시드 채널 (검증됨, used=1 마킹된 거): {len(seed_channels)}", flush=True)

    # 3. 시드 채널 description에서 mention된 채널 추출
    print("\n=== description에서 채널 mention 추출 ===", flush=True)
    mentioned_handles = set()
    for sc in seed_channels[:10]:
        sid = sc.get("id", "")
        if sid.startswith("@"):
            url = "https://www.youtube.com/" + sid
        else:
            url = "https://www.youtube.com/channel/" + sid
        print(f"  {sc.get('name', '?')} ({sc.get('source', '?')}) description scan...", flush=True)
        descs = fetch_video_descriptions(url, n=15)
        for desc in descs:
            mentioned_handles.update(extract_channels_from_description(desc))
    print(f"description에서 추출: {len(mentioned_handles)}개 handle", flush=True)

    # 4. 키워드 검색 결과
    print("\n=== 키워드 검색 ===", flush=True)
    search_channels = {}
    for kw in keywords[:8]:
        print(f"  검색: {kw}", flush=True)
        results = yt_search_channels(kw, n=50)
        for ch in results:
            search_channels[ch["channel_id"]] = ch
        await asyncio.sleep(2)
    print(f"검색 결과: {len(search_channels)} 유니크 채널", flush=True)

    # 5. mention된 handle을 채널 메타로 해석
    print(f"\n=== mention handle → 채널 메타 해석 ({len(mentioned_handles)}) ===", flush=True)
    mention_channels = {}
    for handle in list(mentioned_handles)[:30]:
        meta = fetch_channel_meta(handle)
        if meta and meta.get("channel_id") and meta["channel_id"] not in search_channels:
            mention_channels[meta["channel_id"]] = meta
    print(f"해석된 mention 채널: {len(mention_channels)}", flush=True)

    # 6. 모든 후보 합치고 기존/거부 제외
    all_candidates = {**search_channels, **mention_channels}
    existing = get_existing_channels(job_id)
    rej_ids, rej_names = get_rejected_channels()
    new_candidates = [
        ch for cid, ch in all_candidates.items()
        if cid not in existing
        and cid not in rej_ids
        and ch["channel_name"] not in rej_names
        and ch.get("channel_url")
    ]
    print(f"\n새 채널 후보 (기존/거부 제외 후): {len(new_candidates)}", flush=True)

    # mention 출처 우선 정렬 (시드 채널이 mention한 거니까 높은 가능성)
    mention_ids = set(mention_channels.keys())
    new_candidates.sort(key=lambda c: 0 if c["channel_id"] in mention_ids else 1)

    # 7. Pro 검증
    print("\n=== Pro 검증 ===", flush=True)
    passed = []
    failed = []
    for idx, ch in enumerate(new_candidates[:target_new * 3]):
        source = "mention" if ch["channel_id"] in mention_ids else "search"
        print(f"[{idx+1}/{min(len(new_candidates), target_new*3)}] {ch['channel_name']} ({source})", flush=True)
        frames = channel_short_frames(ch["channel_url"], n=3)
        if not frames:
            failed.append({**ch, "verdict": "no_frames", "source": source})
            continue
        result = await verify_channel_with_pro(ch, dna_text, frames)
        ch.update(result)
        ch["source"] = source
        if result["verdict"] == "yes" and result["confidence"] >= 5:
            passed.append(ch)
            print(f"  ✓ {result['reason']}", flush=True)
        else:
            failed.append(ch)
            print(f"  ✗ ({result['verdict']}) {result['reason']}", flush=True)
        if len(passed) >= target_new:
            print(f"  목표 {target_new}개 달성", flush=True)
            break
        await asyncio.sleep(8)

    result = {
        "ok": True,
        "job_id": job_id,
        "diss_id": diss_id,
        "cate_name": cate_name,
        "keywords": keywords[:8],
        "seed_channels": len(seed_channels),
        "mentioned_handles": len(mentioned_handles),
        "mention_resolved": len(mention_channels),
        "search_channels": len(search_channels),
        "new_candidates": len(new_candidates),
        "passed": len(passed),
        "failed": len(failed),
        "passed_channels": passed,
        "failed_sample": failed[:15],
    }
    print(f"\n=== 끝 — 통과 {len(passed)} / 실패 {len(failed)} ===", flush=True)
    for ch in passed:
        print(f"  ⭐ {ch['channel_name']} ({ch.get('source', '?')}) — {ch.get('reason', '')}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python channel_discovery_v2.py <job_id> <diss_id> [target=20]")
        sys.exit(1)
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    result = asyncio.run(discover_v2(sys.argv[1], sys.argv[2], target_new=target))
    Path("/tmp/channel_discovery_v2_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print("\n저장: /tmp/channel_discovery_v2_result.json")
