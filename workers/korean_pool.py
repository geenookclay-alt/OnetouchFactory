"""Korean pool builder — discover → classify → index Korean reference channels.

End-to-end orchestrator that takes a dissection's DNA and builds out the
visual-match pool: KR keyword search → channel discovery → DNA classification
→ frame extraction + CLIP indexing. Idempotent — already-indexed videos
are skipped so you can re-run to top up.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import korean_discovery
from . import channel_classifier
from . import youtube_client
from . import visual_match

sys.path.insert(0, str(Path(__file__).parent.parent))
from api import database as db


ProgressCb = Callable[[int, str], Awaitable[None]]


async def _noop_progress(pct: int, msg: str) -> None:
    return None


async def build_korean_pool(
    dissection_id: str,
    target_dna: dict,
    progress_cb: ProgressCb | None = None,
    max_channels: int = 50,
    videos_per_channel: int = 8,
    max_frames: int = 40,
    max_keywords: int = 10,
    classify_concurrency: int = 4,
    index_concurrency: int = 2,
) -> dict:
    """Run the full pool-build pipeline.

    Returns aggregate stats: discovered/reference/indexed counts.
    """
    progress = progress_cb or _noop_progress

    # 1. Korean keywords + discovery
    await progress(5, "한국어 키워드 생성")
    keywords = await korean_discovery.generate_korean_keywords(
        target_dna, n=max_keywords)
    if not keywords:
        await progress(100, "키워드 생성 실패 — 풀 빌드 종료")
        return {"keywords": [], "discovered": 0, "reference": 0, "indexed": 0}

    await progress(15, f"한국 채널 발굴 ({len(keywords)}개 키워드)")
    pool = await korean_discovery.discover_korean_channels(
        target_dna, keywords=keywords, max_channels=max_channels)
    await progress(30, f"발굴 채널: {len(pool)}개 → 결 판단 중")

    # 2. Classify each channel (bounded concurrency)
    sem_classify = asyncio.Semaphore(classify_concurrency)

    async def _classify_one(entry: dict) -> tuple[dict, dict]:
        async with sem_classify:
            try:
                result = await channel_classifier.classify_channel(
                    entry["channel_id"], target_dna)
            except Exception as e:
                result = {"is_reference": False, "matching_count": 0,
                          "total_sampled": 0, "matching_ratio": 0.0,
                          "videos": [], "error": str(e)}
            return entry, result

    classify_tasks = [_classify_one(e) for e in pool]
    references: list[tuple[dict, dict]] = []
    done = 0
    for coro in asyncio.as_completed(classify_tasks):
        entry, result = await coro
        done += 1
        db.add_korean_pool_channel(
            dissection_id=dissection_id,
            channel_id=entry["channel_id"],
            channel_name=entry.get("name", ""),
            is_reference=bool(result.get("is_reference")),
            matching_count=int(result.get("matching_count", 0)),
            total_sampled=int(result.get("total_sampled", 0)),
            matching_ratio=float(result.get("matching_ratio", 0.0)),
        )
        if result.get("is_reference"):
            references.append((entry, result))
        if done % 5 == 0:
            pct = 30 + int(20 * (done / max(len(pool), 1)))
            await progress(pct, f"결 판단 {done}/{len(pool)} → 레퍼런스 {len(references)}")

    await progress(50, f"레퍼런스 채널 확정: {len(references)}개")

    # 3. Pull recent shorts for each reference channel + index frames
    sem_index = asyncio.Semaphore(index_concurrency)
    indexed_total = 0

    async def _index_channel(entry: dict, classify_result: dict) -> int:
        ch_id = entry["channel_id"]
        ch_name = entry.get("name", "")
        # Reuse shorts from classification when available, otherwise fetch
        videos = [v for v in classify_result.get("videos", [])
                  if v.get("match")] or []
        if len(videos) < videos_per_channel:
            try:
                more = await youtube_client.get_channel_videos(
                    ch_id, max_results=videos_per_channel * 2)
                seen = {v.get("video_id") for v in videos}
                for v in more:
                    if v.get("video_id") in seen:
                        continue
                    if 0 < (v.get("duration") or 0) <= 65:
                        videos.append(v)
                    if len(videos) >= videos_per_channel:
                        break
            except Exception:
                pass
        videos = videos[:videos_per_channel]

        count = 0
        for v in videos:
            vid = v.get("video_id")
            url = v.get("url") or (vid and f"https://www.youtube.com/shorts/{vid}")
            if not vid or not url:
                continue
            if db.is_video_in_pool(vid):
                count += 1
                continue
            async with sem_index:
                try:
                    n = await visual_match.index_video(
                        url, vid, channel_id=ch_id,
                        channel_handle=entry.get("handle", ""),
                        max_frames=max_frames,
                    )
                except Exception:
                    n = 0
            if n > 0:
                db.add_korean_pool_video(
                    dissection_id=dissection_id,
                    channel_id=ch_id,
                    channel_handle=entry.get("handle", ""),
                    channel_name=ch_name,
                    video_id=vid,
                    video_url=url,
                    title=v.get("title"),
                    frames_count=n,
                )
                count += 1
        return count

    index_tasks = [_index_channel(e, r) for e, r in references]
    done = 0
    for coro in asyncio.as_completed(index_tasks):
        n = await coro
        indexed_total += n
        done += 1
        if done % 2 == 0 or done == len(index_tasks):
            pct = 50 + int(45 * (done / max(len(index_tasks), 1)))
            await progress(pct, f"인덱싱 {done}/{len(index_tasks)} 채널 ({indexed_total} 영상)")

    final = db.pool_stats()
    await progress(100,
                   f"완료: 풀 영상 {final['indexed_videos']}개 / 채널 {final['indexed_channels']}개")
    return {
        "keywords": keywords,
        "discovered": len(pool),
        "reference": len(references),
        "indexed_this_run": indexed_total,
        **final,
    }
