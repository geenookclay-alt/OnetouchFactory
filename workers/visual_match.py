"""Visual match — does a candidate video already exist in the KR pool?

Pipeline:
  1. Pull the candidate's storyboard frames
  2. CLIP-embed them
  3. Per-frame Qdrant search → aggregate scores by matched video_id
  4. Apply thresholds (EXCLUDE / SUSPECT / NEW)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from . import video_storyboard
from . import clip_engine
from . import qdrant_index


# Aggregated similarity score thresholds (sum of per-frame cosine sims, scaled).
# With 30~50 frames and cosine [0,1], a strong full match lands ~25~40.
EXCLUDE_THRESHOLD = 30.0
SUSPECT_THRESHOLD = 15.0

PER_FRAME_TOP_K = 5
# Per-frame cosine threshold. 0.60 catches Korean re-uploads with subtitle/crop
# transformations (validated 2026-05-07: A↔B same-video variant scores 85.9
# EXCLUDE, while unrelated pairs and even same-channel-different-videos stay
# below 2.0 NEW). Pre-2026-05-07 value was 0.78 which missed transformed
# re-uploads (A↔B scored 0.0).
MIN_PER_FRAME_SCORE = 0.60


async def index_video(video_url: str, video_id: str, channel_id: str,
                      channel_handle: str = "",
                      max_frames: int = 50) -> int:
    """Pull storyboard, embed, push into Qdrant. Returns # frames indexed."""
    frames = await video_storyboard.extract_frames(video_url, max_frames=max_frames)
    if not frames:
        return 0
    vectors = await clip_engine.embed_images_async(frames)
    items = [
        {
            "vector": vectors[i],
            "video_id": video_id,
            "frame_idx": i,
            "channel_id": channel_id,
            "video_url": video_url,
            "channel_handle": channel_handle,
        }
        for i in range(len(vectors))
    ]
    return qdrant_index.upsert_frames(items)


async def match_candidate(video_url: str,
                          max_frames: int = 50) -> dict:
    """Score how strongly the candidate overlaps with the indexed KR pool."""
    frames = await video_storyboard.extract_frames(video_url, max_frames=max_frames)
    if not frames:
        return {"verdict": "NEW", "score": 0.0, "reason": "no_frames",
                "best_match_video": None, "matched_channel": None,
                "frames_used": 0, "matches": []}

    vectors = await clip_engine.embed_images_async(frames)

    # Aggregate: sum top-k similarities per candidate video, weighted by score
    by_video: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"score": 0.0, "frames_hit": 0,
                 "channel_id": "", "channel_handle": "", "video_url": ""}
    )
    for vec in vectors:
        hits = qdrant_index.search(vec, limit=PER_FRAME_TOP_K)
        for h in hits:
            s = h["score"]
            if s < MIN_PER_FRAME_SCORE:
                continue
            p = h["payload"]
            vid = p.get("video_id", "")
            if not vid:
                continue
            agg = by_video[vid]
            agg["score"] += s
            agg["frames_hit"] += 1
            agg["channel_id"] = p.get("channel_id", agg["channel_id"])
            agg["channel_handle"] = p.get("channel_handle", agg["channel_handle"])
            agg["video_url"] = p.get("video_url", agg["video_url"])

    if not by_video:
        return {"verdict": "NEW", "score": 0.0, "reason": "no_hits",
                "best_match_video": None, "matched_channel": None,
                "frames_used": len(vectors), "matches": []}

    ranked = sorted(by_video.items(), key=lambda kv: -kv[1]["score"])
    best_id, best = ranked[0]
    score = round(best["score"], 2)
    if score >= EXCLUDE_THRESHOLD:
        verdict = "EXCLUDE"
    elif score >= SUSPECT_THRESHOLD:
        verdict = "SUSPECT"
    else:
        verdict = "NEW"

    matches = [
        {
            "video_id": vid,
            "score": round(d["score"], 2),
            "frames_hit": d["frames_hit"],
            "channel_id": d["channel_id"],
            "channel_handle": d["channel_handle"],
            "video_url": d["video_url"],
        }
        for vid, d in ranked[:5]
    ]
    return {
        "verdict": verdict,
        "score": score,
        "best_match_video": best_id,
        "matched_channel": best["channel_id"],
        "matched_handle": best["channel_handle"],
        "matched_url": best["video_url"],
        "frames_used": len(vectors),
        "matches": matches,
    }
