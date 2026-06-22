"""NexLev N8N For YouTube REST client (https://prod.dashboard.nexlev.io).

All endpoints require the NEXLEV_API_KEY env var (sent in the `x-api-key`
header). Quota is consumed against the user's NexLev plan — see
https://dashboard.nexlev.io/n8n-youtube/manage-api-key for usage.

Endpoint costs (🥞 = quota units):
  channel-analytics      🥞 10
  similar-channels       🥞 20
  similar-videos         🥞 10
  similar-thumbnails     🥞 10
  videos/details         🥞  1
  videos/transcript      🥞  1
"""
import os
import httpx


NEXLEV_API_KEY = os.getenv("NEXLEV_API_KEY", "")
NEXLEV_BASE = "https://prod.dashboard.nexlev.io"
DEFAULT_TIMEOUT = 30.0


def _headers() -> dict:
    return {
        "x-api-key": NEXLEV_API_KEY,
        "Content-Type": "application/json",
    }


async def _post(path: str, payload: dict, timeout: float = DEFAULT_TIMEOUT) -> dict | list | None:
    if not NEXLEV_API_KEY:
        return None
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{NEXLEV_BASE}{path}",
                              headers=_headers(), json=payload)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None


async def _get(path: str, params: dict, timeout: float = DEFAULT_TIMEOUT) -> dict | list | None:
    if not NEXLEV_API_KEY:
        return None
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{NEXLEV_BASE}{path}",
                             headers=_headers(), params=params)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None


# ============================================================
# Channel analytics & similarity
# ============================================================

async def get_channel_analytics(channel_id: str) -> dict | None:
    """🥞 10 — rich channel metadata."""
    data = await _post("/api/external/analytics/channel-analytics",
                       {"channelId": channel_id})
    if isinstance(data, list) and data:
        return data[0]
    return data


async def get_similar_channels(channel_id: str, channel_type: str = "all",
                               level: int = 1) -> list[dict]:
    """🥞 20 — find channels similar to a given channel.

    channel_type: "all" or "short"
    level: 1 (best matches) ~ 3 (broader). Default 1 recommended.
    Returns list of {about, tags, similarityScore, ...}.
    """
    data = await _post("/api/external/similar-channels/search", {
        "channelId": channel_id,
        "channelType": channel_type,
        "level": level,
    }, timeout=120.0)  # similarity search can be slow
    if isinstance(data, dict):
        return data.get("data") or data.get("channels") or []
    if isinstance(data, list):
        return data
    return []


async def get_similar_videos(video_id: str) -> list[dict]:
    """🥞 10 — find videos semantically similar to a given video.

    Returns list of {title, similarity_score, format_score, videoId,
                     channelTitle, channelId, viewCount, lengthText, ...}.
    """
    data = await _post("/api/external/similar-videos/videos",
                       {"videoId": video_id}, timeout=60.0)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or []
    return []


async def get_similar_thumbnails(query: str,
                                 query_type: str = "text",
                                 limit: int = 10) -> list[dict]:
    """🥞 10 — find videos with visually similar thumbnails.

    query_type: "text" (description) or "url" (image URL).
    Returns list of {similarity, id, youtubeVideoUrl, videoData, channelData,
                     calculatedData{outlierScore, isOutlier}, ...}.
    """
    if query_type not in ("text", "url"):
        raise ValueError("query_type must be 'text' or 'url'")
    payload = {"text": query} if query_type == "text" else {"image_url": query}
    data = await _post("/api/external/similar-thumbnails/search", payload,
                       timeout=60.0)
    if isinstance(data, dict):
        items = data.get("results") or data.get("thumbnails") or data.get("data") or []
        return items[:limit] if limit else items
    return []


# ============================================================
# Video details & transcripts
# ============================================================

async def get_video_details(video_id: str) -> dict | None:
    """🥞 1 — comprehensive metadata (title, viewCount, channelId, hasCaption, ...)."""
    data = await _get("/api/external/videos/details", {"videoId": video_id})
    if isinstance(data, list) and data:
        return data[0]
    return data


async def get_video_transcript(video_id: str) -> dict | None:
    """🥞 1 — transcript with timestamps. Returns None if video has no captions."""
    return await _get("/api/external/videos/transcript", {"videoId": video_id})


# ============================================================
# Normalizers — turn NexLev responses into our candidate schema
# ============================================================

def _safe_int(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _parse_length_text(s: str | None) -> int:
    """Parse "HH:MM:SS"/"MM:SS"/"SS" → total seconds. Empty → 0."""
    if not s:
        return 0
    try:
        parts = [int(p) for p in s.split(":")]
    except Exception:
        return 0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return 0


def normalize_thumbnail_to_candidate(item: dict) -> dict | None:
    """Convert a similar_thumbnails result row → candidate dict.

    Mirrors the shape produced by youtube_client._normalize_youtube so the
    rest of the pipeline (filter, dedupe, DB insert) treats NexLev results
    identically.
    """
    vid = item.get("id") or item.get("videoId")
    url = item.get("youtubeVideoUrl") or (vid and f"https://youtube.com/shorts/{vid}")
    if not vid or not url:
        return None
    vd = item.get("videoData") or {}
    cd = item.get("channelData") or {}
    return {
        "platform": "youtube",
        "video_id": vid,
        "url": url,
        "title": (item.get("metadata") or {}).get("title") or vd.get("title", ""),
        "caption": (vd.get("description") or "")[:500],
        "channel_id": vd.get("channelId") or cd.get("channelId"),
        "channel_name": vd.get("channelTitle") or cd.get("title"),
        "view_count": _safe_int(vd.get("viewCount")),
        "like_count": _safe_int(vd.get("likeCount")),
        "duration": _safe_int(vd.get("lengthSeconds")),
        "published_at": vd.get("publishDate") or vd.get("uploadDate"),
        "thumbnail_url": item.get("youtubeThumbnailUrl") or item.get("youtubeMaxResUrl"),
    }


def normalize_similar_video_to_candidate(item: dict) -> dict | None:
    """Convert a similar_videos result row → candidate dict."""
    vid = item.get("videoId")
    if not vid:
        return None
    return {
        "platform": "youtube",
        "video_id": vid,
        "url": f"https://youtube.com/shorts/{vid}",
        "title": item.get("title", ""),
        "caption": (item.get("description") or "")[:500],
        "channel_id": item.get("channelId"),
        "channel_name": item.get("channelTitle"),
        "view_count": _safe_int(item.get("viewCount")),
        "like_count": 0,
        "duration": _parse_length_text(item.get("lengthText")),
        "published_at": None,
        "thumbnail_url": (item.get("thumbnail") or [{}])[0].get("url") if item.get("thumbnail") else None,
    }


# ============================================================
# Pipeline-friendly wrappers (returns normalized candidate lists)
# ============================================================

async def search_videos_by_text(query: str, limit: int = 15) -> list[dict]:
    """🥞 10 — NexLev similar_thumbnails text search → candidate list.

    Use this as a keyword-search source when YouTube API quota is exhausted
    or to broaden coverage with NexLev's visual index.
    """
    items = await get_similar_thumbnails(query, "text", limit=limit)
    out = []
    for it in items:
        n = normalize_thumbnail_to_candidate(it)
        if n:
            out.append(n)
    return out


async def search_similar_videos(video_id: str) -> list[dict]:
    """🥞 10 — NexLev similar_videos → candidate list (seed-based discovery)."""
    items = await get_similar_videos(video_id)
    out = []
    for it in items:
        n = normalize_similar_video_to_candidate(it)
        if n:
            out.append(n)
    return out
