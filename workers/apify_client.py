"""Apify adapter — TikTok and Instagram scrapers."""
import os
import httpx
import asyncio
from typing import Any


APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_BASE = "https://api.apify.com/v2"


async def _run_actor_sync(actor_id: str, input_data: dict, timeout: int = 240) -> list[dict]:
    """Run an Apify Actor synchronously and return dataset items."""
    if not APIFY_TOKEN:
        raise RuntimeError("APIFY_TOKEN not set")
    url = (f"{APIFY_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
           f"?token={APIFY_TOKEN}&timeout={timeout}")
    async with httpx.AsyncClient(timeout=timeout + 30) as client:
        r = await client.post(url, json=input_data)
        r.raise_for_status()
        return r.json()


async def search_tiktok(keywords: list[str], results_per_keyword: int = 30) -> list[dict]:
    """Search TikTok by multiple keywords. Returns normalized video list."""
    raw = await _run_actor_sync(
        "clockworks~tiktok-scraper",
        {
            "searchQueries": keywords,
            "resultsPerPage": results_per_keyword,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
        },
    )
    # filter ads + photo carousel (slideshow) — not real videos
    items = []
    for v in raw:
        if v.get("isAd"):
            continue
        # Skip TikTok photo carousel (no real video)
        if v.get("isSlideshow") or v.get("isPhoto") or v.get("imagePost"):
            continue
        # Also skip if duration is 0 (likely photo post)
        dur = (v.get("videoMeta") or {}).get("duration") or 0
        if dur <= 0:
            continue
        items.append(_normalize_tiktok(v))
    return items


def _normalize_tiktok(v: dict) -> dict:
    return {
        "platform": "tiktok",
        "video_id": v.get("id"),
        "url": v.get("webVideoUrl"),
        "title": (v.get("text") or "")[:200],
        "caption": v.get("text", ""),
        "channel_name": v.get("authorMeta", {}).get("name"),
        "channel_id": v.get("authorMeta", {}).get("id"),
        "view_count": v.get("playCount", 0),
        "like_count": v.get("diggCount", 0),
        "duration": v.get("videoMeta", {}).get("duration", 0),
        "published_at": v.get("createTimeISO"),
        "thumbnail_url": v.get("videoMeta", {}).get("coverUrl"),
        "search_query": v.get("searchQuery"),
    }


async def search_instagram_reels(keyword: str, results: int = 50) -> list[dict]:
    """Search Instagram Reels by keyword (uses patient_discovery actor)."""
    raw = await _run_actor_sync(
        "patient_discovery~instagram-search-reels",
        {"keyword": keyword, "resultsLimit": results},
    )
    items = [_normalize_instagram(v) for v in raw if v.get("media_type") in (2, None)]
    # Generic hashtag spam filter
    items = [it for it in items if not _is_generic_ig(it.get("caption", ""))]
    return items


import re as _re

GENERIC_TAGS = {
    'fyp','foryou','foryoupage','trending','viral','explore','explorepage','reels',
    'reel','instamood','instagood','love','instalike','followforfollow','f4f',
    'shorts','short','viralreels','viralvideos','reelsinstagram','reelsvideo',
    'reelitfeelit','trend','trendingreels','trendingnow','viralpost','viralreel',
    'foryourpage','tiktok','dance','funny','meme','memes',
}

def _is_generic_ig(caption: str) -> bool:
    """True if caption is mostly generic hashtags / no meaningful words."""
    if not caption or len(caption.strip()) < 10:
        return True
    tags = [t.lower().lstrip('#') for t in _re.findall(r'#[A-Za-z0-9_]+', caption)]
    non_tag = _re.sub(r'#[A-Za-z0-9_]+', ' ', caption)
    non_tag_words = [w for w in _re.findall(r"[A-Za-z가-힣]{3,}", non_tag)]
    # 95%+ tags, no meaningful words: generic
    if tags and len(non_tag_words) <= 2:
        spam_ratio = sum(1 for t in tags if t in GENERIC_TAGS) / max(1, len(tags))
        if spam_ratio >= 0.5:
            return True
    return False


def _normalize_instagram(v: dict) -> dict:
    code = v.get("code")
    return {
        "platform": "instagram",
        "video_id": code or v.get("id"),
        "url": f"https://www.instagram.com/reel/{code}/" if code else v.get("video_url"),
        "title": (v.get("caption", {}).get("text", "") if isinstance(v.get("caption"), dict)
                  else "")[:200],
        "caption": v.get("caption", {}).get("text", "") if isinstance(v.get("caption"), dict) else "",
        "channel_name": v.get("user", {}).get("username"),
        "channel_id": str(v.get("user", {}).get("pk", "")),
        "view_count": v.get("play_count") or v.get("ig_play_count") or 0,
        "like_count": v.get("like_count", 0),
        "duration": v.get("video_duration", 0),
        "published_at": v.get("taken_at_date"),
        "thumbnail_url": v.get("thumbnail_url"),
    }


async def search_instagram_hashtag(hashtag: str, results: int = 30) -> list[dict]:
    """Search Instagram by hashtag (returns posts + reels)."""
    raw = await _run_actor_sync(
        "apify~instagram-hashtag-scraper",
        {"hashtags": [hashtag], "resultsLimit": results},
    )
    out = []
    for v in raw:
        if v.get("type") != "Video":
            continue
        out.append({
            "platform": "instagram",
            "video_id": v.get("shortCode"),
            "url": v.get("url"),
            "title": (v.get("caption") or "")[:200],
            "caption": v.get("caption", ""),
            "channel_name": v.get("ownerUsername"),
            "channel_id": v.get("ownerId"),
            "view_count": v.get("videoPlayCount") or v.get("videoViewCount") or 0,
            "like_count": v.get("likesCount", 0),
            "duration": v.get("videoDuration", 0),
            "published_at": v.get("timestamp"),
            "thumbnail_url": v.get("displayUrl"),
        })
    return out


async def get_youtube_via_apify(query: str, results: int = 50) -> list[dict]:
    """Search YouTube via Apify (fallback if NexLev unavailable)."""
    raw = await _run_actor_sync(
        "streamers~youtube-scraper",
        {
            "searchKeywords": [query],
            "maxResults": results,
            "uploadDate": "all",
            "duration": "short",
            "sortBy": "viewCount",
        },
    )
    out = []
    for v in raw:
        out.append({
            "platform": "youtube",
            "video_id": v.get("id"),
            "url": v.get("url"),
            "title": v.get("title"),
            "caption": v.get("description", ""),
            "channel_name": v.get("channelName"),
            "channel_id": v.get("channelId"),
            "view_count": v.get("viewCount", 0),
            "like_count": v.get("likes", 0),
            "duration": v.get("duration", 0),
            "published_at": v.get("date"),
            "thumbnail_url": v.get("thumbnailUrl"),
        })
    return out
