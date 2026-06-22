"""YouTube adapter — currently stubbed; integrates with NexLev/youtube-analyzer
via the user's existing Claude Code MCPs. For the standalone backend, we use
Apify YouTube scraper as the primary path (see apify_client.get_youtube_via_apify).

In the future this can be replaced with direct YouTube Data API v3 calls
(quota-based) or a local youtube-analyzer service.
"""
import os
import sys
import asyncio
import shutil
import httpx
import re
from typing import Any
import urllib.parse
from pathlib import Path


YT_DLP = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")


# YouTube Data API v3 - public, $0 within quota (10,000 units/day)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
_API_KEYS_CACHE = None
_KEY_IDX = 0  # round-robin index
_DEAD_KEYS = set()  # 403/quota된 키들 (이번 process)


def _load_all_keys():
    global _API_KEYS_CACHE
    if _API_KEYS_CACHE is not None:
        return _API_KEYS_CACHE
    keys = []
    # env에서 직접
    for name in ("YOUTUBE_API_KEY", "YOUTUBE_API_KEY_2", "YOUTUBE_API_KEY_3"):
        v = os.getenv(name)
        if v: keys.append(v)
    # .env 파일에서 보강
    if len(keys) < 2:
        try:
            for line in open(Path(__file__).resolve().parent.parent / ".env", encoding="utf-8"):
                if line.startswith("YOUTUBE_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[1].strip()
                        if k and k not in keys:
                            keys.append(k)
        except Exception:
            pass
    _API_KEYS_CACHE = keys
    return keys


def _get_key():
    """Round-robin으로 살아있는 키 반환."""
    global _KEY_IDX, YOUTUBE_API_KEY
    keys = _load_all_keys()
    if not keys:
        return ""
    live = [k for k in keys if k not in _DEAD_KEYS]
    if not live:
        # 다 죽으면 reset (다음 회복 시도)
        _DEAD_KEYS.clear()
        live = keys
    k = live[_KEY_IDX % len(live)]
    _KEY_IDX = (_KEY_IDX + 1) % len(live)
    YOUTUBE_API_KEY = k  # legacy global 동기화
    return k


def _mark_dead(key: str):
    _DEAD_KEYS.add(key)
YT_API_BASE = "https://www.googleapis.com/youtube/v3"


async def search_youtube(query: str, max_results: int = 50,
                         duration: str = "short", region: str = "US") -> list[dict]:
    """Search YouTube via Data API v3. 403/quota 시 다른 키 자동 시도."""
    keys = _load_all_keys()
    if not keys:
        return []
    last_err = None
    for attempt_i in range(len(keys)):
        key = _get_key()
        if not key:
            return []
        try:
            return await _do_search(query, key, max_results, duration, region)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                _mark_dead(key)
                last_err = e
                print(f"[yt-search] key dead ({e.response.status_code}), 다음 키 시도", flush=True)
                continue
            raise
        except Exception as e:
            raise
    print(f"[yt-search] 모든 키 죽음: {last_err}", flush=True)
    return []




async def search_channel_videos_via_api(channel_id: str, max_results: int = 50,
                                          order: str = "viewCount") -> list[dict]:
    """YouTube Data API로 채널 영상 인기순 가져옴. yt-dlp 못하는 거.
    order: viewCount / date / rating / relevance
    """
    keys = _load_all_keys()
    if not keys or not channel_id:
        return []
    for attempt_i in range(len(keys)):
        key = _get_key()
        if not key:
            return []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {
                    "key": key, "part": "snippet", "channelId": channel_id,
                    "type": "video", "videoDuration": "short",
                    "maxResults": min(max_results, 50), "order": order,
                }
                r = await client.get(f"{YT_API_BASE}/search", params=params)
                r.raise_for_status()
                items = r.json().get("items", [])
                video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
                if not video_ids:
                    return []
                params2 = {"key": key, "part": "snippet,statistics,contentDetails",
                           "id": ",".join(video_ids)}
                r2 = await client.get(f"{YT_API_BASE}/videos", params=params2)
                r2.raise_for_status()
                videos = r2.json().get("items", [])
                return [_normalize_youtube(v) for v in videos]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                _mark_dead(key)
                continue
            print(f"[api-channel] fail: {e}", flush=True)
            return []
        except Exception as e:
            print(f"[api-channel] fail: {e}", flush=True)
            return []
    return []


async def _do_search(query: str, key: str, max_results: int, duration: str, region: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Search to get videoIds
        params = {
            "key": key,
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoDuration": duration,  # short = <4min
            "maxResults": min(max_results, 50),
            "order": "viewCount",
            "regionCode": region,
        }
        r = await client.get(f"{YT_API_BASE}/search", params=params)
        r.raise_for_status()
        items = r.json().get("items", [])
        video_ids = [it["id"]["videoId"] for it in items]
        if not video_ids:
            return []

        # 2. Batch fetch full stats for ranking
        params = {
            "key": key,
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
        }
        r = await client.get(f"{YT_API_BASE}/videos", params=params)
        r.raise_for_status()
        videos = r.json().get("items", [])

    return [_normalize_youtube(v) for v in videos]


def _normalize_youtube(v: dict) -> dict:
    sn = v.get("snippet", {})
    st = v.get("statistics", {})
    cd = v.get("contentDetails", {})
    return {
        "platform": "youtube",
        "video_id": v["id"],
        "url": f"https://youtube.com/shorts/{v['id']}",
        "title": sn.get("title"),
        "caption": sn.get("description", ""),
        "channel_name": sn.get("channelTitle"),
        "channel_id": sn.get("channelId"),
        "view_count": int(st.get("viewCount", 0)),
        "like_count": int(st.get("likeCount", 0)),
        "duration": _parse_iso8601_duration(cd.get("duration", "PT0S")),
        "published_at": sn.get("publishedAt"),
        "thumbnail_url": (sn.get("thumbnails", {}).get("high") or
                          sn.get("thumbnails", {}).get("default", {})).get("url"),
    }


def _parse_iso8601_duration(s: str) -> int:
    """Parse PT1M30S to total seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h, mi, sec = (int(g or 0) for g in m.groups())
    return h * 3600 + mi * 60 + sec


async def resolve_channel_id(handle_or_url: str) -> str | None:
    """@handle, URL → UC... channelId. Public wrapper — key round-robin/fallback.
    API 다 죽으면 yt-dlp fallback (첫 영상의 channel_id 추출)."""
    keys = _load_all_keys()
    if not keys:
        return await _resolve_channel_id_via_ytdlp(handle_or_url)
    last_err = None
    for _ in range(len(keys)):
        key = _get_key()
        if not key:
            break
        try:
            return await _resolve_channel_id(handle_or_url, key)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                _mark_dead(key)
                last_err = e
                print(f"[yt-resolve] key dead ({e.response.status_code}), 다음 키 시도", flush=True)
                continue
            raise
    print(f"[yt-resolve] 모든 키 죽음 → yt-dlp fallback (last_err={last_err})", flush=True)
    return await _resolve_channel_id_via_ytdlp(handle_or_url)


async def _resolve_channel_id_via_ytdlp(handle_or_url: str) -> str | None:
    """yt-dlp fallback — 채널 첫 영상의 channel_id 추출."""
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "--flat-playlist", "--playlist-items", "1",
            "--print", "%(channel_id)s",
            handle_or_url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        cid = out.decode().strip().split("\n")[0].strip()
        if cid and cid.startswith("UC"):
            print(f"[yt-resolve-ytdlp] {handle_or_url} → {cid}", flush=True)
            return cid
    except Exception as e:
        print(f"[yt-resolve-ytdlp] 실패: {e}", flush=True)
    return None


async def get_channel_meta(channel_id: str) -> dict | None:
    """채널 snippet+statistics 메타. key round-robin/fallback."""
    keys = _load_all_keys()
    if not keys:
        return None
    last_err = None
    for _ in range(len(keys)):
        key = _get_key()
        if not key:
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    f"{YT_API_BASE}/channels",
                    params={"key": key, "part": "snippet,statistics", "id": channel_id},
                )
                r.raise_for_status()
                items = r.json().get("items", [])
                return items[0] if items else None
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                _mark_dead(key)
                last_err = e
                print(f"[yt-meta] key dead ({e.response.status_code}), 다음 키 시도", flush=True)
                continue
            raise
    print(f"[yt-meta] 모든 키 죽음: {last_err}", flush=True)
    return None


async def _resolve_channel_id(handle_or_url: str, key: str) -> str | None:
    """@handle, /channel/UCxxx, URL, 또는 channelId 입력을 UC... channelId로 변환."""
    decoded = urllib.parse.unquote(handle_or_url).strip()
    # 이미 channelId 형식 (UC + 22자)이면 그대로
    if decoded.startswith("UC") and len(decoded) == 24:
        return decoded
    # /channel/UCxxx 형식
    if "/channel/" in decoded:
        cid = decoded.split("/channel/")[-1].split("/")[0].split("?")[0]
        if cid.startswith("UC") and len(cid) == 24:
            return cid
    # @handle 추출
    handle = None
    if "@" in decoded:
        handle = decoded.split("@")[-1].split("/")[0].split("?")[0]
    elif "youtube.com" not in decoded and "/" not in decoded:
        handle = decoded.lstrip("@")
    if not handle:
        return None
    # forHandle API로 channelId 해석
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{YT_API_BASE}/channels",
            params={"key": key, "part": "id", "forHandle": f"@{handle}"},
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        if items:
            return items[0]["id"]
    return None


async def get_channel_videos(channel_id_or_url: str, max_results: int = 100,
                             order: str = "viewCount") -> list[dict]:
    """채널 영상 가져오기.

    order='viewCount' → yt-dlp 우선 (API search.list는 viral 영상 누락 많음. 정확성 ↑).
    order='date'      → YouTube Data API uploads playlist (빠름, 시간순).

    API 죽으면 yt-dlp fallback.
    """
    # 인기순 — yt-dlp 우선 (정확성. API search.list는 channel의 viral 영상 못 잡아옴)
    if order == "viewCount":
        return await _get_channel_videos_via_ytdlp(channel_id_or_url, max_results, order)

    # 시간순 — API 우선 (빠름)
    keys = _load_all_keys()
    if not keys:
        print("[yt-channel] no API key → yt-dlp fallback", flush=True)
        return await _get_channel_videos_via_ytdlp(channel_id_or_url, max_results, order)
    last_err = None
    for _ in range(len(keys)):
        key = _get_key()
        if not key:
            break
        try:
            return await _do_get_channel_videos(channel_id_or_url, max_results, order, key)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                _mark_dead(key)
                last_err = e
                print(f"[yt-channel] key dead ({e.response.status_code}), 다음 키 시도", flush=True)
                continue
            raise
        except Exception:
            raise
    print(f"[yt-channel] 모든 키 죽음 → yt-dlp fallback (last_err={last_err})", flush=True)
    return await _get_channel_videos_via_ytdlp(channel_id_or_url, max_results, order)


async def _ytdlp_video_meta(video_id: str, timeout: int = 15) -> dict | None:
    """yt-dlp single video 메타. view_count + duration + 채널 정보."""
    return await _ytdlp_video_meta_by_url(
        f"https://www.youtube.com/watch?v={video_id}", timeout=timeout
    )


async def _ytdlp_channel_list(channel_url: str, max_videos: int = 50,
                               timeout: int = 60) -> list[dict]:
    """yt-dlp로 채널 영상 목록 받기. TikTok / YouTube 채널 URL 다 지원.
    flat-playlist 한 번 호출로 view_count + duration + title 받음.
    채널의 최신 영상부터 max_videos개까지.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "--flat-playlist",
            "--playlist-end", str(max_videos),
            "--print",
            "%(extractor)s|%(id)s|%(title)s|%(view_count)s|%(duration)s|%(channel)s|"
            "%(channel_id)s|%(thumbnail)s|%(timestamp)s|%(uploader)s|%(uploader_id)s|"
            "%(webpage_url)s",
            channel_url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except Exception as e:
        print(f"[ytdlp-channel] fail: {e}", flush=True)
        return []

    results = []
    for line in out.decode().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 11)
        if len(parts) < 5 or not parts[1]:
            continue
        def _safe_int(x):
            try: return int(x) if x not in ("NA", "") else 0
            except Exception: return 0
        def _safe_float(x):
            try: return float(x) if x not in ("NA", "") else 0.0
            except Exception: return 0.0
        def _v(idx, default=""):
            return parts[idx] if len(parts) > idx and parts[idx] != "NA" else default

        extractor = (parts[0] or "").lower()
        if "youtube" in extractor:
            platform = "youtube"
            canonical_url = f"https://youtube.com/shorts/{parts[1]}"
        elif "tiktok" in extractor:
            platform = "tiktok"
            uploader_id = _v(10, "")
            if uploader_id and not uploader_id.startswith("@"):
                uploader_id = "@" + uploader_id
            canonical_url = (
                f"https://www.tiktok.com/{uploader_id}/video/{parts[1]}"
                if uploader_id else _v(11, "")
            )
        else:
            platform = extractor or "unknown"
            canonical_url = _v(11, "")

        # timestamp → ISO 변환
        ts = _safe_int(_v(8, ""))
        if ts > 0:
            from datetime import datetime, timezone
            published_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            published_iso = None

        results.append({
            "platform": platform,
            "video_id": parts[1],
            "url": canonical_url,
            "title": _v(2, ""),
            "view_count": _safe_int(_v(3, "")),
            "duration": int(_safe_float(_v(4, ""))),
            "channel_name": _v(5, "") or _v(9, ""),
            "channel_id": _v(6, "") or _v(10, ""),
            "thumbnail_url": _v(7, None) or None,
            "published_at": published_iso,
        })
    return results


async def _ytdlp_video_meta_by_url(url: str, timeout: int = 20) -> dict | None:
    """다중 플랫폼 (YouTube / TikTok / Instagram) URL → 메타.
    yt-dlp가 자동 검출. extractor 이름으로 platform 결정.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "--skip-download", "--print",
            "%(extractor)s|%(id)s|%(title)s|%(view_count)s|%(duration)s|%(channel)s|"
            "%(channel_id)s|%(thumbnail)s|%(upload_date)s|%(description)s|%(uploader)s|"
            "%(uploader_id)s|%(webpage_url)s",
            url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except Exception:
        return None
    parts = out.decode().strip().split("|", 12)
    if len(parts) < 5 or not parts[1]:
        return None

    def _safe_int(x):
        try: return int(x) if x not in ("NA", "") else 0
        except Exception: return 0
    def _safe_float(x):
        try: return float(x) if x not in ("NA", "") else 0.0
        except Exception: return 0.0
    def _v(idx, default=""):
        return parts[idx] if len(parts) > idx and parts[idx] != "NA" else default

    extractor = (parts[0] or "").lower()
    if "youtube" in extractor:
        platform = "youtube"
        canonical_url = f"https://youtube.com/shorts/{parts[1]}"
    elif "tiktok" in extractor:
        platform = "tiktok"
        # TikTok URL 보존 — uploader가 있으면 깔끔한 URL 만듦
        uploader_id = _v(11, "")
        if uploader_id and not uploader_id.startswith("@"):
            uploader_id = "@" + uploader_id
        canonical_url = (
            f"https://www.tiktok.com/{uploader_id}/video/{parts[1]}"
            if uploader_id else _v(12, url)
        )
    elif "instagram" in extractor:
        platform = "instagram"
        # Instagram은 webpage_url 보존
        canonical_url = _v(12, url)
    else:
        platform = extractor or "unknown"
        canonical_url = _v(12, url)

    return {
        "platform": platform,
        "video_id": parts[1],
        "url": canonical_url,
        "title": _v(2, ""),
        "view_count": _safe_int(_v(3, "")),
        "like_count": 0,
        "duration": int(_safe_float(_v(4, ""))),
        "channel_name": _v(5, "") or _v(10, ""),  # channel 없으면 uploader
        "channel_id": _v(6, "") or _v(11, ""),
        "thumbnail_url": _v(7, None) or None,
        "published_at": _v(8, None) or None,
        "caption": _v(9, "")[:500],
    }


async def _get_channel_videos_via_ytdlp(channel_url: str, max_results: int = 100,
                                         order: str = "viewCount") -> list[dict]:
    """yt-dlp로 채널 영상 가져오기. flat-playlist 한 번 호출로 view_count + duration + title 다 받음.
    인기순일 때 API search.list보다 더 정확 (API는 search index 기반이라 viral 영상 누락 많음).
    """
    # /shorts 탭 우선. channelId/handle/URL 입력 모두 정상화.
    if channel_url.startswith("UC") and len(channel_url) == 24:
        # 순수 channelId — URL로 변환
        url = f"https://www.youtube.com/channel/{channel_url}/shorts"
    elif "/shorts" in channel_url:
        url = channel_url
    elif ("@" in channel_url or "/channel/" in channel_url):
        url = channel_url.rstrip("/") + "/shorts"
    elif "youtube.com" not in channel_url and "/" not in channel_url:
        # 그냥 handle 문자열 — @ 붙여서 URL로
        h = channel_url.lstrip("@")
        url = f"https://www.youtube.com/@{h}/shorts"
    else:
        url = channel_url

    # 같은 채널 영상 list이므로 channel name을 URL에서 추출해서 fallback
    fallback_channel_name = ""
    try:
        decoded = urllib.parse.unquote(channel_url)
        if "@" in decoded:
            fallback_channel_name = decoded.split("@")[-1].split("/")[0].split("?")[0]
        elif "/channel/" in decoded:
            fallback_channel_name = decoded.split("/channel/")[-1].split("/")[0].split("?")[0]
    except Exception:
        pass

    # flat-playlist에서 메타 한꺼번에 받음 — view_count, duration, title, channel
    try:
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "--flat-playlist",
            "--print", "%(id)s|%(view_count)s|%(duration)s|%(channel)s|%(channel_id)s|%(title)s",
            url, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
    except Exception as e:
        print(f"[yt-dlp-fetch] flat-list 실패: {e}", flush=True)
        return []

    def _safe_int(x):
        try: return int(x) if x not in ("NA", "") else 0
        except Exception: return 0
    def _safe_float(x):
        try: return float(x) if x not in ("NA", "") else 0.0
        except Exception: return 0.0

    videos = []
    for line in out.decode().split("\n"):
        if not line.strip(): continue
        parts = line.split("|", 5)
        if len(parts) < 1 or not parts[0]: continue
        vid = parts[0].strip()
        ch_name = parts[3] if len(parts) > 3 and parts[3] != "NA" else ""
        if not ch_name:
            ch_name = fallback_channel_name
        videos.append({
            "platform": "youtube",
            "video_id": vid,
            "url": f"https://youtube.com/shorts/{vid}",
            "title": parts[5] if len(parts) > 5 else "",
            "view_count": _safe_int(parts[1] if len(parts) > 1 else "0"),
            "like_count": 0,
            "duration": int(_safe_float(parts[2] if len(parts) > 2 else "0")),
            "channel_name": ch_name,
            "channel_id": parts[4] if len(parts) > 4 and parts[4] != "NA" else "",
            "thumbnail_url": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            "published_at": None,
            "caption": "",
        })

    if not videos:
        print(f"[yt-dlp-fetch] 영상 0개: {url}", flush=True)
        return []

    if order == "viewCount":
        videos.sort(key=lambda v: v.get("view_count") or 0, reverse=True)

    vc_max = max((v.get("view_count") or 0) for v in videos)
    print(f"[yt-dlp-fetch] {url} → {len(videos)}개, top view {vc_max:,}, return {min(len(videos), max_results)}개", flush=True)
    return videos[:max_results]


async def _do_get_channel_videos(channel_id_or_url: str, max_results: int,
                                  order: str, key: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 0. handle/URL → channelId 해석
        channel_id = await _resolve_channel_id(channel_id_or_url, key)
        if not channel_id:
            print(f"[yt-channel] channelId 못 찾음: {channel_id_or_url}", flush=True)
            return []

        all_video_ids: list[str] = []
        if order == "viewCount":
            # search.list로 인기순. 페이징.
            next_token = None
            while len(all_video_ids) < max_results:
                params = {
                    "key": key,
                    "part": "snippet",
                    "channelId": channel_id,
                    "type": "video",
                    "order": "viewCount",
                    "maxResults": min(50, max_results - len(all_video_ids)),
                }
                if next_token:
                    params["pageToken"] = next_token
                r = await client.get(f"{YT_API_BASE}/search", params=params)
                r.raise_for_status()
                data = r.json()
                for it in data.get("items", []):
                    vid = (it.get("id") or {}).get("videoId")
                    if vid:
                        all_video_ids.append(vid)
                next_token = data.get("nextPageToken")
                if not next_token:
                    break
        else:
            # uploads playlist (시간순) — 기존 동작
            r = await client.get(
                f"{YT_API_BASE}/channels",
                params={"key": key, "part": "contentDetails", "id": channel_id},
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                return []
            uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            next_token = None
            while len(all_video_ids) < max_results:
                params = {
                    "key": key,
                    "part": "contentDetails",
                    "playlistId": uploads,
                    "maxResults": 50,
                }
                if next_token:
                    params["pageToken"] = next_token
                r = await client.get(f"{YT_API_BASE}/playlistItems", params=params)
                r.raise_for_status()
                data = r.json()
                for it in data.get("items", []):
                    all_video_ids.append(it["contentDetails"]["videoId"])
                next_token = data.get("nextPageToken")
                if not next_token:
                    break

        # 메타+stats batch fetch
        result = []
        for i in range(0, min(len(all_video_ids), max_results), 50):
            chunk = all_video_ids[i : i + 50]
            r = await client.get(
                f"{YT_API_BASE}/videos",
                params={
                    "key": key,
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(chunk),
                },
            )
            r.raise_for_status()
            for v in r.json().get("items", []):
                result.append(_normalize_youtube(v))
        return result


def extract_channel_handle(url: str) -> str:
    """Extract @handle or channelId from YouTube URL.

    Percent-decodes the URL first so CJK handles (arriving as
    `@%EC%87%BC...` from form submissions) become the raw unicode YouTube's
    search API expects.
    """
    decoded = urllib.parse.unquote(url)
    if "@" in decoded:
        return decoded.split("@")[-1].split("/")[0]
    if "/channel/" in decoded:
        return decoded.split("/channel/")[-1].split("/")[0]
    return decoded
