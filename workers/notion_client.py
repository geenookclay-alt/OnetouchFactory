"""Notion adapter — push discovered candidates to a Notion database."""
import os
import httpx
from typing import Any


NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_VERSION = "2022-06-28"
NOTION_BASE = "https://api.notion.com/v1"


async def _request(method: str, path: str, **kwargs) -> dict:
    if not NOTION_TOKEN:
        raise RuntimeError("NOTION_TOKEN not set")
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(method, f"{NOTION_BASE}{path}",
                                 headers=headers, **kwargs)
        r.raise_for_status()
        return r.json()


async def search_pages(query: str = "", filter_object: str | None = None) -> list[dict]:
    payload: dict = {"query": query, "page_size": 25}
    if filter_object:
        payload["filter"] = {"property": "object", "value": filter_object}
    data = await _request("POST", "/search", json=payload)
    return data.get("results", [])


async def query_database(database_id: str, filters: dict | None = None) -> list[dict]:
    payload: dict = {"page_size": 100}
    if filters:
        payload["filter"] = filters
    data = await _request("POST", f"/databases/{database_id}/query", json=payload)
    return data.get("results", [])


async def create_page_in_database(database_id: str, properties: dict) -> dict:
    """Create a row in a Notion database."""
    return await _request(
        "POST", "/pages",
        json={"parent": {"database_id": database_id}, "properties": properties},
    )


def make_props(title_field: str, video: dict, dna: dict | None = None,
               classification: str = "키핑") -> dict:
    """Build Notion properties from a candidate video."""
    props = {
        title_field: {
            "title": [{"text": {"content": (video.get("title") or "")[:200]}}]
        },
        "URL": {"url": video.get("url")},
        "채널": {"rich_text": [{"text": {"content": video.get("channel_name") or ""}}]},
        "조회수": {"number": video.get("view_count", 0)},
        "길이": {"rich_text": [{"text": {"content": f"{video.get('duration', 0)}s"}}]},
        "결분류": {"select": {"name": classification}},
    }
    if dna:
        notes = dna.get("summary_kr", "") or dna.get("primary_dna", "")
        props["비고"] = {"rich_text": [{"text": {"content": notes[:200]}}]}
    return props


async def append_candidates(database_id: str, candidates: list[dict],
                            title_field: str = "제목") -> int:
    """Bulk append candidates to a Notion database. Returns count added."""
    added = 0
    for c in candidates:
        try:
            await create_page_in_database(
                database_id,
                make_props(title_field, c, c.get("dna"), c.get("classification", "키핑")),
            )
            added += 1
        except Exception as e:
            print(f"Notion append failed for {c.get('url')}: {e}")
    return added
