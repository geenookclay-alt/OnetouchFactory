"""YouTube storyboard frame extractor.

Uses yt-dlp to fetch the storyboard sprite (the tiny thumbnail grid YouTube
serves for scrubber previews), then slices it into individual PIL frames.
Doesn't download the actual video — sprites are tens of KB.

YouTube storyboard format metadata (yt-dlp):
  - format_note == "storyboard"
  - width / height: per-tile size in pixels
  - rows / columns: tile grid per sprite fragment
  - fragments[]: list of sprite image URLs (one per fragment)
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def _resolve_yt_dlp() -> str:
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    return shutil.which("yt-dlp") or "yt-dlp"


YT_DLP_BIN = os.getenv("YT_DLP_BIN") or _resolve_yt_dlp()


async def _run(*args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", "timeout"
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _pick_best_storyboard(info: dict) -> dict | None:
    """Pick the highest-resolution storyboard format from yt-dlp output."""
    formats = info.get("formats") or []
    sb = [f for f in formats
          if (f.get("format_note") or "").lower().startswith("storyboard")
          or str(f.get("format_id") or "").startswith("sb")]
    if not sb:
        return None
    # Sort by per-tile resolution descending — bigger tiles = better CLIP quality
    sb.sort(key=lambda f: (f.get("width") or 0) * (f.get("height") or 0),
            reverse=True)
    return sb[0]


async def extract_frames(video_url: str,
                         max_frames: int = 50) -> list[Image.Image]:
    """Return up to max_frames PIL images extracted from the YouTube storyboard."""
    rc, out, err = await _run(YT_DLP_BIN, "--dump-json", "--no-warnings",
                              "--skip-download", video_url, timeout=45.0)
    if rc != 0 or not out.strip():
        return []
    try:
        info = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return []

    sb = _pick_best_storyboard(info)
    if not sb:
        return []

    tile_w = int(sb.get("width") or 0)
    tile_h = int(sb.get("height") or 0)
    rows = int(sb.get("rows") or 0)
    cols = int(sb.get("columns") or 0)
    fragments = sb.get("fragments") or []

    if tile_w <= 0 or tile_h <= 0 or not fragments:
        return []

    urls = [f.get("url") for f in fragments if f.get("url")]
    if not urls:
        return []

    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for idx, url in enumerate(urls):
            if len(frames) >= max_frames:
                break
            sprite_path = td_path / f"sb_{idx}.jpg"
            rc, _, _ = await _run("curl", "-sSL", "--max-time", "20",
                                  "-o", str(sprite_path), url, timeout=25.0)
            if rc != 0 or not sprite_path.exists() or sprite_path.stat().st_size == 0:
                continue
            try:
                img = Image.open(sprite_path).convert("RGB")
            except Exception:
                continue
            frames.extend(_split_sprite(img, tile_w, tile_h, rows, cols,
                                        max_frames - len(frames)))
    return frames[:max_frames]


def _split_sprite(sprite: Image.Image, tile_w: int, tile_h: int,
                  rows: int, cols: int, limit: int) -> list[Image.Image]:
    """Slice a sprite image into rows×cols tiles of (tile_w, tile_h)."""
    sw, sh = sprite.size
    if rows <= 0 or cols <= 0:
        rows = max(1, sh // tile_h)
        cols = max(1, sw // tile_w)
    out: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            if len(out) >= limit:
                return out
            x0 = c * tile_w
            y0 = r * tile_h
            x1 = min(x0 + tile_w, sw)
            y1 = min(y0 + tile_h, sh)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            try:
                tile = sprite.crop((x0, y0, x1, y1))
                if tile.getbbox() is None:  # all-black padding
                    continue
                out.append(tile.resize((100, 100)))
            except Exception:
                continue
    return out
