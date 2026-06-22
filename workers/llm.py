"""LLM adapter — Ollama (local) primary, Gemini (cloud) fallback."""
import os
import re
import sys
import json
import shutil
import asyncio
import base64
import tempfile
import subprocess
from pathlib import Path
import httpx
from typing import Any
from dataclasses import dataclass


_YOUTUBE_RE = re.compile(r'^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/', re.I)


def _resolve_yt_dlp() -> str:
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    return shutil.which("yt-dlp") or "yt-dlp"


async def _has_video_frames(path: Path) -> bool:
    """ffprobe quick check — returns True if video has at least 1 frame."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        n = int((out or b"0").decode().strip() or "0")
        return n > 0
    except Exception:
        return False


async def _ffmpeg_reencode(src_path: Path, dst_path: Path) -> bool:
    """Re-encode src to clean H.264 mp4. Returns True on success."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(src_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            str(dst_path),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
        return proc.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 1000
    except Exception:
        return False


async def _ytdlp_with_validate(url: str, out_path: Path, fmt: str, timeout: float) -> bool:
    proc = await asyncio.create_subprocess_exec(
        _resolve_yt_dlp(),
        "-f", fmt, "--merge-output-format", "mp4",
        "--force-overwrites", "--no-warnings",
        "-o", str(out_path), url,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return False
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
        return False
    return await _has_video_frames(out_path)


async def _ytdlp_download(url: str, out_path: Path, timeout: float = 180.0) -> None:
    """Download from any yt-dlp-supported site, validating with ffprobe.
    Tries multiple formats + ffmpeg re-encode fallback for broken mp4."""
    # Try several format strings — first valid one wins.
    formats = [
        "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "best[ext=mp4][vcodec*=avc1]/best[ext=mp4]",
        "best",
    ]
    last_err = ""
    for fmt in formats:
        if await _ytdlp_with_validate(url, out_path, fmt, timeout):
            return
        last_err = f"format {fmt!r} failed (no frames)"
    # Last resort: download with "best" then ffmpeg re-encode
    raw_path = out_path.with_suffix(".raw.mp4")
    proc = await asyncio.create_subprocess_exec(
        _resolve_yt_dlp(), "-f", "best",
        "--force-overwrites", "--no-warnings",
        "-o", str(raw_path), url,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"yt-dlp timeout after {timeout}s")
    if proc.returncode != 0 or not raw_path.exists():
        raise RuntimeError(f"yt-dlp failed all formats: {last_err} | {err.decode()[:200]}")
    # Force ffmpeg re-encode
    ok = await _ffmpeg_reencode(raw_path, out_path)
    raw_path.unlink(missing_ok=True)
    if not ok or not await _has_video_frames(out_path):
        raise RuntimeError("video has 0 frames even after re-encode")


async def _gemini_upload_file(file_path: Path, mime_type: str = "video/mp4") -> str:
    """Upload local file to Gemini Files API via resumable upload.
    Polls until state == ACTIVE. Returns file URI usable in file_data.file_uri.
    Used for videos > 20MB (inline_data limit).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    file_size = file_path.stat().st_size

    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": file_path.name}},
        )
        r.raise_for_status()
        session_url = r.headers.get("X-Goog-Upload-URL") or r.headers.get("x-goog-upload-url")
        if not session_url:
            raise RuntimeError(f"no upload URL header: {dict(r.headers)}")

        with file_path.open("rb") as f:
            data = f.read()
        r2 = await client.post(
            session_url,
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=data,
        )
        r2.raise_for_status()
        info = r2.json().get("file") or r2.json()
        name = info.get("name")
        uri = info.get("uri")
        if not (name and uri):
            raise RuntimeError(f"upload returned no name/uri: {info}")

        for _ in range(60):
            rs = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/{name}",
                headers={"x-goog-api-key": GEMINI_API_KEY},
            )
            rs.raise_for_status()
            state = rs.json().get("state")
            if state == "ACTIVE":
                return uri
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing FAILED: {rs.json()}")
            await asyncio.sleep(2)
        raise RuntimeError(f"Gemini file did not become ACTIVE within 120s: {name}")


async def _gemini_video_part(video_url: str) -> dict:
    """Build the `parts[0]` dict for a Gemini video chat request.
    YouTube → file_data.file_uri direct. Others → yt-dlp download then either
    inline_data (≤18MB) or Files API upload.
    """
    if _YOUTUBE_RE.match(video_url):
        return {"file_data": {"file_uri": video_url, "mime_type": "video/*"}}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        await _ytdlp_download(video_url, tmp_path)
        size = tmp_path.stat().st_size
        if size <= 18 * 1024 * 1024:
            data_b64 = base64.b64encode(tmp_path.read_bytes()).decode()
            return {"inline_data": {"mime_type": "video/mp4", "data": data_b64}}
        uri = await _gemini_upload_file(tmp_path, mime_type="video/mp4")
        return {"file_data": {"file_uri": uri, "mime_type": "video/mp4"}}
    finally:
        tmp_path.unlink(missing_ok=True)


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_used: int = 0


async def ollama_chat(prompt: str, system: str = "", json_mode: bool = False,
                      model: str | None = None) -> LLMResponse:
    """Call Ollama generate API."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
    return LLMResponse(
        text=data["response"],
        model=payload["model"],
        tokens_used=data.get("eval_count", 0),
    )


async def ollama_embed(text: str) -> list[float]:
    """Get embedding for text using nomic-embed-text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def gemini_video_chat(youtube_url: str, prompt: str, system: str = "",
                             model: str = "gemini-3.5-flash",
                             max_tokens: int = 16384,
                             json_mode: bool = False,
                             temperature: float = 0.3) -> LLMResponse:
    """Call Gemini with a video as input. YouTube URLs go in by file_uri.
    TikTok/Instagram/etc. are downloaded via yt-dlp first and sent as
    inline_data (≤18MB) or Files API upload.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen_config: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
    video_part = await _gemini_video_part(youtube_url)
    body = {
        "contents": [{
            "parts": [video_part, {"text": prompt}],
        }],
        "generationConfig": gen_config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            url, json=body,
            headers={"x-goog-api-key": GEMINI_API_KEY,
                     "Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
    cand = data.get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts) if parts else ""
    return LLMResponse(
        text=text, model=model,
        tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
    )


async def gemini_chat(prompt: str, system: str = "",
                      model: str = "gemini-3.5-flash",
                      max_tokens: int = 16384,
                      json_mode: bool = False) -> LLMResponse:
    """Call Gemini API. JSON mode forces JSON output (more parseable)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen_config: dict = {
        "temperature": 0.3,
        "maxOutputTokens": max_tokens,
        "thinkingConfig": {"thinkingBudget": 0},
    }
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(
            url,
            json=body,
            headers={"x-goog-api-key": GEMINI_API_KEY,
                     "Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
    cand = data.get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts) if parts else ""
    return LLMResponse(
        text=text, model=model,
        tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
    )


# ===== High-level analysis functions =====

async def analyze_video_dna(title: str, caption: str, transcript: str = "",
                            channel: str = "") -> dict:
    """Extract DNA categories/tags from a video's metadata."""
    system = (
        "You are a YouTube/TikTok content analyst. Given video metadata, "
        "extract the content DNA in structured JSON. Output JSON only, no commentary."
    )
    prompt = f"""Analyze this video and output JSON with these fields:
{{
  "primary_dna": "one short tag like 'naughty_horse' or 'gender_reveal_balloon'",
  "categories": ["list", "of", "tags"],
  "format_type": "one of: reaction|tutorial|prank|edit|compilation|story|original",
  "is_original": true/false,
  "is_ranking_or_compilation": true/false,
  "is_ai_or_cgi": true/false,
  "language": "en/ko/es/etc",
  "summary_kr": "한 줄 한국어 요약"
}}

Channel: {channel}
Title: {title}
Caption: {caption}
Transcript: {transcript[:1000]}"""

    resp = await ollama_chat(prompt, system, json_mode=True)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        return {"primary_dna": "unknown", "raw": resp.text}


async def match_dna_score(candidate_dna: dict, reference_dnas: list[dict]) -> dict:
    """Score how well a candidate matches the reference channel's DNA pool."""
    system = (
        "You are a content matching analyst. Given a candidate video's DNA "
        "and a list of reference DNAs from a successful channel, score the match."
    )
    prompt = f"""Score the candidate against the reference DNA pool.

Candidate DNA:
{json.dumps(candidate_dna, ensure_ascii=False)}

Reference Pool (successful videos from the target channel):
{json.dumps(reference_dnas[:20], ensure_ascii=False)}

Output JSON only:
{{
  "score": 0.0 to 1.0,
  "matched_categories": ["..."],
  "best_match_dna": "primary_dna of closest reference",
  "reasoning_kr": "한 줄 한국어 설명"
}}"""
    resp = await ollama_chat(prompt, system, json_mode=True)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        return {"score": 0.0, "raw": resp.text}
