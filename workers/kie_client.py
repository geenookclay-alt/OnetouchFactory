"""Kie.ai client — Runway Gen-4 Turbo image-to-video.

Cloud GPU, dedicated/always-warm. ~5~15s per clip vs fal cold start 200s+.
Quality: high (Runway안정적, 캐릭터 일관성 강함).

Auth: header "Authorization: Bearer <KIE_API_KEY>".
Submit returns taskId; we poll record-detail until state == "success".
Pricing: credit-based (~₩140/2s clip 추정).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


KIE_API_KEY = os.getenv("KIE_API_KEY", "")
KIE_BASE = "https://api.kie.ai/api/v1"
KIE_RUNWAY_GENERATE = f"{KIE_BASE}/runway/generate"
KIE_RUNWAY_DETAIL = f"{KIE_BASE}/runway/record-detail"
KIE_KONTEXT_GENERATE = f"{KIE_BASE}/flux/kontext/generate"
KIE_KONTEXT_DETAIL = f"{KIE_BASE}/flux/kontext/record-info"
KIE_TIMEOUT_SEC = float(os.getenv("KIE_TIMEOUT_SEC", "300"))
KIE_DUMMY_CALLBACK = "https://example.com/kie-callback-unused"


def _ffmpeg() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return shutil.which("ffmpeg") or "ffmpeg"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {KIE_API_KEY}",
        "Content-Type": "application/json",
    }


async def healthcheck(timeout: float = 5.0) -> bool:
    """Verify KIE_API_KEY is set. We don't actually hit a paid endpoint."""
    return bool(KIE_API_KEY)


async def _normalize_mp4(src_path: Path, dst_path: Path,
                         target_w: int = 1080, target_h: int = 1920,
                         target_fps: int = 30) -> None:
    """Re-encode Kie mp4 → 1080x1920 @ 30fps h264 yuv420p, audio dropped.
    Matches sprite/cut params so concat -c copy stitches cleanly.
    """
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1"
    )
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-y",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "18",
        "-r", str(target_fps),
        "-vf", vf,
        "-an",
        "-movflags", "+faststart",
        str(dst_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dst_path.exists():
        raise RuntimeError(f"ffmpeg normalize mp4 failed: {err.decode()[:300]}")


async def _upload_image_url(image_path: Path) -> str:
    """Kie.ai requires a publicly accessible image URL (data URIs rejected
    with 422 'ImageUrl is illegal'). We piggyback on fal's storage upload
    SDK which the project already has installed.
    """
    from . import fal_client as _fc
    return await _fc.storage_upload(image_path)


async def _wait_for_runway(client: httpx.AsyncClient, task_id: str,
                           timeout: float = KIE_TIMEOUT_SEC) -> dict:
    """Poll record-detail until state == "success" or timeout. Returns the
    full data dict (videoInfo.videoUrl is the result).
    """
    deadline = time.monotonic() + timeout
    delay = 2.0
    while time.monotonic() < deadline:
        r = await client.get(
            KIE_RUNWAY_DETAIL,
            params={"taskId": task_id},
            headers=_headers(),
        )
        r.raise_for_status()
        body = r.json()
        data = body.get("data") or {}
        state = data.get("state")
        if state == "success":
            return data
        if state == "fail":
            raise RuntimeError(f"Kie Runway failed: {body}")
        await asyncio.sleep(delay)
        delay = min(delay * 1.2, 5.0)
    raise TimeoutError(f"Kie Runway taskId {task_id} did not finish in {timeout}s")


async def runway_i2v(image_path: Path, duration_sec: float, out_mp4: Path,
                     *, action_prompt: str = "",
                     duration: int = 5,
                     quality: str = "720p",
                     aspect_ratio: str = "9:16",
                     watermark: str = "") -> dict:
    """Runway Gen-4 Turbo i2v via Kie.ai.

    duration: 5 or 10 (10 incompatible with 1080p).
    quality: "720p" or "1080p".
    Default 720p+5s = ~₩140 / clip estimate.
    Returns {task_id, video_url, frames, fps, duration_sec, wall_sec, prompt}.
    """
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY not set")
    image_path = Path(image_path)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Korean shorts "byeong-mat" (cheesy/wonky) comic style — intentionally
    # rough motion, exaggerated reaction, hand-drawn line art feel. Avoid
    # words like "smooth/clear/vibrant" that push Runway toward polished output.
    full_prompt = (
        f"low-effort hand-drawn 2D sketch animation, "
        f"{action_prompt}, "
        f"wobbly bouncy exaggerated motion, cheesy comic reaction, "
        f"raw amateur Korean meme video style, "
        f"black ink line art on white, simple flat shading, "
        f"choppy frames like flipbook, no smoothness"
    )[:1800] if action_prompt else "wobbly comic mascot animation, raw 2D sketch"

    image_url = await _upload_image_url(image_path)

    payload = {
        "prompt": full_prompt,
        "duration": duration,
        "quality": quality,
        "imageUrl": image_url,
        "aspectRatio": aspect_ratio,
        "waterMark": watermark,
        "callBackUrl": KIE_DUMMY_CALLBACK,
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(KIE_RUNWAY_GENERATE, json=payload, headers=_headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Kie generate {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"Kie generate code {body.get('code')}: {body}")
        task_id = (body.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"Kie generate missing taskId: {body}")

        result = await _wait_for_runway(client, task_id)
        video_url = (result.get("videoInfo") or {}).get("videoUrl") or result.get("videoUrl")
        if not video_url:
            raise RuntimeError(f"Kie missing videoUrl: {result}")

        raw_path = out_mp4.with_name(out_mp4.stem + ".raw.mp4")
        async with httpx.AsyncClient(timeout=180.0) as dlc:
            rv = await dlc.get(video_url)
            rv.raise_for_status()
            raw_path.write_bytes(rv.content)

    await _normalize_mp4(raw_path, out_mp4)
    raw_path.unlink(missing_ok=True)

    return {
        "task_id": task_id,
        "video_url": video_url,
        "frames": duration * 24,
        "fps": 24,
        "duration_sec": float(duration),
        "wall_sec": round(time.monotonic() - t0, 2),
        "prompt": full_prompt,
    }


async def flux_kontext_img2img(image_path: Path, out_png: Path, *,
                               prompt: str, aspect_ratio: str = "9:16",
                               poll_sec: float = 3.0,
                               timeout: float = 720.0,
                               retry: int = 1) -> dict:
    """Flux Kontext Pro img2img via Kie.ai — same character, new style/expression.

    aspect_ratio: "9:16" / "1:1" / "3:4" / "16:9" / "4:3" / "1:2"
    Cost ~$0.025/image (yangbong v9 verified).
    timeout: per-attempt polling deadline (default 360s — Kie 가끔 200s+ 늦음).
    retry: timeout/RuntimeError 시 새 task로 재시도 횟수 (default 1).
    Returns {task_id, image_url, cost_usd, wall_sec, prompt}.
    """
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY not set")
    image_path = Path(image_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    image_url = await _upload_image_url(image_path)
    payload = {
        "prompt": prompt[:1800],
        "inputImage": image_url,
        "aspectRatio": aspect_ratio,
        "model": "flux-kontext-pro",
        "outputFormat": "png",
    }

    last_err: Exception | None = None
    for attempt in range(retry + 1):
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(KIE_KONTEXT_GENERATE, json=payload, headers=_headers())
                if r.status_code >= 400:
                    raise RuntimeError(f"Kie kontext {r.status_code}: {r.text[:500]}")
                body = r.json()
                if body.get("code") != 200:
                    raise RuntimeError(f"Kie kontext code {body.get('code')}: {body}")
                task_id = (body.get("data") or {}).get("taskId")
                if not task_id:
                    raise RuntimeError(f"Kie kontext missing taskId: {body}")

                deadline = time.monotonic() + timeout
                result_url = None
                while time.monotonic() < deadline:
                    rs = await client.get(KIE_KONTEXT_DETAIL,
                                          params={"taskId": task_id},
                                          headers=_headers())
                    rs.raise_for_status()
                    d = (rs.json().get("data") or {})
                    flag = d.get("successFlag")
                    if flag == 1:
                        result_url = (d.get("response") or {}).get("resultImageUrl")
                        if not result_url:
                            raise RuntimeError(f"Kie kontext: no resultImageUrl in {d}")
                        break
                    if flag in (2, 3):
                        raise RuntimeError(f"Kie kontext failed: {d}")
                    await asyncio.sleep(poll_sec)
                if not result_url:
                    raise TimeoutError(f"Kie kontext task {task_id} timeout")

                async with httpx.AsyncClient(timeout=120.0) as dlc:
                    rv = await dlc.get(result_url)
                    rv.raise_for_status()
                    out_png.write_bytes(rv.content)

            return {
                "task_id": task_id,
                "image_url": result_url,
                "cost_usd": 0.025,
                "wall_sec": round(time.monotonic() - t0, 2),
                "prompt": payload["prompt"],
                "attempts": attempt + 1,
            }
        except (TimeoutError, RuntimeError) as e:
            last_err = e
            if attempt < retry:
                logger.warning("Kie kontext attempt %d failed (%s) — retrying", attempt + 1, e)
                await asyncio.sleep(2.0)
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Kie kontext: unreachable")


# ============================================================
# Nano Banana (Google Gemini 2.5 Flash Image) via Kie unified jobs API
# ============================================================
KIE_JOBS_CREATE = f"{KIE_BASE}/jobs/createTask"
KIE_JOBS_INFO = f"{KIE_BASE}/jobs/recordInfo"


async def nano_banana_img2img(
    image_path: Path | None,
    out_png: Path,
    *,
    prompt: str,
    extra_image_paths: list[Path] | None = None,
    aspect_ratio: str = "9:16",
    poll_sec: float = 2.0,
    timeout: float = 600.0,  # Kie 실제 200~500초 — 10분으로 늘림
) -> dict:
    """Google Nano Banana (gemini-3.5-flash-image-edit) img2img via Kie.

    image_path: 1차 reference (정체성 anchor). None이면 text2img (model=google/nano-banana).
    extra_image_paths: 추가 reference (예: baseline + 표정 ref). 모두 image_urls에 들어감.
    Use case: angel raw → ghibli baseline → expression variant. 정체성 99% 유지.
    Cost: ~4 credits/img (Kie credits — Kontext보다 빠름 ~8s).
    Returns {task_id, image_url, cost_credits, wall_sec, prompt}.
    """
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY not set")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    image_urls: list[str] = []
    if image_path is not None:
        image_urls.append(await _upload_image_url(Path(image_path)))
    for p in extra_image_paths or []:
        image_urls.append(await _upload_image_url(Path(p)))

    if image_urls:
        model = "google/nano-banana-edit"
        input_payload: dict = {
            "prompt": prompt[:3000],
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
    else:
        model = "google/nano-banana"
        input_payload = {
            "prompt": prompt[:3000],
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }

    payload = {"model": model, "input": input_payload}

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(KIE_JOBS_CREATE, json=payload, headers=_headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Kie nano-banana create {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"Kie nano-banana code {body.get('code')}: {body}")
        task_id = (body.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"Kie nano-banana missing taskId: {body}")

        deadline = time.monotonic() + timeout
        result_url = None
        credits = 0.0
        while time.monotonic() < deadline:
            rs = await client.get(KIE_JOBS_INFO,
                                  params={"taskId": task_id},
                                  headers=_headers())
            rs.raise_for_status()
            d = (rs.json().get("data") or {})
            state = d.get("state")
            if state == "success":
                import json as _json
                rj = d.get("resultJson")
                if isinstance(rj, str):
                    try: rj = _json.loads(rj)
                    except Exception: rj = {}
                urls = (rj or {}).get("resultUrls") or []
                if not urls:
                    raise RuntimeError(f"Kie nano-banana no resultUrls: {d}")
                result_url = urls[0]
                credits = float(d.get("creditsConsumed") or 0)
                break
            if state in ("fail", "failed"):
                raise RuntimeError(f"Kie nano-banana failed: {d.get('failMsg')} | {d}")
            await asyncio.sleep(poll_sec)
        if not result_url:
            raise TimeoutError(f"Kie nano-banana task {task_id} timeout")

        async with httpx.AsyncClient(timeout=120.0) as dlc:
            rv = await dlc.get(result_url)
            rv.raise_for_status()
            out_png.write_bytes(rv.content)

    return {
        "task_id": task_id,
        "image_url": result_url,
        "cost_credits": credits,
        "wall_sec": round(time.monotonic() - t0, 2),
        "prompt": input_payload["prompt"],
        "model": model,
        "n_input_images": len(image_urls),
    }


# ============================================================
# OpenAI GPT-Image-2 (i2i + t2i) via Kie unified jobs API
# Slugs from docs: gpt-image-2-image-to-image / gpt-image-2-text-to-image
# Strong style transfer (better than nano-banana for ghibli/painterly).
# ============================================================
async def gpt_image_2_i2i(
    image_paths: list[Path],
    out_png: Path,
    *,
    prompt: str,
    quality: str = "high",       # "low" / "medium" / "high"
    aspect_ratio: str = "9:16",  # "auto" / "1:1" / "9:16" / "16:9" / "3:4" / "4:3" — Kie UI와 동일 필드
    resolution: str = "1K",      # "1K" / "2K" / "4K"
    poll_sec: float = 3.0,
    timeout: float = 720.0,      # Kie 실제 처리 175~513초 — 12분으로 늘림. credit 낭비 방지
) -> dict:
    """OpenAI GPT-Image-2 image-to-image via Kie. Multi-image input supported.

    2026-05-28 저녁 검증: image_size 필드는 Kie API가 무시 (default 가로로 처리됨).
    UI와 같이 aspect_ratio + resolution 조합으로 호출해야 의도한 사이즈 나옴.

    Pricing (per Kie docs): 6 credits (1k) / 10 credits (2k) / 16 credits (4k).
    Strong painterly style transfer, much stronger than nano-banana.
    """
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY not set")
    if not image_paths:
        raise ValueError("at least one input image required for i2i")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # parallel upload — N장이면 N번 sequential은 시간 병목 (1~3s × N)
    image_urls = await asyncio.gather(
        *[_upload_image_url(Path(p)) for p in image_paths]
    )

    payload = {
        "model": "gpt-image-2-image-to-image",
        "input": {
            "prompt": prompt[:5000],
            "image_urls": image_urls,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "num_images": 1,
            "output_format": "png",
        },
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(KIE_JOBS_CREATE, json=payload, headers=_headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Kie gpt-image-2 create {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"Kie gpt-image-2 code {body.get('code')}: {body}")
        task_id = (body.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"Kie gpt-image-2 missing taskId: {body}")

        deadline = time.monotonic() + timeout
        result_url = None
        credits = 0.0
        last_state = None
        while time.monotonic() < deadline:
            rs = await client.get(KIE_JOBS_INFO,
                                  params={"taskId": task_id},
                                  headers=_headers())
            rs.raise_for_status()
            d = (rs.json().get("data") or {})
            state = d.get("state")
            last_state = state
            if state == "success":
                import json as _json
                rj = d.get("resultJson")
                if isinstance(rj, str):
                    try: rj = _json.loads(rj)
                    except Exception: rj = {}
                urls = (rj or {}).get("resultUrls") or []
                if not urls:
                    raise RuntimeError(f"gpt-image-2 no resultUrls: {d}")
                result_url = urls[0]
                credits = float(d.get("creditsConsumed") or 0)
                break
            if state in ("fail", "failed"):
                raise RuntimeError(f"gpt-image-2 failed: {d.get('failMsg')} | {d}")
            await asyncio.sleep(poll_sec)
        if not result_url:
            raise TimeoutError(f"gpt-image-2 task {task_id} timeout (last_state={last_state})")

        async with httpx.AsyncClient(timeout=120.0) as dlc:
            rv = await dlc.get(result_url)
            rv.raise_for_status()
            out_png.write_bytes(rv.content)

    return {
        "task_id": task_id,
        "image_url": result_url,
        "cost_credits": credits,
        "wall_sec": round(time.monotonic() - t0, 2),
        "prompt": payload["input"]["prompt"],
        "n_input_images": len(image_urls),
    }


async def gpt_image_2_t2i(
    out_png: Path,
    *,
    prompt: str,
    quality: str = "high",
    image_size: str = "1024x1024",
    poll_sec: float = 3.0,
    timeout: float = 720.0,      # Kie 실제 처리 200~500초 — 12분
) -> dict:
    """OpenAI GPT-Image-2 text-to-image via Kie. Slug: gpt-image-2-text-to-image."""
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY not set")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "gpt-image-2-text-to-image",
        "input": {
            "prompt": prompt[:5000],
            "quality": quality,
            "image_size": image_size,
            "num_images": 1,
            "output_format": "png",
        },
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(KIE_JOBS_CREATE, json=payload, headers=_headers())
        if r.status_code >= 400:
            raise RuntimeError(f"gpt-image-2 t2i create {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"gpt-image-2 t2i code {body.get('code')}: {body}")
        task_id = (body.get("data") or {}).get("taskId")

        deadline = time.monotonic() + timeout
        result_url = None
        credits = 0.0
        while time.monotonic() < deadline:
            rs = await client.get(KIE_JOBS_INFO, params={"taskId": task_id}, headers=_headers())
            rs.raise_for_status()
            d = (rs.json().get("data") or {})
            state = d.get("state")
            if state == "success":
                import json as _json
                rj = d.get("resultJson")
                if isinstance(rj, str):
                    try: rj = _json.loads(rj)
                    except Exception: rj = {}
                urls = (rj or {}).get("resultUrls") or []
                if not urls:
                    raise RuntimeError(f"gpt-image-2 t2i no resultUrls: {d}")
                result_url = urls[0]
                credits = float(d.get("creditsConsumed") or 0)
                break
            if state in ("fail", "failed"):
                raise RuntimeError(f"gpt-image-2 t2i failed: {d.get('failMsg')} | {d}")
            await asyncio.sleep(poll_sec)
        if not result_url:
            raise TimeoutError(f"gpt-image-2 t2i task {task_id} timeout")

        async with httpx.AsyncClient(timeout=120.0) as dlc:
            rv = await dlc.get(result_url)
            rv.raise_for_status()
            out_png.write_bytes(rv.content)

    return {
        "task_id": task_id,
        "image_url": result_url,
        "cost_credits": credits,
        "wall_sec": round(time.monotonic() - t0, 2),
        "prompt": payload["input"]["prompt"],
    }
