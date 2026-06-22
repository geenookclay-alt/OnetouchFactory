"""Replicate client — 우리 양봉컴퍼니 LoRA로 ControlNet-스타일 변환.

흐름:
1. ostris/flux-dev-lora-trainer로 학습된 모델은 dongret-max/banbaji-{angel,reaper}
2. 그 모델은 Flux Dev + LoRA가 통합된 형태 — image + prompt 받음
3. 우리는 img2img 모드(prompt_strength 조정)로 실사 frame → 통합 일러스트 변환
4. trigger_word(TOK_BANBAJI_ANGEL/REAPER)를 prompt에 넣어 LoRA 효과 발동

Inference 비용: ~$0.04/call (Flux Dev pricing)
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Trained LoRA model slugs (set after training completes)
TRAINED_MODELS = {
    "angel": {
        "slug": "dongret-max/banbaji-angel",
        "trigger_word": "TOK_BANBAJI_ANGEL",
    },
    "reaper": {
        "slug": "dongret-max/banbaji-reaper",
        "trigger_word": "TOK_BANBAJI_REAPER",
    },
}

# ControlNet Canny — pose 100% 보존 (LoRA 미지원, bg 변환용)
FLUX_CANNY_MODEL = "black-forest-labs/flux-canny-dev"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }


async def healthcheck() -> bool:
    if not REPLICATE_API_TOKEN:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{REPLICATE_API_BASE}/account",
                            headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"})
            return r.status_code == 200
    except Exception:
        return False


async def _upload_image_url(image_path: Path) -> str:
    """Replicate accepts public URLs. We piggyback on fal storage."""
    from . import fal_client as _fc
    return await _fc.storage_upload(Path(image_path))


async def _wait_for_prediction(prediction_id: str, *,
                               timeout: float = 300.0,
                               poll_sec: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    delay = poll_sec
    async with httpx.AsyncClient(timeout=30.0) as c:
        while time.monotonic() < deadline:
            r = await c.get(f"{REPLICATE_API_BASE}/predictions/{prediction_id}",
                            headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"})
            r.raise_for_status()
            d = r.json()
            status = d.get("status")
            if status == "succeeded":
                return d
            if status in ("failed", "canceled"):
                raise RuntimeError(
                    f"Replicate prediction {prediction_id} {status}: {d.get('error')}"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.15, 4.0)
    raise TimeoutError(f"Replicate prediction {prediction_id} timeout after {timeout}s")


async def _get_latest_version(model_slug: str) -> str:
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{REPLICATE_API_BASE}/models/{model_slug}",
                        headers=_headers())
        r.raise_for_status()
        v = (r.json().get("latest_version") or {}).get("id")
        if not v:
            raise RuntimeError(f"No latest_version for {model_slug}")
        return v


async def lora_img2img(character: str, image_path: Path, out_png: Path, *,
                      action_prompt: str = "",
                      bg_style_prompt: str = "",
                      prompt_strength: float = 0.85,
                      guidance: float = 3.5,
                      num_inference_steps: int = 28,
                      lora_scale: float = 1.0,
                      seed: int | None = None,
                      aspect_ratio: str = "9:16") -> dict:
    """trained LoRA 모델로 실사 frame → 양봉 스타일 일러스트 (마스코트 통합).

    character: "angel" / "reaper"
    action_prompt: 마스코트 동작/감정 (예: "raising arms in despair, crying")
    bg_style_prompt: 그림체 가이드 (default = B급 sketch)
    prompt_strength: 0.5(원본 보존) ~ 1.0(전부 새로 그림)
                     0.85가 우리 use case(스타일 변환+마스코트 추가)에 sweet spot

    Returns {prediction_id, image_url, cost_usd, wall_sec}.
    """
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    if character not in TRAINED_MODELS:
        raise ValueError(f"unknown character: {character}")

    info = TRAINED_MODELS[character]
    slug = info["slug"]
    trigger = info["trigger_word"]

    image_path = Path(image_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bg_style = bg_style_prompt.strip() or (
        "rough hand-drawn sketch line art, thick black ink outlines on white background, "
        "B-grade Korean web comic mascot style, low-effort doodle aesthetic, "
        "wobbly hand-drawn lines, no shading, no detail"
    )

    prompt = (
        f"{bg_style}. The scene from the photo redrawn in this style "
        f"with {trigger} character {action_prompt or 'standing in the scene'}, "
        f"reacting to the situation."
    )[:2000]

    image_url = await _upload_image_url(image_path)
    payload = {
        "input": {
            "image": image_url,
            "prompt": prompt,
            "prompt_strength": prompt_strength,
            "guidance": guidance,
            "num_inference_steps": num_inference_steps,
            "lora_scale": lora_scale,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 95,
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    version = await _get_latest_version(slug)

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            f"{REPLICATE_API_BASE}/models/{slug}/versions/{version}/predictions",
            json=payload, headers=_headers(),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Replicate predict {r.status_code}: {r.text[:500]}")
        body = r.json()
        pid = body.get("id")
        if not pid:
            raise RuntimeError(f"Replicate missing id: {body}")

    result = await _wait_for_prediction(pid, timeout=240.0)
    output = result.get("output")
    image_url_out = output[0] if isinstance(output, list) else output
    if not image_url_out:
        raise RuntimeError(f"Replicate no output: {result}")

    async with httpx.AsyncClient(timeout=120.0) as c:
        rv = await c.get(image_url_out)
        rv.raise_for_status()
        out_png.write_bytes(rv.content)

    return {
        "prediction_id": pid,
        "image_url": image_url_out,
        "cost_usd": 0.04,
        "wall_sec": round(time.monotonic() - t0, 2),
        "character": character,
        "trigger_word": trigger,
    }


async def flux_canny_sketch(image_path: Path, out_png: Path, *,
                            prompt: str = "",
                            guidance: float = 30.0,
                            num_inference_steps: int = 28,
                            seed: int | None = None,
                            aspect_ratio: str = "9:16") -> dict:
    """ControlNet Canny — 실사 frame → sketch (구도 100% 보존, LoRA X).

    bg 변환용. 마스코트는 별도 lora_text2img로 생성.
    """
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    image_path = Path(image_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bg_prompt = prompt.strip() or (
        "rough hand-drawn sketch line art, thick black ink outlines on plain white background, "
        "B-grade Korean web comic doodle style, wobbly hand-drawn lines, "
        "no shading, no detail, no color"
    )

    control_url = await _upload_image_url(image_path)
    payload = {
        "input": {
            "control_image": control_url,
            "prompt": bg_prompt[:2000],
            "guidance": guidance,
            "num_inference_steps": num_inference_steps,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 95,
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            f"{REPLICATE_API_BASE}/models/{FLUX_CANNY_MODEL}/predictions",
            json=payload, headers=_headers(),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Replicate canny {r.status_code}: {r.text[:500]}")
        body = r.json()
        pid = body.get("id")
        if not pid:
            raise RuntimeError(f"Replicate missing id: {body}")

    result = await _wait_for_prediction(pid, timeout=240.0)
    output = result.get("output")
    image_url_out = output[0] if isinstance(output, list) else output
    if not image_url_out:
        raise RuntimeError(f"Replicate canny no output: {result}")

    async with httpx.AsyncClient(timeout=120.0) as c:
        rv = await c.get(image_url_out)
        rv.raise_for_status()
        out_png.write_bytes(rv.content)

    return {
        "prediction_id": pid,
        "image_url": image_url_out,
        "cost_usd": 0.04,
        "wall_sec": round(time.monotonic() - t0, 2),
    }


async def lora_text2img(character: str, out_png: Path, *,
                       action_prompt: str = "",
                       num_inference_steps: int = 28,
                       guidance: float = 3.5,
                       lora_scale: float = 1.0,
                       seed: int | None = None,
                       aspect_ratio: str = "1:1") -> dict:
    """trained LoRA로 마스코트 캐릭터만 생성 (text-to-image, no input image).

    흰 배경에 깨끗한 캐릭터 → ffmpeg colorkey/PIL alpha로 누끼 → composite.
    """
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    if character not in TRAINED_MODELS:
        raise ValueError(f"unknown character: {character}")

    info = TRAINED_MODELS[character]
    slug = info["slug"]
    trigger = info["trigger_word"]

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"{trigger} character {action_prompt or 'standing facing forward'}, "
        f"full body visible, clean white background, "
        f"rough hand-drawn sketch line art, thick black ink outlines, "
        f"B-grade Korean web comic mascot doodle style"
    )[:2000]

    payload = {
        "input": {
            "prompt": prompt,
            "guidance": guidance,
            "num_inference_steps": num_inference_steps,
            "lora_scale": lora_scale,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 95,
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    version = await _get_latest_version(slug)
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            f"{REPLICATE_API_BASE}/models/{slug}/versions/{version}/predictions",
            json=payload, headers=_headers(),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Replicate predict {r.status_code}: {r.text[:500]}")
        body = r.json()
        pid = body.get("id")
        if not pid:
            raise RuntimeError(f"Replicate missing id: {body}")

    result = await _wait_for_prediction(pid, timeout=240.0)
    output = result.get("output")
    image_url_out = output[0] if isinstance(output, list) else output
    if not image_url_out:
        raise RuntimeError(f"Replicate no output: {result}")

    async with httpx.AsyncClient(timeout=120.0) as c:
        rv = await c.get(image_url_out)
        rv.raise_for_status()
        out_png.write_bytes(rv.content)

    return {
        "prediction_id": pid,
        "image_url": image_url_out,
        "cost_usd": 0.04,
        "wall_sec": round(time.monotonic() - t0, 2),
        "character": character,
        "trigger_word": trigger,
    }


async def lora_img2img_batch(character: str, image_path: Path,
                             out_pngs: list[Path], *,
                             action_prompt: str = "",
                             bg_style_prompt: str = "",
                             prompt_strength: float = 0.85,
                             seeds: list[int] | None = None,
                             aspect_ratio: str = "9:16") -> list[dict]:
    """Boiling Lines용 — 같은 input, 다른 seed N장 병렬 생성."""
    n = len(out_pngs)
    if seeds is None:
        seeds = [1234 + i * 9999 for i in range(n)]
    elif len(seeds) != n:
        raise ValueError("seeds length mismatch")

    tasks = [
        lora_img2img(character, image_path, out_pngs[i],
                     action_prompt=action_prompt,
                     bg_style_prompt=bg_style_prompt,
                     prompt_strength=prompt_strength,
                     seed=seeds[i], aspect_ratio=aspect_ratio)
        for i in range(n)
    ]
    return await asyncio.gather(*tasks)


# Training helpers (called once per character setup)

async def get_training(training_id: str) -> dict:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{REPLICATE_API_BASE}/trainings/{training_id}",
                        headers=_headers())
        r.raise_for_status()
        return r.json()
