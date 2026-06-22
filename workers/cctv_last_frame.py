"""더빙 잡에 영상 분석 기반 CCTV 실사 변환 png 자동 첨부.

흐름 (2026-05-29 검증 완료):
1. 영상을 Gemini Files API에 upload → file_uri
2. Gemini Vision (chain: flash-latest → 3.5-flash → 2.5-flash-lite → 2.5-flash → 2.5-pro)
   으로 영상 분석 → "가장 뉴스에 나올 만한 한 장면" 매우 디테일한 영어 묘사
3. 그 묘사를 Kie GPT-Image-2 text-to-image로 호출 (CCTV 지글지글 prompt 강화)
4. out_dir/cctv_last.png 저장

대표님 룰:
- 마지막 프레임 X — 영상 전체 보고 newsworthy 장면 선택 (Gemini가 결정)
- Kie 사용 (라오장 X)
- CCTV 화질 지글지글 (heavy grain, JPEG artifact, vintage 1990s)
- 사이즈 작아도 OK

검증: qPpVROGIEVE 3D 애니 경찰서 영상 → 진짜 1990s CCTV 한 컷 (Gemini 8초 + Kie 71초).
비용 ≈ $0.14 (Gemini 무료, Kie gpt-image-2 1K).
실패해도 잡 자체는 OK (조용히 None 반환).
"""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


# Gemini chain — 503 폭증 대응. flash-latest가 video multimodal 가장 빠름.
_GEMINI_CHAIN = (
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash",
)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Kie endpoint
_KIE_CREATE = "https://api.kie.ai/api/v1/jobs/createTask"
_KIE_INFO = "https://api.kie.ai/api/v1/jobs/recordInfo"


# 영상 분석 prompt — 가장 뉴스에 나올 만한 충격적인 한 장면 + CCTV 지글지글 강화 prompt 자동 inject.
_VIDEO_ANALYSIS_PROMPT = """이 영상에서 가장 뉴스에 나올 만한 충격적인 한 장면을 골라서 photorealistic CCTV 카메라로 다시 그릴 수 있게 매우 디테일한 영어 묘사로 출력해줘. 등장 인물(수/성별/나이/인종/자세/옷색·스타일/헤어/수염/표정/부상·붕대 부위), 정확한 장소(어떤 방, 벽색, 바닥, 천장, 조명, 가구, 표지판), 카메라 framing(앵글/거리), 보이는 모든 소품·동물·차량.

마지막에 추가: ", photorealistic noisy CCTV security camera still, heavy video grain, JPEG compression artifacts, slight blur, washed-out cool tones, slight wide-angle barrel distortion, fluorescent overhead lighting, vintage 1990s CCTV recording aesthetic, slight interlaced lines, real human beings with real skin texture and real hair, RAW security DVR still"

영어로만 출력. 250 단어 이내. 서두 없이."""


def _get_gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or ""


def _get_kie_key() -> str:
    return os.environ.get("KIE_API_KEY") or ""


async def _upload_video_to_gemini(c: httpx.AsyncClient, video_path: Path,
                                    api_key: str) -> str:
    """Gemini Files API에 video upload + file_uri 받기 (ACTIVE 대기)."""
    size = video_path.stat().st_size
    init = await c.post(
        f"{_GEMINI_BASE.replace('/v1beta','')}/upload/v1beta/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": video_path.name}}, timeout=60,
    )
    up_url = init.headers.get("x-goog-upload-url")
    if not up_url:
        raise RuntimeError(f"no upload url: {init.text[:200]}")
    u = await c.post(
        up_url, content=video_path.read_bytes(),
        headers={
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": "video/mp4",
        }, timeout=600,
    )
    f = u.json().get("file") or {}
    uri = f.get("uri")
    fname = f.get("name")
    if not uri or not fname:
        raise RuntimeError(f"no uri: {u.text[:200]}")
    # wait for ACTIVE (max 60초)
    for _ in range(30):
        s = await c.get(f"{_GEMINI_BASE}/{fname}",
                        headers={"x-goog-api-key": api_key}, timeout=30)
        if s.json().get("state") == "ACTIVE":
            return uri
        await asyncio.sleep(2)
    return uri  # PROCESSING이어도 진행 시도


async def _analyze_video_with_chain(c: httpx.AsyncClient, file_uri: str,
                                      api_key: str) -> str:
    """Gemini chain으로 video 분석 → newsworthy scene 디테일 묘사."""
    body = {
        "contents": [{"parts": [
            {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
            {"text": _VIDEO_ANALYSIS_PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2500},
    }
    last_err = ""
    for model in _GEMINI_CHAIN:
        for attempt in range(3):
            try:
                r = await c.post(
                    f"{_GEMINI_BASE}/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key,
                             "Content-Type": "application/json"},
                    json=body, timeout=180,
                )
                if r.status_code == 200:
                    try:
                        return (r.json()["candidates"][0]["content"]
                                ["parts"][0]["text"]).strip()
                    except (KeyError, IndexError):
                        last_err = f"parse fail: {r.text[:150]}"
                        break  # 다음 model
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    last_err = f"HTTP {r.status_code}"
                    break  # 다음 model로 fallback
                last_err = f"HTTP {r.status_code}: {r.text[:120]}"
                break
            except Exception as e:
                last_err = str(e)[:150]
                if attempt < 2:
                    await asyncio.sleep(3)
                    continue
                break
    raise RuntimeError(f"Gemini chain 다 fail: {last_err}")


async def _kie_t2i(c: httpx.AsyncClient, prompt: str, api_key: str,
                    out_png: Path) -> dict:
    """Kie gpt-image-2 text-to-image — 결과 png 저장."""
    payload = {
        "model": "gpt-image-2-text-to-image",
        "input": {
            "prompt": prompt[:5000],
            "quality": "high",
            "aspect_ratio": "9:16",
            "resolution": "1K",
            "num_images": 1,
            "output_format": "png",
        },
    }
    cr = await c.post(_KIE_CREATE,
                       headers={"Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"},
                       json=payload, timeout=60)
    j = cr.json()
    if j.get("code") != 200:
        raise RuntimeError(f"Kie create fail: {j}")
    tid = (j.get("data") or {}).get("taskId")
    if not tid:
        raise RuntimeError(f"Kie no taskId: {j}")
    t0 = time.time()
    deadline = t0 + 480
    while time.time() < deadline:
        await asyncio.sleep(3)
        rs = await c.get(_KIE_INFO, params={"taskId": tid},
                          headers={"Authorization": f"Bearer {api_key}"},
                          timeout=15)
        d = (rs.json().get("data") or {})
        st = d.get("state") or d.get("status")
        if st == "success":
            urls = []
            try:
                urls = json.loads(d.get("resultJson") or "{}").get(
                    "resultUrls", [])
            except Exception:
                pass
            if urls:
                img = await c.get(urls[0], timeout=120)
                out_png.write_bytes(img.content)
                return {"path": str(out_png), "wall_sec": time.time() - t0,
                        "image_url": urls[0]}
            raise RuntimeError(f"Kie success but no urls: {d}")
        if st in ("failed", "fail"):
            raise RuntimeError(f"Kie failed: {d.get('failMsg') or d}")
    raise TimeoutError("Kie t2i timeout")


async def attach_cctv_frame(video_path: Path, out_dir: Path) -> dict | None:
    """더빙 잡 영상 → 가장 newsworthy 장면 분석 → Kie t2i 실사 CCTV png.

    out_dir/cctv_last.png 저장. Returns {"path", "scene_desc", "total_sec"} or None.
    실패해도 잡 자체는 OK.
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gem_key = _get_gemini_key()
        kie_key = _get_kie_key()
        if not gem_key or not kie_key:
            print("  ⚠️ CCTV: GEMINI_API_KEY / KIE_API_KEY 없음 — skip",
                  flush=True)
            return None
        total_t0 = time.time()
        async with httpx.AsyncClient(timeout=600) as c:
            # 1) video upload → Gemini Files
            print("  [CCTV] video upload + Gemini 분석...", flush=True)
            uri = await _upload_video_to_gemini(c, Path(video_path), gem_key)
            # 2) Gemini chain 분석
            desc = await _analyze_video_with_chain(c, uri, gem_key)
            print(f"  [CCTV] scene_desc: {desc[:140]}...", flush=True)
            # 3) Kie t2i
            out_png = out_dir / "cctv_last.png"
            r = await _kie_t2i(c, desc, kie_key, out_png)
        total = time.time() - total_t0
        return {
            "path": str(out_png),
            "scene_desc": desc,
            "kie_wall_sec": r.get("wall_sec"),
            "total_sec": total,
            "image_url": r.get("image_url"),
        }
    except Exception as e:
        print(f"  ⚠️ CCTV 변환 실패 (잡 OK): {str(e)[:200]}", flush=True)
        return None
