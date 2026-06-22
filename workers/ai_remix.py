"""AI 변형 제작 — 후보 영상 분석 + 마스코트 합본 렌더링.

Phase 1: analyze_for_remix → spec JSON (Gemini 영상 직접 분석)
Phase 2: render_remix → 마스코트 이미지 + sprite 클립 + 원본 다운 + ffmpeg 합본

비용: 분석 ~$0.05 + 이미지 N장 ~$0.003*N (sprite 트릭은 영상 모델 0원).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from . import comfy_client
from . import fal_client
from . import kie_client
from . import llm
from . import mascot as mascot_worker


REMIX_SYSTEM_PROMPT = """너는 한국 양봉컴퍼니 YouTube Shorts 채널의 영상 편집 전문가야.
해외 viral 쇼츠 영상에 한국 양봉컴퍼니의 카테고리 시그니처 마스코트 1쌍
(SAVIOR 구원/보호 + VICTIM 위기/사고)이 정적 컷으로 끼어드는 webtoon_static 변형본을 만든다.

⭐ 핵심 사상 — 마스코트는 단순 reaction이 아니라 "사건의 숨은 주체"다.
- 마스코트가 invisible actor로 사고/사건의 원인이거나 책임자.
- narrative_role 필드에 그 역할을 한 줄로 명시.

⭐ 클립 개수 — 영상 결에 맞게 0~3개. 억지로 변곡점 만들지 마.
다음 패턴 중 영상에 가장 어울리는 거 1개 골라:

  (A) **변곡점 없는 단조로운 영상** → "인트로 + 아웃트로" 2개 컷:
      - clip 1: 시작 0~3초 사이 (시작부) — 영상 도입에 마스코트가 setup
      - clip 2: 마지막 -3초~끝 사이 (마무리) — 영상 끝에 마스코트가 punchline/coda
      예: 단순 풍경/일상/제품 리뷰 등 사건이 명확하지 않은 영상

  (B) **명확한 단일 변곡점** (1개) → "변곡점 직전" 1개 컷:
      - clip 1: 변곡점 0.5~2초 전 — 마스코트가 사건 막 일으키려는/막 막으려는 그 순간
      예: 단순 트릭/원샷 사고/한 방 결말

  (C) **빌드업 + 페이오프** (2개) → 2개 컷:
      - clip 1: 빌드업 (긴장/기대/공포) — 시청자 대신 마스코트가 반응
      - clip 2: 페이오프 (결말 증폭, 환호/허망/충격)
      예: 위험 직전 → 결과

  (D) **3-막 구조** (드물게 — 빌드업/위기/결말) → 3개 컷:
      - clip 1, 2, 3 각 막의 전환점

  (E) **마스코트가 어색한 영상** → **0개 (clips=[])**. 거짓 변곡점 만들지 마.
      예: 마스코트 narrative와 결이 안 맞는 영상, 사람 클로즈업만 가득한 영상

⚠️ 가장 중요한 룰:
  1. 변곡점이 없으면 (A) 인트로/아웃트로 패턴 사용 — 억지로 중간에 끼우지 마.
  2. 영상 흐름과 마스코트 narrative가 안 맞으면 (E) 0개 반환.
  3. 컷 개수보다 컷의 자연스러움이 우선.
  4. 형님이 결과 보고 "왜 여기?"라고 묻지 않을 자리만 추천.

⚠️ TIMESTAMP:
  • 반드시 영상 시작부터 **절대 초 단위** (예: 4.5, 14.0, 22.5)
  • **0~1 비율(normalized) 절대 금지**
  • 각 클립 duration 1.5~3초 권장 (정적 컷이라 너무 길면 지루)
  • estimated_total_seconds 필드에 영상 전체 길이 명시

⭐ webtoon_static 흐름 (default motion_mode) 필수 필드:
- character: "savior" 또는 "victim" — 컷마다 어느 마스코트 등장
  · savior = 카테고리의 SAVIOR 마스코트 (구원/보호/안도/희망 — 사고 막거나 구원해줌)
  · victim = 카테고리의 VICTIM 마스코트 (위기/사고/좌절 — 사건의 책임자/피해자)
  · 카테고리 마스코트 정보 받았으면 그 캐릭터의 컨셉/외형에 맞춰 narrative 추천
- expression: 영문 표정 (GPT-image-2 i2i lockdown 입력)
  · 예: "shocked dismay with eyes wide open and mouth gaping in surprise"
  · 예: "exhausted regret with eyes closed and mouth pressed flat"
  · 표정 명세를 매우 구체적으로 (눈/입/눈썹/눈물/홍조 등 visual cue 모두 영문)
- hand_action: 영문 손/팔 자세 (선택, 필요 시만)
  · 표정과 맞는 손동작이 있으면 명시. 없으면 빈 문자열 (baseline 자세 유지).
  · 예: "BOTH hands raised up to cover the eyes (palms facing inward)"
  · 예: "ONE hand (right hand) covering the forehead in a facepalm gesture"

⭐ mascot_placement (매우 중요 — Gemini가 영상 frame 보고 직접 추천):
  영상에서 사람/주제 객체의 위치 보고 마스코트 자리를 자연스럽게 정해.
  • x_center: 마스코트 가운데 가로 좌표 (0~1080, 캔버스 1080 너비 기준)
    - 사람이 화면 가운데 (~540)면, 마스코트는 사람 머리 옆 (~250 또는 ~830)
    - 화면 왼쪽 끝 = 0, 오른쪽 끝 = 1080
  • y_center: 마스코트 가운데 세로 좌표 (0~1920, 캔버스 1920 높이 기준)
    - 사람 머리 = 보통 화면 위쪽 1/3 (~500~700)
    - 사람 어깨 = ~700~900, 사람 몸 = ~900~1200, 사람 발 = ~1500~1800
  • mirror: true/false — 마스코트 좌우 반전 여부
    - 마스코트가 사람 쪽 (또는 객체 쪽) 향해야 자연스러우면 결정
    - 마스코트 baseline이 보통 정면 또는 한쪽 향함. 사람이 마스코트의 오른쪽에 있으면 mirror=false 유지 (마스코트 자연스럽게 오른쪽 향해), 사람이 왼쪽에 있으면 mirror=true
    - 동작 방향이 명확할 때 (예: "오른쪽으로 바람 분다") 그 방향에 맞춤
  • size: 마스코트 너비 px (기본 480, 작게 하려면 320 / 더 작게 240)
  예시: {"x_center": 270, "y_center": 600, "mirror": false, "size": 480}
  반드시 영상 frame 분석 결과 — 사람/객체 위치 보고 적합한 자리 추천.

⭐ Legacy 필드 (호환용):
- mascot_action_kr / mascot_action_en: webtoon_static에서는 expression이 우선
- korean_subtitle: 한국어 자막 15자 이하 (캡컷 후처리용)
- mascot_position / mascot_size_ratio: 옛 필드 — mascot_placement가 우선

JSON만 출력. 설명/주석/마크다운 X. 영상 결이 안 맞으면 clips=[] 빈 배열도 OK."""


REMIX_USER_TEMPLATE = """이 YouTube Shorts에 카테고리 시그니처 마스코트 1쌍이 등장할 위치를 추천해 (0~3개).

⭐ 카테고리 마스코트 1쌍:
{mascot_pair_block}

위 마스코트의 컨셉/외형/narrative 역할에 맞게 영상 흐름과 자연스럽게 어울리는 자리만 추천.
변곡점 없으면 인트로(시작) + 아웃트로(마지막) 패턴 OK. 영상 결과 안 맞으면 빈 배열도 OK.

JSON 형식:
{{
  "concept": "전체 변형본 컨셉 한 줄 요약 (한국어)",
  "estimated_total_seconds": 25.0,
  "pattern": "A_intro_outro / B_single / C_buildup_payoff / D_three_act / E_skip 중 1개",
  "clips": [
    {{
      "start": 3.5,
      "end": 6.0,
      "purpose": "왜 이 구간에 마스코트 끼우는지 (한국어 한 줄)",
      "character": "savior",
      "narrative_role": "마스코트가 사건의 숨은 주체로서 역할 한 줄 (한국어)",
      "actor_emotion": "안도 / 자책 / 체념 / 좌절 / 환희 / 충격 등 (한국어 1~2개)",
      "expression": "영문 표정 명세 (눈/입/눈썹/visual cue 구체적으로)",
      "hand_action": "영문 손/팔 자세 (선택, 비우면 baseline 자세 유지)",
      "mascot_action_kr": "마스코트가 뭘 하는지 (한국어 한 줄)",
      "mascot_action_en": "legacy 영어 동작 (선택)",
      "korean_subtitle": "한국어 자막 15자 이하 (예: 오늘 가시려나)",
      "mascot_position": "bottom_right / bottom_left / bottom_center 등",
      "mascot_size_ratio": 0.42
    }}
  ]
}}

영상 결이 마스코트와 안 맞거나 아예 끼우면 어색하면 clips=[] 빈 배열로 반환."""


def _format_mascot_block(mascot_roles: list[dict] | None,
                          mascot_pair: dict | None,
                          legacy_mascot: dict | None) -> str:
    """카테고리 마스코트 정보를 prompt block으로 포맷.
    1순위: 동적 N roles list. 2순위: legacy savior/victim pair. 3순위: 단일 mascot.
    """
    # 1. dynamic roles
    if mascot_roles:
        lines = ["사용 가능한 마스코트 (이 중에서만 character role_id 선택):"]
        for i, r in enumerate(mascot_roles, 1):
            rid = r.get("role_id") or f"role_{i}"
            name = r.get("name_kr") or rid
            label = r.get("role_label_kr") or ""
            narrative = r.get("narrative_role") or ""
            ck = r.get("concept_kr") or ""
            ce = r.get("concept_en") or ""
            lines.append(f"  · role_id=\"{rid}\" — {label} {name}")
            if narrative: lines.append(f"    역할: {narrative}")
            if ck: lines.append(f"    컨셉: {ck}")
            if ce: lines.append(f"    영문: {ce}")
        return "\n".join(lines)
    # 2. legacy pair
    if mascot_pair:
        savior = mascot_pair.get("savior") or {}
        victim = mascot_pair.get("victim") or {}
        return (
            f"사용 가능한 마스코트 1쌍 (legacy):\n"
            f"  · role_id=\"savior\" (구원/보호): {savior.get('concept_kr') or '(미설정)'}\n"
            f"  · role_id=\"victim\" (위기/사고): {victim.get('concept_kr') or '(미설정)'}"
        )
    # 3. legacy single
    if legacy_mascot:
        m = legacy_mascot
        return (
            f"단일 마스코트 (legacy):\n"
            f"  이름: {m.get('name') or '(미정)'} / 컨셉: {m.get('concept') or '(미정)'}\n"
            f"  → role_id는 단일이라 자유 (보통 savior 또는 mascot 등)"
        )
    return (
        "(마스코트 미설정) 추측 archetype: savior=구원, victim=위기. "
        "character는 narrative에 맞게 자유 선택."
    )


def _strip_json_fence(text: str) -> str:
    """Strip ```json ... ``` fences if Gemini wraps output."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


async def refine_clip_with_korean(
    *, purpose_kr: str = "",
    mascot_action_kr: str = "",
    character_role_id: str = "savior",
    mascot_roles: list[dict] | None = None,
) -> dict:
    """형님이 한글로 적은 마스코트 동작/목적 → Gemini가 상황 분석 → 영어 expression + hand_action.

    수동 추가 클립이거나 Gemini 자동 분석이 마음에 안 들 때 한글 한 줄 입력으로 영어 prompt 재생성.
    Returns {expression, hand_action, purpose_refined, mascot_action_kr}
    """
    # 마스코트 컨셉 컨텍스트
    role_ctx = ""
    if mascot_roles:
        for r in mascot_roles:
            if r.get("role_id") == character_role_id:
                role_ctx = (
                    f"이 컷에 등장할 마스코트:\n"
                    f"- role_id: {r.get('role_id')}\n"
                    f"- 라벨: {r.get('role_label_kr', '')}\n"
                    f"- 컨셉: {r.get('concept_kr', '')}\n"
                    f"- 영문 컨셉: {r.get('concept_en', '')}\n"
                )
                break

    system = (
        "너는 GPT-image-2 image-to-image용 영어 프롬프트 전문가야. "
        "한국어로 짧게 묘사된 마스코트 상황을 받아서, GPT-image-2가 정확히 그릴 수 있게 "
        "구체적인 영어 표정(expression)과 영어 손/팔 자세(hand_action)를 만들어내. "
        "표정은 visual cue (눈/입/눈썹/눈물/홍조 등) 모두 영문으로 명세. "
        "손/팔 자세는 baseline pose에서 변경되어야 할 때만 영문으로 명세, 변경 불필요 시 빈 문자열.\n\n"
        "출력은 JSON만:\n"
        '{"expression": "...", "hand_action": "...", "purpose_refined": "...", "mascot_action_kr_refined": "..."}\n'
        "purpose_refined: 상황 목적 한국어 한 줄 정제. mascot_action_kr_refined: 마스코트 동작 한국어 한 줄 정제."
    )
    user = (
        f"{role_ctx}\n"
        f"형님이 적은 한국어:\n"
        f"- 목적: {purpose_kr}\n"
        f"- 마스코트 동작: {mascot_action_kr}\n\n"
        "위 한국어를 분석해서 GPT-image-2 영문 프롬프트 + 정제된 한국어 출력 (JSON only)."
    )
    resp = await llm.gemini_chat(user, system=system,
                                  model="gemini-3.5-flash",
                                  max_tokens=2048)
    text = (resp.text or "").strip()
    start = text.find("{"); end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Gemini did not return JSON: {text[:200]}")
    try:
        data = json.loads(text[start:end+1])
    except Exception as e:
        raise RuntimeError(f"Gemini JSON parse fail: {e}")
    return {
        "expression": str(data.get("expression") or "").strip(),
        "hand_action": str(data.get("hand_action") or "").strip(),
        "purpose_refined": str(data.get("purpose_refined") or "").strip(),
        "mascot_action_kr_refined": str(data.get("mascot_action_kr_refined") or "").strip(),
    }


def _normalize_mascot_placement(mp) -> dict:
    """Gemini 응답의 mascot_placement 안전하게 정규화 — 캔버스 범위 내로 강제."""
    if not isinstance(mp, dict):
        return {"x_center": 540, "y_center": 1340, "mirror": False, "size": 480, "motion_type": "static"}
    def _clip(v, lo, hi, default):
        try: return max(lo, min(int(v), hi))
        except Exception: return default
    return {
        "x_center": _clip(mp.get("x_center"), 0, 1080, 540),
        "y_center": _clip(mp.get("y_center"), 0, 1920, 1340),
        "mirror": bool(mp.get("mirror", False)),
        "size": _clip(mp.get("size"), 200, 800, 480),
        "motion_type": str(mp.get("motion_type") or "static"),
    }


async def analyze_for_remix(video_url: str, mascot: dict | None = None,
                             mascot_pair: dict | None = None,
                             mascot_roles: list[dict] | None = None,
                             video_duration: float | None = None) -> dict:
    """Gemini가 YouTube 영상 직접 보고 마스코트 등장 spec 자동 추천.

    mascot_roles (동적 N개) 우선. 없으면 mascot_pair → legacy mascot.
    Returns spec dict (concept, pattern, clips). Clips 빈 배열도 valid.

    video_duration: 영상 길이 (초). 주어지면 norm_clips의 start/end를
    [0, duration] 안으로 clamp + start>=end 클립은 drop (preview-frame fail 예방).
    """
    user_prompt = REMIX_USER_TEMPLATE.format(
        mascot_pair_block=_format_mascot_block(mascot_roles, mascot_pair, mascot),
    )

    # retry 3회 — Gemini 빈 응답 (safety filter / rate limit) 대응
    raw = ""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = await llm.gemini_video_chat(
                youtube_url=video_url,
                prompt=user_prompt,
                system=REMIX_SYSTEM_PROMPT,
                model="gemini-3.5-flash",
                max_tokens=16384 if attempt > 0 else 8192,
                json_mode=True,
                temperature=0.3 + 0.05 * attempt,
            )
            raw = _strip_json_fence(resp.text or "")
            if raw.strip():
                break
            last_err = RuntimeError(f"Gemini 빈 응답 (시도 {attempt+1}/3)")
        except Exception as e:
            last_err = e
            continue

    if not raw.strip():
        return {
            "error": f"Gemini 빈 응답 (safety filter 또는 rate limit, 3회 retry 모두 fail): {last_err}",
            "raw_text": "",
            "concept": "",
            "clips": [],
        }

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse failed: {e}",
            "raw_text": raw[:2000],
            "concept": "",
            "clips": [],
        }

    # Normalize
    clips = spec.get("clips") or []
    norm_clips = []
    valid_positions = {
        "bottom_right", "bottom_left", "bottom_center",
        "top_right", "top_left", "top_center",
        "right", "left", "center",
    }
    for c in clips:
        try:
            # character: 자유 role_id (snake_case ASCII). legacy 호환: angel→savior/reaper→victim.
            ch_raw = str(c.get("character", "savior")).strip()
            ch_legacy = {"angel": "savior", "reaper": "victim"}
            ch = ch_legacy.get(ch_raw.lower(), ch_raw)
            ch = re.sub(r"[^a-zA-Z0-9_]+", "_", ch).strip("_").lower() or "savior"
            mp = str(c.get("mascot_position", "bottom_right")).strip().lower()
            if mp not in valid_positions:
                mp = "bottom_right"
            # caption_box: [top_pct, bot_pct] or null
            cb_raw = c.get("caption_box")
            if isinstance(cb_raw, list) and len(cb_raw) >= 2:
                caption_box = [float(cb_raw[0]), float(cb_raw[1])]
            else:
                caption_box = None
            # mascot_pixel_position: pixel-precise override (Phase 3)
            mpp = c.get("mascot_pixel_position")
            if isinstance(mpp, dict) and all(k in mpp for k in ("x", "y", "w", "h")):
                mascot_pixel_position = {
                    "x": int(mpp["x"]), "y": int(mpp["y"]),
                    "w": int(mpp["w"]), "h": int(mpp["h"]),
                }
            else:
                mascot_pixel_position = None
            c_start = float(c.get("start", 0))
            c_end = float(c.get("end", 0))
            # 영상 길이 알면 clamp + invalid clip drop (preview-frame fail 예방)
            if video_duration and video_duration > 0:
                # 원본부터 영상 길이 초과(=clamp 후에도 의미 X) 클립은 drop
                if c_start >= video_duration - 0.5:
                    continue
                c_start = max(0.0, min(c_start, video_duration - 0.2))
                c_end = max(0.0, min(c_end, video_duration))
                if c_end <= c_start + 0.5:
                    # 0.5초 미만 마스코트 클립은 의미 X (보통 2~5초) → drop
                    continue
            norm_clips.append({
                "start": c_start,
                "end": c_end,
                "purpose": str(c.get("purpose", "")).strip(),
                "character": ch,
                # v10.2: force boiling=1 (LoRA boiling = 그림 자체 변경 부작용. Phase 4에서 img2img 떨림 재구현 예정)
                "boiling_frames": 1,
                "narrative_role": str(c.get("narrative_role", "")).strip(),
                "actor_emotion": str(c.get("actor_emotion", "")).strip(),
                "actor_action": str(c.get("actor_action", "")).strip(),
                "mascot_action_kr": str(c.get("mascot_action_kr", "")).strip(),
                "mascot_action_en": str(c.get("mascot_action_en", "")).strip(),
                "korean_subtitle": str(c.get("korean_subtitle", "")).strip()[:30],
                "caption_box": caption_box,
                "mascot_position": mp,
                "mascot_size_ratio": max(0.20, min(float(c.get("mascot_size_ratio") or 0.42), 0.60)),
                "mascot_pixel_position": mascot_pixel_position,
                # webtoon_static 새 필드 — Gemini 추천 픽셀 좌표 + 좌우 반전.
                # 형님이 모달에서 직접 수정 가능. 없으면 default (가운데 하단).
                "mascot_placement": _normalize_mascot_placement(c.get("mascot_placement")),
                "bg_style_prompt": str(c.get("bg_style_prompt") or "").strip(),
                # webtoon_static 필드 (Gemini 추천 또는 형님 직접 입력)
                "expression": str(c.get("expression") or "").strip(),
                "hand_action": str(c.get("hand_action") or "").strip(),
            })
        except (ValueError, TypeError):
            continue

    return {
        "concept": spec.get("concept", ""),
        "estimated_total_seconds": spec.get("estimated_total_seconds"),
        "pattern": str(spec.get("pattern") or "").strip(),  # A_intro_outro / B / C / D / E_skip
        "clips": norm_clips,
        "tokens_used": resp.tokens_used,
    }


# ============================================================
# Phase 2 — Render: 마스코트 이미지 + ffmpeg 합본
# ============================================================

REMIX_OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "remixes"
REMIX_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_yt_dlp() -> str:
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    return shutil.which("yt-dlp") or "yt-dlp"


def _ffmpeg() -> str:
    # uvicorn started before brew install may not have /opt/homebrew/bin in PATH
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return shutil.which("ffmpeg") or "ffmpeg"


async def _run(*args: str, timeout: float = 300.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", "timeout"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


ORIGINALS_DIR = Path(__file__).parent.parent / "data" / "originals"


def _orig_cache_path(url: str) -> Path:
    import hashlib
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return ORIGINALS_DIR / f"{h}.mp4"


async def _validate_video_file(path: Path) -> bool:
    """ffprobe 검증 — 영상 stream + frame 수 + 길이 일관성 검사.
    video duration이 format duration 절반 미만이면 partial 다운으로 보고 거절."""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        ffprobe_bin = _ffmpeg().replace("ffmpeg", "ffprobe")
        # count_frames 빼고 codec/duration만 (1초 이내). 긴 영상 count_frames=매우 느림(60s+).
        rc, out, err = await _run(
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,duration",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
            timeout=10.0,
        )
        if rc != 0:
            return False
        import json as _json
        try:
            data = _json.loads(out or "{}")
            streams = data.get("streams", [])
            if not streams:
                return False
            # codec_name 있어야 valid video stream
            if not streams[0].get("codec_name"):
                return False
            # video stream 길이 vs 전체 파일 길이 (partial 다운 검출)
            try:
                vid_dur = float(streams[0].get("duration") or 0)
                fmt_dur = float(data.get("format", {}).get("duration") or 0)
                if fmt_dur >= 5.0 and vid_dur > 0 and vid_dur < fmt_dur * 0.5:
                    return False
            except (TypeError, ValueError):
                pass
            return True
        except Exception:
            return False
    except Exception:
        return False


# url별 다운로드 lock — 같은 영상에 동시 호출 들어와도 한 번만 다운, 나머지는 대기.
# preview-frame + source-video + render-clip이 동시에 같은 영상 요청해도 race condition 없음.
_download_locks: dict[str, asyncio.Lock] = {}


async def get_or_download_original(url: str) -> Path:
    """원본 mp4 cache 반환. 없거나 망가지면 yt-dlp 다운.
    같은 url에 동시 호출 들어오면 lock으로 직렬화 (race condition 방지).
    다운 직후 검증 + 안 되면 최대 5회 재시도 (백오프).
    """
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    cache = _orig_cache_path(url)
    # 빠른 path — 이미 cache 있으면 바로 반환 (lock 없이)
    if cache.exists() and cache.stat().st_size > 1000:
        if await _validate_video_file(cache):
            return cache

    # url별 lock 획득 — 동시 다운 직렬화 (race condition 방지)
    lock = _download_locks.setdefault(url, asyncio.Lock())
    async with lock:
        # lock 안에서 다시 확인 — 다른 호출이 이미 다운했을 수도
        if cache.exists() and cache.stat().st_size > 1000:
            if await _validate_video_file(cache):
                return cache
        # 망가진 cache면 삭제
        try:
            if cache.exists():
                cache.unlink()
        except Exception:
            pass
        # 5번 재시도 흐름 — lock 안에서 (race 없음)
        return await _do_download_with_retry(url, cache)


async def _do_download_with_retry(url: str, cache: Path) -> Path:
    """다운로드 + 검증 + 5번 재시도. lock 안에서만 호출됨."""
    last_err = ""
    import asyncio as _aio
    # 시도 5번 — YouTube 일시 장애 회복 시간 확보 (백오프):
    # 0: default (1080p 우선) — 즉시
    # 1: default — 5초 후
    # 2: safe (VP9 제외) — 15초 후
    # 3: safe — 30초 후
    # 4: bare (해상도/코덱 제한 없이 그냥 best) — 60초 후 (최후)
    attempts = [
        (0, "default"),
        (5, "default"),
        (15, "safe"),
        (30, "safe"),
        (60, "bare"),
    ]
    for sleep_sec, preset in attempts:
        if sleep_sec > 0:
            print(f"[download] {url} 재시도 {sleep_sec}초 대기 (preset={preset})", flush=True)
            await _aio.sleep(sleep_sec)
        try:
            await download_youtube_video(url, cache, format_preset=preset)
        except Exception as e:
            last_err = f"[{preset}] {str(e)[:200]}"
            try: cache.unlink()
            except Exception: pass
            continue
        if await _validate_video_file(cache):
            return cache
        last_err = f"[{preset}] partial video stream (validator 거절)"
        try: cache.unlink()
        except Exception: pass

    try:
        from . import notify as _n
        await _n.notify_error("영상 다운로드 실패",
                              f"{url}\n5번 시도 다 실패\n마지막 사유: {last_err}")
    except Exception:
        pass
    raise RuntimeError(f"다운로드 실패 5회: {url} — {last_err}")


async def download_youtube_video(url: str, out_path: Path,
                                 max_height: int = 1920,
                                 format_preset: str = "default") -> None:
    """yt-dlp 원본 영상 다운 — 그 영상의 최고 화질 자동 선택.

    format_preset:
      - "default": bestvideo+bestaudio/best — 그 영상의 최고 화질 자동 (1080p 있으면 1080p,
        없으면 720p, 그것도 없으면 360p — yt-dlp가 알아서 가장 좋은 거 선택)
      - "safe":    VP9 제외 (VP9 fragment 깨지는 영상 회피)
      - "bare":    best 단일 stream (분리 다운 X, 최후 fallback)
    """
    bin_ = _resolve_yt_dlp()
    if format_preset == "bare":
        # 최후 fallback — 단일 stream best (분리 다운 X, fragment 깨짐 회피)
        fmt = "best"
    elif format_preset == "safe":
        # VP9 제외 — VP9 fragment 못 받는 영상용. 그 영상의 best (VP9 빼고)
        fmt = (
            "bestvideo[vcodec!*=vp9]+bestaudio/"
            "best[vcodec!*=vp9]/"
            "best"
        )
    else:
        # default — 그 영상의 최고 화질 자동 (yt-dlp 표준 fallback)
        # 분리 stream 우선 (1080p 받기 위해), 안 되면 단일 best로 자동 fallback
        # 즉 1080p 있는 영상 = 1080p, 720p만 있는 영상 = 720p, 360p만 = 360p
        fmt = "bestvideo+bestaudio/best"
    ffmpeg_bin = _ffmpeg()
    rc, _, err = await _run(
        bin_, url,
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--ffmpeg-location", ffmpeg_bin,
        "--force-overwrites",
        "--no-warnings",
        "--no-part",
        "--retries", "10",
        "--fragment-retries", "10",
        "--retry-sleep", "linear=1::2",
        "--socket-timeout", "30",
        "--remote-components", "ejs:github",  # YouTube JS challenge 해결 (2026-05-18)
        "-o", str(out_path),
        timeout=300.0,
    )
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"yt-dlp failed: {err[:300]}")


async def make_sprite_clip(image_path: Path, duration: float,
                           out_path: Path, width: int = 1080,
                           height: int = 1920, fps: int = 30) -> None:
    """정적 이미지 → ken-burns 줌인 효과 영상.

    영상 모델 안 쓰는 무료 sprite 트릭. duration 초 동안 살짝 줌인.
    """
    total_frames = max(int(duration * fps), 30)
    # zoompan z increment scaled so we don't over-zoom on long clips
    zoom_speed = 0.0015 if duration <= 4 else 0.0008
    vf = (
        f"scale={width*2}:{height*2}:force_original_aspect_ratio=increase,"
        f"crop={width*2}:{height*2},"
        f"zoompan=z='min(zoom+{zoom_speed},1.3)':d={total_frames}:"
        f"s={width}x{height}:fps={fps}"
    )
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-loop", "1", "-i", str(image_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-r", str(fps),
        "-an",  # no audio
        str(out_path),
        timeout=120.0,
    )
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg sprite failed: {err[:300]}")


async def cut_segment(input_path: Path, start: float, end: float,
                      out_path: Path, width: int = 1080,
                      height: int = 1920, fps: int = 30) -> None:
    """원본 영상의 [start, end] 구간 잘라서 표준 해상도. 오디오 살림 (합본에서 원본 부분 소리 들어감).
    합본 호환 형식 (1080x1920, 30fps, libx264, aac stereo 44100, 기준 컷 간격 1초).
    결과 video stream 길이 검증 — 기대 길이의 절반 미만이면 망가진 cache로 보고 거절.
    """
    duration = max(end - start, 0.1)
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-i", str(input_path),
        "-ss", str(start),
        "-t", str(duration),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,"
               f"crop={width}:{height},setsar=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        "-r", str(fps),
        "-g", str(fps),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        str(out_path),
        timeout=180.0,
    )
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg cut failed: {err[:300]}")
    # 결과 검증: input 영상 실제 길이 기준 expected duration 계산.
    # tail cut처럼 end가 영상 길이 초과해도 정상 (영상 끝까지). 검증은 expected 기준.
    input_dur = await _probe_duration(input_path)
    if input_dur > 0:
        expected_dur = min(duration, max(input_dur - start, 0.0))
    else:
        expected_dur = duration
    if expected_dur >= 3.0:
        result_vid_dur = await _probe_video_duration(out_path)
        if result_vid_dur > 0 and result_vid_dur < expected_dur * 0.5:
            try: out_path.unlink()
            except Exception: pass
            raise RuntimeError(
                f"cut 결과 짤림 (기대 {expected_dur:.1f}s, 결과 video {result_vid_dur:.1f}s) — "
                f"원본 cache 망가진 듯. 다시 시도 시 cache 자동 재다운됨"
            )


async def _has_audio_stream(path: Path) -> bool:
    """ffprobe로 audio stream 존재 여부 검사."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1",
            str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return out.decode().strip() == "audio"
    except Exception:
        return False


async def _probe_duration(path: Path) -> float:
    """ffprobe로 영상 길이 (초)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return float(out.decode().strip() or 0)
    except Exception:
        return 0.0


async def _probe_video_duration(path: Path) -> float:
    """ffprobe로 video stream 길이 (초). format duration과 다를 수 있음 (partial 다운 검출용)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=nw=1:nk=1", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return float(out.decode().strip() or 0)
    except Exception:
        return 0.0


async def _normalize_segment(in_path: Path, out_path: Path,
                             width: int = 1080, height: int = 1920,
                             fps: int = 30) -> None:
    """영상 1개를 합본 호환 형식으로 정규화. 1080x1920 + 30fps + libx264 + aac stereo 44100.
    audio 없으면 silent 추가."""
    has_a = await _has_audio_stream(in_path)
    args = [_ffmpeg(), "-y", "-i", str(in_path)]
    if not has_a:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    args += [
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
    ]
    if not has_a:
        args += ["-map", "0:v", "-map", "1:a", "-shortest"]
    else:
        args += ["-map", "0:v", "-map", "0:a"]
    args += [
        "-g", str(fps),  # 기준 컷 간격 = 1초 (이어붙이기 안전)
        "-movflags", "+faststart",
        str(out_path),
    ]
    rc, _, err = await _run(*args, timeout=300.0)
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"normalize 실패 ({in_path.name}): {err[:200]}")


async def concat_segments(segment_paths: list[Path], out_path: Path,
                          width: int = 1080, height: int = 1920,
                          fps: int = 30,
                          progress_cb=None) -> None:
    """빠른 흐름: 각 segment 정규화 (병렬) → concat demuxer 단순 이어붙이기 (재인코딩 X).
    원본 cut의 audio 유지, 마스코트 클립엔 silent audio 추가.
    영상 합본 = 정규화 시간 (가장 긴 segment 기준) + 이어붙이기 (수 초). 30초 안."""
    if not segment_paths:
        raise ValueError("no segments to concat")
    if len(segment_paths) == 1:
        shutil.copy(segment_paths[0], out_path)
        if progress_cb:
            await progress_cb(100, "✅ 합본 완료 (단일 segment)")
        return

    n = len(segment_paths)
    durations = await asyncio.gather(*[_probe_duration(p) for p in segment_paths])
    total_duration = sum(durations) or 1.0
    if progress_cb:
        await progress_cb(5, f"🎬 합본 시작 — {n}개 영상, 총 {total_duration:.1f}초")

    # 1단계: 각 segment 정규화 (병렬, ultrafast)
    work_dir = out_path.parent / f"_norm_{out_path.stem}"
    work_dir.mkdir(exist_ok=True)
    norm_paths = [work_dir / f"norm_{i:02d}.mp4" for i in range(n)]

    done = [0]
    async def normalize_one(idx: int):
        await _normalize_segment(segment_paths[idx], norm_paths[idx], width, height, fps)
        done[0] += 1
        if progress_cb:
            pct = 5 + int(done[0] / n * 60)  # 5~65%
            await progress_cb(pct, f"🎬 영상 정규화 {done[0]}/{n}")

    try:
        await asyncio.gather(*[normalize_one(i) for i in range(n)])

        # 2단계: concat demuxer로 재인코딩 없이 이어붙이기
        if progress_cb:
            await progress_cb(70, "🎬 이어붙이는 중 (재인코딩 X)")
        list_file = work_dir / "concat_list.txt"
        list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in norm_paths),
                              encoding="utf-8")

        args = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(out_path)]
        rc, _, err = await _run(*args, timeout=120.0)
        if rc != 0 or not out_path.exists():
            raise RuntimeError(f"이어붙이기 실패: {err[:300]}")
        # 결과 검증 1: video stream 길이가 audio 길이 절반 미만이면 망가진 합본
        result_vid_dur = await _probe_video_duration(out_path)
        result_fmt_dur = await _probe_duration(out_path)
        if (result_fmt_dur >= 5.0 and result_vid_dur > 0
                and result_vid_dur < result_fmt_dur * 0.5):
            try: out_path.unlink()
            except Exception: pass
            err_msg = (f"합본 짤림 (video {result_vid_dur:.1f}s vs 전체 {result_fmt_dur:.1f}s) — "
                       f"원본 cache 망가진 듯")
            try:
                from . import notify as _n
                await _n.notify_error("합본 결과 짤림", err_msg)
            except Exception:
                pass
            raise RuntimeError(err_msg)
        # 결과 검증 2: 합본 길이가 입력 segments 합보다 2초 이상 짧으면 segment 빠진 것
        if result_fmt_dur > 0 and total_duration - result_fmt_dur > 2.0:
            try: out_path.unlink()
            except Exception: pass
            err_msg = (f"합본 segment 빠짐 (기대 {total_duration:.1f}s, 실제 {result_fmt_dur:.1f}s, "
                       f"{total_duration - result_fmt_dur:.1f}s 짤림)")
            try:
                from . import notify as _n
                await _n.notify_error("합본 segment 누락", err_msg)
            except Exception:
                pass
            raise RuntimeError(err_msg)
    finally:
        # 임시 파일 정리
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    if progress_cb:
        await progress_cb(100, "✅ 합본 완료")


# Variation suffixes for multi-pose: same character, micro-different poses to
# fake frame-by-frame animation when stitched. Same seed → same character;
# different prompt suffix → slightly different pose/expression.
_POSE_VARIANTS = [
    "",
    ", slight head tilt to right",
    ", looking slightly up, mouth open in surprise",
    ", eyes more closed, head tilt to left",
    ", body leaning forward, hands raised slightly higher",
    ", subtle expression shift, eyes wide",
    ", different angle, three-quarter view",
]


async def generate_pose_variants(dissection_id: str, base_action: str,
                                 count: int = 5) -> list[dict]:
    """Generate N variations of the same action (parallel). Same mascot seed
    so character stays consistent — only pose/expression varies per frame.
    """
    suffixes = _POSE_VARIANTS[:count] if count <= len(_POSE_VARIANTS) else (
        _POSE_VARIANTS + [_POSE_VARIANTS[-1]] * (count - len(_POSE_VARIANTS))
    )
    prompts = [f"{base_action}{s}" for s in suffixes]
    results = await asyncio.gather(*[
        mascot_worker.generate_pose_image(dissection_id, p) for p in prompts
    ], return_exceptions=True)
    out: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        out.append(r)
    return out


async def make_multipose_clip(image_paths: list[Path], duration: float,
                              out_path: Path, width: int = 1080,
                              height: int = 1920, fps: int = 30) -> None:
    """N image들을 균등 시간 분배 + 각각 ken-burns → concat = 가짜 프레임 애니.

    형님 reference 영상의 저승사자 (3~4 포즈 컷 전환) 동급 스타일.
    """
    n = max(len(image_paths), 1)
    per = max(duration / n, 0.3)  # 각 포즈 최소 0.3s
    work = out_path.parent / f"_multi_{out_path.stem}"
    work.mkdir(exist_ok=True)
    sub_clips: list[Path] = []
    try:
        for i, img in enumerate(image_paths):
            sub = work / f"pose_{i}.mp4"
            await make_sprite_clip(img, per, sub, width=width, height=height, fps=fps)
            sub_clips.append(sub)
        await concat_segments(sub_clips, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def extract_frame(video_path: Path, time_sec: float, out_png: Path,
                        width: int = 1080, height: int = 1920) -> None:
    """Single-frame snapshot at time_sec, normalized to 1080x1920.
    영상 길이 초과 시 자동 clamp (no packets 에러 방지) + fast seek (-ss 앞)."""
    # 영상 길이 초과 clamp — 오래된 spec의 영상 끝 초과 clip 방어
    try:
        dur = await _probe_video_duration(Path(video_path))
        if dur and dur > 0 and time_sec > dur - 0.2:
            time_sec = max(0.0, dur - 0.2)
    except Exception:
        pass
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,"
               f"crop={width}:{height},setsar=1",
        "-q:v", "2",
        str(out_png),
        timeout=60.0,
    )
    if rc != 0 or not out_png.exists():
        raise RuntimeError(f"ffmpeg frame extract failed (t={time_sec:.1f}): {err[:300]}")


async def compose_freeze_overlay(bg_image: Path, alpha_mov: Path,
                                 duration: float, out_mp4: Path,
                                 width: int = 1080, height: int = 1920,
                                 fps: int = 30) -> None:
    """Frozen background image (loop for `duration` seconds) +
    PRORES4444 mascot video with alpha overlaid on top, centered.
    Output is the same params as sprite/cut so concat -c copy works.
    """
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-loop", "1", "-t", str(duration), "-i", str(bg_image),
        "-i", str(alpha_mov),
        "-filter_complex",
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1[bg];"
        f"[1:v]scale={width}:-1:force_original_aspect_ratio=decrease[mascot];"
        f"[bg][mascot]overlay=(W-w)/2:(H-h)/2:format=auto:shortest=1[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "18",
        "-r", str(fps), "-t", str(duration),
        "-an",
        "-movflags", "+faststart",
        str(out_mp4),
        timeout=180.0,
    )
    if rc != 0 or not out_mp4.exists():
        raise RuntimeError(f"ffmpeg freeze-overlay failed: {err[:400]}")


async def compose_freeze_runway(bg_image: Path, mascot_mp4: Path,
                                duration: float, out_mp4: Path,
                                width: int = 1080, height: int = 1920,
                                fps: int = 30,
                                colorkey: str = "0xFFFFFF",
                                similarity: float = 0.18,
                                blend: float = 0.10) -> None:
    """Compose: frozen bg image looped + mascot mp4 (white-background)
    chroma-keyed to transparent + overlaid centered. Single ffmpeg pass —
    no separate matting step needed.

    similarity (0.01~0.99): how close to colorkey color counts as background.
    blend (0~1): edge softness. Tune if mascot edges look fringy.
    """
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-loop", "1", "-t", str(duration), "-i", str(bg_image),
        "-i", str(mascot_mp4),
        "-filter_complex",
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1[bg];"
        f"[1:v]colorkey={colorkey}:{similarity}:{blend},"
        f"scale={width}:-1:force_original_aspect_ratio=decrease[mascot];"
        f"[bg][mascot]overlay=(W-w)/2:(H-h)/2:shortest=1[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "18",
        "-r", str(fps), "-t", str(duration),
        "-an",
        "-movflags", "+faststart",
        str(out_mp4),
        timeout=180.0,
    )
    if rc != 0 or not out_mp4.exists():
        raise RuntimeError(f"ffmpeg freeze-runway compose failed: {err[:400]}")


async def make_freeze_runway_clip(orig_video: Path, start_sec: float,
                                  duration: float, mascot_pose: Path,
                                  action_en: str, out_clip: Path,
                                  work_dir: Path,
                                  use_kontext: bool = False) -> dict:
    """Cheaper/simpler reference-style clip using Kie Runway:
       1. extract original frame at start_sec
       2. (optional) Kie Flux Kontext img2img → black-and-white line art
       3. Kie Runway Gen-4 Turbo i2v on mascot reference (5s 720p)
       4. ffmpeg colorkey + overlay → composite mp4

    Cheaper than freeze_overlay (no BiRefNet, no LTX cold start).
    Returns {cost_usd}.
    """
    from . import kie_client
    work_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0

    orig_frame = work_dir / "frame.png"
    bg_image = work_dir / ("illustration.png" if use_kontext else "frame.png")
    mascot_mp4 = work_dir / "mascot.mp4"

    await extract_frame(orig_video, start_sec, orig_frame)

    if use_kontext:
        # Hook for later: Kie Flux Kontext img2img once we wire it up.
        # For now skip, use raw frame as bg.
        bg_image = orig_frame

    fr = await kie_client.runway_i2v(mascot_pose, duration, mascot_mp4,
                                     action_prompt=action_en,
                                     duration=5, quality="720p",
                                     aspect_ratio="9:16")
    cost += 0.06   # ~$0.06/clip

    await compose_freeze_runway(bg_image, mascot_mp4, duration, out_clip)
    return {"cost_usd": round(cost, 4)}


# ============================================================
# yangbong motion_mode v2 — Replicate trained LoRA (ControlNet-style 통합)
# ============================================================

# v9 (Kontext 합성) 단점:
#   - 마스코트가 별도 layer = 스티커 느낌
#   - pose 보존 약함
# v10 (LoRA img2img):
#   - 우리 trained Flux+LoRA 모델로 단일 호출 → 사람+마스코트 통합 일러스트
#   - img2img prompt_strength로 구도 보존 + LoRA로 캐릭터 일관성

YB_CANVAS_W, YB_CANVAS_H = 1080, 1920


def _yb_resolve_mascot_box(canvas_w: int, canvas_h: int, *,
                           position: str, size_ratio: float,
                           pixel_position: dict | None = None) -> tuple[int, int, int, int]:
    """6-corner code + size_ratio (또는 pixel override)을 box 좌표로.

    Returns (x, y, w, h). pixel_position이 있으면 그게 우선.
    """
    if pixel_position:
        return (int(pixel_position["x"]), int(pixel_position["y"]),
                int(pixel_position["w"]), int(pixel_position["h"]))
    margin = int(canvas_w * 0.04)
    mw = int(canvas_w * size_ratio)
    mh = mw  # square approx based on canvas width
    pos = (position or "bottom_right").lower()
    if pos in ("right", "bottom_right"):
        return (canvas_w - mw - margin, canvas_h - mh - margin, mw, mh)
    if pos in ("left", "bottom_left"):
        return (margin, canvas_h - mh - margin, mw, mh)
    if pos == "bottom_center":
        return ((canvas_w - mw) // 2, canvas_h - mh - margin, mw, mh)
    if pos == "top_right":
        return (canvas_w - mw - margin, margin, mw, mh)
    if pos == "top_left":
        return (margin, margin, mw, mh)
    if pos == "top_center":
        return ((canvas_w - mw) // 2, margin, mw, mh)
    if pos == "center":
        return ((canvas_w - mw) // 2, (canvas_h - mh) // 2, mw, mh)
    return (canvas_w - mw - margin, canvas_h - mh - margin, mw, mh)


def _yb_clean_frame(frame_path: Path, out_path: Path, *,
                    caption_box: list | None = None,
                    mascot_box: tuple[int, int, int, int] | None = None,
                    soft_buffer: int = 12) -> None:
    """Frame 전처리:
    1. caption_box ([top_pct, bot_pct]) 영역 흰색 fill (Canny가 자막 외곽선 잡지 못하게).
       caption_box=None이면 자막 fill 안 함.
    2. mascot_box ((x, y, w, h)) 영역 + buffer 흰색 fill (Flux가 거기서 hallucinate 방지).
       mascot_box=None이면 fill 안 함.
    """
    from PIL import Image, ImageDraw
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)

    if caption_box and len(caption_box) >= 2:
        top_y = int(h * float(caption_box[0]))
        bot_y = int(h * float(caption_box[1]))
        if 0 <= top_y < bot_y <= h:
            d.rectangle([0, top_y, w, bot_y], fill=(255, 255, 255))

    if mascot_box:
        mx, my, mw, mh = mascot_box
        # add small buffer so canny doesn't catch the box edge
        d.rectangle([max(mx - soft_buffer, 0), max(my - soft_buffer, 0),
                     min(mx + mw + soft_buffer, w), min(my + mh + soft_buffer, h)],
                    fill=(255, 255, 255))

    img.save(out_path)


def _yb_composite_mascot_on_bg(bg_path: Path, mascot_path: Path,
                               out_path: Path, *,
                               box: tuple[int, int, int, int],
                               white_threshold: int = 250) -> None:
    """PIL composite: bg sketch 위에 mascot transparent로 올림.

    box: (x, y, w, h) — 마스코트가 들어갈 영역.
    누끼: 거의 순백 픽셀만 transparent (threshold 250 = 흰 robe 보존).
    Scale: box 안에 비율 유지로 fit (가로/세로 중 작은 쪽 기준).
    Paste: 중앙 정렬, 발이 box 바닥에 닿게.
    """
    from PIL import Image
    bg = Image.open(bg_path).convert("RGBA")
    # Flux Canny may return 832x1472 etc — resize to canonical 1080x1920
    if bg.size != (YB_CANVAS_W, YB_CANVAS_H):
        bg = bg.resize((YB_CANVAS_W, YB_CANVAS_H), Image.LANCZOS)
    mascot = Image.open(mascot_path).convert("RGBA")

    # 1. mascot pure-white → alpha (only near-pure-white, preserve robe colors)
    px = mascot.load()
    mw0, mh0 = mascot.size
    for y in range(mh0):
        for x in range(mw0):
            r, g, b, a = px[x, y]
            if r >= white_threshold and g >= white_threshold and b >= white_threshold:
                px[x, y] = (r, g, b, 0)

    # 1b. crop to alpha bounding box — LoRA generates char in center with white padding,
    # cropping makes the character fill the target box instead of being tiny inside.
    alpha = mascot.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        mascot = mascot.crop(bbox)
        mw0, mh0 = mascot.size

    # 2. scale: fit inside box preserving aspect (use the more constraining axis)
    bx, by, bw, bh = box
    scale = min(bw / max(mw0, 1), bh / max(mh0, 1))
    target_w = max(int(mw0 * scale), 32)
    target_h = max(int(mh0 * scale), 32)
    mascot = mascot.resize((target_w, target_h), Image.LANCZOS)

    # 3. paste — h-center + bottom-align (마스코트 발이 box 바닥에 닿게)
    paste_x = bx + (bw - target_w) // 2
    paste_y = by + (bh - target_h)
    bg.paste(mascot, (paste_x, paste_y), mascot)
    bg.convert("RGB").save(out_path, "PNG")


async def make_yangbong_clip(orig_video: Path, start_sec: float,
                             duration: float,
                             out_clip: Path, work_dir: Path,
                             *,
                             character: str = "reaper",
                             action_prompt: str = "",
                             bg_style_prompt: str = "",
                             caption_box: list | None = None,
                             mascot_position: str = "bottom_right",
                             mascot_size_ratio: float = 0.42,
                             mascot_pixel_position: dict | None = None,
                             boiling_frames: int = 1,
                             boiling_fps: int = 6,
                             zoom: bool = True,
                             zoom_to: float = 1.15,
                             style_seed: int = 0) -> dict:
    """v10.2 — spec-driven dynamic pipeline:
       1. 실사 frame 추출 (start_sec) → decrease+pad with white
       2. _yb_clean_frame: caption_box (자막) + mascot_box (마스코트 영역) 흰색 fill
       3. ControlNet Canny → bg sketch + LoRA text2img → mascot (sequential)
       4. PIL composite — mascot 누끼 + bg에 overlay (mascot_box 좌표)
       5. ffmpeg zoom-in + (boiling) cycle → mp4

    caption_box: [top_pct, bot_pct] (0~1) 또는 None (자막 없음).
    mascot_position: "bottom_right" / "bottom_left" / ... 6 corners.
    mascot_size_ratio: 0.20~0.60 (캔버스 가로 대비).
    mascot_pixel_position: {x, y, w, h} — Phase 3 visual override (있으면 우선).
    """
    from . import replicate_client
    work_dir.mkdir(parents=True, exist_ok=True)

    # mascot box 미리 계산 (clean_frame + composite 둘 다 같은 좌표 사용)
    mascot_box = _yb_resolve_mascot_box(
        YB_CANVAS_W, YB_CANVAS_H,
        position=mascot_position,
        size_ratio=mascot_size_ratio,
        pixel_position=mascot_pixel_position,
    )

    # 1. extract frame at exact start_sec
    # decrease + pad with white = 비율 안 맞으면 crop 대신 흰 padding (다리 잘림 방지)
    frame_raw = work_dir / "1_frame_raw.png"
    rc, _, err = await _run(
        _ffmpeg(), "-y",
        "-i", str(orig_video),
        "-ss", str(start_sec),
        "-frames:v", "1",
        "-vf", f"scale={YB_CANVAS_W}:{YB_CANVAS_H}:force_original_aspect_ratio=decrease,"
               f"pad={YB_CANVAS_W}:{YB_CANVAS_H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
        "-q:v", "2", str(frame_raw), timeout=60.0,
    )
    if rc != 0 or not frame_raw.exists():
        raise RuntimeError(f"yangbong frame extract failed: {err[:300]}")

    # 1b. clean: caption + mascot area both → white fill
    frame = work_dir / "1_frame.png"
    _yb_clean_frame(frame_raw, frame,
                    caption_box=caption_box,
                    mascot_box=mascot_box)

    # 2. (parallel) bg ControlNet + mascot LoRA
    n = max(boiling_frames, 1)
    bg_pngs = [work_dir / f"2_bg_{i}.png" for i in range(n)]
    mascot_pngs = [work_dir / f"3_mascot_{i}.png" for i in range(n)]

    bg_prompt = (bg_style_prompt or "").strip() or (
        "rough hand-drawn sketch line art of people in a room, "
        "thick black ink outlines on plain white background, "
        "B-grade Korean web comic doodle style, wobbly hand-drawn lines, "
        "minimal background, simple flat doodle"
    )

    # bg seeds + mascot seeds — 별도 (boiling means different randomness for both)
    # All clips of same render share style_seed → consistent line style across video.
    # Within a clip, boiling frames vary by tiny offset (boiling pattern only).
    bg_seeds = [(style_seed or 17_000) + i * 7919 for i in range(n)]
    mascot_seeds = [(style_seed or 17_000) + 50_000 + i * 7919 for i in range(n)]

    # Strictly sequential — Replicate "1 burst" rate limit. Sleep between calls.
    bg_results = []
    mascot_results = []
    for i in range(n):
        bg_r = await replicate_client.flux_canny_sketch(
            frame, bg_pngs[i], prompt=bg_prompt, seed=bg_seeds[i],
        )
        bg_results.append(bg_r)
        await asyncio.sleep(2)
        m_r = await replicate_client.lora_text2img(
            character, mascot_pngs[i],
            action_prompt=action_prompt or "standing facing forward",
            seed=mascot_seeds[i],
        )
        mascot_results.append(m_r)
        if i < n - 1:
            await asyncio.sleep(2)
    cost = sum(float(r.get("cost_usd") or 0)
               for r in (bg_results + mascot_results))

    # 3. PIL composite per frame — uses pre-resolved mascot_box for exact alignment
    composed_pngs = [work_dir / f"4_composed_{i}.png" for i in range(n)]
    for i in range(n):
        _yb_composite_mascot_on_bg(
            bg_pngs[i], mascot_pngs[i], composed_pngs[i],
            box=mascot_box,
        )

    # 4. ffmpeg compose
    canvas_filter = (
        f"scale={YB_CANVAS_W}:{YB_CANVAS_H}:force_original_aspect_ratio=decrease,"
        f"pad={YB_CANVAS_W}:{YB_CANVAS_H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1"
    )

    fps = 24
    if n > 1:
        loops_needed = int(duration * boiling_fps / n) + 2
        if zoom:
            zoom_filter = (
                f",zoompan=z='min(zoom+{(zoom_to - 1.0) / max(duration, 1.0) / boiling_fps:.5f},{zoom_to})'"
                f":d=1:s={YB_CANVAS_W}x{YB_CANVAS_H}:fps={boiling_fps}"
            )
        else:
            zoom_filter = ""
        rc, _, err = await _run(
            _ffmpeg(), "-y",
            "-framerate", str(boiling_fps),
            "-stream_loop", str(loops_needed),
            "-i", str(work_dir / "4_composed_%d.png"),
            "-t", str(duration),
            "-vf", canvas_filter + zoom_filter,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "20",
            "-r", str(fps), "-an",
            "-movflags", "+faststart",
            str(out_clip), timeout=180.0,
        )
    else:
        if zoom:
            total_frames = max(int(duration * fps), 48)
            zoom_increment = (zoom_to - 1.0) / max(total_frames, 1)
            zoom_filter = (
                f",zoompan=z='min(zoom+{zoom_increment:.5f},{zoom_to})'"
                f":d={total_frames}:s={YB_CANVAS_W}x{YB_CANVAS_H}:fps={fps}"
            )
        else:
            zoom_filter = ""
        rc, _, err = await _run(
            _ffmpeg(), "-y",
            "-loop", "1", "-i", str(composed_pngs[0]),
            "-t", str(duration),
            "-vf", canvas_filter + zoom_filter,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "20",
            "-r", str(fps), "-an",
            "-movflags", "+faststart",
            str(out_clip), timeout=180.0,
        )

    if rc != 0 or not out_clip.exists():
        raise RuntimeError(f"yangbong compose failed: {err[-2000:]}")

    return {
        "cost_usd": round(cost, 4),
        "character": character,
        "boiling_frames": n,
    }


# Legacy v9 (Kontext) — 보존: 필요시 motion_mode="yangbong_legacy" 추가 시 활용 가능

async def make_freeze_overlay_clip(orig_video: Path, start_sec: float,
                                   duration: float, mascot_pose: Path,
                                   action_en: str, out_clip: Path,
                                   work_dir: Path) -> dict:
    """Reference-style clip: original frame at start_sec → Flux Kontext
    line-art transform → LTX i2v of mascot → BiRefNet video matting →
    ffmpeg overlay. Returns {cost_usd}.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0

    orig_frame = work_dir / "frame.png"
    illustration = work_dir / "illustration.png"
    mascot_mp4 = work_dir / "mascot.mp4"
    mascot_alpha = work_dir / "mascot_alpha.mov"

    await extract_frame(orig_video, start_sec, orig_frame)
    kr = await fal_client.flux_kontext_img2img(orig_frame, illustration)
    cost += float(kr.get("cost_usd") or 0)

    fr = await fal_client.i2v(mascot_pose, duration, mascot_mp4,
                              action_prompt=action_en)
    cost += 0.05 * max(duration, 1.0) / 2.0

    br = await fal_client.birefnet_video_alpha(mascot_mp4, mascot_alpha)
    cost += float(br.get("cost_usd") or 0)

    await compose_freeze_overlay(illustration, mascot_alpha, duration, out_clip)
    return {"cost_usd": round(cost, 4)}


async def make_i2v_clip_wan(image_path: Path, action_prompt: str,
                            duration: float, out_path: Path) -> float:
    """Wan 2.1 I2V — 진짜 motion. 정적 이미지 + prompt → 영상.

    DeepInfra Wan-AI/Wan2.1-I2V-14B. 비용 ~$0.18/5초.
    Returns cost_usd.
    """
    import base64
    DEEPINFRA_KEY = os.getenv("DEEPINFRA_API_KEY", "")
    if not DEEPINFRA_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY not set")
    img_bytes = image_path.read_bytes()
    img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()

    import httpx
    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.post(
            "https://api.deepinfra.com/v1/inference/Wan-AI/Wan2.1-I2V-14B-720P",
            headers={"Authorization": f"Bearer {DEEPINFRA_KEY}",
                     "Content-Type": "application/json"},
            json={
                "image": img_b64,
                "prompt": action_prompt,
                "num_frames": min(int(duration * 16), 81),
                "fps": 16,
                "guidance_scale": 5.0,
            },
        )
        r.raise_for_status()
        d = r.json()

    video_field = d.get("video_url") or d.get("video") or d.get("videos")
    if isinstance(video_field, list):
        video_field = video_field[0] if video_field else None
    if not video_field:
        raise RuntimeError(f"Wan returned no video. keys={list(d.keys())}")

    if isinstance(video_field, str) and video_field.startswith("http"):
        async with httpx.AsyncClient(timeout=120.0) as c:
            rv = await c.get(video_field)
        out_path.write_bytes(rv.content)
    elif isinstance(video_field, str) and video_field.startswith("data:"):
        b64 = video_field.split(",", 1)[1]
        out_path.write_bytes(base64.b64decode(b64))
    else:
        raise RuntimeError("unknown video field format")

    return float(d.get("inference_status", {}).get("cost") or 0)


async def render_remix(remix_id: int, dissection_id: str, candidate_url: str,
                       spec: dict,
                       progress_cb=None,
                       motion_mode: str = "sprite",   # "sprite" or "wan"
                       make_combined: bool = True,
                       ) -> dict:
    """spec 기반 렌더 — 마스코트 이미지 + 클립 N개 영구 저장 + (옵션) 합본.

    motion_mode:
      - "sprite":         무료 ken-burns (정적 이미지 줌인). 빠름.
      - "sprite_multi":   5포즈 컷 전환 (가짜 프레임 애니).
      - "comfy":          로컬 ComfyUI LTX 2B distilled — 진짜 motion. 무료. ~30~60s/clip.
      - "fal":            Fal.ai LTX 13B distilled — 진짜 motion. ~₩100/clip. ~5~10s/clip.
      - "freeze_overlay": Reference 스타일 — 원본 freeze + 흑백 일러스트 변환 +
                          LTX 마스코트 + BG 제거 + overlay 합성. ~₩200/clip.
      - "runway":         Kie.ai Runway Gen-4 Turbo i2v — always warm. ~₩84/clip.
      - "freeze_runway":  양봉 reference 스타일 — 원본 freeze + Kie Runway 마스코트 +
                          ffmpeg colorkey 합성. ~₩84/clip.
      - "yangbong":       ★ 양봉 reference 직접 모방 (Kontext bg + Kontext mascot +
                          tear sprites + ffmpeg compose). 사람 위에 흰 패딩 + 마스코트 swing.
                          ~₩70/clip, ~50s/clip. v9 검증된 prototype port.
      - "wan":            DeepInfra Wan 2.1 I2V (현재 미지원).

    각 clip은 /remixes/{remix_id}/clip_{i}.mp4 로 영구 저장 → 다운로드 가능.
    spec.clips[i] 에 output_url, image_url, duration_sec 자동 추가.

    Returns {clips: [...], combined_url, cost_usd}.
    """
    async def emit(pct: int, msg: str):
        if progress_cb:
            await progress_cb(pct, msg)

    clips_in = spec.get("clips") or []
    if not clips_in:
        raise ValueError("spec has no clips")
    clips_in = sorted(clips_in, key=lambda c: float(c.get("start", 0)))

    # 렌더 직전 clip 시간 clamp — 오래된 spec의 영상 길이 초과 clip 방어
    # (분석 때 video_duration clamp 놓친 경우 — no packets 에러 원천 차단)
    try:
        _orig_for_dur = await get_or_download_original(candidate_url)
        _vdur = await _probe_video_duration(_orig_for_dur)
        if _vdur and _vdur > 0:
            _fixed = []
            for _c in clips_in:
                _s = float(_c.get("start", 0)); _e = float(_c.get("end", 0))
                if _s >= _vdur - 0.5:
                    print(f"  [clamp] clip start={_s} >= dur={_vdur:.1f} → drop", flush=True)
                    continue
                _c["start"] = max(0.0, min(_s, _vdur - 0.5))
                _c["end"] = max(_c["start"] + 0.5, min(_e, _vdur))
                _fixed.append(_c)
            if _fixed:
                clips_in = _fixed
            print(f"  [clamp] 영상 {_vdur:.1f}s 기준 clip {len(clips_in)}개 유효", flush=True)
    except Exception as _e:
        print(f"  [clamp] skip ({str(_e)[:80]})", flush=True)

    # Permanent per-remix dir (clips kept for download)
    final_dir = REMIX_OUT_DIR / f"remix_{remix_id}"
    final_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0
    enriched_clips: list[dict] = []

    # 1. Mascot image(s) per clip
    #    - sprite_multi: 5 pose variants per clip (parallel) → 가짜 프레임 애니
    #    - yangbong: Kontext does style+expression on the base reference; pose-gen skipped
    #    - others: 1 pose per clip
    poses_per_clip = (
        5 if motion_mode == "sprite_multi"
        else 0 if motion_mode in ("yangbong", "yangbong_v14", "webtoon_static")
        else 1
    )
    clip_pose_paths: list[list[Path]] = []   # per-clip list of pose paths
    clip_pose_urls: list[list[str]] = []

    if motion_mode == "yangbong":
        # No Flux pose gen — Kontext goes directly from base mascot reference.
        ref = mascot_worker.get_reference(dissection_id)
        for _ in clips_in:
            clip_pose_paths.append([ref["path"]])
            clip_pose_urls.append([ref["url"]])
        await emit(20, f"양봉 스타일: 베이스 마스코트 사용 (포즈 생성 스킵)")
    elif motion_mode == "yangbong_v14":
        # v14: yangbong_v14 모듈이 자체적으로 LoRA + Kontext sprite 생성. pose-gen skipped.
        for _ in clips_in:
            clip_pose_paths.append([])
            clip_pose_urls.append([])
        await emit(20, f"양봉 v14 스타일: 자체 sprite 생성 (포즈 스킵)")
    elif motion_mode == "webtoon_static":
        # webtoon_static: per-mascot baseline pair (savior/victim) cached in DB,
        # 표정만 GPT-image-2 i2i로 변경. pose-gen skipped.
        for _ in clips_in:
            clip_pose_paths.append([])
            clip_pose_urls.append([])
        await emit(20, f"웹툰 정적컷: 마스코트 baseline → 표정 변경 (포즈 스킵)")
    else:
        await emit(5, f"마스코트 이미지 생성 중 (clip {len(clips_in)}개 × pose {poses_per_clip})…")
        for i, c in enumerate(clips_in):
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "standing"
            try:
                if motion_mode == "sprite_multi":
                    results = await generate_pose_variants(
                        dissection_id, action, count=poses_per_clip)
                    if not results:
                        raise RuntimeError("0 pose variants generated")
                else:
                    results = [await mascot_worker.generate_pose_image(
                        dissection_id, action)]
            except Exception as e:
                raise RuntimeError(f"마스코트 이미지 (clip {i+1}) 실패: {e}")
            clip_pose_paths.append([Path(r["image_path"]) for r in results])
            clip_pose_urls.append([r["image_url"] for r in results])
            cost += sum(float(r.get("cost") or 0) for r in results)
            await emit(5 + int(20 * (i + 1) / len(clips_in)),
                       f"이미지 {i+1}/{len(clips_in)} ({len(results)}장)")

    # 2. Per-clip mp4 — parallel for cloud modes (fal/wan); serial for local
    #    (comfy/sprite/sprite_multi) since they fight for the same MPS GPU.
    await emit(25, f"클립 영상 생성 중 ({motion_mode}, "
                   f"{'병렬' if motion_mode in {'fal', 'wan', 'runway', 'freeze_runway', 'freeze_overlay', 'yangbong_v14'} else '직렬'})…")
    completed = {"n": 0}

    async def _gen_one(idx: int, c: dict, poses: list[Path], urls: list[str]):
        duration = max(float(c["end"]) - float(c["start"]), 1.0)
        clip_path = final_dir / f"clip_{idx}.mp4"
        local_cost = 0.0
        if motion_mode == "comfy":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "idle pose"
            try:
                await comfy_client.i2v(poses[0], duration, clip_path,
                                       action_prompt=action)
            except Exception as e:
                await emit(25, f"⚠️ Comfy 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "fal":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "idle pose"
            try:
                await fal_client.i2v(poses[0], duration, clip_path,
                                     action_prompt=action)
                local_cost = 0.05 * max(duration, 1.0) / 2.0
            except Exception as e:
                await emit(25, f"⚠️ Fal 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "runway":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "idle pose"
            try:
                # Runway Gen-4 Turbo only supports 5s or 10s. Pick 5s for shorts.
                await kie_client.runway_i2v(poses[0], duration, clip_path,
                                            action_prompt=action, duration=5,
                                            quality="720p", aspect_ratio="9:16")
                local_cost = 0.06
            except Exception as e:
                await emit(25, f"⚠️ Runway 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "freeze_runway":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "idle pose"
            try:
                orig = await get_or_download_original(candidate_url)
                clip_work = final_dir / f"_work_freezerw_{idx}"
                start_sec = float(c["start"])
                fr = await make_freeze_runway_clip(
                    orig, start_sec, duration, poses[0], action,
                    clip_path, clip_work, use_kontext=False,
                )
                local_cost = float(fr.get("cost_usd") or 0)
                shutil.rmtree(clip_work, ignore_errors=True)
            except Exception as e:
                await emit(25, f"⚠️ freeze_runway 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "yangbong":
            try:
                orig = await get_or_download_original(candidate_url)
                clip_work = final_dir / f"_work_yangbong_{idx}"
                start_sec = float(c["start"])
                # spec.character: "angel" / "reaper" — Gemini 결정. fallback "reaper"
                ch = (c.get("character") or "reaper").lower()
                if ch not in ("angel", "reaper"):
                    ch = "reaper"
                # action prompt — 영어 우선 (LoRA 학습이 영어 caption)
                action_kr = c.get("mascot_action_kr") or c.get("actor_action") or ""
                action_en = c.get("mascot_action_en") or ""
                action = (action_en or action_kr or "reacting to the situation").strip()
                # boiling: spec.boiling_frames (default 1 = 정적, 4~6 = 떨림 cycle)
                boiling = int(c.get("boiling_frames") or 1)
                # spec-driven: caption_box, mascot_position, mascot_size_ratio, mascot_pixel_position
                fr = await make_yangbong_clip(
                    orig, start_sec, duration,
                    out_clip=clip_path,
                    work_dir=clip_work,
                    character=ch,
                    action_prompt=action,
                    bg_style_prompt=c.get("bg_style_prompt") or "",
                    caption_box=c.get("caption_box"),
                    mascot_position=c.get("mascot_position") or "bottom_right",
                    mascot_size_ratio=float(c.get("mascot_size_ratio") or 0.42),
                    mascot_pixel_position=c.get("mascot_pixel_position"),
                    boiling_frames=boiling,
                    zoom=True,
                    style_seed=remix_id * 1000,  # consistent style across clips
                )
                local_cost = float(fr.get("cost_usd") or 0)
                shutil.rmtree(clip_work, ignore_errors=True)
            except Exception as e:
                logger.exception("yangbong clip %s failed", idx + 1)
                await emit(25, f"⚠️ yangbong 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "yangbong_v14":
            from . import yangbong_v14 as ybk14
            try:
                orig = await get_or_download_original(candidate_url)
                clip_work = final_dir / f"_work_v14_{idx}"
                start_sec = float(c["start"])
                ch = (c.get("character") or "angel").lower()
                if ch not in ("angel", "reaper"):
                    ch = "angel"
                # Phase 12: spec 필드 매핑 (frontend 새 옵션 + 기존 호환)
                pose_prompts = (
                    c.get("pose_prompts")
                    or c.get("yangbong_v14_poses")
                    or None
                )
                if isinstance(pose_prompts, list):
                    pose_prompts = [str(p).strip() for p in pose_prompts if str(p).strip()]
                    if not pose_prompts:
                        pose_prompts = None
                bg_prompt = (
                    c.get("scene_full_prompt")
                    or c.get("yangbong_v14_bg_prompt")
                    or c.get("bg_style_prompt")
                    or None
                )
                # mascot_size_ratio → mascot_w (1080 canvas 기준)
                try:
                    sr = float(c.get("mascot_size_ratio") or 0.42)
                except (TypeError, ValueError):
                    sr = 0.42
                sr = max(0.15, min(sr, 0.70))
                mascot_w = max(180, min(int(1080 * sr), 760))
                try:
                    pis = float(c.get("mascot_pop_in_sec") or 0.3)
                except (TypeError, ValueError):
                    pis = 0.3
                pis = max(0.0, min(pis, 5.0))
                try:
                    cfps = int(c.get("pose_cycle_fps") or 8)
                except (TypeError, ValueError):
                    cfps = 8
                cfps = max(2, min(cfps, 24))
                fr = await ybk14.make_yangbong_v14_clip(
                    orig, start_sec, duration,
                    out_clip=clip_path,
                    work_dir=clip_work,
                    character=ch,
                    bg_prompt=bg_prompt,
                    pose_prompts=pose_prompts,
                    mascot_w=mascot_w,
                    pop_in_sec=pis,
                    sprite_fps=cfps,
                    seed=remix_id * 1000 + idx,
                )
                local_cost = float(fr.get("cost_usd") or 0)
                shutil.rmtree(clip_work, ignore_errors=True)
            except Exception as e:
                logger.exception("yangbong_v14 clip %s failed", idx + 1)
                await emit(25, f"⚠️ yangbong_v14 실패 → 원본 컷 사용 (clip {idx+1}): {e}")
                try:
                    orig_fb = await get_or_download_original(candidate_url)
                    await cut_segment(orig_fb, float(c["start"]),
                                      float(c["start"]) + duration, clip_path)
                except Exception:
                    raise e
        elif motion_mode == "webtoon_static":
            # webtoon_static: per-mascot N roles (자유 archetype) baseline + 표정만 변경.
            # spec.clips[i].character = role_id → DB mascot에서 해당 role의 baseline 매핑.
            from . import webtoon_static as wts
            from api import database as _db
            try:
                # 동적 roles 우선 (mascot.roles_json), 없으면 legacy savior/victim pair fallback
                ch_raw = (c.get("character") or "").strip()
                # legacy 호환 매핑
                ch_legacy_map = {"angel": "savior", "reaper": "victim"}
                ch = ch_legacy_map.get(ch_raw.lower(), ch_raw or "savior")
                role_data = None
                # 1. dynamic roles
                if hasattr(_db, "get_mascot_roles"):
                    roles = _db.get_mascot_roles(dissection_id) or []
                    for r in roles:
                        if (r or {}).get("role_id") == ch:
                            role_data = r
                            break
                    # role_id 매칭 안 되면 1번째 role fallback
                    if not role_data and roles:
                        role_data = roles[0]
                # 2. legacy pair fallback
                if not role_data:
                    pair = _db.get_mascot_pair(dissection_id) if hasattr(_db, "get_mascot_pair") else None
                    if pair:
                        if ch.lower() in ("savior", "victim"):
                            cand_role = pair.get(ch.lower()) or {}
                            if cand_role.get("baseline_path"):
                                role_data = cand_role
                        # 그래도 없으면 baseline 있는 role 우선 사용
                        if not role_data:
                            for k in ("savior", "victim"):
                                if (pair.get(k) or {}).get("baseline_path"):
                                    role_data = pair.get(k); break
                if not role_data:
                    raise RuntimeError(
                        "이 카테고리에 마스코트가 아직 확정되지 않았습니다. "
                        "🎭 모달에서 추천 → 시안 generate → 확정 먼저."
                    )
                baseline = role_data.get("baseline_path")
                spec = role_data.get("character_spec") or ""
                # baseline 없으면 다른 role의 baseline 차용 (graceful)
                if not baseline:
                    for fallback in (roles or []) + [
                        (pair or {}).get("savior") if 'pair' in dir() else {},
                        (pair or {}).get("victim") if 'pair' in dir() else {},
                    ]:
                        if fallback and fallback.get("baseline_path"):
                            baseline = fallback["baseline_path"]
                            spec = fallback.get("character_spec") or spec
                            break
                # spec 비어있어도 baseline 있으면 진행 — minimal default spec
                if not spec.strip():
                    # 다른 role의 spec 차용 시도
                    for fb in (roles or []):
                        if fb.get("character_spec") and fb.get("character_spec").strip():
                            spec = fb["character_spec"]
                            break
                    if not spec.strip() and pair:
                        for k in ("savior", "victim"):
                            sp = (pair.get(k) or {}).get("character_spec") or ""
                            if sp.strip():
                                spec = sp
                                break
                    if not spec.strip():
                        # 진짜 마지막 fallback: minimal generic spec
                        spec = (
                            "- Body: chibi proportions, head-to-body 1:1.5\n"
                            "- 2D Korean webtoon style, flat clean colors, clear black ink line outlines\n"
                            "- White background, isolated character"
                        )
                if not baseline:
                    raise RuntimeError(f"마스코트 baseline이 어디에도 없음 (role_id={ch})")
                expression = (
                    c.get("expression")  # Gemini 추천 또는 형님 직접 입력
                    or c.get("actor_emotion")
                    or c.get("mascot_action_kr")
                    or "calm neutral expression"
                )
                hand_action = c.get("hand_action") or None
                expr_prompt = mascot_worker.expression_prompt_for(
                    character_spec=spec, expression=expression, hand_action=hand_action,
                )
                # 마스코트 위치/크기 — spec.clip.mascot_placement (가운데 좌표)
                # webtoon_static.py 안에서 sprite 실제 크기로 ffmpeg overlay 좌표 변환
                xc = None
                yc = None
                mirror = False
                mascot_w = 480
                mp = c.get("mascot_placement")
                if isinstance(mp, dict):
                    sz = mp.get("size")
                    if isinstance(sz, (int, float)) and 100 <= int(sz) <= 1080:
                        mascot_w = int(sz)
                    _xc = mp.get("x_center")
                    if isinstance(_xc, (int, float)):
                        xc = int(_xc)
                    _yc = mp.get("y_center")
                    if isinstance(_yc, (int, float)):
                        yc = int(_yc)
                    if mp.get("mirror"):
                        mirror = True
                clip_work = final_dir / f"_work_webtoon_{idx}"
                start_sec = float(c["start"])
                # 정적 이미지 컷은 최소 2.5초 유지 — 너무 짧으면 시청자가 못 봄.
                effective_duration = max(float(duration), 2.5)
                if effective_duration > duration + 0.1:
                    logger.info("clip %s duration %.2fs → forced to %.2fs (min 2.5)",
                                idx + 1, duration, effective_duration)
                orig_path = await get_or_download_original(candidate_url)
                fr = await wts.make_webtoon_static_clip(
                    orig_path,
                    start_sec, effective_duration,
                    out_clip=clip_path,
                    work_dir=clip_work,
                    baseline_path=Path(baseline),
                    expression_prompt=expr_prompt,
                    x_center=xc,
                    y_center=yc,
                    mirror=mirror,
                    mascot_w=mascot_w,
                )
                # enriched에 실제 사용 duration 반영 (concat 흐름이 일관)
                duration = effective_duration
                local_cost = float(fr.get("cost_usd") or 0)
                shutil.rmtree(clip_work, ignore_errors=True)
            except Exception as e:
                logger.exception("webtoon_static clip %s failed", idx + 1)
                await emit(25, f"⚠️ webtoon_static 실패 → 원본 컷 사용 (clip {idx+1}): {e}")
                try:
                    orig_fb = await get_or_download_original(candidate_url)
                    await cut_segment(orig_fb, float(c["start"]),
                                      float(c["start"]) + duration, clip_path)
                except Exception:
                    raise e
        elif motion_mode == "freeze_overlay":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "idle pose"
            try:
                # Need the original video on disk to extract a freeze frame.
                # Fetched lazily so we only pay yt-dlp once per render.
                if "_orig_for_overlay" not in c:
                    orig = await get_or_download_original(candidate_url)
                else:
                    orig = Path(c["_orig_for_overlay"])
                clip_work = final_dir / f"_work_overlay_{idx}"
                start_sec = float(c["start"])
                fr = await make_freeze_overlay_clip(
                    orig, start_sec, duration, poses[0], action,
                    clip_path, clip_work,
                )
                local_cost = float(fr.get("cost_usd") or 0)
                shutil.rmtree(clip_work, ignore_errors=True)
            except Exception as e:
                await emit(25, f"⚠️ freeze_overlay 실패 → sprite fallback (clip {idx+1}): {e}")
                await make_sprite_clip(poses[0], duration, clip_path)
        elif motion_mode == "wan":
            action = c.get("mascot_action_en") or c.get("mascot_action_kr") or "standing"
            local_cost = await make_i2v_clip_wan(poses[0], action, duration, clip_path)
        elif motion_mode == "sprite_multi" and len(poses) > 1:
            await make_multipose_clip(poses, duration, clip_path)
        else:
            await make_sprite_clip(poses[0], duration, clip_path)
        enriched = {
            **c,
            "output_url": f"/remixes/remix_{remix_id}/clip_{idx}.mp4?v={int(time.time())}",
            "output_path": str(clip_path),
            "image_url": urls[0] if urls else None,  # yangbong_v14는 pose pre-gen skip → urls 빈 list
            "image_urls": urls,
            "duration_sec": round(duration, 2),
            "motion_mode": motion_mode,
        }
        completed["n"] += 1
        await emit(25 + int(45 * completed["n"] / len(clips_in)),
                   f"클립 {completed['n']}/{len(clips_in)} ({motion_mode})")
        return idx, enriched, local_cost

    tasks = [_gen_one(i, c, p, u) for i, (c, p, u)
             in enumerate(zip(clips_in, clip_pose_paths, clip_pose_urls))]
    # yangbong: 무조건 sequential. Replicate "1 burst" 제한이 크레딧과 무관하게 적용됨.
    if motion_mode in {"fal", "wan", "freeze_overlay", "runway", "freeze_runway",
                        "webtoon_static"}:
        gen_results = await asyncio.gather(*tasks)
    else:
        gen_results = []
        for t in tasks:
            gen_results.append(await t)
    gen_results.sort(key=lambda r: r[0])
    enriched_clips = [r[1] for r in gen_results]
    cost += sum(r[2] for r in gen_results)

    # Update spec on disk so frontend re-load shows clip URLs even if combined fails
    spec["clips"] = enriched_clips

    combined_url = None
    if make_combined:
        # 3. Download original + cut + concat (best-effort — failure doesn't kill clips)
        try:
            await emit(75, "원본 영상 준비 + 합본 시도…")
            work_dir = final_dir / "_work"
            work_dir.mkdir(exist_ok=True)
            orig_path = await get_or_download_original(candidate_url)

            # Insert mode (yangbong v10.2+): mascot clip is INSERTED at start_sec
            # without cutting the original. Total length = orig + sum(clip durations).
            # cursor advances to s (not e) so original 4~6s portion stays intact.
            sequence: list[Path] = []
            cursor = 0.0
            for i, c in enumerate(clips_in):
                s = float(c["start"])
                if s > cursor + 0.05:
                    seg = work_dir / f"orig_{i}.mp4"
                    await cut_segment(orig_path, cursor, s, seg)
                    sequence.append(seg)
                sequence.append(final_dir / f"clip_{i}.mp4")
                cursor = s  # ← v10.2: insert (don't replace original window)
            tail = work_dir / "orig_tail.mp4"
            try:
                await cut_segment(orig_path, cursor, cursor + 600, tail)
                sequence.append(tail)
            except Exception:
                pass

            combined_path = final_dir / "combined.mp4"
            logger.info("concat sequence (%d): %s", len(sequence),
                        [p.name for p in sequence])
            await concat_segments(sequence, combined_path)
            combined_url = f"/remixes/remix_{remix_id}/combined.mp4?v={int(time.time())}"
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.exception("combined.mp4 build failed for remix %s", remix_id)
            await emit(95, f"⚠️ 합본 실패 (클립은 정상): {e}")

    await emit(100, "✅ 완료")
    return {
        "clips": enriched_clips,
        "combined_url": combined_url,
        "cost_usd": round(cost, 4),
        "clip_count": len(enriched_clips),
    }
