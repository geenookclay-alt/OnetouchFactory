"""Mascot generation — per-dissection 2D 캐릭터 일관 관리.

DeepInfra Flux Schnell로 시안 N장 생성 → 형님이 1장 선택 → seed 저장.
이후 모든 동작 image는 같은 seed + 같은 base prompt → 일관 캐릭터 유지.

Flux Schnell 비용: ~$0.003/image. 시안 5장 ≈ ₩20.
"""
from __future__ import annotations

import asyncio
import base64
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from api import database as db


DEEPINFRA_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1"

# Where mascot reference images live on disk. Mounted at /mascots/...
MASCOT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "mascots"
MASCOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Korean concept → English ImageGen prompt (Gemini Flash)
# Flux/SDXL은 영어로 학습돼서 한국어 프롬프트 거의 안 먹음. 자동 번역.
# ============================================================

async def translate_concept_for_imagegen(concept_kr: str) -> str:
    """Convert a Korean character concept into a detailed English Flux prompt."""
    text = (concept_kr or "").strip()
    if not text:
        return ""
    # Already mostly English? skip translation
    if all(ord(c) < 128 for c in text):
        return text

    from . import llm  # local import to avoid cycle
    system = (
        "You convert Korean character concepts into detailed English prompts "
        "for Flux image generation. Output ONE line, English only, no markdown, "
        "no quotes, no explanation.\n\n"
        "RULES:\n"
        "1. Identify EVERY visual element in the Korean concept and write each "
        "explicitly in English (wings, halo, hat, scythe, crown, etc).\n"
        "2. Use the formula: <character species/identity>, <clothing>, "
        "<accessories — list each explicitly>, <facial expression>, <pose>.\n"
        "3. For small or subtle elements (small wings, tiny crown, single feather), "
        "add 'clearly visible' so the model doesn't drop them.\n"
        "4. DO NOT repeat the style (line art / black ink / white background) — "
        "that's added separately."
    )
    prompt = (
        f"Korean concept: {text}\n\n"
        "English Flux prompt (one short line, every visual element explicit):"
    )
    try:
        # Gemini 2.5 Flash uses internal thinking tokens — give plenty of room
        resp = await llm.gemini_chat(prompt, system=system,
                                     model="gemini-3.5-flash",
                                     max_tokens=2048)
        out = (resp.text or "").strip().strip('"').strip("'")
        for prefix in ("English prompt:", "Prompt:", "Output:"):
            if out.lower().startswith(prefix.lower()):
                out = out[len(prefix):].strip()
        return out or text
    except Exception as e:
        print(f"translate_concept_for_imagegen failed: {e}")
        return text  # fallback raw


# ============================================================
# DeepInfra image generation
# ============================================================

async def _flux_schnell(prompt: str, seed: int | None = None,
                        width: int = 1024, height: int = 1024) -> dict:
    """Single Flux Schnell call. Returns response dict (raises on error)."""
    if not DEEPINFRA_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY not set")
    body = {
        "prompt": prompt,
        "num_inference_steps": 4,
        "width": width,
        "height": height,
    }
    if seed is not None:
        body["seed"] = int(seed)
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(
            f"{DEEPINFRA_BASE}/inference/black-forest-labs/FLUX-1-schnell",
            headers={"Authorization": f"Bearer {DEEPINFRA_KEY}",
                     "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        return r.json()


def _decode_image(image_field: str) -> bytes:
    """Flux returns 'data:image/png;base64,...' or raw base64."""
    if image_field.startswith("data:"):
        image_field = image_field.split(",", 1)[1]
    return base64.b64decode(image_field)


# ============================================================
# Public API: 시안 5~10장 생성, 마스코트 저장, 동작 이미지 생성
# ============================================================

DEFAULT_STYLE_BASE = (
    "simple hand-drawn line art, black ink on white background, "
    "minimalist 2D character mascot, expressive face, full body with all "
    "accessories clearly visible, clean composition, no shading, no color"
)


async def generate_mascot_options(dissection_id: str, concept: str,
                                  count: int = 5) -> list[dict]:
    """Generate N candidate mascot images for a dissection.

    Concept can be Korean — gets auto-translated to English first because
    Flux is trained on English. Returns list with both the original concept
    and the English prompt actually used.
    """
    concept_en = await translate_concept_for_imagegen(concept)
    base_prompt = f"{DEFAULT_STYLE_BASE}, {concept_en}"
    seeds = random.sample(range(1, 10**8), count)

    out_dir = MASCOT_DIR / dissection_id / "options"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _one(seed: int) -> dict | None:
        try:
            resp = await _flux_schnell(base_prompt, seed=seed)
            imgs = resp.get("images") or []
            if not imgs:
                return None
            img_bytes = _decode_image(imgs[0])
            fname = f"{seed}.png"
            fpath = out_dir / fname
            fpath.write_bytes(img_bytes)
            return {
                "seed": seed,
                "image_path": str(fpath),
                "image_url": f"/mascots/{dissection_id}/options/{fname}",
                "prompt": base_prompt,
                "cost": resp.get("inference_status", {}).get("cost", 0),
            }
        except Exception as e:
            print(f"flux call failed seed={seed}: {e}")
            return None

    results = await asyncio.gather(*[_one(s) for s in seeds])
    return [r for r in results if r]


def select_mascot(dissection_id: str, name: str, concept: str,
                  style_prompt: str, seed: int,
                  reference_image_path: str) -> dict:
    """Persist the user's chosen mascot. Moves the chosen option image to a
    permanent location and updates the DB.
    """
    chosen_src = Path(reference_image_path)
    if not chosen_src.exists():
        raise FileNotFoundError(reference_image_path)

    final_dir = MASCOT_DIR / dissection_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "reference.png"
    final_path.write_bytes(chosen_src.read_bytes())

    final_url = f"/mascots/{dissection_id}/reference.png"
    return db.upsert_mascot(
        dissection_id=dissection_id,
        name=name,
        concept=concept,
        style_prompt=style_prompt,
        reference_image_path=str(final_path),
        reference_image_url=final_url,
        seed=int(seed),
    )


def get_reference(dissection_id: str) -> dict:
    """Return mascot reference info: {path, url, name, concept, style_prompt}.

    yangbong motion_mode skips Flux pose generation and feeds the reference
    image directly into Kontext for style/expression transformation. This
    matches v9 prototype behavior — the base mascot is enough.
    """
    m = db.get_mascot(dissection_id)
    if not m:
        raise ValueError(f"No mascot set for dissection {dissection_id}")
    return {
        "path": Path(m["reference_image_path"]),
        "url": m["reference_image_url"],
        "name": m.get("name"),
        "concept": m.get("concept"),
        "style_prompt": m.get("style_prompt"),
        "seed": m.get("seed"),
    }


async def generate_pose_image(dissection_id: str, action_prompt: str) -> dict:
    """Generate one mascot image in a specific pose. Uses the saved seed +
    style_prompt so the character stays visually consistent across poses.
    """
    mascot = db.get_mascot(dissection_id)
    if not mascot:
        raise ValueError(f"No mascot set for dissection {dissection_id}")

    full_prompt = f"{mascot['style_prompt']}, {action_prompt}"
    resp = await _flux_schnell(full_prompt, seed=int(mascot["seed"]))
    imgs = resp.get("images") or []
    if not imgs:
        raise RuntimeError("Flux returned no image")
    img_bytes = _decode_image(imgs[0])

    poses_dir = MASCOT_DIR / dissection_id / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    fpath = poses_dir / f"{ts}.png"
    fpath.write_bytes(img_bytes)

    return {
        "image_path": str(fpath),
        "image_url": f"/mascots/{dissection_id}/poses/{ts}.png",
        "action_prompt": action_prompt,
        "full_prompt": full_prompt,
        "cost": resp.get("inference_status", {}).get("cost", 0),
    }


# ============================================================
# Webtoon mascot pair system (savior + victim) — webtoon_static motion_mode
# Gemini가 카테고리 dna 분석 → 1쌍 컨셉 추천 → GPT-image-2 t2i baseline.
# ============================================================

import json as _json
from . import kie_client
from . import cost_tracker


WEBTOON_BASELINE_TEMPLATE = (
    "Generate a 2D Korean webtoon-style chibi character (네이버웹툰 style). "
    "Render with: flat clean colors, clear black ink line outlines, simple cell shading. "
    "NO watercolor, NO photo, NO 3D, NO sketch.\n"
    "\n"
    "EXACT character spec:\n"
    "{character_spec}\n"
    "\n"
    "Pose: facing forward, calm neutral pose, full body visible from head to feet, "
    "centered framing, hands visible.\n"
    "Background: pure pristine WHITE, isolated character, no scenery.\n"
    "\n"
    "Output: 1024x1024 webtoon illustration, single character on white background."
)


WEBTOON_EXPRESSION_TEMPLATE = (
    "TASK: Make a minimal expression and pose edit to the character. "
    "This is an EDIT task, NOT a redraw task.\n"
    "\n"
    "🚨 CRITICAL — IDENTITY LOCKDOWN (HIGHEST PRIORITY):\n"
    "The character in the reference images is the ABSOLUTE ground truth. The output MUST be the SAME character.\n"
    "- SPECIES: If references show an ANIMAL → output the same animal. If a HUMAN → output a human. NEVER swap species.\n"
    "- POSTURE: If references show a QUADRUPED (4-legged animal, e.g., dog, cat) → output stays 4-legged. "
    "NEVER convert a quadruped to a 2-legged anthropomorphic / humanoid / furry character.\n"
    "- NO ANTHROPOMORPHIC TRANSFORMATION: If references show a normal animal (dog standing on 4 paws, no clothes), "
    "the output is a NORMAL animal — NEVER add human clothes (coat, shirt, pants, turtleneck), NEVER make it stand upright on hind legs, "
    "NEVER give it human posture (hands clasped, arms folded, standing tall like a person).\n"
    "- CLOTHING: NEVER add clothes/accessories not in references. If references show a dog with only a bandana, "
    "output stays a dog with only a bandana — no coat, no shirt, no pants.\n"
    "- GENDER: NEVER change male ↔ female. Output the same gender as references.\n"
    "- AGE: NEVER change age range. Child stays child, adult stays adult, elderly stays elderly.\n"
    "- ETHNICITY: NEVER change race or ethnic features. Asian stays Asian, etc.\n"
    "- BODY TYPE: NEVER change body proportions (slim/heavy, tall/short, muscular/lean)\n"
    "- HAIR/FUR: NEVER change hair or fur color or length. White stays white, grey stays grey.\n"
    "- DO NOT add a second character (human, animal, plushie, doll, Kuromi, Hello Kitty, Sanrio, etc.) — single character output only\n"
    "- DO NOT replace the character with a different character even if the expression text hints otherwise\n"
    "\n"
    "Multiple reference images may be provided showing this character from various angles "
    "(front, 3/4, profile, back, face close-up). Use ALL of them as the canonical character anchor — "
    "every visible detail appears in these references and must stay IDENTICAL in the output.\n"
    "\n"
    "The output character must be INSTANTLY RECOGNIZABLE as the SAME character (same species, same identity) from the references:\n"
    "- For ANIMAL characters: same fur/feather color and texture, ear shape, tail shape, breed traits, body proportions\n"
    "- For HUMAN characters: same face shape, skin tone, age, hair color/length/style, facial hair (mustache/beard), eyewear\n"
    "- Same clothing or accessories if any — every layer's color, length, sleeves, collar, pattern\n"
    "- Same props (bandana, collar, tag, origami crane, paper stack, etc.) — size + color + position\n"
    "- Same body proportions (chibi ratio, head size)\n"
    "- Same line thickness and color palette\n"
    "\n"
    "Change the facial expression to convey: {expression}.\n"
    "{hand_clause}"
    "IMPORTANT — EXPRESSION DESCRIBES EMOTION ONLY, NOT IDENTITY OR POSTURE:\n"
    "The expression text describes the EMOTION to convey. It does NOT describe a different character or posture. "
    "Apply the emotion to the EXACT character from references — never use it as an excuse to change identity, species, or posture:\n"
    "- If expression says 'youthful' / 'elderly' / 'masculine' / 'feminine' — IGNORE as identity hints. "
    "The character's age/gender stays as in references.\n"
    "- If expression uses human-only words ('blush on cheeks', 'looking up at the person', 'blissful eyes', "
    "'hands clasped', 'arms folded', 'standing tall') and the character is a 4-LEGGED ANIMAL: "
    "TRANSLATE to natural animal body language — NEVER make the animal stand on 2 legs like a human.\n"
    "  • 'blush on cheeks' → soft pink tint on cheeks/muzzle\n"
    "  • 'hands clasped' / 'BOTH hands' / 'hand action' → IGNORE for 4-legged animals — keep all 4 paws on ground naturally\n"
    "  • 'looking up at the person' → head tilted up, eyes looking up (4 paws still on ground)\n"
    "  • 'smile' → open mouth + tongue out + relaxed jaw\n"
    "  • 'standing' → standing on 4 paws (NOT 2 legs)\n"
    "- If expression mentions a person/people/another character ('looking at the man', 'the woman beside'), "
    "treat as background context — do NOT add that person to the output. Single character only.\n"
    "- DO NOT change the character or its posture because the expression hints at human-like body language.\n"
    "- The character's posture (4-legged vs 2-legged) is FIXED by the references — expression CANNOT override this.\n"
    "Body pose may change minimally to support the expression, but species, posture, clothing, and identity stay identical.\n"
    "\n"
    "STRICT — DO NOT ADD or CHANGE:\n"
    "- DO NOT change the species (animal ↔ human swap is FORBIDDEN)\n"
    "- DO NOT make a 4-legged animal stand on 2 legs (NO anthropomorphic / furry / humanoid animal transformation)\n"
    "- DO NOT add clothes to an animal that has no clothes in references (no coat, shirt, pants, turtleneck, suit, dress)\n"
    "- DO NOT add a hat / 갓 / hood / crown / headwear if none in references\n"
    "- DO NOT add halo / wings / scythe / sword / weapon if none in references\n"
    "- DO NOT add a second character (human, mascot, plushie, doll) beside the main character\n"
    "- DO NOT add toys/plushies/dolls (Kuromi, Hello Kitty, Sanrio characters, etc.) if none in references\n"
    "- DO NOT add traditional Korean clothing (한복 / 도포) if none in references\n"
    "- DO NOT swap the character. DO NOT redesign. DO NOT generate a generic chibi version\n"
    "The character's identity, species, posture (4-legged vs 2-legged), and clothing from references must be preserved 100%.\n"
    "\n"
    "Output: 1024x1024 PNG, single character on pure white background, 2D Korean webtoon style "
    "matching the references' line art and color palette.\n"
    "\n"
    "🚨 FINAL LOCKDOWN — BASELINE IMAGE IS THE ONLY SOURCE OF TRUTH:\n"
    "The reference images are the SINGLE source of truth for the character's appearance. "
    "DO NOT change ANY visual aspect of the character's overall look from how it appears in the references:\n"
    "- preserve fur color, fur texture, skin color EXACTLY as in references\n"
    "- preserve breed, species, body type, proportions EXACTLY\n"
    "- preserve all accessories (bandana, collar, tag, clothes if any) EXACTLY\n"
    "- preserve ear shape, tail shape, face shape EXACTLY\n"
    "The ONLY thing you may change is the FACIAL EXPRESSION to match the requested emotion. "
    "Body pose may change minimally to support the expression, but the character's overall appearance "
    "must look IDENTICAL to the references — same character, same color, same shape, same everything.\n"
    "\n"
    "(Optional context only — if it conflicts with the references in ANY way, IGNORE this and follow the references):\n"
    "{character_spec}"
)


async def recommend_mascot_pair(dissection_id: str) -> dict:
    """Gemini로 카테고리(dissection)의 narrative archetype을 분석 → savior + victim
    1쌍 컨셉 + 캐릭터 spec을 자동 추천. 결과는 형님이 확정 전에 review 가능."""
    diss = db.get_dissection(dissection_id)
    if not diss:
        raise ValueError(f"dissection {dissection_id} not found")

    name = diss.get("name") or ""
    raw = diss.get("dissection_result") or "{}"
    try:
        dna = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        dna = {}
    summary_kr = (
        dna.get("summary_kr") or
        (dna.get("common_dna") or {}).get("summary_kr") or
        ""
    )
    items = dna.get("items") or (dna.get("common_dna") or {}).get("items") or {}
    narrative_clues = []
    for key in ("emotional_arc", "narrative_pattern", "trigger_event",
                "main_subject", "humor_pattern", "viewer_reaction"):
        v = items.get(key)
        if isinstance(v, dict):
            narrative_clues.append(f"- {key}: {v.get('value', '')}")
        elif v:
            narrative_clues.append(f"- {key}: {v}")
    clues_text = "\n".join(narrative_clues) or "(no narrative items extracted)"

    from . import llm
    system = (
        "You are designing a pair of mascot characters (savior + victim) for a Korean shorts "
        "channel. The mascots will appear as static reaction cuts in every video of one category. "
        "Your job: analyze the category's narrative archetype, then propose ONE savior character "
        "(rescue/protect/empathize role) and ONE victim character (caught-in-trouble/regret/"
        "in-distress role). Both should fit the SPECIFIC theme of the category, not generic.\n\n"
        "Output strictly in JSON with this schema:\n"
        '{\n'
        '  "savior": {\n'
        '    "name_kr": "<short Korean name, ~6 chars>",\n'
        '    "concept_kr": "<one-line Korean concept including narrative role>",\n'
        '    "concept_en": "<one-line English concept used as image-gen prompt>",\n'
        '    "character_spec": "<5~8 bullet points describing visible features for lockdown '
        'prompt — body proportions, head/hair/hood, face, clothing, accessories, weapons/props, '
        'pose. Be very specific so GPT-image-2 can replicate consistently.>"\n'
        '  },\n'
        '  "victim": { same fields }\n'
        '}\n\n'
        "Rules:\n"
        "- Both characters chibi style, head-to-body 1:1.5, designed for static webtoon cuts.\n"
        "- Avoid copyrighted characters / brand mascots.\n"
        "- character_spec must be detailed enough that two separate generations yield the SAME "
        "character (e.g. specify hair color/length, clothing color/pattern, exact accessory "
        "shapes). Use bullet lines starting with '- '."
    )
    user = (
        f"Category name: {name}\n"
        f"Category summary (Korean): {summary_kr}\n"
        f"Narrative clues:\n{clues_text}\n\n"
        "Propose savior + victim mascot pair as JSON."
    )
    resp = await llm.gemini_chat(user, system=system,
                                 model="gemini-3.5-flash",
                                 max_tokens=4096)
    text = (resp.text or "").strip()
    # extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Gemini did not return JSON: {text[:300]}")
    try:
        pair = _json.loads(text[start:end + 1])
    except Exception as e:
        raise RuntimeError(f"Gemini JSON parse fail: {e} | raw: {text[:400]}")
    return pair


async def generate_baseline_for_role(
    dissection_id: str,
    role: str,
    character_spec: str,
    count: int = 3,
) -> list[dict]:
    """GPT-image-2 t2i로 baseline 시안 N장 generate (병렬, **누적 저장**).
    파일명 timestamp 사용 → 다시 generate해도 이전 시안 보존. 형님이 모달 열면
    누적된 모든 시안에서 1장 골라 쓸 수 있음.
    """
    if role not in ("savior", "victim"):
        raise ValueError(f"role must be savior or victim, got {role}")

    out_dir = MASCOT_DIR / dissection_id / f"webtoon_{role}_options"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = WEBTOON_BASELINE_TEMPLATE.format(character_spec=character_spec)

    base_ts = int(time.time() * 1000)

    async def _one(i: int):
        # 누적 timestamp + 인덱스 — 충돌 회피, 정렬 가능
        fname = f"opt_{base_ts}_{i+1}.png"
        out_png = out_dir / fname
        try:
            fr = await kie_client.gpt_image_2_t2i(
                out_png, prompt=prompt,
                quality="high", image_size="1024x1024",
            )
            return {
                "image_path": str(out_png),
                "image_url": f"/mascots/{dissection_id}/webtoon_{role}_options/{fname}",
                "filename": fname,
                "created_ts": base_ts + i,
                "task_id": fr["task_id"],
                "credits": fr["cost_credits"],
                "wall_sec": fr["wall_sec"],
            }
        except Exception as e:
            print(f"baseline {role} option {i+1} fail: {e}")
            return None

    results = await asyncio.gather(*[_one(i) for i in range(count)])
    return [r for r in results if r]


def list_baseline_options(dissection_id: str, role: str) -> list[dict]:
    """디스크에 누적된 모든 시안 list. 모달 열면 자동 표시.
    파일명 mtime 기준 최신 순. 빈 list 가능 (아직 generate 안 함).
    """
    if role not in ("savior", "victim"):
        raise ValueError(f"role must be savior or victim, got {role}")
    out_dir = MASCOT_DIR / dissection_id / f"webtoon_{role}_options"
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{
        "image_path": str(p),
        "image_url": f"/mascots/{dissection_id}/webtoon_{role}_options/{p.name}",
        "filename": p.name,
        "created_ts": int(p.stat().st_mtime * 1000),
    } for p in files]


def delete_baseline_option(dissection_id: str, role: str, filename: str) -> bool:
    """단일 시안 디스크 삭제 (형님이 마음에 안 드는 시안 정리용)."""
    if role not in ("savior", "victim"):
        raise ValueError(f"role must be savior or victim, got {role}")
    if "/" in filename or ".." in filename:
        raise ValueError("invalid filename")
    target = MASCOT_DIR / dissection_id / f"webtoon_{role}_options" / filename
    if target.exists():
        target.unlink()
        return True
    return False


def select_baseline(
    dissection_id: str,
    role: str,
    chosen_image_path: str,
    concept_kr: str,
    concept_en: str,
    character_spec: str,
) -> dict:
    """legacy savior/victim — webtoon_{role}_baseline.png로 cache + DB upsert."""
    if role not in ("savior", "victim"):
        raise ValueError(f"role must be savior or victim, got {role}")
    src = Path(chosen_image_path)
    if not src.exists():
        raise FileNotFoundError(chosen_image_path)

    final_dir = MASCOT_DIR / dissection_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"webtoon_{role}_baseline.png"
    final_path.write_bytes(src.read_bytes())
    final_url = f"/mascots/{dissection_id}/webtoon_{role}_baseline.png"

    return db.upsert_mascot_pair_role(
        dissection_id=dissection_id,
        role=role,
        concept_kr=concept_kr,
        concept_en=concept_en,
        character_spec=character_spec,
        baseline_path=str(final_path),
        baseline_url=final_url,
        chosen_filename=src.name,
    )


async def auto_describe_baseline(image_path: Path) -> str:
    """Gemini Vision으로 baseline image 분석 → character_spec 자동 생성.
    한국 전통 의상 / chibi 외형 / accessories 등 visual detail 영문 bullet."""
    import base64
    from . import llm
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
    system = (
        "You are describing a chibi character image for use as a reference description "
        "in image-edit prompts. List EVERY visible detail of the character so a generative model "
        "can preserve identity. Use bullet points starting with '- '. Be very specific:\n"
        "- Body proportions (chibi ratio, head size)\n"
        "- Hair (color, length, style, parting, braids, twin-tails, etc.)\n"
        "- Headwear: hats, hoods, crowns. If Korean traditional 갓 (gat), specify wide-brim, "
        "  black color, chin ties, etc. If hood, describe shape/depth.\n"
        "- Face: skin tone, eye shape/color, nose, mouth, blush, freckles, special markings\n"
        "- Clothing: every layer (outer robe / dopo / hanbok jeogori / pants / dress) with "
        "  color, length, sleeves, collar, sash, pattern\n"
        "- Accessories: halo, wings, scythe, brushes, scrolls (명부), props, weapons, jewelry — "
        "  size + color + position\n"
        "- Pose silhouette\n"
        "- Color palette + line thickness style (webtoon flat colors etc.)\n"
        "\n"
        "If the character is a Korean grim reaper (저승사자), explicitly mention 갓, 도포, 명부, "
        "흰 종이 두루마리, 붓 등 Korean cultural elements. Do NOT describe generic Western reaper "
        "elements (scythe, dark robe) unless they are actually visible.\n"
        "\n"
        "Output ONLY the bullet list, 8~12 lines. No headers, no commentary."
    )
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            {"text": "Describe this chibi character in 8~12 detailed bullet points."},
        ]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        "systemInstruction": {"parts": [{"text": system}]},
    }
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            headers={"x-goog-api-key": llm.GEMINI_API_KEY,
                     "Content-Type": "application/json"},
            json=body,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Gemini Vision HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        raise RuntimeError(f"Gemini Vision invalid response: {str(data)[:300]}")
    return text


def expression_prompt_for(character_spec: str, expression: str,
                           hand_action: str | None = None) -> str:
    """Return lockdown expression prompt to feed to gpt_image_2_i2i (with baseline as input)."""
    hand_clause = (
        f"Hand and arm position: {hand_action}.\n"
        if hand_action else
        "Hand and arm position: SAME as reference (no change unless the expression dictates).\n"
    )
    return WEBTOON_EXPRESSION_TEMPLATE.format(
        character_spec=character_spec,
        expression=expression,
        hand_clause=hand_clause,
    )


# ============================================================
# Dynamic N-roles system (자유 archetype, 1~3 마스코트)
# ============================================================

async def recommend_mascot_roles(dissection_id: str) -> list[dict]:
    """Gemini로 카테고리 narrative 분석 → 1~3개 마스코트 자유 archetype 추천.
    savior/victim 강제 없음 — 카테고리에 맞으면 1개, 대립 narrative면 2~3개.
    """
    diss = db.get_dissection(dissection_id)
    if not diss:
        raise ValueError(f"dissection {dissection_id} not found")

    name = diss.get("name") or ""
    raw = diss.get("dissection_result") or "{}"
    try:
        dna = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        dna = {}
    summary_kr = (
        dna.get("summary_kr") or
        (dna.get("common_dna") or {}).get("summary_kr") or ""
    )
    items = dna.get("items") or (dna.get("common_dna") or {}).get("items") or {}
    narrative_clues = []
    for key in ("emotional_arc", "narrative_pattern", "trigger_event",
                "main_subject", "humor_pattern", "viewer_reaction"):
        v = items.get(key)
        if isinstance(v, dict):
            narrative_clues.append(f"- {key}: {v.get('value', '')}")
        elif v:
            narrative_clues.append(f"- {key}: {v}")
    clues_text = "\n".join(narrative_clues) or "(narrative items 없음)"

    from . import llm
    system = (
        "You design mascot characters for a Korean shorts channel category. "
        "Each category has a different narrative archetype — sometimes 1 mascot is enough, "
        "sometimes 2 (rescue vs danger / champion vs loser / success vs failure), "
        "and sometimes 3 (act-1 / act-2 / act-3). Decide the count based on the category's "
        "narrative.\n\n"
        "Output strict JSON: {\"roles\": [array of 1~3 mascot objects]}\n\n"
        "Each mascot object schema:\n"
        '{\n'
        '  "role_id": "<short ASCII identifier, snake_case, e.g. champion / gutter_pin / chef / hero>",\n'
        '  "role_label_kr": "<short Korean label, ~6 chars, e.g. 스트라이크왕 / 거터핀 / 셰프>",\n'
        '  "narrative_role": "<one-line Korean role description>",\n'
        '  "name_kr": "<short Korean name>",\n'
        '  "concept_kr": "<one-line Korean concept>",\n'
        '  "concept_en": "<one-line English concept used as image-gen prompt>",\n'
        '  "character_spec": "<5~8 bullet lines starting with - describing visible features '
        'for lockdown prompt — chibi proportions, hair/hat, face, clothing, accessories, '
        'weapons/props, pose. Be very specific so GPT-image-2 can replicate consistently.>"\n'
        "}\n\n"
        "Rules:\n"
        "- Choose 1 mascot if category needs no narrative opposition (e.g. cooking, vlog).\n"
        "- Choose 2 if category has a clear pair archetype (e.g. winner/loser, hero/villain, "
        "savior/victim, success/failure).\n"
        "- Choose 3 only if category has a 3-act structure (rare).\n"
        "- All chibi style, head-to-body 1:1.5.\n"
        "- Avoid copyrighted characters / brand mascots.\n"
        "- character_spec detailed enough that two separate generations yield the SAME character."
    )
    user = (
        f"Category name: {name}\n"
        f"Category summary (Korean): {summary_kr}\n"
        f"Narrative clues:\n{clues_text}\n\n"
        "Decide count and propose mascot roles JSON."
    )
    resp = await llm.gemini_chat(user, system=system,
                                 model="gemini-3.5-flash",
                                 max_tokens=4096)
    text = (resp.text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Gemini did not return JSON: {text[:300]}")
    try:
        data = _json.loads(text[start:end + 1])
    except Exception as e:
        raise RuntimeError(f"Gemini JSON parse fail: {e} | raw: {text[:400]}")
    roles = data.get("roles") or []
    if not isinstance(roles, list) or not roles:
        raise RuntimeError(f"Gemini roles empty: {data}")
    # character_spec list → string join
    for r in roles:
        spec = r.get("character_spec")
        if isinstance(spec, list):
            r["character_spec"] = "\n".join(str(s) for s in spec)
    return roles


def _safe_role_id(role_id: str) -> str:
    """ASCII snake_case만 허용 (디스크 폴더명 안전)."""
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9_]+", "_", str(role_id or "")).strip("_").lower()
    return s or "role"


async def generate_baseline_for_role_id(
    dissection_id: str,
    role_id: str,
    character_spec: str,
    count: int = 3,
) -> list[dict]:
    """role_id별 GPT-image-2 t2i baseline 시안 N장 (병렬, 누적)."""
    role_id = _safe_role_id(role_id)
    out_dir = MASCOT_DIR / dissection_id / f"webtoon_{role_id}_options"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = WEBTOON_BASELINE_TEMPLATE.format(character_spec=character_spec)

    base_ts = int(time.time() * 1000)

    async def _one(i: int):
        fname = f"opt_{base_ts}_{i+1}.png"
        out_png = out_dir / fname
        try:
            fr = await kie_client.gpt_image_2_t2i(
                out_png, prompt=prompt, quality="high", image_size="1024x1024",
            )
            return {
                "image_path": str(out_png),
                "image_url": f"/mascots/{dissection_id}/webtoon_{role_id}_options/{fname}",
                "filename": fname,
                "created_ts": base_ts + i,
                "task_id": fr["task_id"],
                "credits": fr["cost_credits"],
                "wall_sec": fr["wall_sec"],
            }
        except Exception as e:
            print(f"baseline {role_id} option {i+1} fail: {e}")
            return None

    results = await asyncio.gather(*[_one(i) for i in range(count)])
    success_n = len([r for r in results if r])
    try:
        cost_tracker.add_cost_by_label(dissection_id, "gpt_image_2_t2i_high_1024", count=success_n)
    except Exception:
        pass
    return [r for r in results if r]


def list_baseline_options_for_role_id(dissection_id: str, role_id: str) -> list[dict]:
    role_id = _safe_role_id(role_id)
    out_dir = MASCOT_DIR / dissection_id / f"webtoon_{role_id}_options"
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{
        "image_path": str(p),
        "image_url": f"/mascots/{dissection_id}/webtoon_{role_id}_options/{p.name}",
        "filename": p.name,
        "created_ts": int(p.stat().st_mtime * 1000),
    } for p in files]


async def select_baseline_for_role_id(
    dissection_id: str,
    role_id: str,
    chosen_image_path: str,
    name_kr: str,
    role_label_kr: str,
    narrative_role: str,
    concept_kr: str,
    concept_en: str,
    character_spec: str,
) -> list[dict]:
    """role_id별 baseline 확정 + roles JSON update + Gemini Vision 자동 spec 보강.
    character_spec이 비어있거나 너무 짧으면 자동으로 Gemini Vision이 baseline image
    분석해서 디테일 spec 생성. 형님이 갓/도포 등 디테일 안 적어도 일관성 유지.
    """
    role_id_safe = _safe_role_id(role_id)
    src = Path(chosen_image_path)
    if not src.exists():
        raise FileNotFoundError(chosen_image_path)
    final_dir = MASCOT_DIR / dissection_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"webtoon_{role_id_safe}_baseline.png"
    final_path.write_bytes(src.read_bytes())
    final_url = f"/mascots/{dissection_id}/webtoon_{role_id_safe}_baseline.png"

    # 시안 확정 시 무조건 이미지 분석 AI 호출 — 캐릭터 디테일 자세히 자동 추출.
    # 화면 입력 spec이 있어도 합쳐서 더 자세하게 (지금까지 짧은 spec으로 일관성 망함).
    # 분석 실패 시 텔레그램 알림 (침묵 실패 X — 일관성 망가지는 원인).
    spec_clean = (character_spec or "").strip()
    try:
        auto_spec = await auto_describe_baseline(final_path)
        if auto_spec and auto_spec.strip():
            if spec_clean:
                character_spec = spec_clean + "\n" + auto_spec
            else:
                character_spec = auto_spec
    except Exception as e:
        print(f"⚠️ auto_describe_baseline fail (role={role_id_safe}): {e}")
        try:
            from . import notify
            await notify.notify_error(
                "⚠️ 캐릭터 자동 분석 실패",
                f"역할: {role_id_safe} (카테 {dissection_id})\n"
                f"사유: {str(e)[:200]}\n"
                "→ 짧은 설명만 들어감. 일관성 약해질 수 있음 — 자세한 설명 직접 채워주세요"
            )
        except Exception:
            pass

    # preserve existing turnaround on re-select (디스크 8각 살아있으면 보존)
    existing_roles = db.get_mascot_roles(dissection_id) or []
    existing = next((r for r in existing_roles if r.get("role_id") == role_id_safe), None)
    extra_kwargs = {}
    if existing:
        ta_paths = existing.get("turnaround_paths") or []
        # filter to existing files only
        ta_paths = [p for p in ta_paths if Path(p).exists()]
        # also auto-discover from disk (in case DB lost track)
        disk_ta = sorted((final_dir).glob(f"webtoon_{role_id_safe}_turnaround_*.png"))
        all_ta = sorted(set([str(p) for p in (list(map(Path, ta_paths)) + disk_ta)]))
        if all_ta:
            extra_kwargs["turnaround_paths"] = all_ta
            extra_kwargs["turnaround_status"] = "ready" if len(all_ta) == 8 else "partial"
    return db.upsert_mascot_role(
        dissection_id=dissection_id,
        role_id=role_id_safe,
        role_label_kr=role_label_kr,
        narrative_role=narrative_role,
        name_kr=name_kr,
        concept_kr=concept_kr,
        concept_en=concept_en,
        character_spec=character_spec,
        baseline_path=str(final_path),
        baseline_url=final_url,
        chosen_filename=src.name,
        **extra_kwargs,
    )


def delete_baseline_option_for_role_id(
    dissection_id: str, role_id: str, filename: str,
) -> bool:
    role_id = _safe_role_id(role_id)
    if "/" in filename or ".." in filename:
        raise ValueError("invalid filename")
    target = MASCOT_DIR / dissection_id / f"webtoon_{role_id}_options" / filename
    if target.exists():
        target.unlink()
        return True
    return False


# ============================================================
# 8각 turnaround sheet — multi-angle reference for i2i anchor
# ============================================================
TURNAROUND_ANGLES = [
    ("01", "front view facing the camera directly, full body visible from head to feet"),
    ("02", "front 3/4 view, body turned 30-45 degrees, full body visible from head to feet"),
    ("03", "full left profile, body facing camera's left, side silhouette, full body from head to feet"),
    ("04", "back view, character facing directly away from camera, full body from head to feet"),
    ("05", "face close-up portrait, head and shoulders only, looking at camera, identical hairstyle and facial features as the reference, no background"),
]


async def generate_turnaround_for_role(
    dissection_id: str, role_id: str,
    progress_cb=None,
) -> dict:
    """role의 baseline → 8각 turnaround generate + DB 저장.

    비용: $0.03 × 8 = $0.24 일회성. 이후 영상 만들 때 baseline + 8각 = 9 refs anchor.
    """
    role_id = _safe_role_id(role_id)
    roles = db.get_mascot_roles(dissection_id) or []
    role = next((r for r in roles if r.get("role_id") == role_id), None)
    if not role:
        raise ValueError(f"role {role_id} not found in {dissection_id}")
    baseline_path = role.get("baseline_path")
    if not baseline_path:
        raise ValueError(f"role {role_id} baseline not confirmed")
    baseline = Path(baseline_path)
    if not baseline.exists():
        raise ValueError(f"baseline file missing: {baseline}")

    spec_str = role.get("character_spec") or "- chibi character"
    LOCKDOWN = (
        "CRITICAL TASK — Make this output 99% visually IDENTICAL to the reference image. "
        "Only the camera angle changes. Every other detail MUST stay exactly the same.\n\n"
        "STRICT RULES — DO NOT CHANGE:\n"
        "- Hairstyle: same length, same texture, same parting, same color (do NOT tie up if loose, do NOT shorten if long, do NOT make tidy if messy)\n"
        "- Facial hair: if mustache/beard visible in reference, MUST appear identical in output\n"
        "- Face age: if elderly with wrinkles, MUST stay elderly (do not make younger)\n"
        "- Clothing colors: every layer's color matches exactly (do NOT substitute white for light blue, do NOT change brown vest to black, etc.)\n"
        "- Accessory colors: origami crane / paper / props colors match exactly (do NOT change pink to blue, etc.)\n"
        "- Body proportions: same chibi ratio\n\n"
        f"Character spec — every detail below MUST be preserved:\n{spec_str}\n\n"
        "Pure white background, full body visible, no shadows, "
        "clean 2D Korean webtoon (네이버웹툰) style with thin black ink line outlines and flat color fills."
    )

    out_dir = baseline.parent
    db.upsert_mascot_role(
        dissection_id, role_id, turnaround_status="generating",
    )

    # 새로 N각 만들기 전 옛 turnaround_*.png 다 청소 (각도 갯수 줄였을 때 옛 파일 남는 거 방지).
    # 옛 파일 남으면 webtoon_static이 영상 만들 때 디스크 ls로 다 ref 사용 → 캐릭터 일관성 망가짐.
    try:
        for old_png in out_dir.glob(f"webtoon_{role_id}_turnaround_*.png"):
            old_png.unlink()
    except Exception as _e:
        print(f"⚠️ 옛 turnaround 청소 실패: {_e}", flush=True)

    paths: list[str] = []
    failed: list[str] = []
    for i, (angle_id, angle_prompt) in enumerate(TURNAROUND_ANGLES):
        if progress_cb:
            await progress_cb(
                int(i / len(TURNAROUND_ANGLES) * 100),
                f"각도 {i+1}/{len(TURNAROUND_ANGLES)} generate 중 ({angle_id})",
            )
        out_png = out_dir / f"webtoon_{role_id}_turnaround_{angle_id}.png"
        full_prompt = (
            f"{LOCKDOWN}\n\n"
            f"Camera angle: {angle_prompt}. "
            "ONLY the camera angle changes — character design, outfit, props, and color palette "
            "must stay IDENTICAL to the reference image."
        )
        # angle별 자동 retry 1번 — Kie task hang/timeout 케이스 회복
        # timeout 4분 (240초) — Kie i2i 평균 90초인데 가끔 3분 넘게 걸림.
        # 2분(default)으로 자르면 적합한 task도 timeout으로 죽여서 retry 무한 반복.
        #
        # ⭐ snowball multi-ref — baseline + 이전에 만든 각도 다 reference로 사용.
        # 매 각도가 이전 결과들 다 보고 만들어짐 → 캐릭터 일관성 누적.
        refs = [baseline] + [Path(p) for p in paths if Path(p).exists()]
        last_err = None
        for attempt in range(2):
            try:
                await kie_client.gpt_image_2_i2i(
                    refs, out_png,
                    prompt=full_prompt,
                    quality="high", image_size="1024x1024",
                    timeout=240.0,
                )
                paths.append(str(out_png))
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[turnaround] {role_id} angle {angle_id} attempt {attempt+1} failed: {e}",
                      flush=True)
                if attempt < 1 and progress_cb:
                    await progress_cb(
                        int(i / len(TURNAROUND_ANGLES) * 100),
                        f"각도 {i+1}/{len(TURNAROUND_ANGLES)} 재시도 ({angle_id})",
                    )
        if last_err is not None:
            failed.append(angle_id)

    status = "ready" if len(paths) == len(TURNAROUND_ANGLES) else (
        "partial" if paths else "failed"
    )
    db.upsert_mascot_role(
        dissection_id, role_id,
        turnaround_paths=paths,
        turnaround_status=status,
    )
    if progress_cb:
        await progress_cb(
            100,
            f"✅ {len(paths)}/{len(TURNAROUND_ANGLES)} 완료" + (
                f" (실패: {','.join(failed)})" if failed else ""
            ),
        )
    try:
        cost_tracker.add_cost(dissection_id, 0.03 * (len(paths) + len(failed)))
    except Exception:
        pass
    # 텔레그램 알림 — 모두 성공 / 일부 실패 / 다 실패 따라 다르게
    try:
        from . import notify
        diss = db.get_dissection(dissection_id) or {}
        diss_name = diss.get("name") or dissection_id
        if status == "ready":
            await notify.notify_success(
                f"🎨 8각 turnaround 완료 — {diss_name}",
                f"역할: {role_id}\n{len(paths)}/{len(TURNAROUND_ANGLES)} 각도 다 만들어짐",
            )
        elif status == "partial":
            await notify.notify_error(
                f"⚠️ 8각 turnaround 일부 실패 — {diss_name}",
                f"역할: {role_id}\n성공 {len(paths)}/{len(TURNAROUND_ANGLES)}, 실패 각도: {', '.join(failed)}",
            )
        else:
            await notify.notify_error(
                f"❌ 8각 turnaround 다 실패 — {diss_name}",
                f"역할: {role_id}\n8각 다 실패 — Kie 서비스 점검 필요",
            )
    except Exception as _e:
        print(f"[turnaround] 텔레그램 알림 실패: {_e}", flush=True)
    return {
        "role_id": role_id,
        "paths": paths,
        "count": len(paths),
        "failed": failed,
        "status": status,
        "cost_usd": round(0.03 * (len(paths) + len(failed)), 4),
    }


async def regenerate_turnaround_angle(
    dissection_id: str, role_id: str, angle_id: str,
) -> dict:
    """특정 각도 1장만 ↻ 재생성. 다른 각도는 보존."""
    role_id = _safe_role_id(role_id)
    if angle_id not in [a[0] for a in TURNAROUND_ANGLES]:
        raise ValueError(f"invalid angle_id: {angle_id}")
    angle_prompt = next(p for a, p in TURNAROUND_ANGLES if a == angle_id)

    roles = db.get_mascot_roles(dissection_id) or []
    role = next((r for r in roles if r.get("role_id") == role_id), None)
    if not role:
        raise ValueError(f"role {role_id} not found")
    baseline = Path(role.get("baseline_path") or "")
    if not baseline.exists():
        raise ValueError(f"baseline missing")
    spec_str = role.get("character_spec") or "- chibi character"

    out_png = baseline.parent / f"webtoon_{role_id}_turnaround_{angle_id}.png"
    full_prompt = (
        "CRITICAL TASK — Make this output 99% visually IDENTICAL to the reference image. "
        "Only the camera angle changes. Every other detail MUST stay exactly the same.\n\n"
        "STRICT RULES — DO NOT CHANGE:\n"
        "- Hairstyle: same length, same texture, same parting, same color (do NOT tie up if loose, do NOT shorten if long, do NOT make tidy if messy)\n"
        "- Facial hair: if mustache/beard visible in reference, MUST appear identical in output\n"
        "- Face age: if elderly with wrinkles, MUST stay elderly (do not make younger)\n"
        "- Clothing colors: every layer's color matches exactly\n"
        "- Accessory colors: origami crane / paper / props colors match exactly\n"
        "- Body proportions: same chibi ratio\n\n"
        f"Character spec — every detail below MUST be preserved:\n{spec_str}\n\n"
        f"Camera angle: {angle_prompt}.\n\n"
        "Pure white background, full body visible, clean 2D Korean webtoon style."
    )
    # ⭐ multi-ref — baseline + 다른 각도 (재생성하는 각도 빼고) 다 reference로.
    # 다른 각도들이 캐릭터 디테일 lockdown 역할 (머리 모양/옷 일관성).
    other_paths = [
        Path(p) for p in (role.get("turnaround_paths") or [])
        if Path(p).exists() and f"_turnaround_{angle_id}.png" not in p
    ]
    refs = [baseline] + other_paths
    await kie_client.gpt_image_2_i2i(
        refs, out_png,
        prompt=full_prompt,
        quality="high", image_size="1024x1024",
        timeout=240.0,
    )
    try:
        cost_tracker.add_cost(dissection_id, 0.03)
    except Exception:
        pass
    # turnaround_paths에 추가 (없으면)
    paths = list(role.get("turnaround_paths") or [])
    if str(out_png) not in paths:
        paths.append(str(out_png))
        paths.sort()
    status = "ready" if len(paths) == len(TURNAROUND_ANGLES) else "partial"
    db.upsert_mascot_role(
        dissection_id, role_id,
        turnaround_paths=paths, turnaround_status=status,
    )
    return {"role_id": role_id, "angle_id": angle_id, "path": str(out_png),
            "cost_usd": 0.03}


def delete_turnaround_angle(
    dissection_id: str, role_id: str, angle_id: str,
) -> dict:
    """특정 각도 ✕ 삭제 — 디스크 파일 unlink + DB sync."""
    role_id = _safe_role_id(role_id)
    if angle_id not in [a[0] for a in TURNAROUND_ANGLES]:
        raise ValueError(f"invalid angle_id: {angle_id}")
    roles = db.get_mascot_roles(dissection_id) or []
    role = next((r for r in roles if r.get("role_id") == role_id), None)
    if not role:
        raise ValueError(f"role {role_id} not found")
    baseline_dir = Path(role.get("baseline_path") or "").parent
    target = baseline_dir / f"webtoon_{role_id}_turnaround_{angle_id}.png"
    unlinked = False
    if target.exists():
        target.unlink()
        unlinked = True
    paths = [p for p in (role.get("turnaround_paths") or [])
             if Path(p).name != target.name]
    status = "ready" if len(paths) == len(TURNAROUND_ANGLES) else (
        "partial" if paths else "none"
    )
    db.upsert_mascot_role(
        dissection_id, role_id,
        turnaround_paths=paths, turnaround_status=status,
    )
    return {"role_id": role_id, "angle_id": angle_id,
            "unlinked": unlinked, "remaining": len(paths)}
