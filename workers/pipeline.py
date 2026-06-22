"""Discovery pipeline orchestrator.

Stages:
  1. Reference channel ingestion (extract used videos → DNA pool)
  2. Multi-platform candidate collection (YT + TikTok + IG)
  3. Filtering (views/duration/excluded)
  4. DNA matching (Ollama Qwen 14B)
  5. Korean repost duplicate check
  6. Persistence (SQLite + optional Notion)
"""
import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Awaitable

from . import llm
from . import apify_client
from . import youtube_client
from . import notion_client
from . import dissection
from . import keyword_generator
from . import visual_match
from . import nexlev_client

# Local imports for typing only
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api import database as db


# ===== Stage helpers =====

def _passes_filter(v: dict, min_views: int, max_duration: int,
                   excluded_kw: list[str], excluded_channels: list[str]) -> bool:
    """Apply hard filters: views, duration, excluded keywords/channels."""
    if (v.get("view_count") or 0) < min_views:
        return False
    dur = v.get("duration") or 0
    # tolerate 0-second slideshow content if user wants to keep them
    if dur > max_duration:
        return False
    title_caption = (v.get("title", "") + " " + v.get("caption", "")).lower()
    for kw in excluded_kw:
        if kw and kw.lower() in title_caption:
            return False
    channel = (v.get("channel_name") or "").lower()
    for ec in excluded_channels:
        if ec and ec.lower() in channel:
            return False
    return True


# ===== Main pipeline =====

ProgressCb = Callable[[int, str], Awaitable[None]]


async def run_pipeline(
    job_id: str,
    keywords: list[str],
    reference_channel: str | None = None,
    platforms: list[str] = ("youtube",),
    min_views: int = 5_000_000,
    max_duration: int = 55,
    excluded_keywords: list[str] | None = None,
    excluded_channels: list[str] | None = None,
    notion_database_id: str | None = None,
    progress_cb: ProgressCb | None = None,
    enable_dna_matching: bool = False,  # default: skip — too slow on local LLM
    enable_visual_match: bool = False,  # default: skip — requires built KR pool
    visual_match_concurrency: int = 3,
) -> dict:
    """Run the full discovery pipeline. Stores results in DB."""
    excluded_keywords = excluded_keywords or [
        "top", "best", "worst", "compilation", "ranking", "funniest",
        "girls vs boys",
        # 인도 시그니처 — 형님 룰 인도 제외
        "hindi", "bollywood", "desi", "bhai", "bhaiya", "namaste",
    ]
    excluded_channels = excluded_channels or []

    async def _emit(pct: int, msg: str):
        db.update_job(job_id, progress=pct, progress_message=msg)
        if progress_cb:
            await progress_cb(pct, msg)

    db.update_job(job_id, status="running",
                  started_at=datetime.utcnow().isoformat())
    await _emit(5, "파이프라인 시작")

    # ===== Stage 1: Reference channel ingestion (optional) =====
    reference_dnas: list[dict] = []
    if reference_channel:
        await _emit(10, f"레퍼런스 채널 분석: {reference_channel}")
        try:
            handle = youtube_client.extract_channel_handle(reference_channel)
            # Try YouTube Data API first; fallback empty if no key
            ref_videos = await youtube_client.get_channel_videos(handle, max_results=100)
            for v in ref_videos[:30]:  # top 30 by recency for DNA pool
                dna = await llm.analyze_video_dna(
                    v.get("title", ""), v.get("caption", ""),
                    channel=v.get("channel_name", ""),
                )
                reference_dnas.append(dna)
                db.add_reference_video(
                    channel_id=v.get("channel_id") or handle,
                    channel_name=v.get("channel_name") or handle,
                    platform="youtube",
                    video_id=v["video_id"],
                    title=v.get("title"),
                    dna_summary=json.dumps(dna, ensure_ascii=False),
                )
        except Exception as e:
            await _emit(15, f"레퍼런스 분석 실패 (계속 진행): {e}")
    await _emit(25, f"레퍼런스 DNA {len(reference_dnas)}개 확보")

    # ===== Stage 2: Multi-platform collection (parallel union + cache) =====
    # Each source goes through a 1-hour DB cache so repeated identical searches
    # are deterministic + free. Same keyword within TTL → cache hit → quota 0.
    # Apify scrapers are inherently ~40% non-deterministic; cache makes the
    # candidate pool fully reproducible across re-runs.
    all_candidates: list[dict] = []
    sources: list[tuple[str, Any]] = []  # [(label, coroutine)]

    async def _cached(source: str, query: str, factory):
        """DB-cached source call. factory() returns a coroutine list."""
        hit = db.cache_get(source, query)
        if hit is not None:
            return hit
        try:
            results = await factory()
        except Exception as e:
            raise  # let outer gather capture failures per-source
        if results:
            db.cache_set(source, query, results, ttl_seconds=3600)
        return results or []

    if "youtube" in platforms:
        for kw in keywords:
            # Phase 11: region US + KR both, then dedup downstream
            sources.append((f"yt_api_us:{kw}",
                            _cached("yt_api_us", kw,
                                    lambda kw=kw: youtube_client.search_youtube(kw, max_results=60, region="US"))))
            sources.append((f"yt_api_kr:{kw}",
                            _cached("yt_api_kr", kw,
                                    lambda kw=kw: youtube_client.search_youtube(kw, max_results=40, region="KR"))))
            # apify_yt (streamers~youtube-scraper) currently returns 400 due
            # to actor input schema drift. NexLev + YouTube API + Apify
            # TikTok/IG cover the gap. Re-enable with ENABLE_APIFY_YT=1 once
            # the actor or input schema is fixed.
            if os.getenv("ENABLE_APIFY_YT") == "1":
                sources.append((f"apify_yt:{kw}",
                                _cached("apify_yt", kw,
                                        lambda kw=kw: apify_client.get_youtube_via_apify(kw, results=30))))

    if False and "tiktok" in platforms:  # 비활성 - Apify quota 절약
        # cache key = sorted joined keywords so same keyword-set hits cache
        kw_key = "|".join(sorted(keywords))
        sources.append(("apify_tiktok",
                        _cached("apify_tiktok", kw_key,
                                lambda: apify_client.search_tiktok(keywords, results_per_keyword=50))))

    if False and "instagram" in platforms:  # 비활성 — 결 안 맞아서 제거
        for kw in keywords[:8]:  # Phase 11: expanded from 5 → 8 for broader IG coverage
            sources.append((f"apify_ig:{kw}",
                            _cached("apify_ig", kw,
                                    lambda kw=kw: apify_client.search_instagram_reels(kw, results=30))))

    # NexLev similar_thumbnails text search — backup discovery source that
    # works even when YouTube API + Apify YouTube are quota-blocked. 🥞10/kw.
    # Capped at top 10 keywords to keep monthly NexLev quota predictable
    # (10 kw × 10 quota = 100 quota per discovery run).
    if "youtube" in platforms:
        for kw in keywords[:10]:
            sources.append((f"nexlev_thumb:{kw}",
                            _cached("nexlev_thumb", kw,
                                    lambda kw=kw: nexlev_client.search_videos_by_text(kw, limit=15))))

    await _emit(30, f"병렬 검색 중 ({len(sources)} sources)")

    sem = asyncio.Semaphore(8)  # bounded concurrency — don't hammer APIs

    async def _bounded(coro):
        async with sem:
            return await coro

    results = await asyncio.gather(
        *[_bounded(coro) for _, coro in sources],
        return_exceptions=True,
    )

    # Collate + per-source stats (so we see which APIs are quota-blocked)
    stats: dict[str, int] = {}
    fail_examples: dict[str, str] = {}
    for (label, _), res in zip(sources, results):
        family = label.split(":", 1)[0]
        if isinstance(res, Exception):
            stats[family + "_fail"] = stats.get(family + "_fail", 0) + 1
            fail_examples.setdefault(family, str(res)[:120])
            continue
        if not res:
            continue
        stats[family] = stats.get(family, 0) + len(res)
        all_candidates.extend(res)

    summary = ", ".join(f"{k}={v}" for k, v in stats.items())
    await _emit(65, f"수집 완료: {len(all_candidates)}개 [{summary}]")
    for fam, err in fail_examples.items():
        await _emit(65, f"⚠️ {fam} 일부 실패: {err}")

    # ===== Stage 3: Filtering =====
    filtered = [
        v for v in all_candidates
        if _passes_filter(v, min_views, max_duration,
                          excluded_keywords, excluded_channels)
    ]
    await _emit(70, f"필터 통과: {len(filtered)}개 / 원본 {len(all_candidates)}")

    # ===== Stage 4: Dedup against reference + Korean repost check =====
    unique = []
    seen = set()
    for v in filtered:
        key = (v["platform"], v["video_id"])
        if key in seen:
            continue
        seen.add(key)
        if db.is_video_in_reference(v["platform"], v["video_id"]):
            continue
        kr_channel = db.is_video_reposted_in_korea(v["platform"], v["video_id"])
        if kr_channel:
            v["notes"] = f"이미 한국 채널 재업: {kr_channel}"
            v["classification"] = "배제"
        unique.append(v)
    await _emit(78, f"중복 제거 후: {len(unique)}개")

    # ===== Stage 4.5: Visual match against Korean pool (optional) =====
    if enable_visual_match:
        await _emit(79, "시각 매칭 (한국 풀 대조) 시작")
        sem = asyncio.Semaphore(visual_match_concurrency)

        async def _visual_one(v: dict) -> None:
            url = v.get("url")
            if not url:
                return
            async with sem:
                try:
                    res = await visual_match.match_candidate(url)
                except Exception as e:
                    v["visual_match_verdict"] = "ERROR"
                    v["notes"] = (v.get("notes", "") + f" 시각매칭실패:{e}").strip()
                    return
            v["visual_match_verdict"] = res.get("verdict", "NEW")
            v["visual_match_score"] = res.get("score", 0.0)
            v["visual_match_video_id"] = res.get("best_match_video")
            v["visual_match_url"] = res.get("matched_url")
            v["visual_match_channel"] = res.get("matched_channel")
            if res.get("verdict") == "EXCLUDE":
                v["classification"] = "배제"
                v["notes"] = (v.get("notes", "") +
                              f" [시각매칭] 한국재업 추정: {res.get('matched_handle') or res.get('matched_channel')} (score {res.get('score')})").strip()
            elif res.get("verdict") == "SUSPECT":
                v["notes"] = (v.get("notes", "") +
                              f" [시각매칭] 의심 score {res.get('score')}").strip()

        tasks = [_visual_one(v) for v in unique]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 5 == 0 or done == len(tasks):
                pct = 79 + int(0.5 * (done / max(len(tasks), 1)))  # 79~79.5
                await _emit(int(pct), f"시각 매칭 {done}/{len(tasks)}")
        excluded = sum(1 for v in unique if v.get("visual_match_verdict") == "EXCLUDE")
        await _emit(80, f"시각 매칭 완료: 배제 {excluded}개")

    # ===== Stage 5: DNA scoring (optional — disabled by default, too slow) =====
    if enable_dna_matching and reference_dnas:
        await _emit(80, "DNA 매칭 점수 산출 중 (Qwen 14B)")
        for i, v in enumerate(unique):
            try:
                v_dna = await llm.analyze_video_dna(
                    v.get("title", ""), v.get("caption", ""),
                    channel=v.get("channel_name", ""),
                )
                v["dna"] = v_dna
                if v_dna.get("is_ranking_or_compilation") or v_dna.get("is_ai_or_cgi"):
                    v["classification"] = "배제"
                else:
                    score_data = await llm.match_dna_score(v_dna, reference_dnas)
                    v["dna_match_score"] = score_data.get("score", 0.0)
                    v["dna_categories"] = score_data.get("matched_categories", [])
                    v.setdefault("classification",
                                 "키핑" if score_data.get("score", 0) >= 0.5 else "의문")
            except Exception as e:
                v["classification"] = "의문"
                v["notes"] = f"DNA 분석 실패: {e}"
            if i % 5 == 0:
                pct = 80 + int(15 * (i / max(len(unique), 1)))
                await _emit(pct, f"DNA 매칭 {i + 1}/{len(unique)}")
    else:
        # No DNA matching → all candidates default to 키핑, user reclassifies in PWA
        for v in unique:
            v.setdefault("classification", "키핑")
        await _emit(85, f"DNA 매칭 생략 ({len(unique)}개 모두 키핑 default)")

    # ===== Stage 6: Persistence =====
    await _emit(95, "DB 저장 중")
    saved = 0
    for v in unique:
        try:
            db.insert_candidate(
                job_id=job_id,
                platform=v["platform"],
                video_id=v["video_id"],
                url=v.get("url"),
                title=v.get("title"),
                caption=v.get("caption"),
                channel_name=v.get("channel_name"),
                channel_id=v.get("channel_id"),
                view_count=v.get("view_count"),
                like_count=v.get("like_count"),
                duration=v.get("duration"),
                published_at=v.get("published_at"),
                thumbnail_url=v.get("thumbnail_url"),
                dna_match_score=v.get("dna_match_score"),
                dna_categories=json.dumps(v.get("dna_categories", []), ensure_ascii=False),
                classification=v.get("classification", "키핑"),
                notes=v.get("notes", ""),
                visual_match_verdict=v.get("visual_match_verdict"),
                visual_match_score=v.get("visual_match_score"),
                visual_match_video_id=v.get("visual_match_video_id"),
                visual_match_url=v.get("visual_match_url"),
                visual_match_channel=v.get("visual_match_channel"),
            )
            saved += 1
        except Exception as e:
            print(f"DB insert failed: {e}")

    # ===== Optional: Notion adapter =====
    if notion_database_id:
        await _emit(98, "노션 적재 중")
        keep = [v for v in unique if v.get("classification") == "키핑"]
        try:
            await notion_client.append_candidates(notion_database_id, keep)
        except Exception as e:
            await _emit(99, f"노션 적재 실패: {e}")

    db.update_job(
        job_id,
        status="completed",
        progress=100,
        completed_at=datetime.utcnow().isoformat(),
        progress_message=f"검색 완료: {saved}개 - 결 매칭 후처리 시작",
    )
    await _emit(100, f"검색 완료: {saved}개 - 결 매칭 후처리")

    # 자동 결 매칭 후처리 (메타 + Pro)
    try:
        from . import video_dna_filter as _vdf
        # 카테 정보
        with db.get_db() as conn:
            diss_row = conn.execute(
                "SELECT id, name, dissection_result FROM dissection_analyses WHERE related_job_id=? LIMIT 1",
                (job_id,),
            ).fetchone()
        if diss_row:
            diss = dict(diss_row)
            cate_name = diss.get("name") or ""
            dna_summary = f"카테 = {cate_name}"
            try:
                d = json.loads(diss.get("dissection_result") or "{}")
                common = d.get("common_dna") or {}
                items = common.get("items") or {}
                for k in ("4_view_drivers", "6_topics_list"):
                    it = items.get(k) or {}
                    v = it.get("value") or ""
                    if v:
                        dna_summary += chr(10) + (it.get("label", k) or k) + ": " + v[:200]
            except Exception:
                pass
            signature_words = _vdf.get_cate_signature_words(cate_name)
            await _emit(100, "결 매칭 후처리 시작 (메타 + Pro)")
            rcheck = await _vdf.recheck_candidate_pool(
                job_id=job_id, diss_id=diss.get("id"),
                cate_name=cate_name, dna_summary=dna_summary,
                signature_words=signature_words, max_pro_checks=50,
            )
            me = rcheck.get('meta_excluded', 0); pe = rcheck.get('pro_excluded', 0); await _emit(100, f"결 매칭 끝: 메타 {me} + Pro {pe} 제외")
    except Exception as e:
        await _emit(100, f"결 매칭 후처리 skip: {e}")
        import traceback; traceback.print_exc()

    return {"job_id": job_id, "candidates": saved, "filtered": len(unique)}


# ============================================================
# Two-stage dissection pipeline
# ============================================================

async def run_dissection_only(
    diss_id: str,
    reference_channels: list[str],
    progress_cb: ProgressCb | None = None,
    topic_hint: str = "",
) -> dict:
    """Stage 1: Analyze 0~5 reference channels and generate keyword recommendations.

    Stops at status='ready' awaiting the user's [Start Search] trigger.
    """
    async def _emit(pct: int, msg: str):
        db.update_dissection(diss_id, progress=pct, progress_message=msg)
        if progress_cb:
            await progress_cb(pct, msg)

    db.update_dissection(diss_id, status="analyzing")
    await _emit(5, "해체 분석 시작")

    if not reference_channels:
        # No reference - skip dissection, return empty
        db.update_dissection(
            diss_id,
            status="ready",
            progress=100,
            progress_message="레퍼런스 없음 - 키워드 직접 입력 필요",
            dissection_result=json.dumps({"channels": [], "common_dna": None}),
            keywords_result=json.dumps({"english": [], "chinese": [], "japanese": []}),
        )
        await _emit(100, "준비 완료 (레퍼런스 없음)")
        return {"diss_id": diss_id, "status": "ready"}

    hint_msg = f" (주제: {topic_hint[:30]})" if topic_hint else ""
    await _emit(15, f"채널 {len(reference_channels)}개 영상 데이터 수집 중{hint_msg}")

    try:
        # Run dissection (parallel for multiple channels). topic_hint 있으면 해당 주제 영상만 분석.
        diss_result = await dissection.dissect_multiple(
            reference_channels, max_videos_each=40, topic_hint=topic_hint
        )
    except Exception as e:
        db.update_dissection(diss_id, status="failed", error=f"분석 실패: {e}")
        await _emit(100, f"분석 실패: {e}")
        return {"diss_id": diss_id, "status": "failed", "error": str(e)}

    await _emit(70, "13개 항목 분석 완료, 키워드 생성 중")

    # Generate keywords from common_dna (or single channel if N=1)
    common = diss_result.get("common_dna") or (
        diss_result["channels"][0] if diss_result.get("channels") else {}
    )

    try:
        kw = await keyword_generator.generate_keywords(common or {})
    except Exception as e:
        kw = {"error": f"Keyword gen failed: {e}",
              "english": [], "chinese": [], "japanese": []}
    # 대키워드 자동 추가 (LLM이 빠뜨려도 카테 이름에서 추출)
    try:
        diss_meta = db.get_dissection(diss_id) or {}
        cate_name_for_kw = diss_meta.get("name") or ""
        if cate_name_for_kw:
            kw = keyword_generator.ensure_core_keywords(kw, cate_name_for_kw)
    except Exception as e:
        print(f"ensure_core_keywords skip: {e}")

    db.update_dissection(
        diss_id,
        status="ready",
        progress=100,
        progress_message="준비 완료 — [검색 시작] 클릭하면 발굴 시작",
        dissection_result=json.dumps(diss_result, ensure_ascii=False),
        keywords_result=json.dumps(kw, ensure_ascii=False),
    )
    await _emit(100, "분석 + 키워드 준비 완료")
    return {"diss_id": diss_id, "status": "ready",
            "channels_analyzed": len(diss_result.get("channels", [])),
            "keywords_generated": len(keyword_generator.flatten_keywords(kw))}


async def run_search_from_dissection(
    diss_id: str,
    extra_keywords: list[str] | None = None,
    progress_cb: ProgressCb | None = None,
    enable_visual_match: bool = False,
) -> dict:
    """Stage 2: Trigger the search using keywords from a completed dissection.
    Creates a new discovery job linked to the dissection.
    """
    diss = db.get_dissection(diss_id)
    if not diss:
        raise ValueError(f"Dissection {diss_id} not found")
    if diss["status"] not in ("ready", "completed"):
        raise ValueError(f"Dissection not ready (status={diss['status']})")

    # Pull keywords from dissection
    kw_data = json.loads(diss.get("keywords_result") or "{}")
    keywords = keyword_generator.flatten_keywords(kw_data)
    if extra_keywords:
        keywords = list(dict.fromkeys(keywords + extra_keywords))  # dedupe, preserve order

    if not keywords:
        raise ValueError("No keywords available — analysis returned empty")

    # Build reference DNA pool from dissection result
    diss_result = json.loads(diss.get("dissection_result") or "{}")
    channel_urls = json.loads(diss.get("reference_channels") or "[]")

    # Job 재사용 — 같은 dissection에서 검색 시작 누르면 기존 job에 dedup으로 append.
    # 없으면 새로 생성.
    existing_job_id = diss.get("related_job_id")
    if existing_job_id and db.get_job(existing_job_id):
        job_id = existing_job_id
        print(f"[search] 기존 job 재사용: {job_id}", flush=True)
    else:
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        db.insert_job(
            job_id=job_id,
            name=f"{diss['name']} - 검색",
            reference_channel=channel_urls[0] if channel_urls else None,
            keywords=json.dumps(keywords, ensure_ascii=False),
            platforms=diss.get("platforms") or '["youtube"]',
            min_views=diss.get("min_views", 5000000),
            max_duration=diss.get("max_duration", 55),
            excluded_keywords=diss.get("excluded_keywords") or '[]',
            excluded_channels=diss.get("excluded_channels") or '[]',
            notion_page_id=diss.get("notion_database_id"),
        )

    db.update_dissection(diss_id, status="searching", related_job_id=job_id)

    # Run discovery pipeline - reuses existing logic
    platforms = json.loads(diss.get("platforms") or '["youtube"]')
    excluded_kw = json.loads(diss.get("excluded_keywords") or '[]')
    excluded_ch = json.loads(diss.get("excluded_channels") or '[]')

    # Reference analysis already done in dissection — skip in pipeline
    result = await run_pipeline(
        job_id=job_id,
        keywords=keywords,
        reference_channel=None,           # already dissected, don't re-analyze
        platforms=platforms,
        min_views=diss.get("min_views", 5000000),
        max_duration=diss.get("max_duration", 55),
        excluded_keywords=excluded_kw,
        excluded_channels=excluded_ch,
        notion_database_id=diss.get("notion_database_id"),
        progress_cb=progress_cb,
        enable_dna_matching=False,        # skip slow per-video DNA scoring
        enable_visual_match=enable_visual_match,  # CLIP frame match vs Korean pool
    )

    db.update_dissection(
        diss_id, status="completed",
        completed_at=datetime.utcnow().isoformat(),
    )
    return result
