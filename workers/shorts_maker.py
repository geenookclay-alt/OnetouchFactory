"""쇼츠 메이커 워커 (banbaji-discover 통합).

긴 영상 URL을 받아 멀티 하이라이트 쇼츠 N개로 양산.
실제 파이프라인은 scripts.test_shorts_maker.run_pipeline 재사용.
이 모듈은 잡 큐(DB) 진행률·결과 저장만 책임진다.
"""
import sys
import json
import asyncio
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from api import database as db
from scripts.test_shorts_maker import (
    run_pipeline, discover_candidates, render_selected,
)


def _make_progress_cb(job_id: int):
    def cb(stage, pct, msg, extra=None):
        kw = {"status": stage, "progress": pct, "progress_message": msg}
        if extra:
            if "pass1" in extra:
                kw["pass1_json"] = json.dumps(extra["pass1"], ensure_ascii=False)
            if "highlights_count" in extra:
                kw["highlights_count"] = extra["highlights_count"]
            if "source_duration" in extra:
                kw["source_duration"] = extra["source_duration"]
            if "source_size_mb" in extra:
                kw["source_size_mb"] = extra["source_size_mb"]
        try:
            db.update_shorts_job(job_id, **kw)
        except Exception as e:
            print(f"  ⚠️ progress 업데이트 실패: {e}", flush=True)
    return cb


def _collect_files(hl_dir: Path):
    files = []
    if hl_dir.exists():
        for f in sorted(hl_dir.iterdir()):
            if f.is_file():
                files.append({"name": f.name, "size": f.stat().st_size})
    return files


# 업로드 원본 삭제 안전 루트 (이 디렉토리 하위 파일만 삭제 — 그 외 경로는 절대 안 건드림)
_UPLOAD_ROOTS = ("/Volumes/BanbajiMedia/uploads",)


def _cleanup_movie_source(job: dict, out_dir: str):
    """영화 잡 완료 후 대용량 원본 정리 (대표님: 영화 원본 커서 공간확보).
    안전가드: ①업로드 루트 하위 파일만 ②영화 타입 호출부에서만 ③실패해도 잡 영향 X.
    """
    import os
    # 1) 업로드 원본 (job url이 업로드 경로일 때만)
    u = (job.get("url") or "").strip()
    if u.startswith("file://"):
        u = u[7:]
    try:
        if u and not u.startswith("http"):
            rp = os.path.realpath(u)
            roots = [os.path.realpath(r) for r in _UPLOAD_ROOTS]
            if any(rp == r or rp.startswith(r + os.sep) for r in roots) and os.path.isfile(rp):
                os.remove(rp)
                print(f"  🧹 업로드 원본 삭제 (공간확보): {rp}", flush=True)
    except Exception as e:
        print(f"  ⚠️ 업로드 원본 삭제 실패(무시): {e}", flush=True)
    # 2) 중간 source.mp4 + vtt (out_dir, 대용량)
    try:
        od = Path(out_dir)
        sp = od / "source.mp4"
        if sp.exists():
            sp.unlink()
            print("  🧹 중간 source.mp4 삭제", flush=True)
        for v in od.glob("*.vtt"):
            v.unlink(missing_ok=True)
    except Exception as e:
        print(f"  ⚠️ 중간본 삭제 실패(무시): {e}", flush=True)


async def run_shorts_maker(job_id: int):
    """쇼츠 메이커 잡 1개 실행 (워커/스크립트 공용 진입점)."""
    job = db.get_shorts_job(job_id)
    if not job:
        print(f"[shorts] job {job_id} 없음", flush=True)
        return
    url = job["url"]
    out_dir = job.get("out_dir") or f"data/shorts/job_{job_id}"
    # pipeline_type: jobs 테이블의 type 컬럼 (highlight / drama). 없으면 highlight default.
    ptype = (job.get("type") or "highlight").lower()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    db.update_shorts_job(
        job_id, out_dir=out_dir, status="downloading",
        progress=1, progress_message=f"시작 (type={ptype})", error=None,
    )
    cb = _make_progress_cb(job_id)
    try:
        result = await run_pipeline(url, out_dir, on_progress=cb, pipeline_type=ptype)
        results = []
        for r in result["results"]:
            d = {"idx": r["idx"]}
            if "error" in r:
                d["error"] = r["error"]
                d["dir"] = r.get("dir", "")
            else:
                d.update({
                    "dur": r.get("dur", 0),
                    "dir": r["dir"],
                    "files": _collect_files(Path(r["dir"])),
                })
            results.append(d)
        # 캐리커처 정보 (잡 공유)
        characters = []
        for c in result.get("characters", []) or []:
            characters.append({
                "role": c.get("role"),
                "name_or_nickname": c.get("name_or_nickname"),
                "appearance": c.get("appearance"),
                "png": c.get("png"),
                "error": c.get("error"),
            })
        ok_n = len([r for r in results if "error" not in r])
        char_n = len([c for c in characters if not c.get("error")])
        # results_json에 characters도 같이 박음 (별도 컬럼 안 만들고 활용)
        payload = {"highlights": results, "characters": characters,
                   "characters_dir": result.get("characters_dir")}
        db.update_shorts_job(
            job_id,
            status="completed",
            progress=100,
            progress_message=f"하이라이트 {ok_n}/{len(results)} + 캐리커처 {char_n}장",
            results_json=json.dumps(payload, ensure_ascii=False),
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        print(f"[shorts] job {job_id} 완료 ({ok_n} hl + {char_n} chars)", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[shorts] job {job_id} 실패: {e}\n{tb}", flush=True)
        db.update_shorts_job(
            job_id, status="failed",
            error=f"{e}\n{tb[:1500]}",
        )


async def run_shorts_discover(job_id: int):
    """v3 발굴 — 후보만 찾아 저장 (제작 X). status=discovered."""
    job = db.get_shorts_job(job_id)
    if not job:
        print(f"[shorts] job {job_id} 없음", flush=True)
        return
    url = job["url"]
    out_dir = job.get("out_dir") or f"data/shorts/job_{job_id}"
    ptype = (job.get("type") or "drama").lower()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    db.update_shorts_job(job_id, out_dir=out_dir, status="discovering",
                         progress=1, progress_message=f"발굴 시작 ({ptype})", error=None)
    cb = _make_progress_cb(job_id)
    try:
        r = await discover_candidates(url, out_dir, pipeline_type=ptype, on_progress=cb)
        cands = r.get("candidates", [])
        payload = {"candidates": cands, "highlights": [], "characters": []}
        # 전 타입 자동 제작 — 발굴 후 사용자 선택 단계 생략, 모든 후보 한 번에 제작
        # (대표님 2026-06-05: "링크/파일 넣으면 알아서 제작 다 들어가도록")
        if cands:
            db.update_shorts_job(
                job_id, status="discovered",
                progress_message=f"하이라이트 {len(cands)}개 발굴 — 전체 자동 제작 시작 (type={ptype})",
                highlights_count=len(cands),
                source_duration=r.get("source_duration"),
                source_size_mb=r.get("source_size_mb"),
                results_json=json.dumps(payload, ensure_ascii=False),
            )
            print(f"[shorts] job {job_id} {ptype} — 발굴 {len(cands)}개 전체 자동 제작 시작", flush=True)
            all_idxs = [int(c.get("idx", i)) for i, c in enumerate(cands)]
            await run_shorts_render_selected(job_id, all_idxs)
            return
        db.update_shorts_job(
            job_id, status="discovered", progress=100,
            progress_message=f"후보 {len(cands)}개 발굴 (선택 대기)",
            highlights_count=len(cands),
            source_duration=r.get("source_duration"),
            source_size_mb=r.get("source_size_mb"),
            results_json=json.dumps(payload, ensure_ascii=False),
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        print(f"[shorts] job {job_id} 발굴 완료 ({len(cands)} 후보)", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[shorts] job {job_id} 발굴 실패: {e}\n{tb}", flush=True)
        db.update_shorts_job(job_id, status="failed", error=f"{e}\n{tb[:1500]}")


async def run_shorts_render_selected(job_id: int, selected_idxs: list):
    """v3 선택 제작 — 발굴된 후보 중 선택한 것만 클립 생성."""
    job = db.get_shorts_job(job_id)
    if not job:
        print(f"[shorts] job {job_id} 없음", flush=True)
        return
    out_dir = job.get("out_dir") or f"data/shorts/job_{job_id}"
    ptype = (job.get("type") or "drama").lower()
    db.update_shorts_job(job_id, status="rendering", progress=1,
                         progress_message=f"선택 {len(selected_idxs)}개 제작 시작", error=None)
    cb = _make_progress_cb(job_id)
    try:
        r = await render_selected(out_dir, selected_idxs, pipeline_type=ptype, on_progress=cb)
        # 🔴 출처 자막 (대표님 0606): 쇼츠메이커 전 출력에 "출처 : 프로그램명" SRT 의무.
        try:
            from scripts.test_shorts_maker import _write_source_srt
            await _write_source_srt(r.get("results", []), job.get("name") or "")
        except Exception as _e:
            print(f"  ⚠️ 출처 자막 실패(계속): {_e}", flush=True)
        # 기존 results_json(candidates)에 highlights 합치기
        prev = {}
        try:
            prev = json.loads(job.get("results_json") or "{}")
        except Exception:
            prev = {}
        results = []
        for rr in r.get("results", []):
            d = {"idx": rr.get("idx")}
            if "error" in rr:
                d["error"] = rr["error"]
            else:
                d.update({"dur": rr.get("dur", 0), "dir": rr.get("dir", ""),
                          "files": _collect_files(Path(rr["dir"])) if rr.get("dir") else [],
                          "candidate": rr.get("candidate")})
            results.append(d)
        # 캐리커처 (선택 제작 후 첫 성공본으로 1회)
        characters = []
        first_final = None
        for rr in r.get("results", []):
            if "error" not in rr and rr.get("final"):
                fp = Path(rr["final"])
                if fp.exists():
                    first_final = fp; break
        if first_final:
            try:
                from workers.character_generator import run_character_generation
                _maxc = 6 if ptype in ("movie", "anime") else (4 if ptype == "folktale" else 3)   # 영화·애니 5~6명 / 동화 주요인물 4명 / 그 외 3명
                chars = await run_character_generation(first_final, Path(out_dir) / "characters", max_chars=_maxc)
                for c in chars:
                    characters.append({"role": c.get("role"),
                                       "name_or_nickname": c.get("name_or_nickname"),
                                       "png": c.get("png"), "error": c.get("error")})
            except Exception as e:
                print(f"  ⚠️ 캐리커처 실패: {e}", flush=True)
        prev["highlights"] = results
        if characters:
            prev["characters"] = characters
        ok_n = len([x for x in results if "error" not in x])
        db.update_shorts_job(
            job_id, status="completed", progress=100,
            progress_message=f"선택 {ok_n}/{len(results)} 제작 완료",
            results_json=json.dumps(prev, ensure_ascii=False),
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        print(f"[shorts] job {job_id} 선택제작 완료 ({ok_n})", flush=True)
        if ptype == "movie":          # 영화 원본 대용량 → 작업 후 자동 공간확보
            _cleanup_movie_source(job, out_dir)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[shorts] job {job_id} 선택제작 실패: {e}\n{tb}", flush=True)
        db.update_shorts_job(job_id, status="failed", error=f"{e}\n{tb[:1500]}")


if __name__ == "__main__":
    asyncio.run(run_shorts_maker(int(sys.argv[1])))
