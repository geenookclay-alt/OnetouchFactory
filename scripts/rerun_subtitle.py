#!/usr/bin/env python3
"""좀비 자막 잡 재실행 — 서버 재시작으로 인프로세스 워커가 죽은 잡을
별도 프로세스에서 다시 돌린다 (서버 재시작 불필요).

사용: venv/bin/python3 scripts/rerun_subtitle.py 172 173
"""
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import database as db
from workers.auto_subtitle import run_auto_subtitle


def _parse_urls(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


async def rerun_one(job_id: int):
    job = db.get_subtitle_job(job_id)
    if not job:
        print(f"[{job_id}] 잡 없음 — 건너뜀", flush=True)
        return
    vp = job.get("video_path")
    if not vp or not Path(vp).exists():
        print(f"[{job_id}] 영상 파일 없음({vp}) — 재업로드 필요", flush=True)
        db.update_subtitle_job(
            job_id, status="failed",
            progress_message="영상 파일 없음 (재업로드 필요)",
        )
        return
    urls = _parse_urls(job.get("original_urls"))
    style = job.get("style") or "shorts"
    print(f"[{job_id}] 재실행 시작 style={style} urls={urls}", flush=True)
    try:
        await run_auto_subtitle(job_id, Path(vp), urls, None, style, "")
        j = db.get_subtitle_job(job_id) or {}
        print(f"[{job_id}] 끝 status={j.get('status')} "
              f"progress={j.get('progress')}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{job_id}] 예외: {e}", flush=True)


async def main():
    ids = []
    for a in sys.argv[1:]:
        try:
            ids.append(int(a))
        except ValueError:
            pass
    if not ids:
        print("사용: rerun_subtitle.py <job_id> [job_id ...]")
        return
    print(f"재실행 대상: {ids}", flush=True)
    await asyncio.gather(*[rerun_one(i) for i in ids])
    print("전체 끝", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
