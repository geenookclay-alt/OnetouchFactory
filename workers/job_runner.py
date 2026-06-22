"""독립 워커 진입점 — uvicorn(웹서버)과 분리된 별도 프로세스로 작업 실행.

대표님 2026-05-29 근본 해결: 작업이 uvicorn 이벤트 루프에서 돌면
 ① 코드 고쳐 재시작 시 작업이 죽고 ② 무거운 작업이 웹서버를 막아 홈페이지가 느려지고
 ③ 작업이 서버에 묶여 취약함.
→ API는 이 스크립트를 subprocess.Popen으로 띄우기만 함. 작업은 독립 프로세스라
   uvicorn을 재시작해도 안 죽고, 웹서버 응답도 안 느려짐.

사용: venv/bin/python -m workers.job_runner <func> <job_id> [extra]
  shorts_discover <job_id>
  shorts_render   <job_id> <idxs(쉼표구분)>
  clip_edit       <job_id>
"""
import sys
import asyncio
from pathlib import Path

# 🔴 임베디드 파이썬(윈도우 pyembed)은 cwd/스크립트 폴더를 sys.path에 안 넣어
#    `from workers.x`/`from api.x` 가 ModuleNotFoundError(workers)로 죽음 → 클립편집 워커가
#    즉시 종료돼 0%에 멈췄음. repo(이 파일의 부모의 부모)를 직접 path에 추가. (2026-06-16 윈도우 근본수정)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 🔴 .env 로드 — 독립 서브프로세스라 uvicorn과 달리 .env를 직접 불러와야 함.
# (안 하면 KIE_API_KEY 등이 비어 캐리커처(Nano Banana) 생성 실패. 2026-05-31 영화/애니 캐리커처 버그)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ── [배포판] venv 내장 ffmpeg/yt-dlp가 PATH에서 먼저 잡히게 ──
import os as _bbenv_os, sys as _bbenv_sys
from pathlib import Path as _bbenv_Path
_bbenv_os.environ["PATH"] = str(_bbenv_Path(_bbenv_sys.executable).parent) + _bbenv_os.pathsep + _bbenv_os.environ.get("PATH", "")


def main():
    if len(sys.argv) < 3:
        print("usage: job_runner <func> <job_id> [extra]", flush=True)
        sys.exit(1)
    func = sys.argv[1]
    job_id = int(sys.argv[2])

    if func == "shorts_discover":
        from workers.shorts_maker import run_shorts_discover
        asyncio.run(run_shorts_discover(job_id))
    elif func == "shorts_render":
        idxs = [int(x) for x in sys.argv[3].split(",") if x.strip()] if len(sys.argv) > 3 else []
        from workers.shorts_maker import run_shorts_render_selected
        asyncio.run(run_shorts_render_selected(job_id, idxs))
    elif func == "clip_edit":
        # 클립편집(스토리 주제형) — DB에서 urls/주제/분량 읽어 독립 실행. TTS 제외(대표님 0613).
        from workers.clip_editor import run_clip_edit
        from api.database import get_clip_edit_job
        import json as _j
        job = get_clip_edit_job(job_id) or {}
        urls = _j.loads(job.get("urls") or "[]")
        asyncio.run(run_clip_edit(
            job_id, urls, job.get("song_title") or "",
            int(job.get("target_duration") or 50), False))
    else:
        print(f"unknown func: {func}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
