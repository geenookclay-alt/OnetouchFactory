# -*- coding: utf-8 -*-
"""ffmpeg / ffprobe 자동 설치 — static-ffmpeg가 받아온 바이너리를 venv 실행폴더에 복사"""
import shutil, sys, os
from pathlib import Path

def main():
    bindir = Path(sys.executable).parent
    ext = ".exe" if os.name == "nt" else ""
    # venv 내장 ffmpeg만 체크 (시스템 brew/PATH 무시) — 배포 컴엔 brew 없으므로 venv에 확실히 설치
    have_ff = (bindir / f"ffmpeg{ext}").exists()
    have_fp = (bindir / f"ffprobe{ext}").exists()
    if have_ff and have_fp:
        print("[OK] ffmpeg / ffprobe 이미 준비됨")
        return
    print("[..] ffmpeg 내려받는 중 (1~2분, 한 번만)")
    from static_ffmpeg import run
    ff, fp = run.get_or_fetch_platform_executables_else_raise()
    for src, name in [(ff, "ffmpeg"), (fp, "ffprobe")]:
        dst = bindir / (name + ext)
        if not dst.exists():
            shutil.copy2(src, dst)
            try: os.chmod(dst, 0o755)
            except Exception: pass
    print(f"[OK] ffmpeg 설치 완료 → {bindir}")

if __name__ == "__main__":
    main()
