# -*- coding: utf-8 -*-
"""딸깍공장 설정 마법사 — 제미나이 키(자막·더빙 공통 필수) + 타입캐스트 키(더빙 쓰려면 필수)."""
import secrets, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVP = ROOT / ".env"

def ask(label, default=""):
    tip = f" [{default}]" if default else ""
    try:
        v = input(f"{label}{tip}: ").strip()
    except EOFError:
        v = ""
    return v or default

def main():
    print()
    print("=" * 46)
    print("  딸깍공장 설정")
    print("=" * 46)
    if ENVP.exists():
        keep = ask("이미 설정(.env)이 있습니다. 다시 만들까요? (y/N)", "N")
        if keep.lower() != "y":
            print("[OK] 기존 설정 유지")
            return
    print()
    print("① 구글 제미나이 API 키 (필수)")
    print("   발급: https://aistudio.google.com/apikey  (구글 로그인 → Create API key)")
    gem = ask("   키 입력 (AIza로 시작)")
    while not gem:
        print("   ⚠ 키가 없으면 자막·더빙이 동작하지 않습니다.")
        gem = ask("   키 입력 (나중에 넣으려면 skip)")
        if gem == "skip":
            gem = ""
            break
    print()
    print("② 타입캐스트 API 키 (더빙 탭 쓰려면 필수 / 자막만 쓰면 그냥 엔터)")
    print("   넣으면 '더빙' 탭에서 영상을 한국어 음성(TTS)으로 더빙해줍니다.")
    print("   ※ 타입캐스트는 가입 후 크레딧(유료)이 있어야 음성이 생성됩니다.")
    print("   발급: https://typecast.ai → 로그인 → API 키")
    tc = ask("   키 입력 (자막만 쓸 거면 엔터)")

    env = f"""GEMINI_API_KEY={gem}
TYPECAST_API_KEY={tc}
SOLO_MODE=1
ADMIN_USERNAME=owner
ADMIN_PASSWORD={secrets.token_hex(8)}
ADMIN_FULL_NAME=사장님
JWT_SECRET={secrets.token_hex(24)}
BACKEND_API_KEY={secrets.token_hex(16)}
SQLITE_PATH=./db/discover.db
QDRANT_PATH=./qdrant_data
HOST=127.0.0.1
PORT=8000
"""
    ENVP.write_text(env, encoding="utf-8")
    print()
    print(f"[OK] 설정 저장 완료 → {ENVP}")
    if tc:
        print("     타입캐스트 키 등록됨 — 더빙 탭 사용 가능")
    else:
        print("     타입캐스트 키 없음 — 자막 탭만 사용 가능 (더빙 쓰려면 나중에 .env에 추가)")
    print("     로그인 없이 바로 쓰면 됩니다.")

if __name__ == "__main__":
    sys.exit(main())
