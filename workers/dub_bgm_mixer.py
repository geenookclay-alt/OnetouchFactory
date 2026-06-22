"""더빙 메뉴 BGM 믹서 — Bon Jovi Runaway 고정 trim + (옵션) SFX 매핑 → mp3 출력.

대표님 룰:
- BGM 고정: data/bgm_fixed/bonjovi_runaway.mp3
- 영상 길이만큼 trim
- 하이라이트에 맞춰 BGM 시작점 자동 (PoC는 단순 30초~)
- SFX는 풀 받은 후 통합 (지금은 BGM만)
"""
import asyncio
import subprocess
from pathlib import Path


BGM_FIXED = Path(__file__).resolve().parent.parent / "data" / "bgm_fixed" / "bonjovi_runaway.mp3"
BGM_START_SEC = 30.0  # 인트로 건너뛰고 후렴 진입 부근 (수동 큐, 추후 자동화)


async def make_dub_bgm_mix(video_dur: float, out_mp3: Path,
                              bgm_start_sec: float | None = None,
                              video_path: Path | None = None,
                              cctv_pad_sec: float = 2.0) -> dict:
    """더빙용 BGM 믹스 mp3 생성.

    video_dur: 영상 길이 (초)
    out_mp3: 출력 mp3 경로
    bgm_start_sec: BGM trim 시작점 (None이면 BGM_START_SEC 사용)
    video_path: 있으면 영상 분석 → SFX 자동 매핑 추가
    cctv_pad_sec: 영상 끝에 CCTV 컷 붙일 길이 (그만큼 BGM도 더 길게). 기본 2.0초.

    v3: Runaway trim + SFX + 영상 끝 CCTV 2초까지 BGM 깔리게 +pad.
    """
    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    start = float(bgm_start_sec if bgm_start_sec is not None else BGM_START_SEC)
    # total = 영상 + CCTV 컷 길이. BGM/fade-out 다 이 기준.
    total_dur = float(video_dur) + float(cctv_pad_sec)
    end = start + total_dur

    # SFX 매핑 (영상 있으면)
    sfx_matches = []
    if video_path:
        try:
            from workers.bgm_for_subtitle import attach_bgm_mix as _abm  # noqa
            from workers.bgm_sfx_selector import load_meta, match_sfx_multi
            from workers.auto_subtitle import (
                call_gemini, upload_video_to_gemini, ensure_inline_video,
                GEMINI_PRO_MODEL,
            )
            meta = load_meta()
            if meta:
                prompt = """이 영상의 SFX 매핑용. JSON만:
{"sfx_points":[{"time_sec":0.0,"action":"한줄","sfx_keyword":"팝/두둥탁/와우/짝/와장창/퍽/지이잉/오우예/멈춰 등"}]}
규칙: 5~8개. 액션 시점 정확."""
                inline = await ensure_inline_video(Path(video_path))
                file_uri = await upload_video_to_gemini(inline)
                data = await call_gemini(GEMINI_PRO_MODEL, file_uri, prompt,
                                          temperature=0.2)
                if isinstance(data, dict):
                    sfx_matches = match_sfx_multi(
                        data.get("sfx_points", []) or [], meta, limit=8)
        except Exception as e:
            print(f"  ⚠️ SFX 매핑 실패 (BGM만 진행): {e}", flush=True)

    # ffmpeg 구성: BGM (Runaway trim, 영상+CCTV pad만큼) + SFX
    # fade-out 시점은 total_dur 기준 (영상 끝 CCTV 컷까지 BGM 들리고 마지막 0.3초만 fade)
    inputs = ["-i", str(BGM_FIXED.absolute())]
    filter_parts = [
        f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d=0.3,afade=t=out:st={total_dur-0.3:.2f}:d=0.3,"
        f"volume=0.40[bgm]"  # BGM 0.40 (SFX 0.55) — 효과음이 잘 들리게
    ]
    mix_inputs = ["[bgm]"]
    in_idx = 1
    # 반복형 SFX ("뭐였지?뭐였지?뭐였지?") 첫 발성만 자르기
    from workers.bgm_sfx_selector import trim_first_utterance
    for j, sp in enumerate(sfx_matches):
        sp_file = Path(sp.get("file", ""))
        if not sp_file.exists():
            continue
        sec = float(sp.get("sec", 0))
        # SFX는 영상 본편 안에서만 (CCTV 컷에 SFX 안 깖)
        if sec >= video_dur:
            continue
        sp_file = await trim_first_utterance(sp_file)
        inputs += ["-i", str(sp_file.absolute())]
        t_ms = int(sec * 1000)
        filter_parts.append(
            f"[{in_idx}:a]adelay={t_ms}|{t_ms},volume=0.55[s{j}]"
        )
        mix_inputs.append(f"[s{j}]")
        in_idx += 1

    filter_complex = ";".join(filter_parts) + ";" + "".join(mix_inputs) + \
        f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0[aout]"
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "libmp3lame", "-q:a", "2",
        str(out_mp3),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        # 에러 마지막 부분 (실제 원인) 우선 출력
        e_full = err.decode()
        e_tail = e_full[-1500:] if len(e_full) > 1500 else e_full
        raise RuntimeError(f"BGM mix ffmpeg fail: ...{e_tail}")
    return {
        "path": str(out_mp3),
        "bgm_start_sec": start,
        "bgm_end_sec": end,
        "video_dur": video_dur,
        "cctv_pad_sec": cctv_pad_sec,
        "total_dur": total_dur,
        "sfx_count": len(sfx_matches),
    }


if __name__ == "__main__":
    import sys
    async def _cli():
        dur = float(sys.argv[1])
        out = Path(sys.argv[2])
        r = await make_dub_bgm_mix(dur, out)
        print(r)
    asyncio.run(_cli())
