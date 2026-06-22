"""BGM/SFX 메타데이터 기반 자동 매칭 + ffmpeg 믹싱 워커.

흐름:
1. 영상 분석 결과 (mood_keywords + sfx_points + duration)를 받음
2. 메타 DB(data/bgm_library_meta.json)에서 가장 fit한 BGM 1곡 + SFX N개 매칭
3. ffmpeg으로 BGM trim + SFX 시점 매핑 → mp3 출력

대표님 룰:
- 제목 매칭 X, 메타 학습된 fits_scenarios/mood/tags 기반.
- SFX 반복형 ("뭐였지?뭐였지?뭐였지?")은 첫 발성만 잘라서 재생.
"""
import hashlib
import json
import random
import subprocess
import asyncio
from pathlib import Path
from typing import Optional


# 모듈 위치 기준 절대경로 — cwd 의존 X (외부 스크립트/워커가 다른 cwd로 호출해도 안전)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
META_PATH = _DATA_DIR / "bgm_library_meta.json"
LIB_ROOT = _DATA_DIR / "bgm_library"
SFX_TRIM_CACHE = _DATA_DIR / "bgm_library_sfx_trimmed"


def _ffmpeg_bin() -> str:
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return "ffmpeg"


def _ffprobe_bin() -> str:
    for p in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"):
        if Path(p).exists():
            return p
    return "ffprobe"


async def trim_first_utterance(
        sfx_path: Path,
        min_dur_trim: float = 2.0,
        silence_thr: str = "-20dB",
        silence_min: float = 0.15,
        edge_pad: float = 0.05) -> Path:
    """SFX가 "뭐였지?뭐였지?뭐였지?" 같이 반복되면 첫 발성만 자르기.

    - min_dur_trim 이하 SFX는 자르지 X (이미 짧음, 자연스럽게 한 번)
    - silencedetect로 첫 무음 구간 시작점 = 첫 발성 끝
    - 너무 늦은 silence (전체의 70%+ 이후) → 한 발성으로 짐작, 자르지 X
    - 너무 빠른 silence (시작 0.2초 이내) → 잡음, 자르지 X
    - trim 결과는 캐시 (같은 SFX는 매번 같은 결과)

    Returns: trim된 캐시 파일 또는 원본 (자를 필요 없으면).
    """
    sfx_path = Path(sfx_path)
    if not sfx_path.exists():
        return sfx_path
    # 1) duration
    try:
        pr = await asyncio.create_subprocess_exec(
            _ffprobe_bin(), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(sfx_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, _ = await pr.communicate()
        dur = float(out_b.decode().strip())
    except Exception:
        return sfx_path
    if dur <= min_dur_trim:
        return sfx_path  # 이미 짧음
    # 2) silencedetect — 첫 silence_start = 첫 발성 끝
    try:
        pr2 = await asyncio.create_subprocess_exec(
            _ffmpeg_bin(), "-hide_banner", "-i", str(sfx_path),
            "-af", f"silencedetect=n={silence_thr}:d={silence_min}",
            "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err_b = await pr2.communicate()
    except Exception:
        return sfx_path
    first_silence = None
    for line in err_b.decode().split("\n"):
        if "silence_start:" in line:
            try:
                first_silence = float(line.split("silence_start:")[1].strip())
                break
            except (ValueError, IndexError):
                continue
    if first_silence is None:
        return sfx_path  # 한 발성으로 짐작
    if first_silence > dur * 0.7 or first_silence < 0.2:
        return sfx_path  # 자를 필요 X (한 발성 or 잡음)
    trim_end = first_silence + edge_pad
    # 3) 캐시
    SFX_TRIM_CACHE.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(
        f"{sfx_path}|{trim_end:.3f}".encode()
    ).hexdigest()[:16]
    cached = SFX_TRIM_CACHE / f"{cache_key}.mp3"
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    # 4) ffmpeg trim → mp3 cache
    try:
        pr3 = await asyncio.create_subprocess_exec(
            _ffmpeg_bin(), "-y", "-hide_banner",
            "-i", str(sfx_path), "-t", f"{trim_end:.3f}",
            "-c:a", "libmp3lame", "-q:a", "2",
            str(cached),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err3_b = await pr3.communicate()
    except Exception:
        return sfx_path
    if pr3.returncode != 0 or not cached.exists() or cached.stat().st_size == 0:
        return sfx_path  # trim fail → 원본
    return cached


def load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # [배포판] _file 절대경로를 이 설치 위치(data/bgm_library/<kind>/) 기준으로 재해석
    lib = META_PATH.parent / "bgm_library"
    for _t, _m in meta.items():
        if not isinstance(_m, dict):
            continue
        _f = _m.get("_file")
        if not _f:
            continue
        _sub = _m.get("_kind", "") or ""
        _cand = lib / _sub / Path(_f).name
        if _cand.exists():
            _m["_file"] = str(_cand)
    return meta


def _score_match(keywords: list[str], tags: list[str]) -> int:
    """키워드 vs 태그 매칭 점수."""
    if not keywords or not tags:
        return 0
    score = 0
    kws = [str(k).lower().strip() for k in keywords if k]
    tgs = [str(t).lower().strip() for t in tags if t]
    for k in kws:
        for t in tgs:
            if k == t:
                score += 3
            elif k in t or t in k:
                score += 1
    return score


def match_bgm(mood_keywords: list[str], meta: dict,
              video_genre_hint: str = "") -> Optional[dict]:
    """mood에 가장 fit한 BGM 1곡 선택. 동점 시 top-5 중 랜덤(다양성)."""
    candidates = []
    for title, m in meta.items():
        if m.get("_kind") != "bgm":
            continue
        tags = (m.get("mood_keywords", []) +
                m.get("tags", []) +
                m.get("fits_scenarios", []) +
                [m.get("genre", ""), m.get("sound_description", "")])
        score = _score_match(mood_keywords, tags)
        if video_genre_hint:
            if str(m.get("genre", "")).lower().find(video_genre_hint.lower()) >= 0:
                score += 2
        if score > 0:
            candidates.append((score, title, m))
    if not candidates:
        # fallback: 랜덤 BGM
        bgms = [(t, m) for t, m in meta.items() if m.get("_kind") == "bgm"]
        if not bgms:
            return None
        t, m = random.choice(bgms)
        return {**m, "_title": t, "_score": 0}
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:5]
    score, title, m = random.choice(top)
    return {**m, "_title": title, "_score": score}


def match_sfx(sfx_keyword: str, meta: dict,
              action_desc: str = "") -> Optional[dict]:
    """SFX 키워드+액션에 fit한 SFX 1개."""
    candidates = []
    base_kws = [sfx_keyword] + (action_desc.split() if action_desc else [])
    for title, m in meta.items():
        if m.get("_kind") != "sfx":
            continue
        tags = (m.get("tags", []) +
                m.get("fits_scenarios", []) +
                m.get("mood_keywords", []) +
                [m.get("sound_description", "")])
        score = _score_match(base_kws, tags)
        if score > 0:
            candidates.append((score, title, m))
    if not candidates:
        # fallback: 랜덤 SFX
        sfxs = [(t, m) for t, m in meta.items() if m.get("_kind") == "sfx"]
        if not sfxs:
            return None
        t, m = random.choice(sfxs)
        return {**m, "_title": t, "_score": 0}
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:3]
    score, title, m = random.choice(top)
    return {**m, "_title": title, "_score": score}


def match_sfx_multi(sfx_points: list[dict], meta: dict,
                     limit: int = 10) -> list[dict]:
    """sfx_points 각각 매칭. 중복 SFX 자제."""
    out = []
    used_titles = set()
    for p in sfx_points[:limit]:
        kw = p.get("sfx_keyword") or p.get("name_or_sound") or ""
        action = p.get("action") or ""
        # 매칭 시 이미 쓴 파일은 약간 감점
        candidates = []
        base_kws = [kw] + (action.split() if action else [])
        for title, m in meta.items():
            if m.get("_kind") != "sfx":
                continue
            tags = (m.get("tags", []) +
                    m.get("fits_scenarios", []) +
                    [m.get("sound_description", "")])
            score = _score_match(base_kws, tags)
            if title in used_titles:
                score -= 5  # 중복 자제
            if score > 0:
                candidates.append((score, title, m))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        top = candidates[:3]
        score, title, m = random.choice(top)
        if title not in used_titles or score > 5:
            used_titles.add(title)
            out.append({
                "sec": p.get("time_sec") or p.get("sec", 0),
                "file": m.get("_file"),
                "title": title,
                "score": score,
                "keyword": kw,
                "action": action,
            })
    return out


async def make_bgm_sfx_mix(
        duration_sec: float,
        bgm_match: Optional[dict],
        sfx_matches: list[dict],
        out_mp3: Path,
        bgm_start_sec: float = 30.0,
        bgm_volume: float = 0.55,
        sfx_volume: float = 0.55,
) -> dict:
    """BGM 1곡 + SFX N개 ffmpeg amix → mp3.

    bgm_match: {"_file": path, ...} 또는 None
    sfx_matches: [{"sec": 3.8, "file": path, ...}, ...]
    """
    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    inputs = []
    filter_parts = []
    mix_inputs = []
    in_idx = 0

    # BGM
    if bgm_match and bgm_match.get("_file"):
        bgm_path = Path(bgm_match["_file"])
        if bgm_path.exists():
            inputs += ["-i", str(bgm_path.absolute())]
            filter_parts.append(
                f"[{in_idx}:a]atrim={bgm_start_sec}:{bgm_start_sec + duration_sec},"
                f"asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d=0.3,afade=t=out:st={duration_sec - 0.3:.2f}:d=0.3,"
                f"volume={bgm_volume}[bgm]"
            )
            mix_inputs.append("[bgm]")
            in_idx += 1

    # SFX (반복형은 첫 발성만 자르기)
    for j, sp in enumerate(sfx_matches):
        sp_file = Path(sp.get("file", ""))
        if not sp_file.exists():
            continue
        sec = float(sp.get("sec", 0))
        if sec >= duration_sec:
            continue
        # "뭐였지?뭐였지?뭐였지?" 같은 반복형 → 첫 발성만
        sp_file = await trim_first_utterance(sp_file)
        inputs += ["-i", str(sp_file.absolute())]
        t_ms = int(sec * 1000)
        filter_parts.append(
            f"[{in_idx}:a]adelay={t_ms}|{t_ms},volume={sfx_volume}[s{j}]"
        )
        mix_inputs.append(f"[s{j}]")
        in_idx += 1

    if not mix_inputs:
        # 빈 mp3 — silence
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={duration_sec}",
            "-c:a", "libmp3lame", "-q:a", "5", str(out_mp3),
        ]
    else:
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
        raise RuntimeError(f"ffmpeg mix fail: {err.decode()[:500]}")
    return {
        "path": str(out_mp3),
        "bgm_title": bgm_match.get("_title") if bgm_match else None,
        "sfx_count": len(sfx_matches),
        "duration": duration_sec,
    }
