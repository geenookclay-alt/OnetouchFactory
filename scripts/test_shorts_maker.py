#!/usr/bin/env python3
"""쇼츠 메이커 v5 — 긴 URL → 멀티 하이라이트(3~7) → 각각 ≤59초 쇼츠 양산.

v5 (2026-05-25):
- 멀티 하이라이트: Pass 1이 3~7개 highlight 식별, 각각 hl_NN/ 폴더로 양산
- 자동 스냅: Pass 2 keep 경계를 silencedetect로 ±1초 윈도우 내 무음 지점에 자동 보정
  → "명화나|이트" 같은 단어 중간 끊김 원천 차단
- Pass 2 프롬프트 강화: 단어/조사 중간 컷 절대 금지 + 클립 길이 명시
- 대사 SRT: Whisper로 final 한국어 받아쓰기 → 03_대사.srt
- source/intermediates 디폴트 보존 (--cleanup-source 옵션으로 강제 삭제)

사용: venv/bin/python scripts/test_shorts_maker.py <url> <out_dir> [--cleanup-source]
"""
import sys, json, asyncio, subprocess, re, shutil
from pathlib import Path

sys.path.insert(0, ".")
from api import database as db
from workers.auto_subtitle import (
    ensure_inline_video, upload_video_to_gemini, call_gemini,
    GEMINI_PRO_MODEL, run_auto_subtitle, _get_gemini_key,
)

def _dur(path):
    o = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(o) if o else 0.0


def _strip_periods_in_srt(srt_path):
    """SRT의 텍스트 라인에서 마침표(.)만 제거. 타임스탬프·번호·물음표·느낌표는 유지."""
    try:
        txt = srt_path.read_text(encoding="utf-8")
    except Exception:
        return
    new_lines = []
    for line in txt.split("\n"):
        s = line.strip()
        # 타임스탬프 라인 (00:00:00,000 --> 00:00:01,000) 그대로
        if "-->" in line:
            new_lines.append(line)
        # cue 번호만 있는 라인 그대로
        elif s.isdigit():
            new_lines.append(line)
        # 빈 줄 그대로
        elif not s:
            new_lines.append(line)
        # 텍스트 라인 — 마침표 제거 (...도 같이 사라짐)
        else:
            new_lines.append(line.replace(".", ""))
    srt_path.write_text("\n".join(new_lines), encoding="utf-8")


def _parse_json(raw):
    if isinstance(raw, dict):
        return raw
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", s)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


def parse_vtt(path):
    raw = Path(path).read_text(encoding="utf-8")
    pat = re.compile(
        r"(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\n\d+:\d+:|\Z)",
        re.MULTILINE)

    def to_sec(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    segs = []
    for m in pat.finditer(raw):
        s, e = to_sec(m.group(1)), to_sec(m.group(2))
        txt = re.sub(r"<[^>]+>", "", m.group(3)).strip().replace("\n", " ")
        txt = re.sub(r"\s+", " ", txt)
        if txt:
            segs.append({"start": s, "end": e, "text": txt})
    out = []
    for s in segs:
        if out and out[-1]["text"] == s["text"]:
            out[-1]["end"] = s["end"]
        else:
            out.append(s)
    return out


async def gemini_text(prompt):
    from google import genai
    client = genai.Client(api_key=_get_gemini_key())
    r = await asyncio.to_thread(
        lambda: client.models.generate_content(model=GEMINI_PRO_MODEL, contents=prompt)
    )
    return r.text


def detect_silences(path, threshold_db=-35, min_silence=0.15):
    r = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af",
         f"silencedetect=noise={threshold_db}dB:d={min_silence}",
         "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def speech_ranges(total_dur, silences):
    ranges, cur = [], 0.0
    for s, e in silences:
        if s > cur:
            ranges.append((cur, s))
        cur = e
    if cur < total_dur:
        ranges.append((cur, total_dur))
    return [(a, b) for a, b in ranges if b - a >= 0.3]


def snap_to_silence(t, silences, clip_dur, kind, window=1.0):
    """Gemini keep 시각을 ±window 안에서 가장 가까운 무음 지점에 스냅.
    kind="start": 무음 끝(=발화 시작)에 스냅. "end": 무음 시작(=발화 끝)에 스냅.
    윈도우 내 무음 없으면 원래값 유지."""
    if not silences:
        return max(0.0, min(t, clip_dur))
    candidates = []
    for s, e in silences:
        if kind == "start":
            # 무음 끝 = 발화 시작 직전 (이 시각으로 스냅하면 발화 시작에서 깨끗)
            if abs(e - t) <= window:
                candidates.append((abs(e - t), e))
            # 무음 시작에 스냅하면 직전 발화 끝에서 시작 (덜 좋음, 보조)
            if abs(s - t) <= window:
                candidates.append((abs(s - t) + 0.3, s))
        else:  # end
            # 무음 시작 = 발화 끝 직후 (이 시각으로 스냅하면 발화 끝에서 깨끗)
            if abs(s - t) <= window:
                candidates.append((abs(s - t), s))
            if abs(e - t) <= window:
                candidates.append((abs(e - t) + 0.3, e))
    if not candidates:
        return max(0.0, min(t, clip_dur))
    candidates.sort()
    return max(0.0, min(candidates[0][1], clip_dur))


def align_situation_to_jumps(srt_path, jumps, delay, video_dur):
    """❌ 비활성화·쓰지 말 것 (2026-06-06). 비례매핑이 자막 ts를 망가뜨리는 버그
    (cue 1초/17초 불균등·20~30초 늘어뜨림 → 화면과 어긋남, handoff_2026_06_05).
    folktale 해설은 Gemini가 영상 보고 박은 시각 그대로가 정답 → 호출부 제거됨. 다시 부르지 말 것."""
    from workers.audio_sync import parse_srt, write_srt
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return 0
    cues = parse_srt(srt_path)
    if not cues or not jumps:
        return 0
    video_dur = float(video_dur or 0)
    if video_dur <= 1:
        video_dur = max((c["end"] for c in cues), default=10.0)
    # 앵커 = 각 장면이 화면에 나온 뒤 delay초 (첫 장면은 0초 = 도입 TTS 후크)
    anchors = []
    for k, j in enumerate(jumps):
        at = float(j.get("at", 0) or 0)
        anchors.append(0.0 if k == 0 else round(at + delay, 2))
    anchors = sorted({a for a in anchors if a < video_dur - 0.3}) or [0.0]
    M, N = len(cues), len(anchors)
    # 해설을 장면 앵커에 순서대로 매핑(개수 같으면 1:1, 다르면 비례) — 단조 증가 보장
    for i, c in enumerate(cues):
        ai = 0 if M <= 1 else round(i * (N - 1) / (M - 1))
        c["start"] = anchors[min(ai, N - 1)]
    for i in range(1, M):
        if cues[i]["start"] <= cues[i - 1]["start"]:
            cues[i]["start"] = round(min(cues[i - 1]["start"] + 1.3, video_dur - 0.4), 2)
    # end = 다음 해설 직전까지(연속, 빈 구간 X), 마지막은 영상 끝(여운)
    for i in range(M):
        nxt = cues[i + 1]["start"] - 0.05 if i + 1 < M else video_dur
        cues[i]["end"] = round(max(nxt, cues[i]["start"] + 0.4), 2)
    write_srt(cues, srt_path)
    return M


def _despread_tail_srt(srt_path, min_dur=1.3):
    """[동화 전용] 결말 과밀로 짧게 눌린 자막(align 끝 0.4초 쌓임) → 뒤에서 역방향으로 최소 가독시간 확보.
    align_situation_to_jumps가 _jumps 앵커에 매핑하면 결말 클라이맥스(앵커 촘촘)에서 끝 자막이 0.4초로 눌림 → 앞으로 펼침."""
    from workers.audio_sync import parse_srt, write_srt
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return
    cues = parse_srt(srt_path)
    if len(cues) < 2:
        return
    n = 0
    for i in range(len(cues) - 1, -1, -1):
        if cues[i]["end"] - cues[i]["start"] < min_dur - 1e-3:
            cues[i]["start"] = max(0.0, round(cues[i]["end"] - min_dur, 3))
            n += 1
        if i > 0 and cues[i - 1]["end"] > cues[i]["start"]:
            cues[i - 1]["end"] = round(cues[i]["start"], 3)
    if n:
        write_srt(cues, str(srt_path))
        print(f"  결말몰림 분산: {srt_path.name} {n}개 ≥{min_dur}s 확보", flush=True)


def cut_concat(src, ranges, out_path, work_dir, prefix):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더 보장 (없으면 ffmpeg 에러 → 점프컷 통째 실패)
    pieces = []
    for i, (a, b) in enumerate(ranges):
        p = work_dir / f"{prefix}_{i:03d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(a), "-to", str(b), "-i", str(src),
            # 정규화: fps/짝수픽셀/오디오채널/timebase/ts 통일 → concat 안 튐
            "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-video_track_timescale", "30000",
            "-avoid_negative_ts", "make_zero",
            str(p)
        ], check=True, capture_output=True)
        pieces.append(p)
    list_txt = work_dir / f"{prefix}_list.txt"
    list_txt.write_text("\n".join(f"file '{p.absolute()}'" for p in pieces))
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt),
        "-c", "copy", str(out_path)
    ], check=True, capture_output=True)
    return pieces


def detect_scene_cuts(path, threshold=0.35):
    """ffmpeg scene change 검출 → 장면 전환 타임스탬프 리스트. 애니 컷 편집점용.
    이 지점들이 '자연스러운 컷 경계'라 여기서 자르면 흐름이 안 깨짐."""
    import re
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(path), "-vf",
             f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=180)
        return sorted(float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr or ""))
    except Exception:
        return []


def _pyscene_cuts(video_path, threshold=27.0):
    """PySceneDetect로 장면 전환(컷) 시각 리스트(초). 짧은 클립용(빠름). 대표님: 컷 튐 방지에 활용."""
    try:
        from scenedetect import detect, ContentDetector
        sl = detect(str(video_path), ContentDetector(threshold=threshold))
        return [sc[0].seconds for sc in sl]
    except Exception as e:
        print(f"  ⚠️ PySceneDetect 실패: {str(e)[:60]}", flush=True)
        return []


async def _gen_folktale_subs_chunked(final_mp4, hl_dir, base_prompt, chunk=55.0):
    """🔴근본(2026-06-06, CLI folktale_finalize._gen_subs_chunked와 동일): 긴 영상은 Gemini가 자막
    시각을 흘려 뒤로 갈수록 어긋난다(대표님 지적). ~55초 구간(장면전환 경계)으로 나눠 각 구간 자막을
    따로 생성 후 시각 오프셋 합침 = 어떤 길이든 화면과 일치. 01_상황설명만 재생성(덮어씀)."""
    from workers.audio_sync import parse_srt as _ps, write_srt as _ws
    dur = _dur(final_mp4)
    try:
        cuts = _pyscene_cuts(final_mp4)
    except Exception:
        cuts = []
    bounds = [0.0]
    while bounds[-1] + chunk < dur - 12:
        target = bounds[-1] + chunk
        cand = [c for c in cuts if bounds[-1] + 25 < c < bounds[-1] + chunk + 15]
        bounds.append(round(min(cand, key=lambda c: abs(c - target)) if cand else target, 2))
    bounds.append(round(dur, 2))
    if len(bounds) <= 2:
        print(f"  (folktale 자막: {dur:.0f}초 → 1구간, 단일생성 유지)", flush=True)
        return False
    all_cues = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        clip = hl_dir / f"_subchunk{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-to", f"{b:.2f}",
                        "-i", str(final_mp4), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", str(clip)], check=True)
        with db.get_db() as conn:
            cur = conn.execute("INSERT INTO subtitle_jobs (video_filename, video_path, style, status, progress) "
                               "VALUES (?,?,?,'pending',0)", (clip.name, str(clip.absolute()), "shorts_maker:folktale"))
            sid = cur.lastrowid
        n_sub = max(4, round((b - a) / 6.5))
        chunk_prompt = base_prompt + (
            f"\n\n🔴🔴최우선 개수 규칙: 이 영상은 {b-a:.0f}초다. situation_subtitles를 "
            f"**정확히 {n_sub}개 내외**로만 만들어라(한 자막이 6~7초 떠 있게). 절대 짧게 많이 쪼개지 마라.")
        await run_auto_subtitle(sid, clip, prompt_override=chunk_prompt)
        srt = Path(f"data/subtitles/job_{sid}/01_상황설명.srt")
        if srt.exists():
            cs = _ps(srt)
            for c in cs:
                c["start"] = round(c["start"] + a, 3); c["end"] = round(c["end"] + a, 3)
            all_cues += cs
        clip.unlink(missing_ok=True)
    if all_cues:
        _ws(all_cues, str(hl_dir / "01_상황설명.srt"))
        print(f"  ✅ folktale 자막 {len(bounds)-1}개 구간 재생성 → {len(all_cues)}개 (각 구간 시각 정확·뒤로 안 밀림)", flush=True)
        return True
    return False


async def _extract_program_name(title):
    """유튜브 제목 → 프로그램(작품) 이름 한 줄. Gemini로 뽑고 실패 시 정규식 fallback."""
    import re as _re
    title = (title or "").strip()
    if not title:
        return ""
    try:
        r = await gemini_text(
            "다음 유튜브 영상 제목에서 '프로그램(작품/방송) 이름'만 한 줄로 뽑아라. "
            "회차·부제·방송일·채널명·해시태그·이모지·특수문자는 빼고 작품명만. "
            "예: '🧚 은비까비의 옛날옛적에 2 | 13회 ⭐산부새와 섯하니 | 19920710KBS방송 #만화동산' → '은비까비 옛날옛적에'. "
            f"제목: {title}\n[출력: 작품명 한 줄만. 따옴표·설명·접두어 없이]")
        name = (r or "").strip().strip('"').strip("'").splitlines()[0].strip() if r else ""
        if 1 <= len(name) <= 30:
            return name
    except Exception as _e:
        print(f"  ⚠️ 프로그램명 Gemini 추출 실패 → 정규식: {str(_e)[:60]}", flush=True)
    t = _re.sub(r"#\S+", "", title)
    head = t.split("|")[0]
    head = _re.sub(r"[^가-힣A-Za-z0-9 ]", " ", head)
    name = _re.sub(r"\s+", " ", head).strip()
    name = _re.sub(r"\s+\d+\s*$", "", name).strip()
    return name[:30]


def _fmt_srt_ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def _write_source_srt(results, title):
    """🔴 출처 자막 (대표님 0606): 쇼츠메이커 전 출력에 '출처 : 프로그램명' SRT를 영상 전체 길이로 1개 생성.
    별도 SRT(05_출처.srt)로만 — 캡컷 후처리용(대표님: srt로 그냥 만들어줘)."""
    prog = await _extract_program_name(title)
    if not prog:
        print("  ⚠️ 출처 프로그램명 비어있음 — 출처 자막 생략", flush=True)
        return
    n = 0
    for rr in results or []:
        if not isinstance(rr, dict) or rr.get("error") or not rr.get("dir"):
            continue
        hd = Path(rr["dir"]); fm = hd / "final.mp4"
        dur = _dur(fm) if fm.exists() else 0.0
        if dur <= 0:
            dur = 180.0
        (hd / "05_출처.srt").write_text(
            f"1\n00:00:00,000 --> {_fmt_srt_ts(dur)}\n출처 : {prog}\n", encoding="utf-8")
        n += 1
    print(f"  ✅ 출처 자막 {n}개 생성: '출처 : {prog}'", flush=True)


def _snap_clean(src_path, t, rn, lo, hi):
    """t를 [t+lo, t+hi] 범위에서 가장 가까운 **장면전환(PySceneDetect)**에 스냅, 없으면 **무음(RMS 최저)**에.
    영화 클립 시작/끝이 장면·대사 중간에서 안 끊기게(컷 튐 방지)."""
    import os
    import numpy as np
    a = max(0.0, t + lo)
    b = t + hi
    if b <= a + 0.5:
        return t
    clip = Path(src_path).parent / f"_snap_{int(t)}.mp4"
    cuts = []
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(a), "-to", str(b),
                        "-i", str(src_path), "-an", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "32", "-vf", "scale=480:-2", str(clip)], check=True)
        cuts = [a + c for c in _pyscene_cuts(str(clip)) if 0.15 < c < (b - a - 0.15)]
    except Exception:
        pass
    finally:
        try:
            os.remove(clip)
        except OSError:
            pass
    if cuts:                                  # 장면전환 우선
        return min(cuts, key=lambda c: abs(c - t))
    seg = rn[int(a):int(b)]                    # 폴백: 무음(RMS 최저점)
    if len(seg):
        return float(int(a) + int(np.argmin(seg)))
    return t


def _discover_movie_signal(src_path, cues, n=8, region=58, sep=90):
    """영화/드라마 명장면 신호기반 발굴 (밤샘 movie_engine 이식).

    대표님 발굴엔진: **음량 RMS(드라마틱)** + **자막밀도** 로 top-N 비겹침 명장면 윈도우.
    통합 경로엔 이미 cues(자막)가 있으니 OCR 대신 cue 밀도로 자막밀도 계산(빠름).
    Gemini가 자막 텍스트로 '명장면 추측'하던 걸 → 실제 신호(소리 큰+대사 빽빽한 구간)로 교체.
    + 윈도우 경계를 장면전환(PySceneDetect)+무음에 스냅해 컷이 안 튀게.
    Returns [(start, end), ...] (초). region=편당 길이(≤59초).
    """
    import wave
    import numpy as np
    W = Path(src_path).parent / "_sigwork"
    W.mkdir(exist_ok=True)
    wav = str(W / "sig.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src_path),
                        "-ac", "1", "-ar", "8000", wav], check=True)
        wf = wave.open(wav)
        a = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(float)
        wf.close()
        sr = 8000
        T = max(1, len(a) // sr)
        # 1초 단위 음량 RMS → 0~1 정규화
        rms = np.array([np.sqrt((a[i * sr:(i + 1) * sr] ** 2).mean() + 1) for i in range(T)])
        rn = (rms - rms.min()) / (rms.max() - rms.min() + 1)
        # 자막밀도: 그 초에 대사 cue가 떠 있으면 1 (cue 트랙 기반)
        dens = np.zeros(T)
        for c in cues:
            s = int(c.get("start", 0))
            e = int(c.get("end", c.get("start", 0) + 2))
            for t in range(max(0, s), min(T, e + 1)):
                dens[t] = 1.0
        # top-N 비겹침 윈도우 (대사밀도 × 음량 강도·역동성)
        lo = min(int(T * 0.06), 90)             # 인트로(로고·타이틀)만 스킵 — 긴 영화 초반 명장면 보존
        hi = max(lo + 1, int(T * 0.94) - region)  # 엔딩 크레딧 스킵(6%)
        cand = []
        for s in range(lo, hi, 8):
            e = s + region
            d = dens[s:e].mean()
            if d < 0.35:        # 대사 빈약 구간(인트로·풍경·액션 무대사) 제외
                continue
            score = d * (0.6 + rn[s:e].std() + rn[s:e].mean() * 0.5)
            cand.append((score, s, e))
        cand.sort(reverse=True)
        pick = []
        for sc, s, e in cand:
            if all(abs(s - ps) >= sep for ps, _ in pick):
                pick.append((s, e))
            if len(pick) >= n:
                break
        if not pick:            # 폴백: 음량+밀도 최고 단일 구간
            best = None
            for s in range(lo, hi, 8):
                e = s + region
                sc = dens[s:e].mean() + rn[s:e].mean()
                if best is None or sc > best[0]:
                    best = (sc, s, e)
            if best:
                pick = [(best[1], best[2])]
        pick.sort()
        # 경계 스냅 — 장면전환(PySceneDetect)+무음에 맞춰 클립 시작/끝이 안 튀게 (대표님)
        snapped = []
        for s, e in pick:
            ss = _snap_clean(src_path, float(s), rn, -5, 3)   # 시작: 앞쪽 장면 시작에
            ee = _snap_clean(src_path, float(e), rn, -3, 5)   # 끝: 뒤쪽 장면 끝에
            ee = min(ee, ss + region + 3)                     # 너무 길어지지 않게
            if ee - ss < region * 0.55:                       # 스냅이 과하면 원복
                ss, ee = float(s), float(e)
            snapped.append((int(round(ss)), int(round(ee))))
        return snapped
    finally:
        shutil.rmtree(W, ignore_errors=True)


async def yt_subs_only(url, out_tpl):
    def _run():
        subprocess.run([
            "venv/bin/yt-dlp", "--skip-download",
            "--write-auto-subs", "--sub-lang", "ko",
            "--convert-subs", "vtt",
            "-o", str(out_tpl), url
        ], check=True, capture_output=True)
    await asyncio.to_thread(_run)


async def yt_source(url, out_tpl):
    def _run():
        subprocess.run([
            "venv/bin/yt-dlp",
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", str(out_tpl), url
        ], check=True)
    await asyncio.to_thread(_run)


PROMPT_PASS1_TPL = """다음은 한국 영상의 자동자막(타임스탬프 + 대사)이야. 한국 쇼츠로 만들 **재밌는 하이라이트 단락**들을 다 골라줘.

조건:
- 각 하이라이트는 **하나의 코너·빌드업·빵 터지는 단락 전체**를 통째로 (몽타주 X, 흐름이 이어지는 한 단위)
- 각 길이 약 90~180초 (이후 무음제거+문맥컷으로 ≤59초 쇼츠로 축약 예정)
- 도입·인사·잡담·광고 X
- 진짜 빵 터지는 펀치라인·폭로·충격 발언·케미 폭발·디스전·반전 위주
- **3~7개** 골라줘 (영상에서 진짜 재밌는 단락 다, 없으면 적게)
- 서로 겹치지 않게, 시간순

각 하이라이트마다 **type** 필드도 분류해줘:
- "talk" = 토크쇼·인터뷰·예능 대화 중심 (사람 얼굴 + 대화가 핵심. 짠한형/유퀴즈/뜬뜬 등)
- "info" = 정보·리액션·액션 중심 (사물/현장/액션이 핵심. 1분기악·뉴스·다큐·실험·사출좌석 등)
- "mixed" = 둘 다 (예능인데 액션도 많거나, 정보 영상인데 토크도 깊이 있음)
→ 토크쇼/인터뷰면 무조건 "talk", 애매하면 "mixed" 줘.

[출력 JSON만]
{{"segments":[
  {{"start": 234.5, "end": 356.0, "type": "talk", "reason": "왜 골랐는지 한 줄"}},
  {{"start": 1234.5, "end": 1394.0, "type": "talk", "reason": "..."}}
]}}
숫자는 영상 내 초(소수 1자리). JSON 외 텍스트 절대 X.

[자막 transcript]
{transcript}
"""


def PROMPT_PASS2_TPL(clip_dur):
    return f"""이 클립은 한국 예능 하이라이트(약 {clip_dur:.1f}초, 무음 빠진 상태)야. ≤59초 쇼츠로 줄여야 해. **흐름·맥락 유지**하면서 버려도 되는 부분만 잘라.

🔴 **절대 금지**:
- 단어 한가운데 자르지 마 ("명화나|이트" 같이 단어 잘리면 안 됨)
- 조사 중간·호흡 중간 자르지 마
- 컷 경계는 **완전한 발화 단위 끝** (문장·리액션·웃음·박수 끝난 직후)에서만
- **이 클립은 총 {clip_dur:.1f}초까지만 존재** — start/end가 그 이내여야 함

✅ 조건:
- 유지(keep)할 구간들의 [start, end] 리스트. 시간순.
- 빌드업·펀치라인·웃긴 리액션은 무조건 살림.
- 잡담·반복·곁가지·헤매는 부분·진행과 무관한 끼어듦 컷.
- 합계 목표 **40~58초** (절대 59 넘기지 마).

[출력 JSON만]
{{"keep":[{{"start":0.0,"end":12.5}}],"reason":"..."}}
숫자는 클립 내 초(소수 1자리). JSON 외 텍스트 절대 X.
"""


def PROMPT_DRAMA_CORE_TPL(clip_dur):
    return f"""이 영상은 드라마의 한 단락(약 {clip_dur:.1f}초)이야. 이 단락에서 **가장 임팩트 있는 핵심 연속 구간**을 ≤58초로 타이트하게 골라줘.

🎯 타이트하게 (루즈 X):
- 늘어지는 도입·반복·뜸들이기·정적인 부분 **과감히 버려라**.
- 핵심 사건/대사/리액션/반전이 **바로 시작되는** 지점부터.
- 시청자가 1초도 지루하지 않게 — 임팩트 밀도 최대.
- 40~58초 (짧아도 됨, 핵심만 있으면 35초도 OK. 절대 59 넘기지 마).

룰:
- **연속된 한 구간** (start~end). 중간 잘라 이어붙이기 X — 흐름 보존.
- 컷 경계는 발화 단위 끝 (말 중간 X).
- start/end는 0~{clip_dur:.1f}초 이내.

[출력 JSON만]
{{{{"start": 12.0, "end": 60.0, "reason": "핵심 사건 — 바로 본론"}}}}
숫자는 초(소수 1자리). JSON 외 텍스트 X.
"""


async def whisper_srt(audio_or_video, out_dir, basename):
    """대사 STT — 한국어는 ElevenLabs scribe_v2 단독(정확한 단어 ms 타임스탬프 보존).
    비한국어(일/영)만 Gemini Pro 합의. 🔴한국어 Gemini 합의 금지: Gemini가 영상 보며
    타임스탬프를 눈대중으로 다시 박아 ElevenLabs 정확한 시각을 덮음 → 대사 미묘하게 어긋남
    (2026-06-06 대표님 지적). force_validation=False면 한국어만 단독, 비한국어는 그대로 합의.
    """
    from workers.multilang_stt import transcribe_with_validation
    out_srt = out_dir / basename
    await transcribe_with_validation(audio_or_video, out_srt, force_validation=False)
    return out_srt


def _ocr_dialogue_srt(video_path, out_srt):
    """대사 SRT = 화면 **한국어 하드섭 OCR**(Vision). 애니는 음성=일본어라 Whisper가 일본어로 받아씀 →
    화면에 박힌 한국어 자막을 OCR해야 한국어로 정확. (영화도 하드섭 OCR이 더 정확). make_card 방식."""
    import os, glob, difflib
    root = Path(__file__).resolve().parent.parent
    ocrbin = next((str(p) for p in [root / "engine" / "ocrbin", Path("/tmp/ocrbin")] if Path(p).exists()), None)
    if not ocrbin:
        print("  ⚠️ ocrbin 없음 — 대사 OCR 스킵", flush=True)
        return 0
    FR = Path(out_srt).parent / "_dlgfr"
    FR.mkdir(parents=True, exist_ok=True)
    try:
        for f in glob.glob(str(FR / "*.png")):
            os.remove(f)
        # 자막이 상단/중앙하단 어디 있든 잡게 — 상단 28% + 하단 50%를 위아래로 합쳐 OCR(중앙 액션만 제외=노이즈↓)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                        "-filter_complex",
                        "[0:v]fps=5,split=2[a][b];[a]crop=iw:ih*0.28:0:0[t];"
                        "[b]crop=iw:ih*0.50:0:ih*0.50[d];[t][d]vstack=inputs=2,scale=1100:-1[out]",
                        "-map", "[out]",
                        str(FR / "f%05d.png")], check=True)
        frames = sorted(glob.glob(str(FR / "f*.png")))
        res = subprocess.run([ocrbin] + frames, capture_output=True, text=True)
        ocr = {Path(l.split("\t")[0]).name: l.split("\t", 1)[1].strip()
               for l in res.stdout.splitlines() if "\t" in l}
        def norm(t): return re.sub(r"\s+", "", t)
        def clean(t): return re.sub(r"^[\s•·.\-!,~]+", "", t).strip()
        def sim(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
        texts = [clean(ocr.get(Path(f).name, "")) for f in frames]
        FPS = 5
        dcues = []
        i = 0
        N = len(texts)
        while i < N:
            t = texts[i]
            if not norm(t) or len(norm(t)) < 2:
                i += 1
                continue
            j = i
            gap = 0
            while j + 1 < N:
                nx = texts[j + 1]
                if norm(nx) and (norm(nx) == norm(t) or sim(nx, t) > 0.8):
                    if len(nx) > len(t):
                        t = nx
                    j += 1
                    gap = 0
                elif not norm(nx) and gap < 1:
                    j += 1
                    gap += 1
                else:
                    break
            end = j
            while end > i and not norm(texts[end]):
                end -= 1
            s, e = i / FPS, (end + 1) / FPS
            if e - s >= 0.3:
                dcues.append([round(s, 2), round(e, 2), t])
            i = j + 1
        mg = []
        for s, e, t in dcues:
            if mg and s - mg[-1][1] < 1.0 and (sim(t, mg[-1][2]) > 0.8 or norm(t) in norm(mg[-1][2]) or norm(mg[-1][2]) in norm(t)):
                mg[-1][1] = e
                if len(t) > len(mg[-1][2]):
                    mg[-1][2] = t
            else:
                mg.append([s, e, t])
        dcues = [c for c in mg if len(norm(c[2])) >= 2]
        def ts(x): return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(x % 1 * 1000):03d}"
        with open(out_srt, "w", encoding="utf-8") as f:
            for k, (s, e, t) in enumerate(dcues, 1):
                f.write(f"{k}\n{ts(s)} --> {ts(e)}\n{t}\n\n")
        return len(dcues)
    finally:
        shutil.rmtree(FR, ignore_errors=True)


async def process_highlight(idx, seg, src, OUT, hl_root, pipeline_type: str = "highlight",
                            on_progress=None):
    """하이라이트 1개 → ≤59초 쇼츠 + 자막·메타.

    pipeline_type: "highlight" / "drama" — 자막 prompt 분기 (workers/shorts_subtitle).
    on_progress(pct, msg): 제작 단계별 진행률 콜백 (rendering % 실시간 표시용).
    """
    def _pr(pct, msg):
        if on_progress:
            try:
                on_progress(pct, msg)
            except Exception:
                pass
    s0, e0 = float(seg["start"]), float(seg["end"])
    clip_orig_start = s0  # final 첫 프레임의 원본 절대시각 (도입 앞부분 추출용). 컷마다 갱신.
    hl_dir = hl_root / f"hl_{idx:02d}"
    hl_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- [{idx}] {s0:.1f}~{e0:.1f} ({e0-s0:.1f}s) | {seg.get('reason','')[:80]} ---", flush=True)
    _pr(8, "영상 컷")

    # a. 통짜 컷 (애니 anime_hl: 이미 비트컷된 pre-cut 클립이면 그대로 사용 — 컷은 발굴이 끝냄)
    hl_raw = hl_dir / "highlight_raw.mp4"
    _precut = seg.get("_precut")
    if _precut and Path(_precut).exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(_precut),
            # 정규화
            "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-video_track_timescale", "30000",
            "-avoid_negative_ts", "make_zero",
            str(hl_raw)
        ], check=True, capture_output=True)
        print(f"  애니 pre-cut 사용 ({_dur(hl_raw):.1f}s, anime_hl 비트컷)", flush=True)
    else:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(s0), "-to", str(e0), "-i", str(src),
            # 정규화
            "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-video_track_timescale", "30000",
            "-avoid_negative_ts", "make_zero",
            str(hl_raw)
        ], check=True, capture_output=True)

    # b. 무음 제거 — drama/anime/movie는 침묵도 흐름이라 skip (점프 방지)
    raw_dur = _dur(hl_raw)
    _flow_types = ("drama", "anime", "movie", "folktale")
    if pipeline_type in _flow_types:
        hl = hl_dir / "highlight_no_silence.mp4"
        shutil.copy(hl_raw, hl)
        hl_dur = raw_dur
        print(f"  {pipeline_type} 모드 — 무음 제거 skip ({raw_dur:.1f}s 그대로)", flush=True)
    else:
        sils_raw = detect_silences(hl_raw, threshold_db=-32, min_silence=0.4)
        speech = speech_ranges(raw_dur, sils_raw)
        hl = hl_dir / "highlight_no_silence.mp4"
        seg_dir = hl_dir / "_segs"
        seg_dir.mkdir(exist_ok=True)
        cut_concat(hl_raw, speech, hl, seg_dir, prefix="sp")
        hl_dur = _dur(hl)
        print(f"  무음제거: {raw_dur:.1f}→{hl_dur:.1f}s", flush=True)

    # c. Pass 2: drama/anime/movie는 단락 핵심 연속 추출 (앞 자르기 X)
    if pipeline_type in _flow_types:
        final = hl_dir / "final.mp4"
        # v3 반전 배치 — seg.twist_sec(절대초) 있으면 twist가 클립의 ~78%에 오게
        # 앞부분만 조정 (뒤=결말은 보존). twist가 이미 후반이면 그대로.
        twist_abs = seg.get("twist_sec")
        if hl_dur <= 59.5 and twist_abs is not None:
            try:
                twist_rel = float(twist_abs) - s0  # clip 내 상대 위치
                if 0 < twist_rel < hl_dur:
                    ratio = twist_rel / hl_dur
                    # twist가 너무 앞(< 65%)이면 앞을 잘라 78%로 당김
                    if ratio < 0.65:
                        target = 0.78
                        new_dur = twist_rel / target
                        trim_front = hl_dur - new_dur
                        if trim_front > 1.0:
                            sils_hl = detect_silences(hl, threshold_db=-35, min_silence=0.15)
                            cs = snap_to_silence(trim_front, sils_hl, hl_dur, kind="start", window=1.5)
                            if hl_dur - cs >= 25:  # 최소 25초 보장
                                tmp = hl_dir / "_twist_cut.mp4"
                                subprocess.run([
                                    "ffmpeg", "-y", "-ss", str(cs), "-i", str(hl),
                                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                    "-c:a", "aac", "-b:a", "192k", str(tmp)
                                ], check=True, capture_output=True)
                                shutil.move(str(tmp), str(hl))
                                hl_dur = _dur(hl)
                                clip_orig_start += cs  # 앞 cs초 잘림 → 원본 시작점 이동
                                print(f"  반전 배치: 앞 {cs:.1f}초 컷 → twist {target*100:.0f}% (now {hl_dur:.1f}s)", flush=True)
            except Exception as _e:
                print(f"  ⚠️ 반전 배치 skip: {str(_e)[:80]}", flush=True)
        if pipeline_type == "folktale" or hl_dur <= 59.5:
            # 동화: 발굴 엔진이 이미 스토리 전체를 1~2분으로 축약한 완성본 → 통째로(59초 캡 X).
            shutil.copy(hl, final)
            print(f"  {pipeline_type} → final.mp4 = {hl_dur:.2f}초 (단락 전체{', 동화 통째' if pipeline_type=='folktale' else ''})", flush=True)
        else:
            # 단락이 59초 초과 → Gemini로 이 단락 안에서 가장 임팩트 있는 연속 구간 추출
            print(f"  drama 단락 {hl_dur:.0f}초 → 핵심 연속 ≤58초 추출 (Pass 1.5)", flush=True)
            picked = None
            try:
                ana2 = await ensure_inline_video(hl)
                uri2 = await upload_video_to_gemini(ana2)
                raw2 = await call_gemini(GEMINI_PRO_MODEL, uri2, PROMPT_DRAMA_CORE_TPL(hl_dur))
                core = _parse_json(raw2)
                cs = float(core.get("start", 0)); ce = float(core.get("end", 0))
                # silence 스냅으로 경계 다듬기 (말 중간 X)
                sils_hl = detect_silences(hl, threshold_db=-35, min_silence=0.15)
                cs = snap_to_silence(cs, sils_hl, hl_dur, kind="start", window=1.5)
                ce = snap_to_silence(ce, sils_hl, hl_dur, kind="end", window=1.5)
                if ce - cs > 58:
                    ce = cs + 58
                # 최소 길이 보장 — 너무 짧으면(쇼츠로 부적합) 핵심 중심으로 확장
                MIN_LEN = 35.0
                if ce - cs < MIN_LEN:
                    if hl_dur <= MIN_LEN + 2:
                        # 단락 자체가 짧음 → 단락 통째
                        cs, ce = 0.0, hl_dur
                    else:
                        # 핵심 구간 중심으로 MIN_LEN까지 양쪽 확장 (영상 범위 안)
                        mid = (cs + ce) / 2
                        cs = max(0.0, mid - MIN_LEN / 2)
                        ce = min(hl_dur, cs + MIN_LEN)
                        cs = max(0.0, ce - MIN_LEN)
                        # 경계 silence 스냅 다시
                        cs = snap_to_silence(cs, sils_hl, hl_dur, kind="start", window=1.5)
                        ce = snap_to_silence(ce, sils_hl, hl_dur, kind="end", window=1.5)
                        if ce - cs > 58:
                            ce = cs + 58
                if ce > cs + 5:
                    picked = (cs, ce)
            except Exception as _e:
                print(f"  ⚠️ drama 핵심 추출 실패 ({str(_e)[:100]}) — 앞 58초", flush=True)
            if not picked:
                picked = (0.0, 58.0)
            cs, ce = picked
            clip_orig_start += cs  # 핵심 구간 앞 cs초 잘림 → 원본 시작점 이동
            print(f"    핵심 구간: {cs:.1f}~{ce:.1f} ({ce-cs:.1f}s)", flush=True)
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(cs), "-to", str(ce), "-i", str(hl),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", str(final)
            ], check=True, capture_output=True)
            print(f"  drama → final.mp4 = {ce-cs:.1f}초 (단락 핵심)", flush=True)
    elif hl_dur <= 59.5:
        final = hl_dir / "final.mp4"
        shutil.copy(hl, final)
        print(f"  → final.mp4 = {hl_dur:.2f}초 (Pass 2 스킵)", flush=True)
    else:
        ana2 = await ensure_inline_video(hl)
        uri2 = await upload_video_to_gemini(ana2)
        raw2 = await call_gemini(GEMINI_PRO_MODEL, uri2, PROMPT_PASS2_TPL(hl_dur))
        trim = _parse_json(raw2)
        keep = [(float(k["start"]), float(k["end"])) for k in (trim.get("keep") or [])]

        # d. 자동 스냅 — silencedetect 결과로 보정
        sils_hl = detect_silences(hl, threshold_db=-35, min_silence=0.15)
        snapped = []
        for a, b in keep:
            a2 = snap_to_silence(a, sils_hl, hl_dur, kind="start", window=1.0)
            b2 = snap_to_silence(b, sils_hl, hl_dur, kind="end", window=1.0)
            if b2 > a2 and b2 <= hl_dur:
                snapped.append((a2, b2))
        keep_total = sum(b - a for a, b in snapped)
        print(f"  keep {len(snapped)}개(스냅 적용) 합 {keep_total:.1f}초", flush=True)
        for a, b in snapped:
            print(f"    {a:.1f}~{b:.1f} ({b-a:.1f}s)", flush=True)

        final = hl_dir / "final.mp4"
        keep_dir = hl_dir / "_keep"
        keep_dir.mkdir(exist_ok=True)
        cut_concat(hl, snapped, final, keep_dir, prefix="k")
        f_dur = _dur(final)
        print(f"  → final.mp4 = {f_dur:.2f}초", flush=True)
        if f_dur > 59.5:
            cap = hl_dir / "final_cap.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(final), "-t", "59",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", str(cap)
            ], check=True, capture_output=True)
            final = cap

    # d2. 컷 변환 — 무음·재미없는 긴 장면 컷. 대표님 2026-05-29: 레퍼런스(컷모아/폭스토리)도
    #     컷이 계속 넘어감. 무음/긴 장면 길어지면 사람들이 안 봄. 단 편집점(무음 끝·장면 전환)에서만
    #     컷 → 액션·대사 흐름은 안 깨짐. drama/anime 적용 (highlight는 위 b에서 무음제거 완료).
    #     영화(movie): 명장면 연속 보존(내부 무음컷 시 장면 중간 점프=튐). 경계는 _discover에서
    #     PySceneDetect+무음으로 스냅 → 시작/끝 깨끗. 내부 압축은 영콕드콕式 별도(대표님 평가 후).
    if pipeline_type in _flow_types and pipeline_type != "movie" and not seg.get("_precut"):
        try:
            _df = _dur(final)
            # 무음(대사 사이 침묵)만 컷 — speech(대사·소리) 구간은 통째 보존해 단어 중간 안 잘림.
            # 긴 장면 4.5초 트림은 대사 단어를 자르는 문제(대표님 2026-05-29 "대사 끝나기 전 짤림") → 제거.
            # 근본(대사 cue 경계 기반 컷): Whisper를 컷 前에 돌려 대사 끝 지점에서만 컷 — 추후.
            _sils = detect_silences(final, threshold_db=-32, min_silence=0.45)
            _sp = speech_ranges(_df, _sils)
            _keep = [(round(_a, 2), round(_b, 2)) for (_a, _b) in _sp if _b - _a >= 0.3]
            _nt = sum(e - s for s, e in _keep)
            if len(_keep) >= 2 and 5 < _nt < _df - 0.8:
                _cut = hl_dir / "_cut.mp4"
                cut_concat(final, _keep, _cut, hl_dir / "_cutsegs", prefix="ct")
                if _dur(_cut) > 5:
                    shutil.move(str(_cut), str(final))
                    print(f"  무음 컷: {_df:.1f}→{_dur(final):.1f}s ({len(_keep)}구간)", flush=True)
        except Exception as _e:
            print(f"  ⚠️ 무음 컷 실패 (계속): {str(_e)[:150]}", flush=True)
        _pr(35, "무음 컷")

    # e. 대사 SRT — 애니/영화는 화면 **한국어 하드섭 OCR**(음성=일본어 등이라 Whisper면 일본어로 나옴!),
    #    그 외(예능/드라마=한국어 음성)는 Whisper+번역/교정.
    try:
        _pr(40, "대사 받아쓰기")
        dsrt = hl_dir / "03_대사.srt"
        _ocr_dlg = pipeline_type in ("anime", "movie")
        if pipeline_type == "folktale":
            # 레퍼런스('와' 채널)式 '해설 자막 중심' 전환 — 대사 싱크 자막을 만들지 않는다.
            #   해설(01_상황설명)이 줄거리를 처음부터 끝까지 들려주므로 대사 트랙이 불필요(타이밍 어긋남 근본 해결).
            if dsrt.exists():
                dsrt.unlink()  # 잔존본(이전 발굴 STT) 제거
            print(f"  동화 = 해설 자막 중심 → 대사 SRT 생성 안 함(03_대사 skip)", flush=True)
        elif _ocr_dlg:
            print(f"  대사 = 화면 한국어 하드섭 OCR (Vision, 음성전사 X)...", flush=True)
            nd = await asyncio.to_thread(_ocr_dialogue_srt, final, dsrt)
            # 🔴워터마크/로고(예 "KBS Archive 옛날티비")만 읽히면 cue 수는 많아도 실제 대사 0 → '고유 한글 대사'로 판별(영문 로고 무시)
            _uniq = 0
            if dsrt.exists():
                import re as _reu
                _seen = set()
                for _ln in dsrt.read_text(encoding="utf-8").splitlines():
                    _ln = _ln.strip()
                    if _ln and "-->" not in _ln and not _ln.isdigit():
                        _seen.add(_reu.sub(r"[^가-힣]", "", _ln))  # 한글만 남김(워터마크 영문·기호 제거)
                _uniq = len([t for t in _seen if len(t) >= 2])
            print(f"  대사 OCR: {nd} cue (고유 한글대사 {_uniq}종)", flush=True)
            if _uniq < 6:   # 워터마크 변형이 한글 3~5종까지 나옴(예 "옛날티비"+노이즈) → 임계 상향. 실제 하드섭 대사는 보통 10종+
                # 🔴하드섭 한국어 대사 없음(OCR이 워터마크/로고만 읽음) → Whisper 음성 받아쓰기 (대표님: OCR 안 되면 위스퍼)
                print(f"  ⚠️ 고유 한글대사 {_uniq}종 = 하드섭 대사 없음 → Whisper 음성 받아쓰기로 전환", flush=True)
                if dsrt.exists(): dsrt.unlink()
                await whisper_srt(final, hl_dir, "03_대사.srt")
                _ocr_dlg = False  # 아래 번역/교정(비한국어 음성→한국어) 타게
        else:
            # 동화 포함 — final.mp4 직접 Whisper (예능과 동일 경로). folktale_hl이 STT '문장 단위'로 컷하므로
            #   컷 경계 = 문장 경계 → final 받아쓰기해도 단어 안 잘리고, 시각이 final과 '정확히' 일치한다.
            #   (엔진 _dialogue 재사용은 컷 경계에서 0.1~0.18초 '유령 자막'을 만들어 타이밍이 어긋났음 — 대표님 지적)
            print(f"  Whisper 대사 SRT 받아쓰기 (final 직접)...", flush=True)
            await whisper_srt(final, hl_dir, "03_대사.srt")
        if dsrt.exists():
            # Whisper 경로(동화·예능·드라마): 언어 감지 → 비한국어면 번역, 한국어면 교정. (OCR은 이미 한국어라 스킵)
            if not _ocr_dlg:
                try:
                    from workers.audio_sync import (
                        detect_srt_lang, translate_dialogue_srt, correct_dialogue_srt,
                    )
                    _lang = detect_srt_lang(dsrt)
                    if _lang != "ko":
                        nt = await translate_dialogue_srt(dsrt, src_lang=_lang)
                        print(f"  대사 {_lang}→한국어 번역: {nt} cue" if nt else f"  ⚠️ 대사 번역 실패 — 원어({_lang}) 유지", flush=True)
                    else:
                        nc = await correct_dialogue_srt(dsrt)
                        if nc:
                            print(f"  대사 맞춤법 교정: {nc} cue", flush=True)
                except Exception as _e:
                    print(f"  ⚠️ 대사 번역/교정 실패: {str(_e)[:120]}", flush=True)
            # 한 줄 16자 이하 분할 (공통 — 화면 3줄 깨짐 방지)
            try:
                from workers.audio_sync import split_long_dialogue_srt
                n = split_long_dialogue_srt(dsrt, max_line_chars=16)
                print(f"  대사 줄 분할: {n} cue", flush=True)
            except Exception as _e:
                print(f"  ⚠️ 대사 줄 분할 실패: {str(_e)[:120]}", flush=True)
    except Exception as ex:
        print(f"  ⚠️ 대사 SRT 실패: {ex}", flush=True)

    # f. auto_subtitle — 쇼츠메이커 자체 자막 prompt (타입별 분기)
    #    [[menu-prompt-separation]]: 자막 메뉴 SUBTITLE_STYLES 침범 X.
    #    pipeline_type="highlight" → SHORTS_HIGHLIGHT_PROMPT (토크쇼/인터뷰 임팩트 후크)
    #    pipeline_type="drama"     → SHORTS_DRAMA_PROMPT (드라마 흐름 + 후킹)
    from workers.shorts_subtitle import get_shorts_prompt
    shorts_prompt = get_shorts_prompt(pipeline_type)
    # 🔴 드라마 MZ력 강화 (대표님 0606): folktale처럼 학습 매너리즘(시전·참교육·X됨 등) inject → 상황설명 MZ 톤 강화.
    if pipeline_type == "drama":
        try:
            from workers.auto_subtitle import get_learning_inject as _gli_d
            shorts_prompt += "\n\n" + _gli_d(mannerism_only=True)
        except Exception as _e:
            print(f"  ⚠️ 드라마 학습 inject 생략: {str(_e)[:60]}", flush=True)
    # 동화: 엔진이 아는 정확한 컷 시각+장면(_jumps)을 주입 → 상황설명 타이밍·내용 일치(Gemini 눈대중 방지)
    _jumps = seg.get("_jumps")
    if pipeline_type == "folktale" and _jumps:
        # 상황설명 타이밍 = 장면이 화면에 '나온 뒤' 살짝 늦게 띄운다(대표님: 장면 나오기 전에 미리 설명 X).
        #   _jumps.at = 컷이 시작되는 시각 → +SIT_DELAY초(장면이 보이고 난 뒤) 지점을 권장 타임스탬프로 준다.
        SIT_DELAY = 1.2  # (참고) 정렬은 align_situation_to_jumps에서 강제 — 여기선 프롬프트 가이드만
        jl = "\n".join(
            f"{i+1}. {j.get('desc', '')}" for i, j in enumerate(_jumps))
        _target_min = max(len(_jumps) * 2, 10)
        _target_max = len(_jumps) * 3
        shorts_prompt += (
            "\n\n═══════════════════════════════════════\n"
            "[🚨 이 영상의 컷 장면 목록 — 자막을 이 흐름 따라 자연스러운 밀도로 깔아라]\n"
            "═══════════════════════════════════════\n"
            f"아래는 이 쇼츠 화면에 시간순으로 나오는 장면 목록이다(총 {len(_jumps)}개).\n"
            f"🔴**개수(필수)**: situation_subtitles를 **{_target_min}~{_target_max}개** 사이로 자연스럽게 깔아라"
            "(장면 1개당 자막 2~3개씩, 시청자가 자막 없이 멍 때리는 구간 만들지 마라).\n"
            "🔴**순서**: 장면 순서대로 따라가되, 한 장면 안에서 핵심 액션이 보일 때마다 자막 추가 박아도 OK.\n"
            "🔴🔴**톤·포맷 (레몬사이다·1분기악 스타일, 절대 룰)**:\n"
            "  · **화자 콜론(예: '막내:', '아빠:') 절대 X**. 시청자 시점 드립/감상평으로 써라.\n"
            "  · **대사 받아쓰기 X** — 캐릭터 말 그대로 옮기지 마라.\n"
            "  · **드립/풍자/현대 비유 중심** — 옛 동화를 현대 MZ 시각으로 비꼬거나 비유 (NPC/히든퀘스트/SSR템/가챠/코인 떡상/영앤리치/주식 물림/오마카세/반포자이 등 자유 응용).\n"
            "  · 글자수 평균 10~15자, 최대 20자 (캡컷 한 줄). 마침표(.) X.\n"
            "  · 예시 톤: '아빠 광탈각', '초상집에서 먹방 찍는 인성ㄷㄷ', 'NPC 스님 발견!', '히든 퀘스트 바로 수락', '전설급 SSR 아이템 획득', '코인 떡상 소문 퍼짐', '나무 구멍에 가챠 돌리기', '강제 비건 다이어트 당첨ㅋㅋ', '주식 물린 자들의 최후.jpg', '착하게 살면 떡상합니다'.\n"
            "🔴**🚫 마법도구/보물/신비한 물건이 여러 개 나오면(2개 이상): 도구마다 반드시 별도 cue로 박아라**. "
            "절대 묶지 마라. 도구 1개당 (사용 동작 cue: 두드림/펼침/흔듦 등) + (효과 cue: 무엇이 나타나는지) 2개 박는 게 기본. "
            "예: 돗자리·항아리·붓·요술방망이·조롱박·젓가락·반지 등 — **이 동화에 실제로 나오는 도구 이름/개수는 영상 보고 알아서 잡아라**(미리 가정 X, 다른 동화 도구 베끼기 X). "
            "도구가 1개면 1개만, 5개면 5개씩 각각 따로. 형들·악인이 그 도구를 잘못 써서 망하는 장면도 도구별 따로.\n"
            "🔴**🚫 스포 절대 금지**: 다음 장면 내용 미리 말하지 마라. "
            "예: 막내가 보물 받기 직전 장면에서 '보물 받는 막내'라고 미리 X. "
            "장면이 화면에 **실제로 나타난 그 순간**의 액션만 자막으로 써라. "
            "앞으로 일어날 일 예고 X, 과거 회상 X, 결말 미리 알리기 X.\n"
            "🔴**시각(타이밍)**: start/end는 신경 쓰지 마라 — 코드가 각 해설을 그 장면이 화면에 나오는 정확한 시각에 자동으로 박는다. 너는 순서·내용만 정확히. "
            "한 장면에 자막 여러 개일 때는 순서대로 적으면 코드가 그 장면 안에서 자동 분배한다.\n"
            "🔴**대화 자막 X**: 등장인물 입에서 들리는 대사는 자막에 쓰지 마라(시청자가 귀로 듣는다). "
            "상황·액션·임팩트·반전·감정만 MZ 톤으로 짚어라.\n"
            "🔴**표현(중요)**: 옆의 장면 설명 문장은 '무슨 일인지' 참고용일 뿐이다. **그 문장을 절대 그대로 베끼지 마라**(베끼면 딱딱해서 실패). "
            "매 자막을 MZ 밈·드립·속담 비틀기로 새로 써라 — 예: '주인공이 선행을 한다'→'마지막 한 톨까지 베푸는 갓생 주인공ㄷㄷ', "
            "'악인이 벌받는다'→'욕심부린 대가=참교육ㅋㅋ 사필귀정' (인물·도구·상황은 이 동화 장면 목록대로 — 다른 동화 베끼기 X).\n"
            "보물·도구 효과는 장면대로 정확히(도구별로 그 효과를 따로따로 — 도구 종류·효과는 동화마다 다르니 위 장면 목록 보고 잡아라, 미리 가정 X). 처음부터 결말(악인 벌·화해)까지 다 메워라.\n" + jl)
        # 🔴 학습 mannerism 신호 inject (folktale_finalize와 동일·2026-06-06). prompt_override는 auto_subtitle 자동 inject 생략 → 직접 붙임. mannerism_only=True(표현풀 족쇄 방지, 3.1-pro 정형화 방지).
        try:
            from workers.auto_subtitle import get_learning_inject
            shorts_prompt += "\n\n" + get_learning_inject(mannerism_only=True)
        except Exception as _e:
            print(f"  ⚠️ 학습 inject 생략: {str(_e)[:60]}", flush=True)
    seg_type = (seg.get("type") or "talk").lower()
    _pr(60, "자막 생성")
    print(f"  자막 스타일: shorts_maker:{pipeline_type} (segment type={seg_type})", flush=True)
    with db.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO subtitle_jobs (video_filename, video_path, style, status, progress) "
            "VALUES (?,?,?,'pending',0)",
            (final.name, str(final.absolute()), f"shorts_maker:{pipeline_type}")
        )
        sub_id = cur.lastrowid
    print(f"  auto_subtitle id={sub_id}...", flush=True)
    await run_auto_subtitle(sub_id, final, prompt_override=shorts_prompt)

    # 결과 복사 to hl_dir
    sub_dir = Path(f"data/subtitles/job_{sub_id}")
    if sub_dir.exists():
        for f in sub_dir.iterdir():
            if f.is_file() and f.suffix in (".srt", ".txt"):
                shutil.copy(f, hl_dir / f.name)

    # 🔴 folktale: 01_상황설명을 ~55초 구간별로 재생성(덮어씀). 단일 생성은 긴 클립서 뒤로 갈수록 시각 밀림(대표님 0606).
    if pipeline_type == "folktale":
        try:
            await _gen_folktale_subs_chunked(final, hl_dir, shorts_prompt)
        except Exception as _e:
            print(f"  ⚠️ folktale 구간별 자막 실패(단일생성 유지): {str(_e)[:140]}", flush=True)

    # 자막 마침표(.) 제거 — 대표님 룰 (물음표/느낌표 등은 유지)
    for srt_file in hl_dir.glob("*.srt"):
        _strip_periods_in_srt(srt_file)

    # 🔴 folktale 해설(01_상황설명) = Gemini가 영상 보고 박은 시각 그대로 (정답·folktale_finalize CLI와 동일).
    #   align_situation_to_jumps(앵커 강제정렬) 제거(2026-06-06): 비례매핑이 자막 ts를 망가뜨림
    #   (cue 1초/17초 불균등·20~30초 늘어뜨려 화면과 어긋남). 01_상황설명은 아래 silence sync에서도
    #   _sync_targets로 제외되므로 Gemini 시각 그대로 보존된다. ❌ align 다시 쓰지 말 것.

    # Phase 1 silence-aware sync — 01/02 자막 cue를 음성 silence 안에서만 pad.
    # 인접 음성 침범 X. 03_대사.srt(Whisper)는 이미 정확하니 건드리지 X.
    try:
        from workers.audio_sync import (
            extract_audio_16k_mono, detect_speech_intervals,
            adjust_cues, parse_srt, write_srt,
        )
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
            _wav = Path(_f.name)
        extract_audio_16k_mono(final, _wav)
        _speech = detect_speech_intervals(_wav)
        _wav.unlink(missing_ok=True)
        _final_dur = _dur(final)
        # folktale은 01_상황설명을 위에서 _jumps로 정렬했으니 sync 제외(재배치되면 다시 어긋남). 02_쨉쨉이만 sync.
        _sync_targets = ("02_쨉쨉이.srt",) if pipeline_type == "folktale" else ("01_상황설명.srt", "02_쨉쨉이.srt")
        for srt_name in _sync_targets:
            srt_p = hl_dir / srt_name
            if srt_p.exists():
                _cues = parse_srt(srt_p)
                if _cues:
                    # 상황설명은 한 자막이 6초 넘게 떠 대사를 계속 덮는 것 방지 (prompt 안전망).
                    # 대표님 룰: 상황설명은 중요한 순간 2~3개 잠깐만, 나머지는 대사가 화면 채움.
                    if srt_name == "01_상황설명.srt":
                        for _c in _cues:
                            if _c["end"] - _c["start"] > 6.0:
                                _c["end"] = _c["start"] + 6.0
                    elif srt_name == "02_쨉쨉이.srt":
                        # 쨉쨉이는 짧은 강조 — 한 cue가 2초 넘게 이어지면 자름 (대표님: 길게 이어짐)
                        for _c in _cues:
                            if _c["end"] - _c["start"] > 2.0:
                                _c["end"] = _c["start"] + 2.0
                    _adj = adjust_cues(_cues, _speech,
                                        pre_pad=0.2, post_pad=0.2,
                                        video_dur=_final_dur)
                    write_srt(_adj, srt_p)
        print(f"  ✅ 자막 silence-aware sync 적용 ({len(_speech)} speech intervals)", flush=True)
    except Exception as _e:
        print(f"  ⚠️ 자막 sync 실패 (계속): {str(_e)[:200]}", flush=True)

    # 쨉쨉이가 대사를 그대로 베낀 것 제거 (대표님: 쨉쨉이=리액션/효과, 대사 단어 반복 X).
    # 별표·괄호·공백 떼고 핵심어가 대사 텍스트에 들어있으면 = 중복 → 제거.
    try:
        import re as _rej
        from workers.audio_sync import parse_srt as _psj, write_srt as _wsj
        jp = hl_dir / "02_쨉쨉이.srt"
        dp = hl_dir / "03_대사.srt"
        if jp.exists() and dp.exists():
            jcs = _psj(jp); dcs = _psj(dp)
            dclean = _rej.sub(r"[^가-힣a-zA-Z0-9]", "", " ".join(c.get("text", "") for c in dcs))
            kept = []
            for c in jcs:
                core = _rej.sub(r"[^가-힣a-zA-Z0-9]", "", c.get("text", ""))
                if core and len(core) >= 2 and core in dclean:
                    continue  # 대사에 있는 말 = 쨉쨉이에서 빼기
                kept.append(c)
            if len(kept) != len(jcs):
                _wsj(kept, jp)
                print(f"  쨉쨉이 대사중복 {len(jcs)-len(kept)}개 제거", flush=True)
    except Exception as _e:
        print(f"  ⚠️ 쨉쨉이 정리 실패: {str(_e)[:120]}", flush=True)

    # 도입 내레이션 TTS — 첫 상황설명을 타입캐스트(필재)로. 대표님 2026-05-29:
    # 정지(freeze) 말고, 원본 앞부분을 TTS 길이만큼 더 가져와 자연스러운 도입을 만들고 그 위에 TTS.
    # 원본 음소거 X (도입 원본 소리 + TTS 합성). 본편은 그대로라 대사 하나도 안 짤림.
    # 영상 맨 앞 클립이라 앞에 가져올 게 없으면 → 첫 프레임 freeze로 폴백.
    try:
        from workers.audio_sync import parse_srt as _ps2, write_srt as _ws2
        sit_p = hl_dir / "01_상황설명.srt"
        sits = _ps2(sit_p) if sit_p.exists() else []
        first_text = sits[0].get("text", "").replace("\n", " ").strip() if sits else ""
        if first_text:
            from workers.tts_dub import _tts_line, DEFAULT_VOICE, TRIM
            raw_mp3 = hl_dir / "_intro_tts_raw.mp3"
            tts_mp3 = hl_dir / "intro_tts.mp3"
            await _tts_line(first_text, DEFAULT_VOICE, raw_mp3)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-i", str(raw_mp3), "-af", TRIM, str(tts_mp3)], check=True)
            T = _dur(tts_mp3)
            if T and T > 0.3:
                mixed = hl_dir / "_final_tts.mp4"
                # drama/anime/movie는 원본 연속 구간이라 final 직전 T초를 원본에서 가져올 수 있음.
                # 동화: 도입은 freeze(정지화면)+TTS만 → 원본 음성과 안 겹침(대표님 "안 겹치게 프리즈").
                #       그 외 타입은 기존(원본 직전 장면+TTS 합성).
                intro_avail = 0.0 if pipeline_type == "folktale" else (min(T, clip_orig_start) if pipeline_type in _flow_types else 0.0)
                shift = 0.0
                if intro_avail >= 0.5:
                    # ① 원본 [clip_orig_start - intro_avail, clip_orig_start] = final 직전 장면 추출 (크롭 X, final과 동일 인코딩)
                    intro_raw = hl_dir / "_intro_raw.mp4"
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", f"{clip_orig_start - intro_avail:.3f}", "-to", f"{clip_orig_start:.3f}",
                        "-i", str(src),
                        "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2",  # fps=30 강제(소스 24fps여도 본편과 통일 — 싱크깨짐 방지)
                        "-r", "30", "-video_track_timescale", "30000",
                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(intro_raw)
                    ], check=True, capture_output=True)
                    shift = _dur(intro_raw)
                    # ② 도입 장면에 TTS 얹기 — 원본 도입 소리 + TTS 합성 (음소거 X, normalize=0)
                    intro_mix = hl_dir / "_intro_mix.mp4"
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(intro_raw), "-i", str(tts_mp3),
                        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(intro_mix)
                    ], check=True, capture_output=True)
                    # ③ 도입 + 본편 concat (재인코딩 — 코덱/해상도 동일)
                    lst = hl_dir / "_concat.txt"
                    lst.write_text(f"file '{intro_mix.resolve()}'\nfile '{final.resolve()}'\n", encoding="utf-8")
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "30", "-video_track_timescale", "30000",
                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(mixed)
                    ], check=True, capture_output=True)
                    shutil.move(str(mixed), str(final))
                    print(f"  ✅ 도입 앞부분+TTS ({shift:.1f}s, 원본 직전 장면 가져옴): {first_text[:22]}", flush=True)
                else:
                    # 폴백: 영상 맨 앞이라 앞에 가져올 게 없음 → 첫 프레임 freeze + TTS
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-i", str(tts_mp3),
                        "-filter_complex",
                        f"[0:v]tpad=start_duration={T:.3f}:start_mode=clone,fps=30[v];[1:a][0:a]concat=n=2:v=0:a=1[a]",
                        "-map", "[v]", "-map", "[a]", "-r", "30", "-video_track_timescale", "30000",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(mixed)
                    ], check=True, capture_output=True)
                    shutil.move(str(mixed), str(final))
                    shift = T
                    print(f"  ✅ 도입 freeze+TTS ({T:.1f}s, 영상 맨앞이라 정지 폴백): {first_text[:22]}", flush=True)
                # 자막 전부 +shift 시프트 (원본이 shift초 뒤로 밀림). 01 첫 cue는 0~shift 도입(TTS 자막).
                for _sn in ("01_상황설명.srt", "02_쨉쨉이.srt", "03_대사.srt"):
                    _sp = hl_dir / _sn
                    if not _sp.exists():
                        continue
                    _ccs = _ps2(_sp)
                    for _c in _ccs:
                        _c["start"] = round(_c.get("start", 0) + shift, 3)
                        _c["end"] = round(_c.get("end", 0) + shift, 3)
                    if _sn == "01_상황설명.srt" and _ccs:
                        _ccs[0]["start"] = 0.0
                        _ccs[0]["end"] = round(shift, 2)
                    _ws2(_ccs, _sp)
                raw_mp3.unlink(missing_ok=True)
    except Exception as _e:
        print(f"  ⚠️ 도입 TTS 실패 (계속): {str(_e)[:160]}", flush=True)

    # 동화: 상단 고정 제목(00_제목.srt) — 레퍼런스('와' 채널)式 상단 제목 + 잔존 대사 트랙 제거.
    #   제목은 클립 전체 길이(TTS 도입 포함)에 걸쳐 상단에 고정으로 띄운다.
    if pipeline_type == "folktale":
        try:
            _d03 = hl_dir / "03_대사.srt"
            if _d03.exists():
                _d03.unlink()  # 해설 중심 — 대사 싱크 자막 폐기
            _ttl = (seg.get("title_hint") or "").strip()
            if not _ttl:
                try:
                    with db.get_db() as _c3:
                        _r3 = _c3.execute("SELECT gemini_results FROM subtitle_jobs WHERE id=?", (sub_id,)).fetchone()
                    if _r3 and _r3[0]:
                        _p3 = (json.loads(_r3[0]).get("primary") or {})
                        _ttl = (_p3.get("title") or "").strip()
                except Exception:
                    pass
            if _ttl:
                _fd = _dur(final)
                def _ts(x):
                    h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
                (hl_dir / "00_제목.srt").write_text(
                    f"1\n{_ts(0.0)} --> {_ts(max(1.0, _fd))}\n{_ttl}\n", encoding="utf-8")
                print(f"  ✅ 동화 상단 제목 고정: {_ttl}", flush=True)
        except Exception as _e:
            print(f"  ⚠️ 동화 제목 생성 실패 (계속): {str(_e)[:120]}", flush=True)

    # meta JSON 추출 (별도 connection — 위 with 블록은 이미 닫힘)
    with db.get_db() as conn2:
        row = conn2.execute("SELECT gemini_results FROM subtitle_jobs WHERE id=?", (sub_id,)).fetchone()
    if row and row[0]:
        d = json.loads(row[0])
        p = d.get("primary", {}) or d
        meta = {
            "youtube_upload_title": p.get("youtube_upload_title", ""),
            "youtube_description": p.get("youtube_description", ""),
            "hashtags": p.get("hashtags", []),
            "title": p.get("title", ""),
            "summary": p.get("summary", ""),
            "title_candidates": p.get("title_candidates", []),
        }
        (hl_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    _pr(96, "마무리")
    print(f"  ✅ hl_{idx:02d} 완료 ({_dur(final):.1f}s)", flush=True)
    return {"idx": idx, "final": str(final.absolute()), "dur": _dur(final), "sub_id": sub_id, "dir": str(hl_dir)}


PROMPT_PASS1_DRAMA_TPL = """다음은 한국 드라마/영화 영상의 자동자막(타임스탬프 + 대사)이야. 이 이야기를 한국 쇼츠 여러 편으로 나눠줘.

🚨 핵심 — 이야기 구조 단위로 나눠라 (시간 등분 X):
- 영상의 **하나의 완결된 이야기**를 기승전결 흐름으로 본다.
- **의미 있는 단락(scene/beat)** 단위로 나눠 — 발단 / 전개 / 위기 / 절정 / 결말 같은 큰 덩어리.
- 잘게 쪼개지 마. 비슷한 장면이 이어지면 한 편으로 묶어.

편 수 기준 (영상 전체 길이 기준, 엄격히 지켜):
- ~5분: 1~2편
- 5~15분: 3~4편
- 15~30분: 4~6편
- 30분+: 6~8편
→ 위 범위 절대 초과 X. 14분이면 최대 4편.

흐름 연속성:
- segment[i+1].start = segment[i].end (중간 점프 X, 빠지는 내용 X)
- 영상 핵심 흐름을 처음부터 끝까지 다 담되, 각 편이 하나의 의미 단락
- 각 편 30~59초 권장 (한 단락이 길면 59초로 자르되 의미 끊기지 않게)

룰:
- 각 컷 최소 30초, 최대 59초
- 도입 타이틀/광고/엔딩 크레딧 부분만 제외, 본 이야기는 다 담기

JSON만 출력 (segments 길이 = 위 편 수 기준):
{{
  "segments": [
    {{"start": 30.0, "end": 85.0, "reason": "발단 — 상황 소개", "type": "drama"}},
    {{"start": 85.0, "end": 140.0, "reason": "전개 — 갈등 시작 (1편 직후)", "type": "drama"}},
    {{"start": 140.0, "end": 195.0, "reason": "절정~결말 (2편 직후)", "type": "drama"}}
  ]
}}

영상 자막 (참고):
{transcript}

[중요] **이야기 구조 단위로 3~4편 (14분 기준)**. 시간 등분 X. segment 시간 연속. 각 30~59초."""


# ════════════════════════════════════════════════════════════
# v3 발굴 prompt — 반전/임팩트 지점 전부 발굴 (제작 X, 후보 목록만)
# 컷모아/폭스토리 분석 기반: 반전을 65~85% 지점에 두는 "한 방 구조".
# 각 후보에 twist_sec(반전 시점) + 기승전결 메타 포함.
# ════════════════════════════════════════════════════════════

# DRAMA 발굴 (컷모아式) — 한 대화 시퀀스 = 완결된 사이다/반전 한 클립.
PROMPT_DISCOVER_DRAMA_TPL = """다음은 한국 드라마 자동자막(타임스탬프+대사)이야. 쇼츠로 만들 **사이다/반전 장면**을 전부 발굴해줘.

🎯 한 후보 = 완결된 기승전결 한 클립 (컷모아 스타일):
- **한 대화 시퀀스/한 사건** 단위 (같은 장소·인물의 갈등→해소 묶음).
- 각 후보는 **반전/펀치라인 한 방**이 있어야 함 (정체 공개, 통념 깨는 대사, 사이다 참교육).
- 길이 30~58초 분량. 갈등 단순하면 30초, 빌런 여럿/복잡하면 50초.
- 반전(twist)은 그 구간의 후반 70~85%에 오는 지점.

발굴 범위:
- 영상 전체에서 임팩트 있는 장면 **전부** (개수 제한 X — 5개든 20개든 있는 만큼).
- 도입·잡담·연결씬 X. 한 방 있는 장면만.
- impact_score: 1~10 (반전 강도·감정 폭발·사이다 정도).

JSON만 출력:
{{
  "candidates": [
    {{"start": 120.0, "end": 168.0, "twist_sec": 158.0, "title": "12자 후크 제목",
      "summary": "이 장면 한 줄 요약 (누가 뭘 하다 어떻게 반전)",
      "structure": "기: ~ / 승: ~ / 전(반전): ~ / 결: ~",
      "impact_score": 9, "type": "drama"}}
  ]
}}

영상 자막:
{transcript}

[중요] 각 후보 30~58초. twist_sec는 start~end 안 후반부. impact 높은 순. 한 방 없으면 빼."""


# MOVIE 발굴 (폭스토리式) — 영화 한 편에서 명장면/반전 多 발굴.
PROMPT_DISCOVER_MOVIE_TPL = """다음은 영화/드라마 자동자막(타임스탬프+대사)이야. 쇼츠로 만들 **명장면·반전 순간**을 전부 발굴해줘.

🎯 한 후보 = 반전 한 방 있는 47~58초 클립 (폭스토리 스타일):
- 영화엔 임팩트 순간이 **수십 개** 있을 수 있음 — 다 찾아.
- 각 후보: 셋업 → 빌드업 → **반전/펀치라인(후반 65~85%)** → 짧은 여운.
- 명장면 유형: 반전, 정체 공개, 충격 대사, 액션 하이라이트, 감정 폭발, 코믹 펀치.
- 길이 40~58초 분량.

발굴 범위:
- 영화 전체에서 "이 부분 쇼츠로 만들면 터지겠다" 싶은 순간 **전부** (개수 제한 X).
- impact_score: 1~10.

JSON만 출력:
{{
  "candidates": [
    {{"start": 1820.0, "end": 1872.0, "twist_sec": 1860.0, "title": "12~16자 후크 제목",
      "summary": "이 장면 한 줄 (상황→반전)",
      "structure": "기: ~ / 승: ~ / 전(반전): ~ / 결: ~",
      "impact_score": 8, "type": "movie"}}
  ]
}}

영상 자막:
{transcript}

[중요] 각 후보 40~58초. twist_sec는 후반부. 명장면 다 발굴 (무제한). impact 높은 순."""


# HIGHLIGHT 발굴 (예능/토크쇼) — 펀치라인 단락 발굴.
PROMPT_DISCOVER_HIGHLIGHT_TPL = """다음은 한국 예능/토크쇼 자동자막(타임스탬프+대사)이야. 쇼츠로 만들 **빵 터지는 단락**을 전부 발굴해줘.

🎯 한 후보 = 펀치라인/케미 폭발 한 클립:
- 한 코너/빌드업/빵 터지는 단락 (흐름 이어지는 한 단위).
- 펀치라인·폭로·충격 발언·디스전·반전이 후반에.
- 길이 30~58초 분량.

발굴 범위:
- 진짜 재밌는 단락 **전부** (개수 제한 X).
- 도입·인사·잡담·광고 X.
- impact_score: 1~10.
- type 분류: talk(대화중심) / info(정보·리액션) / mixed.

JSON만 출력:
{{
  "candidates": [
    {{"start": 234.0, "end": 280.0, "twist_sec": 270.0, "title": "12자 후크 제목",
      "summary": "이 단락 한 줄",
      "structure": "빌드업: ~ / 펀치라인: ~",
      "impact_score": 8, "type": "talk"}}
  ]
}}

영상 자막:
{transcript}

[중요] 각 후보 30~58초. 빵 터지는 단락 다 발굴. impact 높은 순."""


# ANIME 발굴 (추억의투니투니式) — 옛날 더빙 애니 명장면/명대사.
PROMPT_DISCOVER_ANIME_TPL = """다음은 한국어 더빙 애니메이션 자동자막(타임스탬프+대사)이야. 쇼츠로 만들 **명장면(배틀/경기/명대사 한 판)**을 전부 발굴해줘.

🎯 한 후보 = 완결된 한 판 승부/명장면 (투니투니 스타일):
- **하나의 전투·경기·대결·명장면** 단위 (승패/결판이 클립 안에서 완결).
- 셋업(주인공 위기·불리·조롱) → 반전(기술 적중·역전·각성) → 짧은 결말.
- 반전(twist)은 후반 70~95%에. 다음화 떡밥 X, 그 판은 클립 안에서 끝.
- 길이 50~58초 (애니는 길게, 한 판 다 담기).

🚨 애니 특화:
- 원본 시간 순서 보존 (배틀 인과 — 재배열 X).
- 캐릭터 명대사 + 결정적 장면 둘 다 살리기.
- 위기 셋업 충분히(절반 이상) → 후반 한 방 반전 카타르시스.

발굴 범위:
- 영상 전체에서 "한 판/명장면" **전부** (개수 제한 X).
- 늘어지는 대치·설명 구간만 빼고.
- impact_score: 1~10 (역전·각성·명대사 임팩트).

JSON만 출력:
{{
  "candidates": [
    {{"start": 600.0, "end": 658.0, "twist_sec": 650.0, "title": "12~16자 어그로 제목",
      "summary": "이 한 판 한 줄 (누가 위기→어떻게 역전)",
      "structure": "셋업(위기): ~ / 반전(역전): ~ / 결말: ~",
      "impact_score": 9, "type": "anime"}}
  ]
}}

영상 자막:
{transcript}

[중요] 각 후보 50~58초. twist_sec 후반부. 한 판 완결. 시간순. impact 높은 순."""


DISCOVER_PROMPTS_BY_TYPE = {
    "drama": PROMPT_DISCOVER_DRAMA_TPL,
    "movie": PROMPT_DISCOVER_MOVIE_TPL,
    "highlight": PROMPT_DISCOVER_HIGHLIGHT_TPL,
    "anime": PROMPT_DISCOVER_ANIME_TPL,
}


# 영화 신호기반 발굴 윈도우 → 각 명장면 클립을 Gemini에 **영상으로** 줘서 라벨/하이라이트 판단.
# 신호(음량+OCR)로 위치 찾고 + Gemini가 클립 영상을 직접 봄 = 하이브리드 (대사 텍스트만으론 부정확).
PROMPT_LABEL_ONE_MOVIE = """이 영화/드라마 명장면 클립(약 {dur:.0f}초)을 **직접 보고** 세로 쇼츠용으로 라벨링하라.
[이 구간 대사(OCR, 참고용)]
{daesa}

영상을 보고 무슨 장면인지 정확히 파악해서:
[출력 JSON만]
{{
 "title": "영콕드콕式 자극적 후크 제목 12~18자 (궁금증·반전 유발, 영상 내용 기반)",
 "summary": "이게 왜 명장면인지 한 줄 20~40자",
 "twist_sec": 클립 안에서 가장 임팩트 큰 순간의 상대 초(클립 시작=0 기준, 숫자만),
 "impact_score": 1~10 (쇼츠 흥행 잠재력, 영상 임팩트 보고),
 "is_highlight": true/false (이 구간에 진짜 볼만한 임팩트 장면이 있나)
}}
인물·작품용어·맞춤법 정확. 제목에 마침표·이모지 금지."""


async def _label_movie_windows(src_path, windows, cues):
    """신호로 찾은 윈도우 → 각 클립을 Gemini에 **영상으로** 줘서 영상 보고 제목/요약/하이라이트 판단.
    (발굴 위치=음량+OCR 신호 / 라벨·임팩트=Gemini가 클립 영상 직접 봄 = 하이브리드).
    영상 분석 실패 시 대사 텍스트 폴백. Returns candidates.json 후보 형식 리스트."""
    cands = []
    for i, (s, e) in enumerate(windows):
        wc = [c.get("text", "") for c in cues if s <= float(c.get("start", 0)) <= e]
        txt = " ".join(wc).strip()[:600] or "(대사 적음 — 영상으로 파악)"
        l = {}
        clip = Path(src_path).parent / f"_lbl_{i}.mp4"
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(s), "-to", str(e),
                            "-i", str(src_path), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "28", "-vf", "scale=640:-2", "-c:a", "aac", "-b:a", "96k",
                            str(clip)], check=True)
            iv = await ensure_inline_video(clip)
            uri = await upload_video_to_gemini(iv)
            l = await call_gemini(GEMINI_PRO_MODEL, uri,
                                  PROMPT_LABEL_ONE_MOVIE.format(dur=e - s, daesa=txt),
                                  temperature=0.4) or {}
        except Exception as ex:
            print(f"  ⚠️ 윈도우{i} 영상 라벨 실패(대사 폴백): {str(ex)[:70]}", flush=True)
        finally:
            try:
                clip.unlink()
            except OSError:
                pass
        tw = l.get("twist_sec")
        twist_abs = None
        if isinstance(tw, (int, float)):     # 클립내 상대초 → 절대초, 윈도우 안으로 클램프
            twist_abs = max(float(s) + 1, min(float(e) - 1, float(s) + float(tw)))
        isc = l.get("impact_score")
        cands.append({
            "start": float(s), "end": float(e),
            "twist_sec": twist_abs,
            "title": (l.get("title") or f"명장면 {i + 1}"),
            "summary": l.get("summary", ""),
            "structure": "movie-signal+Gemini영상라벨",
            "impact_score": float(isc) if isinstance(isc, (int, float)) else 6.0,
            "type": "movie",
        })
        print(f"  윈도우{i} {int(s)//60}:{int(s)%60:02d} 영상라벨 → {cands[-1]['title'][:24]}", flush=True)
    return cands


def _ocr_movie_cues(src_path, fps_div=6):
    """영화 하드섭(화면에 박힌 자막) → OCR 대사 cue. **Whisper 폐기**(2시간 전사 병목 + 환각).
    하드섭은 OCR이 빠르고 정확(메모리 룰: 하드섭 OCR / 무자막 Whisper). engine/ocrbin 하단밴드 OCR.
    Returns [{"start","end","text"}, ...]."""
    import os
    import glob
    # ocrbin 경로 (engine/ 우선)
    root = Path(__file__).resolve().parent.parent
    ocrbin = None
    for p in [root / "engine" / "ocrbin", Path("engine/ocrbin"), Path("/tmp/ocrbin")]:
        if Path(p).exists():
            ocrbin = str(p)
            break
    if not ocrbin:
        print("  ⚠️ ocrbin 없음 — OCR cue 스킵", flush=True)
        return []
    W = Path(src_path).parent / "_ocrcue"
    W.mkdir(exist_ok=True)
    try:
        for f in glob.glob(str(W / "*.png")):
            os.remove(f)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src_path),
                        "-vf", f"fps=1/{fps_div},crop=iw:ih*0.30:0:ih*0.70,scale=900:-1",
                        str(W / "f%05d.png")], check=True)
        frames = sorted(glob.glob(str(W / "f*.png")))
        res = subprocess.run([ocrbin] + frames, capture_output=True, text=True)
        ocr = {Path(l.split("\t")[0]).name: l.split("\t", 1)[1].strip()
               for l in res.stdout.splitlines() if "\t" in l}
        cues = []
        for i, f in enumerate(frames):
            t = i * fps_div
            txt = re.sub(r"^[\s•·.\-!,~]+", "", ocr.get(Path(f).name, "")).strip()
            if len(re.sub(r"\s", "", txt)) < 2:
                continue
            if cues and cues[-1]["text"] == txt and t - cues[-1]["end"] <= fps_div + 1:
                cues[-1]["end"] = t + fps_div            # 같은 자막 연장(병합)
            else:
                cues.append({"start": t, "end": t + fps_div, "text": txt})
        print(f"  하드섭 OCR cue {len(cues)}개 ({len(frames)}프레임 스캔)", flush=True)
        return cues
    finally:
        shutil.rmtree(W, ignore_errors=True)


def _discover_anime_hl(src_path, OUT, desc="애니", n=5, extra=""):
    """애니 발굴 = anime_hl 엔진(나루토 레퍼런스 방식): Gemini가 영상 보고 액션 클라이맥스
    윈도우 선택 + 핵심 비트 컷. prep(v360+씬+OCR) → anime_hl(비트컷 클립 N편).
    각 후보 = pre-cut 클립(_precut) + shots(절대초 비트). 출력(자막/캐리커처/TTS/카드)은 render가 기존대로.
    """
    import os as _os
    root = Path(__file__).resolve().parent.parent
    eng = root / "engine"
    tag = "anime_" + re.sub(r"[^\w]", "_", Path(OUT).name)[:30]
    env = dict(_os.environ)
    if extra:
        env["HL_EXTRA"] = extra
    # prep (raw부터 v360 + 씬검출 + 전체 OCR)
    subprocess.run([sys.executable, str(eng / "prep_anime_full.py"), str(src_path), tag], check=True)
    # anime_hl → /tmp/{tag}/hl_N.mp4 (비트컷) + hl_N.json
    adir = Path(OUT) / "_anime"
    adir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(eng / "anime_hl.py"), str(src_path), tag, str(adir), desc, str(n)],
                   check=True, env=env)
    # 후보 수집 (pre-cut 클립 + 메타)
    cands = []
    tdir = Path(f"/tmp/{tag}")
    for j in sorted(tdir.glob("hl_*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
            i = int(j.stem.split("_")[1])
        except Exception:
            continue
        clip = tdir / f"hl_{i}.mp4"
        if not clip.exists():
            continue
        precut = adir / f"hl_{i:02d}.mp4"
        shutil.copy(str(clip), str(precut))
        w = d.get("window", [0, 0])
        cands.append({
            "start": float(w[0]), "end": float(w[1]),
            "title": d.get("title", f"명장면 {i}"),
            "summary": " · ".join(d.get("tech", []) or []),
            "tech": d.get("tech", []),
            "shots": d.get("segs", []),
            "_precut": str(precut),
            "structure": "anime_hl(액션 하이라이트)",
            "impact_score": float(max(1, 10 - i)),
            "type": "anime",
        })
    return cands


def _write_srt(cues, path):
    """[[s,e,txt],...] → SRT 파일."""
    def _t(x):
        h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    blocks = []
    for i, c in enumerate(cues, 1):
        s, e, txt = c[0], c[1], c[2]
        blocks.append(f"{i}\n{_t(s)} --> {_t(e)}\n{txt}\n")
    Path(path).write_text("\n".join(blocks), encoding="utf-8")


def _discover_folktale(src_path, OUT, desc="전래동화", n=1, extra=""):
    """동화 발굴 = folktale_hl 엔진(레몬사이다式 스토리 축약): STT 문장단위 컷.
    각 후보 = pre-cut 클립(_precut) + 대사 SRT(_dialogue_srt, STT 기반) + 점프메타(_jumps).
    대사는 옛 음성이라 OCR(하드섭 없음)·Whisper(환각) 부적합 → 발굴 STT SRT를 그대로 사용.
    출력(자막/캐리커처/TTS/카드)은 render가 기존 반바지 그대로.
    """
    import os as _os
    root = Path(__file__).resolve().parent.parent
    eng = root / "engine"
    tag = "folk_" + re.sub(r"[^\w]", "_", Path(OUT).name)[:30]
    env = dict(_os.environ)
    if extra:
        env["HL_EXTRA"] = extra
    fdir = Path(OUT) / "_folktale"
    fdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(eng / "folktale_hl.py"), str(src_path), tag, str(fdir), desc, str(n)],
                   check=True, env=env)
    cands = []
    tdir = Path(f"/tmp/{tag}")
    for j in sorted(tdir.glob("hl_*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
            i = int(j.stem.split("_")[1])
        except Exception:
            continue
        clip = tdir / f"hl_{i}.mp4"
        if not clip.exists():
            continue
        precut = fdir / f"hl_{i:02d}.mp4"
        shutil.copy(str(clip), str(precut))
        # 대사 SRT (STT 기반, 클립 상대시각) — process_highlight가 03_대사.srt로 사용
        dsrt = fdir / f"hl_{i:02d}_대사.srt"
        _write_srt(d.get("_dialogue", []), dsrt)
        jumps = d.get("_jumps", [])
        cands.append({
            "start": float(d.get("window", [0, 0])[0]), "end": float(d.get("window", [0, 0])[1]),
            "title": d.get("title", f"동화 {i}"),
            "summary": " · ".join(jp.get("desc", "") for jp in jumps[:3]),
            "shots": d.get("segs", []),
            "_precut": str(precut),
            "_dialogue_srt": str(dsrt),
            "_jumps": jumps,
            "structure": "folktale(스토리 축약)",
            "impact_score": float(max(1, 10 - i)),
            "type": "folktale",
        })
    return cands


# ════════════════════════════════════════════════════════════
# v3 발굴 함수 — 다운로드 + cue 추출 + Gemini 발굴 (제작 X, 후보 저장만)
# ════════════════════════════════════════════════════════════
async def _download_and_cues(url: str, OUT: Path, _p, skip_transcribe: bool = False):
    """URL or 로컬파일 경로 → source.mp4 + cue list. (run_pipeline과 공유 로직)
    skip_transcribe=True: Whisper/자막 전사 스킵, src만 확보(cues=[]). 영화 하드섭 OCR 경로용.
    Returns (src_path, cues, src_dur, src_mb)."""
    src = OUT / "source.mp4"
    # ── 로컬 파일 입력 (http 아님) → 복사 + Whisper (자막 다운 X) ──
    _u = (url or "").strip()
    if _u.startswith("file://"):
        _u = _u[7:]
    if not _u.startswith("http"):
        local = Path(_u)
        if not local.exists():
            raise RuntimeError(f"로컬 파일 없음: {local}")
        _p("downloading", 8, f"로컬 파일 사용: {local.name}")
        if local.resolve() != src.resolve():
            shutil.copy(str(local), str(src))
        if skip_transcribe:        # 영화 하드섭 → Whisper 스킵, OCR로 cue (호출부에서)
            _p("downloading", 14, "원본 확보 (하드섭 OCR 예정 — Whisper 생략)")
            return src, [], _dur(src), src.stat().st_size // 1024 // 1024
        _p("downloading", 12, "Whisper 받아쓰기 (로컬)")
        audio = OUT / "_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src), "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(audio)
        ], check=True, capture_output=True)
        await whisper_srt(audio, OUT, "source.srt")
        from workers.audio_sync import parse_srt as _psrt
        ssrt = OUT / "source.srt"
        cues = _psrt(ssrt) if ssrt.exists() else []
        audio.unlink(missing_ok=True)
        if not cues:
            raise RuntimeError("Whisper 받아쓰기 실패 (로컬 파일)")
        _p("downloading", 15, f"Whisper {len(cues)}개 cue")
        src_dur = _dur(src)
        src_mb = src.stat().st_size // 1024 // 1024
        return src, cues, src_dur, src_mb
    # ── YouTube URL ──
    subs_task = asyncio.create_task(yt_subs_only(url, str(OUT / "source.%(ext)s")))
    src_task = asyncio.create_task(yt_source(url, str(OUT / "source.%(ext)s")))
    try:
        await subs_task
    except Exception as e:
        print(f"  ⚠️ 자동자막 실패: {e}", flush=True)
    vtt_files = list(OUT.glob("source*.vtt"))
    if vtt_files:
        cues = parse_vtt(vtt_files[0])
        _p("downloading", 15, f"자동자막 {len(cues)}개 cue")
        await src_task
    else:
        await src_task
        if skip_transcribe:        # 영화 하드섭 → Whisper 스킵
            _p("downloading", 15, "자동자막 없음 — 하드섭 OCR 예정 (Whisper 생략)")
            return src, [], _dur(src), src.stat().st_size // 1024 // 1024
        _p("downloading", 15, "자동자막 없음 — Whisper transcript")
        audio = OUT / "_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src), "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(audio)
        ], check=True, capture_output=True)
        await whisper_srt(audio, OUT, "source.srt")
        # source.srt 파싱
        from workers.audio_sync import parse_srt as _psrt
        ssrt = OUT / "source.srt"
        cues = _psrt(ssrt) if ssrt.exists() else []
        audio.unlink(missing_ok=True)
        if not cues:
            raise RuntimeError("자동자막/Whisper 둘 다 실패")
    src_dur = _dur(src)
    src_mb = src.stat().st_size // 1024 // 1024
    return src, cues, src_dur, src_mb


async def _refine_drama_cuts(src, cands, _p=None, max_n=12):
    """🔴 드라마/예능 컷 경계 영상검증 (대표님 0607): 텍스트 발굴은 펀치라인 직전에 끊긴다.
    각 후보 구간을 앞뒤 패딩 줘서 Gemini가 '직접 영상을 보고' 셋업→펀치라인→직후 리액션까지 통째 + 깔끔한 경계로 보정."""
    dur_full = _dur(src)
    for c in cands[:max_n]:
        try:
            s0, e0 = float(c.get("start", 0)), float(c.get("end", 0))
            if e0 <= s0:
                continue
            ps = max(0.0, s0 - 8.0); pe = min(dur_full, e0 + 16.0)
            clip = Path(f"/tmp/_refine_{int(s0)}_{int(e0)}.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ps:.2f}", "-to", f"{pe:.2f}",
                            "-i", str(src), "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "28", "-c:a", "aac", "-b:a", "96k", str(clip)], check=True, capture_output=True)
            ana = await ensure_inline_video(clip)
            uri = await upload_video_to_gemini(ana)
            prompt = (
                f"이 클립(약 {pe-ps:.0f}초)에서 한국 쇼츠로 쓸 **핵심 장면 하나**의 정확한 시작·끝 초를 골라라(이 클립 0초 기준).\n"
                "🔴①셋업(상황 시작)부터 ②펀치라인/사이다 한 방 ③그 직후 인물 리액션(표정·반응)까지 **통째로** 포함.\n"
                "🔴②펀치라인·직후 리액션을 자르지 마라 — 가장 흔한 실패다(한 방 직전에 끊으면 실패).\n"
                "🔴③말·비명 중간에서 시작/종료 금지(문장·장면 경계에서 깔끔히). ④길이 28~58초.\n"
                "[출력 JSON만] {\"start\": 0.0, \"end\": 0.0}")
            raw = await call_gemini(GEMINI_PRO_MODEL, uri, prompt)
            j = _parse_json(raw)
            clip.unlink(missing_ok=True)
            gs, ge = float(j.get("start", -1)), float(j.get("end", -1))
            if ge > gs >= 0 and (ge - gs) >= 10:
                ns, ne = round(ps + gs, 2), round(min(dur_full, ps + ge), 2)
                if _p:
                    _p("refining", 82, f"컷 영상보정 {s0:.0f}~{e0:.0f} -> {ns:.0f}~{ne:.0f}")
                print(f"  \u2702\ufe0f 컷 영상보정 [{str(c.get('title',''))[:14]}] {s0:.0f}~{e0:.0f} -> {ns:.0f}~{ne:.0f}", flush=True)
                c["start"], c["end"] = ns, ne
        except Exception as ex:
            print(f"  \u26a0\ufe0f 컷 보정 실패(원본 유지): {str(ex)[:90]}", flush=True)
    return cands


async def _discover_drama_holistic(src, cues, desc="드라마", n_target=6, chunk_sec=600.0, _p=None):
    """🔴 드라마/예능 발굴 v2 (대표님 0607): 텍스트 추측 폐기. 영상 전체를 청크로 '직접 보고' + 대사 종합
    + 전체 줄거리 파악 후 베스트 선별. (1)~10분 청크 시청(+그 구간 대사)→장면요약+후보모먼트,
    (2)전체 줄거리·후보 종합해 가장 터질 베스트 N개 선별. 특정부분 추측 X, 전체 이해 기반."""
    import math
    dur = _dur(src)
    nchunks = max(1, math.ceil(dur / chunk_sec))
    clen = dur / nchunks
    all_moments = []
    summaries = []

    def mmss(s):
        return f"{int(s // 60)}:{int(s % 60):02d}"

    for i in range(nchunks):
        a = round(i * clen, 1)
        b = round(min(dur, (i + 1) * clen), 1)
        if _p:
            _p("discovering", 28 + int(42 * i / max(1, nchunks)), f"영상 직접 시청+줄거리 파악 ({i+1}/{nchunks}구간)")
        ctext = "\n".join(f"[{c['start']-a:.0f}s] {c['text']}" for c in cues
                          if c['end'] > a and c['start'] < b and c.get('text', '').strip())[:6000]
        clip = Path(f"/tmp/_holi_{i}.mp4")
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.1f}", "-to", f"{b:.1f}",
                            "-i", str(src), "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "30", "-c:a", "aac", "-b:a", "80k", str(clip)], check=True, capture_output=True)
            ana = await ensure_inline_video(clip)
            uri = await upload_video_to_gemini(ana)
            prompt = (
                f"한국 드라마/예능 '{desc}'의 한 부분 영상이다(원본 {mmss(a)}~{mmss(b)}). "
                "**영상을 직접 보고** 아래 대사도 종합해서 답하라:\n"
                "1) story: 이 구간 줄거리 시간순 2~4줄 (인물·사건·감정 흐름).\n"
                "2) moments: 쇼츠로 터질 순간(사이다·반전·충격·명장면·빵터짐). 각 start/end(이 클립 0초 기준 초)·"
                "desc(한 줄)·impact(1~10). 영상에서 시각적으로 터지는 순간 + 대사 한 방 둘 다 근거로. "
                "셋업~펀치라인~직후 리액션이 한 덩어리로 들어가게 start/end 잡기. 없으면 빈 배열.\n"
                f"[이 구간 대사]\n{ctext}\n"
                '[출력 JSON만] {"story":"...","moments":[{"start":0.0,"end":0.0,"desc":"...","impact":8}]}')
            raw = await call_gemini(GEMINI_PRO_MODEL, uri, prompt)
            j = _parse_json(raw)
            if isinstance(j, dict):
                summaries.append(f"[{mmss(a)}~{mmss(b)}] {j.get('story', '')}")
                for m in (j.get("moments") or []):
                    try:
                        ms = float(m["start"]) + a
                        me = float(m["end"]) + a
                        if me > ms + 5:
                            all_moments.append({"start": round(ms, 1), "end": round(me, 1),
                                                "desc": str(m.get("desc", "")), "impact": float(m.get("impact", 6))})
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ⚠️ 청크{i}({mmss(a)}~{mmss(b)}) 시청 실패: {str(e)[:80]}", flush=True)
        finally:
            clip.unlink(missing_ok=True)
    print(f"  \U0001f4d6 {nchunks}구간 직접 시청 완료 — 후보 모먼트 {len(all_moments)}개 수집", flush=True)
    if not all_moments:
        return []
    if _p:
        _p("discovering", 74, f"전체 줄거리 종합 → 베스트 {n_target} 선별")
    mlist = "\n".join(f"{j}. {mmss(m['start'])}~{mmss(m['end'])} imp{m['impact']:.0f}: {m['desc']}"
                      for j, m in enumerate(all_moments))
    story = "\n".join(summaries)
    fprompt = (
        f"한국 드라마/예능 '{desc}' 한 편(약 {dur/60:.0f}분)의 **전체 줄거리**와 쇼츠 후보 모먼트 목록이다.\n"
        f"[전체 줄거리]\n{story}\n\n[후보 모먼트]\n{mlist}\n\n"
        f"이 작품 **전체 흐름을 이해한 상태에서** 쇼츠로 가장 터질 **베스트 {n_target}개**를 골라라. "
        "비슷·중복은 빼고 다양하게(같은 류 반복 X), 후크+사이다+공유각 강한 것만. "
        "각 후보는 셋업~펀치라인~직후 리액션이 통째로 들어가게 start/end(원본 초) 잡기(30~58초).\n"
        '[출력 JSON만] {"picks":[{"id":0,"start":0.0,"end":0.0,"title":"12자 후크 제목","impact":9,"summary":"한 줄 요약"}]}')
    raw = await gemini_text(fprompt)
    fj = _parse_json(raw)
    cands = []
    picks = (fj.get("picks") or []) if isinstance(fj, dict) else []
    for p in picks[:n_target]:
        try:
            s = float(p["start"]); e = float(p["end"])
            if e > s + 8:
                cands.append({"start": round(s, 1), "end": round(e, 1), "title": str(p.get("title", "")),
                              "summary": str(p.get("summary", "")), "reason": str(p.get("title", "")),
                              "impact_score": float(p.get("impact", 7)), "type": "drama"})
        except Exception:
            pass
    print(f"  ✅ 전체 줄거리 기반 베스트 {len(cands)}개 선별 완료", flush=True)
    return cands


async def discover_candidates(url: str, out_dir, pipeline_type: str = "drama",
                                on_progress=None) -> dict:
    """v3 발굴 — 영상에서 임팩트/반전 후보를 전부 찾아 저장 (제작 X).
    Returns {"candidates": [...], "source_duration", "source_size_mb"}.
    각 후보: {start, end, twist_sec, title, summary, structure, impact_score, type}
    """
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    def _p(stage, pct, msg):
        print(f"[{pct:>3}%] {stage}: {msg}", flush=True)
        if on_progress:
            try:
                on_progress(stage, pct, msg)
            except Exception:
                pass

    _p("downloading", 5, f"다운로드 ({pipeline_type})")
    _is_movie = pipeline_type == "movie"
    _is_anime = pipeline_type == "anime"
    _is_folktale = pipeline_type == "folktale"
    src, cues, src_dur, src_mb = await _download_and_cues(url, OUT, _p, skip_transcribe=(_is_movie or _is_anime or _is_folktale))

    if _is_anime:
        # ── 애니 발굴엔진 = anime_hl (나루토 레퍼런스 액션 하이라이트, reference_naruto_edit_method) ──
        # Gemini가 영상 보고 액션 클라이맥스 윈도우 선택 + 핵심 비트 컷(필살기·결정타 통째/늘어짐 컷).
        # 출력(자막·캐리커처·TTS·카드)은 render_selected가 기존 반바지 그대로.
        _p("discovering", 20, "애니 prep (OCR/씬 검출) — 처음부터")
        cands = await asyncio.to_thread(_discover_anime_hl, src, OUT, "애니", 5, "")
        _p("discovering", 92, f"애니 하이라이트 {len(cands)}편 발굴")
    elif _is_folktale:
        # ── 동화 발굴엔진 = folktale_hl (레몬사이다式 스토리 축약, STT 문장단위 컷) ──
        # 진행자 오프닝/엔딩·곁가지 제외, 대사 통째(문장 단위). 컷 점프는 상황설명 자막이 메움(점프메타 제공).
        _p("discovering", 20, "동화 STT(환각방지) + 스토리 비트 선택 — 처음부터")
        cands = await asyncio.to_thread(_discover_folktale, src, OUT, "전래동화", 1, "")
        _p("discovering", 92, f"동화 {len(cands)}편 발굴")
    elif _is_movie:
        # ── 대표님 발굴엔진: 신호기반(음량+자막밀도) 명장면 발굴 ──
        # 영화는 하드섭 → Whisper(2시간 전사 병목) 폐기, OCR로 자막 직접 읽음(빠르고 정확).
        # Gemini가 자막 텍스트로 명장면을 '추측'하면 엉뚱한 구간(관광/한담)을 집기도 해서,
        # 실제 신호(소리 크고 대사 빽빽한 구간)로 명장면을 찾고 Gemini는 '라벨'만 단다.
        _p("discovering", 20, "하드섭 OCR (자막밀도+대사)")
        cues = await asyncio.to_thread(_ocr_movie_cues, src)
        _p("discovering", 45, "신호기반 발굴 (음량 RMS + 자막밀도)")
        windows = await asyncio.to_thread(_discover_movie_signal, src, cues, 8, 58)
        _p("discovering", 65, f"명장면 {len(windows)}개 발굴 — 각 클립 영상 라벨링")
        cands = await _label_movie_windows(src, windows, cues)
    elif pipeline_type in ("drama", "talk"):
        # 🔴 드라마/예능 (대표님 0607): 영상 전체 직접 시청 + 대사 + 전체 줄거리 종합 후 선별 (텍스트 추측 폐기)
        _p("discovering", 28, "영상 전체 직접 시청 + 줄거리 파악 후 선별")
        cands = await _discover_drama_holistic(src, cues, "예능" if pipeline_type == "talk" else "드라마", 6, _p=_p)
        if not cands:  # 폴백: 홀리스틱 실패 시 텍스트 발굴
            transcript = "\n".join(f"[{c['start']:.0f}s] {c['text']}" for c in cues)[:80000]
            raw = await gemini_text(PROMPT_DISCOVER_DRAMA_TPL.format(transcript=transcript))
            cands = ((_parse_json(raw) or {}).get("candidates")) or []
    else:
        _p("discovering", 30, f"Gemini 발굴 ({pipeline_type})")
        transcript = "\n".join(f"[{c['start']:.0f}s] {c['text']}" for c in cues)
        if len(transcript) > 80000:
            transcript = "\n".join(f"[{c['start']:.0f}s] {c['text']}" for c in cues[::2])
        tpl = DISCOVER_PROMPTS_BY_TYPE.get(pipeline_type, PROMPT_DISCOVER_DRAMA_TPL)
        raw = await gemini_text(tpl.format(transcript=transcript))
        parsed = _parse_json(raw)
        cands = parsed.get("candidates") or []
    # 정렬: impact_score 높은 순
    for i, c in enumerate(cands):
        c["idx"] = i
        c.setdefault("impact_score", 5)
    cands.sort(key=lambda c: -float(c.get("impact_score", 5)))
    # 재인덱싱 (정렬 후)
    for i, c in enumerate(cands):
        c["idx"] = i
    (OUT / "candidates.json").write_text(
        json.dumps({"candidates": cands, "source_duration": src_dur,
                    "source_size_mb": src_mb, "type": pipeline_type},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _p("discovered", 100, f"후보 {len(cands)}개 발굴 완료")
    return {"candidates": cands, "source_duration": src_dur,
            "source_size_mb": src_mb, "type": pipeline_type}


async def run_pipeline(url: str, out_dir, cleanup: bool = False,
                       max_highlights: int = 7, on_progress=None,
                       pipeline_type: str = "highlight"):
    """긴 URL → 쇼츠 양산. 워커/스크립트 공용.

    pipeline_type:
      - "highlight": 긴 영상 → 하이라이트 N개 (현재 기본).
      - "drama":    짧은 드라마 → 자연 흐름 1~2편.

    on_progress(stage, percent, message): 옵션 콜백 (DB 진행률).
    반환: {"results": [...], "pass1": {...}, "source_dur": s, "source_mb": m}
    """
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / "source.mp4"

    def _p(stage, pct, msg):
        print(f"[{pct:>3}%] {stage}: {msg}", flush=True)
        if on_progress:
            try:
                on_progress(stage, pct, msg)
            except Exception:
                pass

    _p("downloading", 5, "자막+원본 다운로드 시작")
    subs_task = asyncio.create_task(yt_subs_only(url, str(OUT / "source.%(ext)s")))
    src_task = asyncio.create_task(yt_source(url, str(OUT / "source.%(ext)s")))
    try:
        await subs_task
    except Exception as e:
        print(f"  ⚠️ 자동자막 실패: {e}", flush=True)

    vtt_files = list(OUT.glob("source*.vtt"))
    if vtt_files:
        cues = parse_vtt(vtt_files[0])
        _p("downloading", 15, f"자동자막 {len(cues)}개 cue")
    else:
        await src_task
        _p("downloading", 15, "자동자막 없음 — Whisper transcript")
        audio = OUT / "_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src), "-vn", "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(audio)
        ], check=True, capture_output=True)
        await whisper_srt(audio, OUT, "source.srt")
        audio.unlink(missing_ok=True)
        cues = []
        raise RuntimeError("자동자막 없음 — 별도 처리 필요")

    _p("picking", 20, f"Pass 1 — Gemini ({pipeline_type})")
    transcript = "\n".join(f"[{c['start']:.0f}s] {c['text']}" for c in cues)
    if len(transcript) > 60000:
        transcript = "\n".join(f"[{c['start']:.0f}s] {c['text']}" for c in cues[::2])
    # pipeline_type에 따라 Pass 1 prompt 분기
    if pipeline_type == "drama":
        pass1_prompt_tpl = PROMPT_PASS1_DRAMA_TPL
    else:
        pass1_prompt_tpl = PROMPT_PASS1_TPL
    pass1_task = asyncio.create_task(gemini_text(pass1_prompt_tpl.format(transcript=transcript)))
    res = await asyncio.gather(pass1_task, src_task, return_exceptions=True)
    if isinstance(res[1], Exception):
        raise res[1]
    if isinstance(res[0], Exception):
        raise res[0]
    raw1 = res[0]
    src_mb = src.stat().st_size // 1024 // 1024
    src_dur = _dur(src)
    _p("picking", 30, f"source {src_mb}MB / {src_dur:.0f}초")
    pick = _parse_json(raw1)
    segs = pick.get("segments") or []
    # drama: 영상 길이 기준 편 수 cap (기승전결 단락 — 잘게 쪼개지 않게)
    if pipeline_type == "drama":
        mins = src_dur / 60.0
        if mins <= 5:      cap = 2
        elif mins <= 15:   cap = 4
        elif mins <= 30:   cap = 6
        else:              cap = 8
        if len(segs) > cap:
            print(f"  drama 편 수 {len(segs)}→{cap} (영상 {mins:.0f}분 기준)", flush=True)
            segs = segs[:cap]
    elif len(segs) > max_highlights:
        segs = segs[:max_highlights]
    (OUT / "pass1_picks.json").write_text(json.dumps(pick, ensure_ascii=False, indent=2))
    _p("processing", 35, f"하이라이트 {len(segs)}개 선정")
    if on_progress:
        try:
            on_progress("processing", 35, f"하이라이트 {len(segs)}개", extra={
                "pass1": pick, "highlights_count": len(segs),
                "source_duration": src_dur, "source_size_mb": src_mb,
            })
        except Exception:
            pass

    results = []
    n = max(1, len(segs))
    for i, seg in enumerate(segs, 1):
        try:
            r = await process_highlight(i, seg, src, OUT, OUT, pipeline_type=pipeline_type)
            results.append(r)
        except Exception as ex:
            print(f"  ❌ hl_{i:02d} 실패: {ex}", flush=True)
            results.append({"idx": i, "error": str(ex), "dir": str((OUT / f"hl_{i:02d}").absolute())})
        pct = 35 + int(55 * i / n)
        _p("processing", pct, f"hl_{i:02d} 완료 ({i}/{n})")

    # 캐리커처 생성 — 한 예능당 1번만 (모든 hl 공유, 잡 루트에 저장)
    # 첫 성공한 hl의 final.mp4 사용. 없으면 스킵.
    _p("processing", 92, "캐리커처 인물 분석 중...")
    chars_dir = OUT / "characters"
    char_results = []
    first_final = None
    for r in results:
        if "error" not in r and r.get("final"):
            fp = Path(r["final"])
            if fp.exists():
                first_final = fp
                break
    if first_final:
        try:
            from workers.character_generator import run_character_generation
            _maxc = 6 if pipeline_type == "movie" else (4 if pipeline_type == "folktale" else 3)   # 영화 5~6명 / 동화 주요인물 4명
            char_results = await run_character_generation(first_final, chars_dir, max_chars=_maxc)
            _p("processing", 97, f"캐리커처 {len(char_results)}장 생성")
        except Exception as e:
            print(f"  ⚠️ 캐리커처 생성 실패: {e}", flush=True)
    else:
        print("  ⚠️ 성공한 하이라이트 없음 — 캐리커처 스킵", flush=True)

    if cleanup:
        try:
            src.unlink(missing_ok=True)
            for v in OUT.glob("*.vtt"): v.unlink()
            for v in OUT.glob("_inline_*.mp4"): v.unlink()
        except Exception:
            pass

    _p("completed", 100, f"하이라이트 {len([r for r in results if 'error' not in r])}/{len(segs)} + 캐리커처 {len(char_results)}장")
    return {
        "results": results,
        "pass1": pick,
        "source_duration": src_dur,
        "source_size_mb": src_mb,
        "characters": char_results,
        "characters_dir": str(chars_dir) if chars_dir.exists() else None,
    }


async def render_selected(out_dir, selected_idxs: list[int],
                            pipeline_type: str = "drama",
                            on_progress=None) -> dict:
    """v3 선택 제작 — candidates.json에서 선택한 후보만 클립 생성.
    out_dir에 이미 source.mp4 + candidates.json 있어야 함 (discover 후).
    selected_idxs: 만들 후보의 idx 리스트.
    """
    OUT = Path(out_dir)
    src = OUT / "source.mp4"
    cj = OUT / "candidates.json"
    if not src.exists() or not cj.exists():
        raise RuntimeError("source.mp4 / candidates.json 없음 — 발굴 먼저")

    def _p(stage, pct, msg):
        print(f"[{pct:>3}%] {stage}: {msg}", flush=True)
        if on_progress:
            try:
                on_progress(stage, pct, msg)
            except Exception:
                pass

    data = json.loads(cj.read_text(encoding="utf-8"))
    all_cands = data.get("candidates") or []
    by_idx = {int(c.get("idx", i)): c for i, c in enumerate(all_cands)}
    targets = [by_idx[i] for i in selected_idxs if i in by_idx]
    if not targets:
        raise RuntimeError("선택된 후보 없음")

    results = []
    n = len(targets)
    for k, cand in enumerate(targets, 1):
        # 후보 → process_highlight seg 형식. twist_sec 함께 전달 (반전 배치용).
        seg = {
            "start": float(cand["start"]), "end": float(cand["end"]),
            "type": cand.get("type", pipeline_type),
            "reason": cand.get("summary", ""),
            "twist_sec": cand.get("twist_sec"),
            "title_hint": cand.get("title", ""),
            "_precut": cand.get("_precut"),   # 애니/동화: 비트컷 클립 (있으면 그대로 사용)
            "_dialogue_srt": cand.get("_dialogue_srt"),  # 동화: 발굴 STT 대사 SRT
            "_jumps": cand.get("_jumps"),   # 동화: 정확한 컷 시각+장면 → 상황설명 타이밍/내용 소스
        }
        idx = int(cand.get("idx", k))
        try:
            _base = 90.0 * (k - 1) / n   # 후보 k의 시작 % (여러 개면 분할)
            r = await process_highlight(
                idx + 1, seg, src, OUT, OUT, pipeline_type=pipeline_type,
                on_progress=lambda p, m, _b=_base: _p("rendering", int(_b + p * 0.9 / n), f"{m} ({k}/{n})"))
            r["candidate"] = cand
            results.append(r)
        except Exception as ex:
            print(f"  ❌ 후보{idx} 실패: {ex}", flush=True)
            results.append({"idx": idx, "error": str(ex)})
        _p("processing", int(90 * k / n), f"제작 {k}/{n}")
    # status는 여기서 completed로 바꾸지 X — run_shorts_render_selected이 results_json 저장과
    # 함께 completed 찍음 (저장 전 completed면 재시작 등으로 결과 유실 위험. 2026-05-29 사고).
    _p("processing", 97, f"선택 {len([r for r in results if 'error' not in r])}/{n} 제작 (마무리)")
    return {"results": results}


async def _cli_main():
    url = sys.argv[1]
    out = sys.argv[2]
    cleanup = "--cleanup-source" in sys.argv
    # type 인자: --type=drama / --type=movie / --type=highlight (기본 highlight)
    ptype = "highlight"
    mode = "full"  # full=발굴+자동제작(v2) / discover=발굴만(v3) / render=선택제작(v3)
    render_idxs: list[int] = []
    for a in sys.argv:
        if a.startswith("--type="):
            ptype = a.split("=", 1)[1].strip()
        if a == "--discover":
            mode = "discover"
        if a.startswith("--render="):
            mode = "render"
            render_idxs = [int(x) for x in a.split("=", 1)[1].split(",") if x.strip()]
    if mode == "discover":
        r = await discover_candidates(url, out, pipeline_type=ptype)
        print(f"\n=== 발굴 후보 {len(r['candidates'])}개 ===")
        for c in r["candidates"]:
            print(f"  [{c['idx']}] ({c['start']:.0f}~{c['end']:.0f}s, "
                  f"⭐{c.get('impact_score')}) {c.get('title')}")
            print(f"       {c.get('summary')}")
    elif mode == "render":
        # v3 선택 제작 — out_dir의 candidates.json에서 선택 idx만 제작.
        # url 인자는 무시 (placeholder). 예: ... x <out_dir> --type=anime --render=0,1
        r = await render_selected(out, render_idxs, pipeline_type=ptype)
        ok = [x for x in r["results"] if "error" not in x]
        print(f"\n=== 선택 제작 {len(ok)}/{len(r['results'])}개 완료 ===")
        for x in r["results"]:
            if "error" in x:
                print(f"  ❌ idx={x.get('idx')}: {x['error']}")
            else:
                print(f"  ✅ {x.get('hl_dir', x.get('idx'))}")
    else:
        await run_pipeline(url, out, cleanup=cleanup, pipeline_type=ptype)


if __name__ == "__main__":
    asyncio.run(_cli_main())
