"""자막 자동 학습 시스템 — 297영상 분석 → 표현 풀 + 패턴 + 상황 카테 추출.

대표님 룰:
- 11채널 × 27영상 = 297영상 분석
- 매너리즘 방지 — 표현 풀 누적 (대체 X)
- "친구가 영상 보고 자막 다는 느낌" 자연 다양성
- 학습은 톤·성격만, 자막 문장은 매번 새로
- 매 50영상마다 텔레그램 진행률
- 비용 ~$14.85 / ₩20,000 안

흐름:
1. subtitle_learning_queue에서 comments_done 영상 받음
2. yt-dlp로 영상 다운 (best quality mp4)
3. Gemini Files API 업로드
4. Gemini Flash 학습 prompt 호출 (패턴 추출)
5. 자료에 저장:
   - subtitle_learnings (raw)
   - subtitle_situations (카테)
   - subtitle_expression_pool (표현)
   - subtitle_phrase_frequency (빈도)
6. 영상 파일 삭제
7. 매 50영상마다 텔레그램 알림
"""
import os
import json
import asyncio
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
import httpx

from workers.auto_subtitle import (
    upload_video_to_gemini,
    call_gemini,
    _get_gemini_key,
    GEMINI_FLASH_MODEL,
)
from workers import notify
from api import database as db


DATA_DIR = Path(__file__).parent.parent / "data" / "subtitle_learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp 절대 경로 (venv 안에 설치돼 있음, PATH에 없음)
YT_DLP_BIN = str(Path(__file__).parent.parent / "venv" / "bin" / "yt-dlp")


# 학습 prompt — 패턴 추출 위주 (자막 생성과는 별개)
LEARNING_PROMPT = """이 1분 쇼츠 영상은 한국 인기 채널 "{channel_name}"의 인기 영상이다. 자막 자동 생성 학습을 위해 깊이 분석해서 다음 패턴을 추출해줘.

[목적]
이 영상의 자막 톤·표현·패턴을 학습해서 미래 자막 자동 생성 시 표현 풀에 누적한다. "친구가 영상 보고 자막 다는 느낌"의 자연스러운 톤이 핵심.

[댓글 인사이트]
이 영상의 인기 댓글 top 5:
{top_comments}

[추출할 정보]

1. **상황 카테고리** — 이 영상의 핵심 상황을 한 단어로 (예: "gta_stunt", "baseball_reflex", "animal_unexpected_reaction", "music_skill_shock", "car_crash_revenge" 등)
   - 너무 일반화 X (예: "funny_video" X)
   - 너무 세부 X (예: "gta_npc_kill_user_with_grenade_in_alley" X)
   - 적정한 추상화 수준 (예: "gta_user_vs_npc_revenge")

2. **영상 흐름 요약** — 5~10초 단위로 무슨 일이 일어나는지 간결하게

3. **자막 패턴 추출** — 만약 이 영상에 자막 만든다면 어떤 자막 4종이 자연스러울지:
   - **hook_subtitle**: 첫 후크 자막 1개 (~데/~인데/~한다는데 어미)
   - **situation_subtitles**: 상황 자막 list (한 줄 12자 미만)
   - **jjap_jjap_i**: 쨉쨉이 list (* X * / (X) / ㅋ / ㄷㄷ / 대화 형식)
   - **ending_marker**: 마무리 표현 (ㄷㄷ / ... / 무 마무리 / X됨 등)

4. **MZ 신조어** — 이 영상 분위기에 자연스러운 MZ 표현 list (예: ["광탈", "시전", "참교육", "X됨"])

5. **밈 형식** — 이 영상에 어울리는 밈 (예: "대화 형식", "반전 마무리", "캐릭터 의인화", "트위터 톤" 중 골라)

6. **캐릭터 별명** — 영상에 나오는 인물·동물·물체에 자연스러운 별명 (외모·동작 기반)

7. **분위기 분류**:
   - mood: "강한 충격" / "허무·아이러니" / "코미디" / "감동" / "의외" / "위트"
   - dominant_emotion: 시청자가 느끼는 주된 감정

8. **시청자 해석 (댓글 기반)**:
   - 시청자가 이 영상에서 뭘 보고 웃었는지/감탄했는지
   - 위트 키워드·밈 댓글에서 추출

[출력 — JSON으로만]
```json
{{
  "situation_category": "gta_user_vs_npc_revenge",
  "situation_name": "GTA NPC가 유저 응징",
  "video_summary": "유저가 NPC를 도발하다가 NPC가 수류탄으로 응징하는 흐름. ...",
  "key_moments": [
    {{"time_sec": 2.0, "moment": "유저 발사 시작"}},
    {{"time_sec": 14.5, "moment": "NPC가 수류탄 시전"}}
  ],
  "hook_subtitle": "NPC가 먹이감을 찾고 있는데",
  "situation_subtitles": [
    "갑자기 발사 시작",
    "근데 NPC가 가만 안 있음",
    "참교육 시전"
  ],
  "jjap_jjap_i": ["??", "* 수류탄 *", "ㄷㄷ"],
  "ending_marker": "ㄷㄷ",
  "mz_words": ["시전", "참교육", "X됨", "광탈"],
  "meme_format": "반전 마무리",
  "character_nicknames": ["NPC", "갓NPC", "도발러"],
  "mood": "위트",
  "dominant_emotion": "통쾌함",
  "viewer_insights": [
    "NPC가 진짜로 응징한 의외성",
    "수류탄으로 한 방 처리한 깔끔함"
  ]
}}
```

다른 텍스트 X. JSON으로만.
"""


async def _download_video(video_id: str, video_url: str) -> Path:
    """yt-dlp로 영상 다운 (best quality)"""
    out_path = DATA_DIR / f"{video_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    cmd = [
        YT_DLP_BIN,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--remote-components", "ejs:github",
        "-o", str(out_path),
        "--quiet", "--no-warnings",
        video_url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp 실패: {stderr.decode()[:300]}")
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"영상 다운 실패 — 파일 없음 또는 비어있음")
    return out_path


def _format_top_comments(comments_json: str | None) -> str:
    if not comments_json:
        return "(댓글 없음)"
    try:
        comments = json.loads(comments_json)
    except Exception:
        return "(댓글 자료 파싱 실패)"
    if not comments:
        return "(댓글 없음)"
    lines = []
    for i, c in enumerate(comments[:5], start=1):
        text = (c.get("text") or "").replace("\n", " ")[:200]
        likes = c.get("likes", 0)
        lines.append(f"{i}. ({likes}👍) {text}")
    return "\n".join(lines)


async def _analyze_video(file_uri: str, channel_name: str, top_comments_str: str) -> dict:
    """Gemini Flash로 학습용 분석. call_gemini가 이미 JSON 파싱된 dict 반환."""
    prompt = LEARNING_PROMPT.format(channel_name=channel_name, top_comments=top_comments_str)
    result = await call_gemini(GEMINI_FLASH_MODEL, file_uri, prompt, temperature=0.3)
    if not isinstance(result, dict):
        raise RuntimeError(f"Gemini 응답이 dict 아님: type={type(result)} value={str(result)[:200]}")
    if not result.get("situation_category"):
        raise RuntimeError(f"필수 필드 (situation_category) 없음: keys={list(result.keys())[:10]}")
    return result


def _safe_str(v):
    """list/dict는 JSON으로 변환, None은 None 그대로, 나머지는 str"""
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v) if not isinstance(v, (int, float, bool)) else v


def _safe_json(v, default=None):
    """list/dict는 JSON dump, string이면 그대로 (이미 JSON 가정), 나머지는 default JSON"""
    if v is None:
        return json.dumps(default if default is not None else [], ensure_ascii=False)
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _save_learning(conn, video_id: str, channel_id: str, channel_name: str,
                    analysis: dict, top_comments_json: str):
    cur = conn.cursor()

    # 1) subtitle_learnings INSERT
    cur.execute("""
        INSERT OR REPLACE INTO subtitle_learnings (
            video_id, channel_id, channel_name, situation_category,
            video_summary, key_moments_json,
            hook_subtitle, situation_subtitles_json, jjap_jjap_i_json, dialogue_json,
            ending_marker, mz_words_json, meme_format, character_nicknames_json,
            top_comments_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id, channel_id, channel_name,
        _safe_str(analysis.get("situation_category")),
        _safe_str(analysis.get("video_summary")),
        _safe_json(analysis.get("key_moments")),
        _safe_str(analysis.get("hook_subtitle")),
        _safe_json(analysis.get("situation_subtitles")),
        _safe_json(analysis.get("jjap_jjap_i")),
        _safe_json([]),
        _safe_str(analysis.get("ending_marker")),
        _safe_json(analysis.get("mz_words")),
        _safe_str(analysis.get("meme_format")),
        _safe_json(analysis.get("character_nicknames")),
        top_comments_json or "[]",
    ))

    # 2) subtitle_situations INSERT/UPDATE
    cat_key_raw = analysis.get("situation_category")
    cat_key = _safe_str(cat_key_raw) if cat_key_raw else None
    cat_name_raw = analysis.get("situation_name", cat_key)
    cat_name = _safe_str(cat_name_raw) if cat_name_raw else cat_key
    if cat_key:
        summary_raw = analysis.get("video_summary", "") or ""
        if isinstance(summary_raw, (list, dict)):
            summary_str = json.dumps(summary_raw, ensure_ascii=False)
        else:
            summary_str = str(summary_raw)
        cur.execute("""
            INSERT INTO subtitle_situations (category_key, category_name, description, example_video_ids, usage_count)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(category_key) DO UPDATE SET
                example_video_ids = json_insert(
                    coalesce(example_video_ids, '[]'),
                    '$[#]',
                    ?
                )
        """, (cat_key, cat_name, summary_str[:500],
              json.dumps([video_id]), video_id))

    # 3) subtitle_expression_pool 확장 (mz_words)
    mz_words = analysis.get("mz_words", [])
    if not isinstance(mz_words, list):
        mz_words = [mz_words] if mz_words else []
    for word in mz_words:
        if not word or not isinstance(word, str):
            continue
        meaning_key = _guess_meaning_key(word)
        cur.execute("""
            INSERT INTO subtitle_expression_pool (meaning_key, meaning_name, expression, tone, use_count, source_video_ids_json)
            VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT(meaning_key, expression) DO UPDATE SET
                source_video_ids_json = json_insert(
                    coalesce(source_video_ids_json, '[]'),
                    '$[#]',
                    ?
                )
        """, (meaning_key, _meaning_name(meaning_key), word, "학습", json.dumps([video_id]), video_id))

    # 4) subtitle_phrase_frequency 갱신 (mz_words + ending)
    phrases = [w for w in mz_words if isinstance(w, str) and w]
    ending = analysis.get("ending_marker")
    if isinstance(ending, str) and ending:
        phrases.append(ending)
    for p in phrases:
        if not p or not isinstance(p, str):
            continue
        cur.execute("""
            INSERT INTO subtitle_phrase_frequency (phrase, category, use_count, last_used_at)
            VALUES (?, 'learning', 1, CURRENT_TIMESTAMP)
            ON CONFLICT(phrase) DO UPDATE SET
                use_count = use_count + 1,
                last_used_at = CURRENT_TIMESTAMP
        """, (p,))

    # 5) queue status 갱신
    cur.execute("UPDATE subtitle_learning_queue SET status='analyzed', completed_at=CURRENT_TIMESTAMP WHERE video_id=?", (video_id,))

    conn.commit()


# 의미 추정 (간단 mapping — 학습으로 자동 확장)
MEANING_KEYWORDS = {
    "defeat": ["광탈", "X됨", "골로", "리타이어", "GG", "퇴장", "폭사", "X패", "패배"],
    "victory": ["참교육", "응징", "갓", "1승", "박살", "쓸어버림"],
    "attempt": ["시전", "도전", "출격", "발동"],
    "provocation": ["너냐", "어그로", "시비", "도발"],
    "reveal": ["사실", "알고보니", "진짜로", "X였음"],
    "shock": ["ㄷㄷ", "미친", "헐", "와우", "ㄹㅇ"],
}


def _guess_meaning_key(word: str) -> str:
    for key, keywords in MEANING_KEYWORDS.items():
        for kw in keywords:
            if kw in word or word in kw:
                return key
    return "etc"


def _meaning_name(key: str) -> str:
    names = {
        "defeat": "패배", "victory": "승리", "attempt": "시도",
        "provocation": "도발", "reveal": "정체·반전", "shock": "충격",
        "etc": "기타",
    }
    return names.get(key, key)


async def learn_video(conn, queue_row: tuple, semaphore: asyncio.Semaphore) -> tuple[str, bool, str]:
    """한 영상 학습. (video_id, success, message) 반환"""
    video_id, channel_id, channel_name, video_url, comments_json = queue_row
    async with semaphore:
        try:
            # 1. 영상 다운
            video_path = await _download_video(video_id, video_url)

            # 2. Gemini 업로드
            file_uri = await upload_video_to_gemini(video_path)

            # 3. 분석
            top_comments_str = _format_top_comments(comments_json)
            analysis = await _analyze_video(file_uri, channel_name, top_comments_str)

            # 4. 자료에 저장
            _save_learning(conn, video_id, channel_id, channel_name, analysis, comments_json)

            # 5. 영상 파일 삭제 (디스크 절약)
            try:
                video_path.unlink()
            except Exception:
                pass

            return (video_id, True, analysis.get("situation_category", "unknown"))
        except Exception as e:
            cur = conn.cursor()
            cur.execute(
                "UPDATE subtitle_learning_queue SET status='failed', error=? WHERE video_id=?",
                (str(e)[:500], video_id),
            )
            conn.commit()
            return (video_id, False, str(e)[:200])


async def learn_all(session_id: str | None = None, parallel: int = 5,
                     notify_every: int = 50, notify_chat_ids: list[str] | None = None) -> dict:
    """모든 comments_done 영상 학습. 진행률 자동 알림."""
    import sqlite3

    db_path = Path(__file__).parent.parent / "db" / "discover.db"
    conn = sqlite3.connect(str(db_path))

    if not session_id:
        session_id = f"learn_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # 학습 대상 영상 받기
    cur = conn.cursor()
    cur.execute("""
        SELECT video_id, channel_id, channel_name, video_url, comments_json
        FROM subtitle_learning_queue
        WHERE status='comments_done'
        ORDER BY id
    """)
    queue = cur.fetchall()
    total = len(queue)

    # learning_progress 박기
    cur.execute("""
        INSERT INTO learning_progress (session_id, total_videos, status, started_at)
        VALUES (?, ?, 'running', CURRENT_TIMESTAMP)
    """, (session_id, total))
    conn.commit()

    if total == 0:
        await notify.send_telegram("자막 학습: 학습 대상 영상 없음 (comments_done 영상이 0개)")
        return {"session_id": session_id, "total": 0, "message": "no videos to learn"}

    await notify.send_telegram(f"📚 자막 학습 시작 — 총 {total}영상, 동시 {parallel}개 진행")

    semaphore = asyncio.Semaphore(parallel)
    completed = 0
    failed = 0
    failed_details: list[str] = []
    categories: dict[str, int] = {}

    async def _wrap(qrow):
        nonlocal completed, failed
        result = await learn_video(conn, qrow, semaphore)
        vid, success, msg = result
        if success:
            completed += 1
            categories[msg] = categories.get(msg, 0) + 1
        else:
            failed += 1
            failed_details.append(f"{vid}: {msg}")

        # 자료에 진행 갱신
        cur.execute(
            "UPDATE learning_progress SET completed=?, failed=?, current_video_id=? WHERE session_id=?",
            (completed, failed, vid, session_id),
        )
        conn.commit()

        # 매 notify_every마다 텔레그램 알림
        done = completed + failed
        if done > 0 and done % notify_every == 0:
            top_cats = sorted(categories.items(), key=lambda x: -x[1])[:5]
            cat_summary = ", ".join([f"{k}({v})" for k, v in top_cats])
            await notify.send_telegram(
                f"📚 학습 진행 — {done}/{total} (✅{completed} ❌{failed})\n"
                f"top 카테: {cat_summary}"
            )
        return result

    await asyncio.gather(*[_wrap(qrow) for qrow in queue])

    # 끝 보고
    cur.execute("""
        UPDATE learning_progress SET status='completed', completed_at=CURRENT_TIMESTAMP, summary=?
        WHERE session_id=?
    """, (json.dumps({
        "completed": completed,
        "failed": failed,
        "categories": categories,
    }, ensure_ascii=False), session_id))

    # 통계
    cur.execute("SELECT COUNT(*) FROM subtitle_expression_pool")
    expr_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subtitle_situations")
    sit_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM subtitle_patterns")
    pat_count = cur.fetchone()[0]
    cur.execute("SELECT phrase, use_count FROM subtitle_phrase_frequency ORDER BY use_count DESC LIMIT 10")
    top_phrases = cur.fetchall()
    conn.commit()
    conn.close()

    # 매너리즘 점검
    mannerism_signals = [f"{p}({c}회)" for p, c in top_phrases if c >= max(5, total * 0.15)]
    mannerism_note = ""
    if mannerism_signals:
        mannerism_note = f"\n⚠️ 매너리즘 신호 — {', '.join(mannerism_signals[:5])} 사용 비율 높음. 자막 생성 시 풀에서 다른 표현 자연 선택 권장"

    cost_estimate = completed * 0.05
    summary_msg = (
        f"✅ 자막 학습 끝\n"
        f"- 분석 성공: {completed}/{total} (실패 {failed})\n"
        f"- 상황 카테: {sit_count}개\n"
        f"- 표현 풀: {expr_count}개\n"
        f"- 패턴: {pat_count}개 (압축은 별도 단계)\n"
        f"- 비용 추정: ${cost_estimate:.2f} (~₩{int(cost_estimate * 1380):,})\n"
        f"- top 표현 (학습 빈도): {', '.join([f'{p}({c})' for p, c in top_phrases[:5]])}"
        f"{mannerism_note}"
    )
    await notify.send_telegram(summary_msg)

    return {
        "session_id": session_id,
        "total": total,
        "completed": completed,
        "failed": failed,
        "categories": categories,
        "expr_count": expr_count,
        "situation_count": sit_count,
        "cost_usd": cost_estimate,
        "mannerism_signals": mannerism_signals,
    }


if __name__ == "__main__":
    import sys
    parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    result = asyncio.run(learn_all(parallel=parallel))
    print(json.dumps(result, ensure_ascii=False, indent=2))
