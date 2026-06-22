"""일본어 멀티유즈 — 한국어 자막 박힌 영상 → 일본어 생활어 번역.

대표님 룰:
- 의역 위주, 일본 사람이 헷갈리지 X
- 타이밍 완전 똑같게 (한국어 자막 timing 그대로)
- 출력: 일본어 제목 + 상황 srt + 쨉쨉이 srt

흐름:
1. 영상 → Gemini Files API 업로드
2. Gemini Flash 분석 — 영상 안 한국어 자막 읽고 + 일본어 생활어 번역
3. srt 파일 만들기 (상황 / 쨉쨉이)
4. 결과 자료에 저장
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Any

from workers.auto_subtitle import (
    upload_video_to_gemini,
    call_gemini,
    write_srt,
    GEMINI_FLASH_MODEL,
    apply_user_gemini_key,
    ensure_inline_video,
)
from api import database as db


JAPANESE_TRANSLATION_PROMPT = """この動画は韓国の1分ショート動画で、画面に韓国語字幕が焼き込まれています。

# あなたのタスク
1. 動画内の韓国語字幕を全て読み取り
2. 日本人が見ても自然に分かる「生活日本語」に意訳
3. タイミングは韓国語字幕と完全に同じ
4. 日本語タイトル + 状況説明字幕 + ツッコミ字幕 (쨉쨉이) を出力

# 翻訳ルール (絶対)

## 意訳優先 — 直訳禁止
- 韓国の文化・スラングは日本のものに置き換える
- 日本人が「???」とならないように
- 韓国MZスラング → 日本のネットスラング・若者言葉で対応

## 韓国 MZ表現 → 日本語対応表
| 韓国 | 日本語 (生活·自然) |
|---|---|
| 시전 | やってのける / 発動 / かます |
| X됨 | やられた / 即死 / 終了 |
| 광탈 | 即退場 / リタイア / 瞬殺 |
| 참교육 | 制裁 / お仕置き / 教育 |
| 어그로 | 煽り / 挑発 |
| ㄷㄷ | ヤバい / こわっ / ゾクっ |
| ㄹㅇ | マジ / 本当に |
| ㅋㅋ | www / 笑 |
| 갓X | 神X / 最強X |
| 레전드 | レジェンド / 伝説 |
| 미쳤다 | やばすぎ / 狂ってる |
| 지렸다 | ヤバい / 失神もの |
| 넘사벽 | 別格 / 雲の上 / 越えられない壁 |
| 클라스 | クラス違い / 格が違う |
| 현타 | 我に返る / 現実逃避 |
| 멘붕 | メンタル崩壊 / メンブレ |
| 국룰 | お決まり / 鉄則 |
| 찐 | ガチ / 本物 |
| 꿀팁 | お得情報 / 神Tips |

## キャラクター・あだ名
- 「뚱이」(パトリック・スポンジボブの) → 「パトリック」または「ブタくん」など見た目から自然に
- 「갓비둘」→ 「神鳩」または「鳩神」
- 「차주인」→ 「車のオーナー」または「車主」
- 韓国特有のあだ名 → 日本人にも分かる呼び名で

## 文章スタイル
- TTS用「허경환体」相当の最初の字幕は **「〜という」/「〜って」/「〜らしい」** で終わる
  - 例: "NPCが餌を探していたとか"
  - 例: "GTAでこんなことが可能らしい"
- 一行15文字以内 (日本語は文字数余裕がある、漢字使う)
- 自然なネットスラング・若者言葉を活用
- 強制MZ翻訳ではなく、文脈に合わせて自然に

# 出力 — JSON のみ

```json
{
  "duration_sec": 27.95,
  "korean_title_original": "영상에 박힌 한국어 상단 타이틀 (있으면, 없으면 빈 문자열)",
  "japanese_title": "映像内に焼き込む日本語タイトル (15~20文字、強いフッキング)",
  "youtube_upload_title_jp": "YouTubeアップロード用メイン日本語タイトル (35~60文字、SEO+好奇心+人気キーワード。末尾に #shorts または #ショート inline 1~2個 OK)。例: 'まさかNPCがブチ切れて一発逆転したGTA神回 #shorts #GTA'",
  "youtube_upload_title_candidates_jp": [
    "(SEO+好奇心) YouTube用候補1 (35~60文字)",
    "(キーワード+ミーム) YouTube用候補2",
    "(オチ示唆+ハッシュタグ) YouTube用候補3",
    "(刺激+数字) YouTube用候補4",
    "(質問) YouTube用候補5"
  ],
  "youtube_description_jp": "YouTube用日本語説明文 (200~500文字、ストーリー1~2行 + 視聴者の好奇心 + 最後にハッシュタグ5個 inline)。例: 'ただのNPCかと思ったら... 結局一発でやられた話。最後まで見ると何かが違います。 #shorts #GTA #ミーム #ヤバい #韓国'",
  "hashtags_jp": ["#shorts", "#ショート", "#日本語", "#バズる", "...合計8~12個。日本語+英語 mix"],
  "title_candidates_jp": [
    "(好奇心) タイトル候補1",
    "(オチ示唆) タイトル候補2",
    "(疑問) タイトル候補3?",
    "(衝撃) タイトル候補4",
    "(対比) タイトル候補5"
  ],
  "japanese_situation_subtitles": [
    {"start": 0.0, "end": 2.0, "korean": "원본 한국어 자막 (영상에서 읽은 거)", "japanese": "状況説明の日本語訳"}
  ],
  "japanese_jjap_jjap_i_subtitles": [
    {"start": 1.5, "end": 2.5, "korean": "원본 한국어 쨉쨉이", "japanese": "* ツッコミ * など"}
  ],
  "japanese_dialogue_subtitles": [
    {"start": 5.0, "end": 6.0, "korean": "원본 한국어 대사", "japanese": "セリフの日本語訳"}
  ]
}
```

[重要]
- 각 자막에 `korean` + `japanese` 둘 다 박을 것 (대조 표시용)
- `korean`은 영상에서 실제로 본 한국어 자막 (없으면 빈 문자열)
- `japanese`는 일본어 의역
- 타이밍 (start/end)는 한국어 자막과 정확히 같게
- 上記JSON形式のみ出力。他のテキストは出さない。
"""


async def run_japanese_multiuse(job_id: int, video_path: Path) -> None:
    """영상 → 일본어 자막/제목 자동 생성"""
    OUT_DIR = Path(__file__).parent.parent / "data" / "japanese_multiuse"
    out_dir = OUT_DIR / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이 작업 만든 사용자 개인 Gemini 키 있으면 적용 (프리랜서 비용 분리)
    try:
        apply_user_gemini_key((db.get_japanese_multiuse_job(job_id) or {}).get("user_id"))
    except Exception:
        pass

    try:
        db.update_japanese_multiuse_job(
            job_id, status="uploading", progress=10,
            progress_message="영상 준비 중 (큰 영상은 압축)..",
        )
        # 큰 영상은 저화질 압축본 inline (Files API 회피)
        analysis_video = await ensure_inline_video(video_path)
        file_uri = await upload_video_to_gemini(analysis_video)

        db.update_japanese_multiuse_job(
            job_id, status="analyzing", progress=40,
            progress_message="한국어 자막 읽고 일본어 의역 중..",
        )

        result = await call_gemini(
            GEMINI_FLASH_MODEL, file_uri, JAPANESE_TRANSLATION_PROMPT,
            temperature=0.3, max_retries=3,
        )

        if not isinstance(result, dict):
            raise RuntimeError(f"Gemini 응답이 dict 아님: {type(result)}")

        db.update_japanese_multiuse_job(
            job_id, status="generating", progress=80,
            progress_message="srt 파일 생성 중..",
        )

        # srt 파일 만들기
        situation_subs = result.get("japanese_situation_subtitles", [])
        jjap_subs = result.get("japanese_jjap_jjap_i_subtitles", [])
        dialogue_subs = result.get("japanese_dialogue_subtitles", [])

        situation_srt = out_dir / "situation_jp.srt"
        jjap_srt = out_dir / "jjap_jjap_i_jp.srt"
        dialogue_srt = out_dir / "dialogue_jp.srt"

        if situation_subs:
            write_srt(situation_subs, situation_srt, "japanese")
        if jjap_subs:
            write_srt(jjap_subs, jjap_srt, "japanese")
        if dialogue_subs:
            write_srt(dialogue_subs, dialogue_srt, "japanese")

        # 자료에 저장
        cost = 0.05  # Flash 1번 호출 추정
        db.update_japanese_multiuse_job(
            job_id, status="completed", progress=100,
            progress_message="끝",
            japanese_title=result.get("japanese_title", ""),
            title_candidates_jp=json.dumps(result.get("title_candidates_jp", []), ensure_ascii=False),
            korean_subtitles_extracted=json.dumps(
                [{"start": s.get("start"), "end": s.get("end"), "text": s.get("korean", "")}
                 for s in (result.get("japanese_situation_subtitles") or [])
                 if s.get("korean")],
                ensure_ascii=False),
            japanese_situation_srt_path=str(situation_srt) if situation_subs else None,
            japanese_jjap_jjap_i_srt_path=str(jjap_srt) if jjap_subs else None,
            japanese_dialogue_srt_path=str(dialogue_srt) if dialogue_subs else None,
            gemini_result=json.dumps(result, ensure_ascii=False),
            cost_usd=cost,
            completed_at_now=True,
        )

    except Exception as e:
        db.update_japanese_multiuse_job(
            job_id, status="failed", progress=0,
            progress_message=f"실패: {str(e)[:300]}",
            error=str(e)[:500],
        )
        raise
