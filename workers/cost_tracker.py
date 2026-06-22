"""Cost tracking helper. Each AI worker calls add_cost(diss_id, usd) after a paid call."""
import sqlite3

DB_PATH = "/Users/shortsking/banbaji-discover/db/discover.db"

PRICES_USD = {
    "gpt_image_2_t2i_high_1024": 0.03,
    "gpt_image_2_i2i_high_1024": 0.03,
    "kie_flux_kontext_img2img": 0.025,
    "gemini_2_5_flash_video_60s": 0.002,
    "gemini_2_5_flash_text": 0.0002,
    "replicate_lora_img": 0.04,
    "runway_i2v_5s": 0.50,
}


def add_cost(diss_id: str | None, usd: float, label: str = ""):
    """누적 cost를 dissection_analyses.cost_usd에 더함. diss_id None이면 무시.
    fail-soft (lock 등 발생해도 raise X)."""
    if not diss_id or not usd or usd <= 0:
        return
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute(
                "UPDATE dissection_analyses SET cost_usd = COALESCE(cost_usd, 0) + ? WHERE id = ?",
                (float(usd), diss_id),
            )
    except sqlite3.OperationalError:
        pass


def add_cost_by_label(diss_id: str | None, label: str, count: int = 1):
    usd = PRICES_USD.get(label, 0.0) * count
    add_cost(diss_id, usd, label)


def total_cost_usd():
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as c:
            row = c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM dissection_analyses").fetchone()
            return float(row[0] or 0)
    except Exception:
        return 0.0
