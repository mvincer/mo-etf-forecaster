"""30-day average daily volume (ADV) ranking."""

from __future__ import annotations

import pandas as pd

from data.etf_asset_groups import diverse_top_by_underlying


def adv_30d_from_daily(df: pd.DataFrame) -> float | None:
    """Mean daily volume over last 30 rows (trading days) with data."""
    if df.empty or "volume" not in df.columns:
        return None
    v = pd.to_numeric(df["volume"], errors="coerce").dropna()
    if len(v) < 10:
        return None
    tail = v.tail(30)
    return float(tail.mean())


def rank_by_adv_30d(
    daily_frames: dict[str, pd.DataFrame],
    top_n: int = 100,
) -> list[str]:
    scores: list[tuple[str, float]] = []
    for sym, df in daily_frames.items():
        adv = adv_30d_from_daily(df)
        if adv is None or adv <= 0:
            continue
        scores.append((sym, adv))
    scores.sort(key=lambda x: x[1], reverse=True)
    return diverse_top_by_underlying(scores, top_n)
