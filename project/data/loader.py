"""Load ETF universe + macro by selected data source, bar interval, and bar count."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from data import yahoo
from data.adv_rank import rank_by_adv_30d
from data.preset_pairs import all_preset_tickers
from data.timeframes import (
    BAR_INTERVAL_CHOICES,
    YAHOO_INTERVAL,
    trim_last_n_bars,
    yahoo_period_for_n_bars,
)
from data.universe import ETF_SEED_UNIVERSE

DataSource = Literal["yfinance", "master_polygon", "ibkr"]

N_BARS_DOWNLOAD = 1000


def load_etf_universe(
    source: DataSource,
    bar_interval: str,
    n_bars: int = N_BARS_DOWNLOAD,
) -> dict[str, object]:
    """
    Phase 1: rank ETFs by 30-day average daily volume on daily bars; top 100 with distinct underlyings.
    Downloads last `n_bars` bars at `bar_interval` for top-100 ∪ preset pair tickers (seed universe).
    """
    if bar_interval not in BAR_INTERVAL_CHOICES:
        bar_interval = "5m"
    n_bars = max(50, int(n_bars))

    if source == "ibkr":
        raise RuntimeError(
            "Interactive Brokers data is not wired yet. Use Yahoo Finance or Master (Polygon)."
        )

    daily = yahoo.download_daily_many(ETF_SEED_UNIVERSE, period="90d")
    top100 = rank_by_adv_30d(daily, top_n=100)

    seed = set(ETF_SEED_UNIVERSE)
    tickers = sorted((set(top100) | all_preset_tickers()) & seed)
    yf_interval = YAHOO_INTERVAL[bar_interval]
    period = yahoo_period_for_n_bars(bar_interval, n_bars)

    if source == "yfinance":
        raw = yahoo.download_bars_many(tickers, period=period, interval=yf_interval)
    else:
        from data.polygon import fetch_many_aggs

        raw = fetch_many_aggs(tickers, bar_interval, n_bars=n_bars)

    etf_raw: dict[str, pd.DataFrame] = {}
    for sym, df in raw.items():
        etf_raw[sym] = trim_last_n_bars(df, n_bars) if df is not None else pd.DataFrame()

    etf_top = {s: etf_raw[s] for s in top100 if s in etf_raw and not etf_raw[s].empty}

    min_macro = max(50, min(500, n_bars // 2))
    macro, macro_resolved = yahoo.fetch_macro_bars(
        period=period,
        interval=yf_interval,
        min_bars=min_macro,
    )
    macro = {k: trim_last_n_bars(v, n_bars) for k, v in macro.items()}

    return {
        "etf_raw": etf_raw,
        "top100": top100,
        "etf_top": etf_top,
        "macro": macro,
        "macro_resolved": macro_resolved,
        "bar_interval": bar_interval,
        "bar_period": period,
        "n_bars": n_bars,
    }
