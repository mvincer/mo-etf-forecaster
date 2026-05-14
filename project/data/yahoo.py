"""Yahoo Finance intraday data via yfinance (default 5-minute bars)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from data.universe import ETF_SEED_UNIVERSE

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo caps intraday history at ~60 days for sub-daily intervals.
DEFAULT_BAR_INTERVAL = "5m"
DEFAULT_BAR_PERIOD = "60d"

# Yahoo symbols for macro / sentiment. `us_2y_yield` tries a short list because 2YY=F is often sparse intraday.
MACRO_SYMBOLS: dict[str, str | list[str]] = {
    "gold": "GC=F",
    "crude_oil": "CL=F",
    "us_10y_yield": "^TNX",
    "us_2y_yield": ["2YY=F", "ZT=F"],
    "us_3m_yield": "^IRX",
    "copper": "HG=F",
    "vix": "^VIX",
}

# Minimum 5m bars to accept a macro series (else try next candidate).
_MACRO_MIN_BARS = 2500

RISK_REVERSAL_PLACEHOLDER: dict[str, str] = {
    "spx_rr_35d_call_put_vol": "N/A — not on Yahoo Finance (use CBOE / prime broker options surface)",
    "spx_rr_25d_call_put_vol": "N/A — not on Yahoo Finance",
    "spx_rr_15d_call_put_vol": "N/A — not on Yahoo Finance",
    "spx_rr_10d_call_put_vol": "N/A — not on Yahoo Finance",
    "spx_rr_5d_call_put_vol": "N/A — not on Yahoo Finance",
}


def _normalize_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "adj close" in out.columns:
        out = out.rename(columns={"adj close": "adj_close"})
    out["ticker"] = ticker
    return out


def download_bars_single(
    ticker: str,
    period: str = DEFAULT_BAR_PERIOD,
    interval: str = DEFAULT_BAR_INTERVAL,
) -> pd.DataFrame:
    """Intraday bars (Yahoo intraday cap ~60d for 5m)."""
    t = yf.Ticker(ticker)
    df = t.history(interval=interval, period=period, auto_adjust=True)
    return _normalize_download(df, ticker)


def download_daily_many(
    tickers: Iterable[str],
    period: str = "90d",
    batch_size: int = 20,
) -> dict[str, pd.DataFrame]:
    """Daily bars for ADV / slow features."""
    return download_bars_many(tickers, period=period, interval="1d", batch_size=batch_size)


def download_bars_many(
    tickers: Iterable[str],
    period: str = DEFAULT_BAR_PERIOD,
    interval: str = DEFAULT_BAR_INTERVAL,
    batch_size: int = 20,
) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}
    result: dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in tickers}
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        raw = yf.download(
            tickers=" ".join(chunk),
            interval=interval,
            period=period,
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            continue
        if len(chunk) == 1:
            t = chunk[0]
            result[t] = _normalize_download(raw, t)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for t in chunk:
                try:
                    sub = raw[t]
                except (KeyError, TypeError):
                    sub = pd.DataFrame()
                if not sub.empty:
                    result[t] = _normalize_download(sub, t)
        else:
            t0 = chunk[0]
            result[t0] = _normalize_download(raw, t0)
    return result


def last_n_calendar_days(df: pd.DataFrame, days: int = 20) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    end = df.index.max()
    start = end - pd.Timedelta(days=days)
    return df.loc[df.index >= start]


def rank_etfs_by_volume(
    frames: dict[str, pd.DataFrame],
    top_n: int = 100,
    lookback_days: int = 20,
) -> list[str]:
    from data.etf_asset_groups import diverse_top_by_underlying

    scores: list[tuple[str, float]] = []
    for sym, df in frames.items():
        if df.empty or "volume" not in df.columns:
            continue
        w = last_n_calendar_days(df, lookback_days)
        if w.empty:
            continue
        vol = float(pd.to_numeric(w["volume"], errors="coerce").fillna(0).sum())
        scores.append((sym, vol))
    scores.sort(key=lambda x: x[1], reverse=True)
    return diverse_top_by_underlying(scores, top_n)


def fetch_macro_bars(
    period: str = DEFAULT_BAR_PERIOD,
    interval: str = DEFAULT_BAR_INTERVAL,
    *,
    min_bars: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Keys match MACRO_SYMBOLS. Second return maps each label to the Yahoo symbol actually used."""
    threshold = _MACRO_MIN_BARS if min_bars is None else int(min_bars)
    out: dict[str, pd.DataFrame] = {}
    resolved: dict[str, str] = {}
    for label, spec in MACRO_SYMBOLS.items():
        candidates = spec if isinstance(spec, list) else [spec]
        chosen = candidates[0]
        frame = pd.DataFrame()
        for sym in candidates:
            df = download_bars_single(sym, period=period, interval=interval)
            if len(df) >= threshold:
                frame = df
                chosen = sym
                break
            if len(df) > len(frame):
                frame = df
                chosen = sym
        out[label] = frame
        resolved[label] = chosen
    return out, resolved


def fetch_etf_universe_bars(
    period: str = DEFAULT_BAR_PERIOD,
    interval: str = DEFAULT_BAR_INTERVAL,
) -> dict[str, pd.DataFrame]:
    return download_bars_many(ETF_SEED_UNIVERSE, period=period, interval=interval)


def risk_reversal_status() -> pd.DataFrame:
    """Explain missing Yahoo series for SPX risk reversals."""
    rows = []
    for key, note in RISK_REVERSAL_PLACEHOLDER.items():
        rows.append({"series": key, "yahoo_symbol": None, "note": note})
    return pd.DataFrame(rows)


def cache_parquet(frames: dict[str, pd.DataFrame], name: str) -> Path:
    path = CACHE_DIR / f"{name}.parquet"
    combined = []
    for t, df in frames.items():
        if df.empty:
            continue
        x = df.copy()
        x["symbol"] = t
        combined.append(x.reset_index().rename(columns={"index": "datetime"}))
    if not combined:
        return path
    pd.concat(combined, ignore_index=True).to_parquet(path, index=False)
    return path
