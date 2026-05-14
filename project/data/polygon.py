"""Master / Polygon.io aggregates (stocks & ETFs). API key: POLYGON_API_KEY in environment."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import requests

from data.timeframes import POLYGON_AGG, start_date_for_polygon, trim_last_n_bars

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def get_polygon_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_MASSIVE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set POLYGON_API_KEY in your environment or project/.env for Master (Polygon.io) data."
        )
    return key


def _norm_sym(symbol: str) -> str:
    return symbol.upper().strip().replace("-", ".")


def fetch_aggs_range(
    symbol: str,
    multiplier: int,
    timespan: str,
    start: date | str,
    end: date | str,
    *,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Polygon v2 aggregates. `timespan`: minute, hour, day, week, month, quarter, year.
    Returns DataFrame indexed by timezone-aware timestamps (America/New_York).
    """
    sym = _norm_sym(symbol)
    key = api_key or get_polygon_api_key()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        logger.warning("Polygon %s: %s", symbol, data.get("error"))
        return pd.DataFrame()
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    rows = []
    for b in results:
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert("America/New_York")
        rows.append(
            {
                "datetime": ts,
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b.get("v", 0),
            }
        )
    df = pd.DataFrame(rows).set_index("datetime")
    df.index.name = "datetime"
    return df


def _normalize_polygon_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out["ticker"] = _norm_sym(symbol)
    return out


def fetch_many_aggs(
    tickers: Iterable[str],
    bar_interval: str,
    n_bars: int = 1000,
    *,
    api_key: str | None = None,
    max_workers: int = 4,
    throttle_s: float = 0.12,
) -> dict[str, pd.DataFrame]:
    """Aggregates for many tickers; trims to last `n_bars`. `bar_interval`: 1m,5m,15m,1h,1d."""
    if bar_interval not in POLYGON_AGG:
        bar_interval = "5m"
    mult, timespan = POLYGON_AGG[bar_interval]
    end = date.today()
    start = start_date_for_polygon(bar_interval, n_bars)
    key = api_key or get_polygon_api_key()
    tickers = list(dict.fromkeys(tickers))
    out: dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in tickers}

    def one(sym: str) -> tuple[str, pd.DataFrame]:
        time.sleep(throttle_s)
        try:
            df = fetch_aggs_range(sym, mult, timespan, start, end, api_key=key)
            df = _normalize_polygon_df(df, sym)
            df = trim_last_n_bars(df, n_bars)
            return sym, df
        except Exception as e:
            logger.warning("Polygon failed %s: %s", sym, e)
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(one, t): t for t in tickers}
        for fut in as_completed(futs):
            sym, df = fut.result()
            out[sym] = df
    return out


def fetch_many_5m(
    tickers: Iterable[str],
    lookback_days: int = 60,
    *,
    api_key: str | None = None,
    max_workers: int = 4,
    throttle_s: float = 0.12,
) -> dict[str, pd.DataFrame]:
    """Recent 5-minute bars for many tickers (Polygon cap depends on plan; ~60d typical)."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    key = api_key or get_polygon_api_key()
    tickers = list(dict.fromkeys(tickers))
    out: dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in tickers}

    def one(sym: str) -> tuple[str, pd.DataFrame]:
        time.sleep(throttle_s)
        try:
            df = fetch_aggs_range(sym, 5, "minute", start, end, api_key=key)
            df["ticker"] = sym
            return sym, df
        except Exception as e:
            logger.warning("Polygon failed %s: %s", sym, e)
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(one, t): t for t in tickers}
        for fut in as_completed(futs):
            sym, df = fut.result()
            out[sym] = df
    return out


def fetch_daily_max(
    symbol: str,
    *,
    years_back: int = 25,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Daily bars as far back as Polygon returns (from ~years_back)."""
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    return fetch_aggs_range(symbol, 1, "day", start, end, api_key=api_key)
