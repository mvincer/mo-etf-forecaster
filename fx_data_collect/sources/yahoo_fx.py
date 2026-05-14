"""Daily FX OHLCV via Yahoo Finance (fallback when candledata missing / stale tail)."""

from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


def _naive_normalized_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar dates as timezone-naive (UTC wall clock) — fixes mixing tz-aware FXCM + Yahoo."""
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).tz_localize(None).normalize()
    out.index = idx
    return out


def download_yahoo_fx_daily(
    yahoo_ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    retries: int = 3,
) -> pd.DataFrame:
    """Return Open/High/Low/Close/Volume indexed by normalized calendar date."""
    import yfinance as yf

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    end_s = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(yahoo_ticker)
            raw = t.history(start=start_s, end=end_s, interval="1d", auto_adjust=True)
            break
        except Exception as e:
            last_exc = e
            logger.warning("yfinance %s attempt %s: %s", yahoo_ticker, attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
            raw = None
    else:
        logger.error("yfinance failed %s: %s", yahoo_ticker, last_exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df.index)).normalize())
    for old, new in rename.items():
        if old in df.columns:
            out[new] = pd.to_numeric(df[old], errors="coerce")
    if "Close" not in out.columns:
        return pd.DataFrame()
    for col in ("Open", "High", "Low"):
        if col not in out.columns:
            out[col] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out.dropna(subset=["Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def stitch_fxcm_with_yahoo_tail(
    fxcm: pd.DataFrame,
    yahoo_ticker: str,
    *,
    min_date: pd.Timestamp,
    max_stale_days: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Prefer FXCM candledata; append Yahoo for any gap after last FXCM bar through **today**.
    If FXCM empty, use Yahoo from ``min_date``.
    """
    meta: dict = {"fxcm_rows": 0, "yahoo_rows": 0, "yahoo_only": False}
    today = pd.Timestamp.now(tz=None).normalize()
    min_d = pd.Timestamp(min_date)
    if min_d.tz is not None:
        min_d = min_d.tz_convert("UTC").tz_localize(None)
    min_d = min_d.normalize()

    if fxcm is None or fxcm.empty:
        y = download_yahoo_fx_daily(yahoo_ticker, min_d, today)
        meta["yahoo_only"] = True
        meta["yahoo_rows"] = len(y)
        return y.loc[y.index >= min_d], meta

    fx = fxcm.sort_index().copy()
    fx = _naive_normalized_index(fx)
    fx = fx[~fx.index.duplicated(keep="last")]
    meta["fxcm_rows"] = len(fx)
    last = fx.index.max()

    # Recent enough: trim and return
    if last >= today - pd.Timedelta(days=max_stale_days):
        out = fx.loc[fx.index >= min_d]
        return out, meta

    start_gap = last + pd.Timedelta(days=1)
    y = download_yahoo_fx_daily(yahoo_ticker, start_gap, today)
    meta["yahoo_rows"] = len(y)
    if y.empty:
        return fx.loc[fx.index >= min_d], meta

    combo = pd.concat([fx, y])
    combo = _naive_normalized_index(combo)
    combo = combo[~combo.index.duplicated(keep="last")].sort_index()
    return combo.loc[combo.index >= min_d], meta
