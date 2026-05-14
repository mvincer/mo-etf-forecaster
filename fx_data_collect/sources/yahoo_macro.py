"""Download daily OHLC for macro Yahoo symbols (commodities, indices, ETFs)."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from fx_data_collect.sources.yahoo_fx import _naive_normalized_index

logger = logging.getLogger(__name__)


def download_yahoo_macro_daily(
    label: str,
    ticker: str,
    *,
    start: str,
    end: str | None = None,
    retries: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return OHLCV DataFrame + metadata dict."""
    import yfinance as yf

    meta: dict[str, Any] = {
        "label": label,
        "yahoo_ticker": ticker,
        "source": "yahoo",
        "status": "pending",
        "start": None,
        "end": None,
        "rows": 0,
        "frequency": "daily",
        "error": None,
    }
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz=None)
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    last_exc: Exception | None = None
    raw = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            raw = t.history(start=start_s, end=end_s, interval="1d", auto_adjust=True)
            break
        except Exception as e:
            last_exc = e
            logger.warning("yfinance %s %s attempt %s: %s", label, ticker, attempt + 1, e)
            time.sleep(1.2 * (attempt + 1))

    if raw is None or raw.empty:
        meta["status"] = "empty_or_failed"
        meta["error"] = str(last_exc) if last_exc else "no_rows"
        return pd.DataFrame(), meta

    df = raw.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    out = pd.DataFrame()
    if "close" in df.columns:
        out["Close"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        meta["status"] = "no_close"
        return pd.DataFrame(), meta
    for col, std in ("open", "Open"), ("high", "High"), ("low", "Low"):
        if col in df.columns:
            out[std] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[std] = out["Close"]
    out["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df.columns else 0.0
    out = _naive_normalized_index(out)
    out = out.dropna(subset=["Close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    meta["status"] = "ok"
    meta["start"] = str(out.index.min())
    meta["end"] = str(out.index.max())
    meta["rows"] = int(len(out))
    return out, meta


def download_all_macro(
    label_to_ticker: dict[str, str],
    *,
    start: str,
    pause_sec: float = 0.35,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, Any]] = []
    end = pd.Timestamp.now(tz=None).strftime("%Y-%m-%d")
    for lab, tic in label_to_ticker.items():
        df, meta = download_yahoo_macro_daily(lab, tic, start=start, end=end)
        reports.append(meta)
        if meta["status"] == "ok":
            frames[lab] = df
        time.sleep(pause_sec)
    return frames, reports
