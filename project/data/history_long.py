"""Maximum-history daily series for spread charts (pair alignment)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from data.polygon import fetch_daily_max

logger = logging.getLogger(__name__)


def _align_log_daily(dy: pd.DataFrame, dx: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if dy.empty or dx.empty or "close" not in dy.columns or "close" not in dx.columns:
        return None
    dy = dy.sort_index().copy()
    dx = dx.sort_index().copy()
    dy["k"] = pd.DatetimeIndex(pd.to_datetime(dy.index)).strftime("%Y-%m-%d")
    dx["k"] = pd.DatetimeIndex(pd.to_datetime(dx.index)).strftime("%Y-%m-%d")
    y = dy.drop_duplicates("k").set_index("k")["close"].astype(float)
    x = dx.drop_duplicates("k").set_index("k")["close"].astype(float)
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if len(j) < 60:
        return None
    j.index = pd.DatetimeIndex(j.index)
    return np.log(j["y"]), np.log(j["x"])


def fetch_long_daily_pair(y_sym: str, x_sym: str, source: str) -> tuple[pd.Series, pd.Series] | None:
    """As much overlapping daily history as Yahoo or Polygon provides."""
    if source == "ibkr":
        return None
    try:
        if source == "yfinance":
            ty = yf.Ticker(y_sym)
            tx = yf.Ticker(x_sym)
            dy = ty.history(period="max", interval="1d", auto_adjust=True)
            dx = tx.history(period="max", interval="1d", auto_adjust=True)
            dy.columns = [str(c).lower() for c in dy.columns]
            dx.columns = [str(c).lower() for c in dx.columns]
        else:
            dy = fetch_daily_max(y_sym)
            dx = fetch_daily_max(x_sym)
            dy.columns = [str(c).lower() for c in dy.columns]
            dx.columns = [str(c).lower() for c in dx.columns]
        return _align_log_daily(dy, dx)
    except Exception as e:
        logger.warning("Long daily history failed %s %s: %s", y_sym, x_sym, e)
        return None
