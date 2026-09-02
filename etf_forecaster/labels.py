"""Targets: direction and magnitude of the h-day forward move, h = 1..5.

Vol targets (^VIX, ^VVIX, VIXY) are labelled on log levels - same monotonic direction,
but magnitudes on a scale where their mean reversion is well behaved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_forecaster.config import ticker_meta


def make_labels(bars: pd.DataFrame, ticker: str, horizons: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    is_vol = ticker_meta(ticker).get("target_type") == "vol"
    close = np.log(bars["close"]) if is_vol else np.log(bars["close"])
    out = pd.DataFrame(index=bars.index)
    for h in horizons:
        fwd = close.shift(-h) - close       # h-day forward log return (or log-level change)
        out[f"y_dir_{h}"] = (fwd > 0).astype(float).where(fwd.notna())
        out[f"y_ret_{h}"] = fwd
    return out
