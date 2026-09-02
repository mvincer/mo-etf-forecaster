"""PRICE block (px_): returns, gaps, range-based vol estimators, higher moments,
and the raw last-X-candle lag block."""

from __future__ import annotations

import numpy as np
import pandas as pd

_RET_WINDOWS = (1, 2, 3, 5, 10, 21, 63, 126, 252)


def _range_vols(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    log_oc_prev = np.log(o / c.shift())
    ann = np.sqrt(252)

    park = np.sqrt((log_hl**2).rolling(window).mean() / (4 * np.log(2))) * ann
    gk = np.sqrt((0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
                 .rolling(window).mean().clip(lower=0)) * ann
    rs = np.sqrt((np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
                 .rolling(window).mean().clip(lower=0)) * ann
    sigma_o = (log_oc_prev**2).rolling(window).mean()
    sigma_c = (log_co**2).rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = np.sqrt((sigma_o + k * sigma_c + (1 - k) * rs**2 / 252).clip(lower=0)) * ann
    return pd.DataFrame({
        f"px_park_{window}": park, f"px_gk_{window}": gk,
        f"px_rs_{window}": rs, f"px_yz_{window}": yz,
    })


def price_features(bars: pd.DataFrame, *, n_lags: int = 60, log_level: bool = False) -> pd.DataFrame:
    """log_level=True for mean-reverting vol targets (^VIX/^VVIX): returns computed on
    log levels so features and labels live on the same scale."""
    out = pd.DataFrame(index=bars.index)
    close = np.log(bars["close"]) if log_level else bars["close"]
    logret = close.diff() if log_level else np.log(bars["close"] / bars["close"].shift())

    for w in _RET_WINDOWS:
        out[f"px_ret_{w}"] = logret.rolling(w).sum()
    vol21 = logret.rolling(21).std()
    for w in (5, 21, 63):
        out[f"px_ret_{w}_z"] = out[f"px_ret_{w}"] / (vol21 * np.sqrt(w)).replace(0, np.nan)

    # overnight vs intraday decomposition
    overnight = np.log(bars["open"] / bars["close"].shift())
    intraday = np.log(bars["close"] / bars["open"])
    out["px_overnight_5"] = overnight.rolling(5).sum()
    out["px_overnight_21"] = overnight.rolling(21).sum()
    out["px_intraday_5"] = intraday.rolling(5).sum()
    out["px_intraday_21"] = intraday.rolling(21).sum()
    out["px_gap_abs"] = overnight.abs()

    # location within ranges
    roll_max = bars["close"].rolling(252, min_periods=60).max()
    roll_min = bars["close"].rolling(252, min_periods=60).min()
    out["px_dist_52w_high"] = bars["close"] / roll_max - 1
    out["px_dist_52w_low"] = bars["close"] / roll_min - 1
    out["px_range_pos_252"] = (bars["close"] - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    cummax = bars["close"].cummax()
    out["px_drawdown"] = bars["close"] / cummax - 1

    for w in (10, 21):
        out = out.join(_range_vols(bars, w))
    out["px_skew_63"] = logret.rolling(63).skew()
    out["px_kurt_63"] = logret.rolling(63).kurt()

    vol = bars.get("volume")
    if vol is not None and vol.sum() > 0:
        vmean = vol.rolling(63).mean()
        vstd = vol.rolling(63).std()
        out["px_volume_z"] = (vol - vmean) / vstd.replace(0, np.nan)

    # last-X-candle block: z-scored return lags and normalized range lags
    ret_z = logret / vol21.replace(0, np.nan)
    rng = (bars["high"] - bars["low"]) / bars["close"].replace(0, np.nan)
    rng_z = (rng - rng.rolling(63).mean()) / rng.rolling(63).std().replace(0, np.nan)
    lag_cols = {}
    for k in range(1, n_lags + 1):
        lag_cols[f"px_lag_ret_{k}"] = ret_z.shift(k - 1)   # lag 1 = today's bar
        if k <= max(10, n_lags // 3):
            lag_cols[f"px_lag_rng_{k}"] = rng_z.shift(k - 1)
    out = pd.concat([out, pd.DataFrame(lag_cols, index=bars.index)], axis=1)
    return out
