"""MACRO_FAST (mf_), MACRO_SLOW (ms_) and EARNINGS (earn_) blocks.

MACRO_SLOW is built point-in-time: for every release (vintage) date we recompute the
derived metrics from the series exactly as it was known on that date, then forward-fill
onto the trading calendar. No revised value is ever visible before its vintage date.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from etf_forecaster.config import macro_cfg
from etf_forecaster.data import alfred

log = logging.getLogger(__name__)

_WEEKLY_LAGGED = {"nfci", "fed_bs"}  # published ~1 week after their obs period


def _z(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window, min_periods=window // 4).mean()) / \
        s.rolling(window, min_periods=window // 4).std().replace(0, np.nan)


def fast_macro_features(panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """mf_ block from the daily FRED panel (shared across tickers)."""
    out = pd.DataFrame(index=index)
    for label in panel.columns:
        s = panel[label]
        if label in _WEEKLY_LAGGED:
            s = s.shift(7, freq="D") if isinstance(s.index, pd.DatetimeIndex) else s
        s = s.reindex(index).ffill(limit=10)
        out[f"mf_{label}"] = s
        out[f"mf_{label}_d5"] = s.diff(5)
        out[f"mf_{label}_d21"] = s.diff(21)
        out[f"mf_{label}_z"] = _z(s, 252)
    # curve/credit interactions
    if {"mf_us_10y", "mf_us_2y"} <= set(out.columns):
        out["mf_curve_mom"] = (out["mf_us_10y"] - out["mf_us_2y"]).diff(21)
    if "mf_hy_oas" in out and "mf_ig_oas" in out:
        out["mf_credit_ratio"] = out["mf_hy_oas"] / out["mf_ig_oas"].replace(0, np.nan)
    return out


def _pit_metrics(series_id: str, label: str, freq: str,
                 index: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Point-in-time derived metrics for one vintage series, daily-aligned."""
    mat = alfred.load_vintage_matrix(series_id)
    if mat is None or mat.empty:
        return None
    vdates = alfred.vintage_dates(series_id, matrix=mat)
    vdates = vdates[(vdates >= index.min() - pd.Timedelta(days=3650)) & (vdates <= index.max())]
    per_year = {"m": 12, "w": 52, "q": 4}.get(freq, 12)
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    # before first vintage: synthesize "releases" from the earliest vintage + pub lag
    first_col = mat.iloc[:, 0].dropna()
    pub_lag = int(macro_cfg()["alfred_slow"].get(series_id, {}).get("pub_lag_days", 45))
    pre_dates = [d + pd.Timedelta(days=pub_lag) for d in first_col.index
                 if len(vdates) == 0 or d + pd.Timedelta(days=pub_lag) < vdates.min()]
    events: list[tuple[pd.Timestamp, pd.Series | None]] = \
        [(d, None) for d in pre_dates] + [(v, None) for v in vdates]
    events.sort(key=lambda x: x[0])
    for asof, _ in events:
        known = alfred.as_known_series(series_id, asof, matrix=mat)
        if known is None or len(known) < per_year + 1:
            continue
        latest = known.iloc[-1]
        yoy = latest / known.iloc[-1 - per_year] - 1 if abs(known.iloc[-1 - per_year]) > 1e-9 else np.nan
        k3 = max(1, per_year // 4)
        mom3 = (latest / known.iloc[-1 - k3]) ** (per_year / k3) - 1 \
            if len(known) > k3 and abs(known.iloc[-1 - k3]) > 1e-9 else np.nan
        rows[asof] = {
            f"{label}_lvl": float(latest),
            f"{label}_yoy": float(yoy) if np.isfinite(yoy) else np.nan,
            f"{label}_mom3": float(mom3) if np.isfinite(mom3) else np.nan,
        }
    if not rows:
        return None
    pit = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    pit = pit[~pit.index.duplicated(keep="last")]
    daily = pit.reindex(index.union(pit.index)).ffill().reindex(index)
    # days since last release
    rel = pd.Series(pit.index, index=pit.index).reindex(index.union(pit.index)).ffill().reindex(index)
    daily[f"{label}_days_since"] = (index - pd.DatetimeIndex(rel)).days / 30.0
    return daily


def slow_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """ms_ block: vintage-aligned slow macro (shared across tickers)."""
    cfg = macro_cfg()["alfred_slow"]
    frames: list[pd.DataFrame] = []
    for series_id, meta in cfg.items():
        if meta["label"] in ("corp_profits",):  # earnings block handles this one
            continue
        try:
            df = _pit_metrics(series_id, meta["label"], meta["freq"], index)
            if df is not None:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("PIT metrics failed for %s: %s", series_id, exc)
    if not frames:
        return pd.DataFrame(index=index)
    out = pd.concat(frames, axis=1)
    out.columns = [f"ms_{c}" for c in out.columns]
    # z-score the yoy/mom columns over 10y so scales are comparable
    for c in list(out.columns):
        if c.endswith("_yoy") or c.endswith("_mom3"):
            out[c + "_z"] = _z(out[c], 2520)
    return out


def earnings_features(index: pd.DatetimeIndex, *, shiller: pd.DataFrame | None,
                      real_10y: pd.Series | None) -> pd.DataFrame:
    """earn_ block: Shiller earnings growth/CAPE (3-month publication shift) +
    vintage corporate profits."""
    out = pd.DataFrame(index=index)
    if shiller is not None and len(shiller):
        sh = shiller.copy()
        # monthly values for month m are not fully known until ~3 months later
        known = sh.shift(3)
        daily = known.reindex(index.union(known.index)).ffill().reindex(index)
        earn = daily["earnings"]
        out["earn_growth_yoy"] = earn / earn.shift(252) - 1
        out["earn_cape"] = daily["cape"]
        out["earn_cape_z"] = _z(daily["cape"], 2520)
        ep = earn / daily["sp_price"].replace(0, np.nan)
        out["earn_yield"] = ep
        if real_10y is not None:
            r = real_10y.reindex(index).ffill(limit=10) / 100.0
            out["earn_fed_spread"] = ep - r
        else:
            out["earn_fed_spread"] = ep - daily["gs10"] / 100.0
    try:
        cp = _pit_metrics("CP", "corp_profits", "q", index)
        if cp is not None:
            for c in cp.columns:
                out[f"earn_{c}"] = cp[c]
    except Exception as exc:  # noqa: BLE001
        log.debug("corp profits PIT failed: %s", exc)
    return out
