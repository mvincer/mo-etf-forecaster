"""Support/resistance from clustered pivots — continuous distance/strength features."""

# Vendored from Mo_Dash strategies/chart_patterns/sr_levels.py - keep in sync manually.

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import PIVOT_ORDER, SR_MIN_TOUCHES, SR_CLUSTER_ATR, TP_R
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.pivots import extract_pivots
from etf_forecaster.patterns.swing_points import compute_atr


def detect_sr(
    ohlc: pd.DataFrame,
    pair: str,
    tf: str,
    *,
    only: str | None = None,
    label: str | None = None,
) -> List[PatternSignal]:
    """Detect S/R bounces.

    ``only`` restricts to "support" or "resistance"; ``label`` overrides the
    pattern name written on each signal.
    """
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    order = PIVOT_ORDER.get(tf, 6)
    pivots = extract_pivots(closes, order)
    if len(pivots) < SR_MIN_TOUCHES + 2:
        return []

    idx = ohlc.index
    signals: List[PatternSignal] = []
    mid = len(closes) // 2
    atr_mid = float(np.nanmean(atr[max(0, mid - 50) : mid + 50]))
    if atr_mid <= 0 or np.isnan(atr_mid):
        return []

    # Build levels from pivots confirmed before mid+ (causal for later bars)
    early = [p for p in pivots if p.confirm_i < mid]
    clusters: List[List] = []
    for p in sorted(early, key=lambda x: x.price):
        placed = False
        for cl in clusters:
            if abs(p.price - np.mean([x.price for x in cl])) <= SR_CLUSTER_ATR * atr_mid:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])

    levels = []
    for cl in clusters:
        if len(cl) < SR_MIN_TOUCHES:
            continue
        level = float(np.mean([x.price for x in cl]))
        last_touch = max(x.confirm_i for x in cl)
        strength = float(len(cl))
        kind = "resistance" if np.mean([x.kind for x in cl]) > 0 else "support"
        levels.append((level, last_touch, strength, kind))

    name = label or "support_resistance"
    for level, last_touch, strength, kind in levels:
        if only is not None and kind != only:
            continue
        for k in range(last_touch + 5, min(len(closes), last_touch + 120)):
            if np.isnan(atr[k]) or atr[k] <= 0:
                continue
            tol = 0.2 * atr[k]
            if kind == "support" and lows[k] <= level + tol and closes[k] > level:
                entry = float(closes[k])
                sl = level - 0.5 * atr[k]
                risk = entry - sl
                if risk <= 0:
                    continue
                # require clear rejection: bounce ≥ 0.3 ATR from low
                if closes[k] - lows[k] < 0.3 * atr[k]:
                    continue
                tp = entry + TP_R * risk
                dist_above = np.nan
                dist_below = 0.0
                signals.append(
                    PatternSignal(
                        pattern=name,
                        direction="buy",
                        pair=pair,
                        timeframe=tf,
                        confirm_idx=k,
                        confirm_ts=pd.Timestamp(idx[k]),
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        neckline=level,
                        pattern_high=level + atr[k],
                        pattern_low=level,
                        swing_indices=[],
                        meta={
                            "quality": min(100.0, 30 + 12 * strength),
                            "level_strength": strength,
                            "dist_support_atr": dist_below,
                            "dist_resistance_atr": dist_above,
                            "level_type": 0.0,
                            "level": level,
                            "start_i": last_touch,
                            "peak_equality_atr": 0.0,
                            "depth_atr": risk / atr[k],
                            "time_symmetry": 1.0,
                            "prior_pctile": 0.5,
                            "pattern_r2": 0.5,
                            "pattern_height_atr": risk / atr[k],
                            "pattern_bars": float(k - last_touch),
                            "dist_to_neckline_atr": abs(entry - level) / atr[k],
                            "neckline_slope_atr": 0.0,
                            "head_dominance_atr": 0.0,
                            "line_slope": 0.0,
                            "line_intercept": level,
                            "atr_ref": float(atr[k]),
                        },
                    )
                )
                break
            if kind == "resistance" and highs[k] >= level - tol and closes[k] < level:
                entry = float(closes[k])
                sl = level + 0.5 * atr[k]
                risk = sl - entry
                if risk <= 0:
                    continue
                if highs[k] - closes[k] < 0.3 * atr[k]:
                    continue
                tp = entry - TP_R * risk
                signals.append(
                    PatternSignal(
                        pattern=name,
                        direction="sell",
                        pair=pair,
                        timeframe=tf,
                        confirm_idx=k,
                        confirm_ts=pd.Timestamp(idx[k]),
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        neckline=level,
                        pattern_high=level,
                        pattern_low=level - atr[k],
                        swing_indices=[],
                        meta={
                            "quality": min(100.0, 30 + 12 * strength),
                            "level_strength": strength,
                            "level_type": 1.0,
                            "level": level,
                            "start_i": last_touch,
                            "peak_equality_atr": 0.0,
                            "depth_atr": risk / atr[k],
                            "time_symmetry": 1.0,
                            "prior_pctile": 0.5,
                            "pattern_r2": 0.5,
                            "pattern_height_atr": risk / atr[k],
                            "pattern_bars": float(k - last_touch),
                            "dist_to_neckline_atr": abs(entry - level) / atr[k],
                            "neckline_slope_atr": 0.0,
                            "head_dominance_atr": 0.0,
                            "line_slope": 0.0,
                            "line_intercept": level,
                            "atr_ref": float(atr[k]),
                        },
                    )
                )
                break

    return signals
