"""Causal channel detector — parallel rails with ≥3 touches each.

Uses neurotrader-style slope optimization on a trailing window of pivots.
Features: width_atr, position_in_channel, slope_norm, touch counts, age.
"""

# Vendored from Mo_Dash strategies/chart_patterns/channels.py - keep in sync manually.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import (
    CHANNEL_MIN_QUALITY,
    CHANNEL_MIN_TOUCHES,
    CHANNEL_MIN_WIDTH_ATR,
    CHANNEL_SLOPE_TOL,
    TP_R,
)
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.pivots import alternating_pivots, extract_pivots
from etf_forecaster.patterns.swing_points import compute_atr


def _fit_rail(idxs: np.ndarray, prices: np.ndarray) -> Optional[Tuple[float, float]]:
    if len(idxs) < 2:
        return None
    slope, intercept = np.polyfit(idxs.astype(float), prices.astype(float), 1)
    return float(slope), float(intercept)


def detect_channels(ohlc: pd.DataFrame, pair: str, tf: str) -> List[PatternSignal]:
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    from pattern_config import PIVOT_ORDER

    order = PIVOT_ORDER.get(tf, 6)
    pivots = alternating_pivots(extract_pivots(closes, order))
    idx = ohlc.index
    signals: List[PatternSignal] = []
    seen = set()

    # Use trailing windows of recent pivots
    for end in range(CHANNEL_MIN_TOUCHES * 2, len(pivots) + 1):
        window = pivots[max(0, end - 12) : end]
        tops = [p for p in window if p.kind == 1]
        bots = [p for p in window if p.kind == -1]
        if len(tops) < CHANNEL_MIN_TOUCHES or len(bots) < CHANNEL_MIN_TOUCHES:
            continue

        hi_fit = _fit_rail(np.array([p.idx for p in tops]), np.array([p.price for p in tops]))
        lo_fit = _fit_rail(np.array([p.idx for p in bots]), np.array([p.price for p in bots]))
        if hi_fit is None or lo_fit is None:
            continue
        hs, hi = hi_fit
        ls, li = lo_fit
        # slope agreement
        if abs(hs) + abs(ls) > 1e-12:
            if abs(hs - ls) / (abs(hs) + abs(ls) + 1e-12) > CHANNEL_SLOPE_TOL:
                continue

        confirm_i = max(window[-1].confirm_i, window[-1].idx + order)
        if confirm_i >= len(closes) - 1:
            continue
        atr_ref = atr[confirm_i]
        if np.isnan(atr_ref) or atr_ref <= 0:
            continue

        upper = hs * confirm_i + hi
        lower = ls * confirm_i + li
        width = abs(upper - lower)
        if width < CHANNEL_MIN_WIDTH_ATR * atr_ref:
            continue

        slope_norm = ((hs + ls) / 2.0) / atr_ref
        # bounce signals in next few bars after channel established
        for k in range(confirm_i + 1, min(len(closes), confirm_i + 40)):
            if np.isnan(atr[k]) or atr[k] <= 0:
                continue
            u = hs * k + hi
            l = ls * k + li
            w = abs(u - l)
            if w <= 0:
                continue
            pos = (closes[k] - l) / w
            tol = 0.12 * atr[k]
            key = (k, round(l, 6), round(u, 6))
            if key in seen:
                continue

            if lows[k] <= l + tol and closes[k] > l and pos < 0.35:
                entry = float(closes[k])
                sl = l - 0.5 * atr[k]
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + TP_R * risk
                quality = min(100.0, 40 + 10 * len(tops) + 10 * len(bots) + 20 * min(1.0, w / (2 * atr[k])))
                if quality < CHANNEL_MIN_QUALITY:
                    continue
                seen.add(key)
                signals.append(
                    PatternSignal(
                        pattern="channel",
                        direction="buy",
                        pair=pair,
                        timeframe=tf,
                        confirm_idx=k,
                        confirm_ts=pd.Timestamp(idx[k]),
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        neckline=l,
                        pattern_high=u,
                        pattern_low=l,
                        swing_indices=[p.idx for p in window],
                        meta={
                            "quality": quality,
                            "channel_width_atr": w / atr[k],
                            "position_in_channel": pos,
                            "slope_norm": slope_norm,
                            "touch_upper": float(len(tops)),
                            "touch_lower": float(len(bots)),
                            "channel_age_bars": float(k - window[0].idx),
                            "peak_equality_atr": 0.0,
                            "depth_atr": w / atr[k],
                            "time_symmetry": 1.0,
                            "prior_pctile": 0.5,
                            "pattern_r2": 0.7,
                            "pattern_height_atr": w / atr[k],
                            "pattern_bars": float(k - window[0].idx),
                            "dist_to_neckline_atr": abs(entry - l) / atr[k],
                            "neckline_slope_atr": slope_norm,
                            "head_dominance_atr": 0.0,
                            "line_slope": ls,
                            "line_intercept": li,
                            "atr_ref": float(atr[k]),
                        },
                    )
                )
                break
            if highs[k] >= u - tol and closes[k] < u and pos > 0.65:
                entry = float(closes[k])
                sl = u + 0.5 * atr[k]
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - TP_R * risk
                quality = min(100.0, 40 + 10 * len(tops) + 10 * len(bots) + 20 * min(1.0, w / (2 * atr[k])))
                if quality < CHANNEL_MIN_QUALITY:
                    continue
                seen.add(key)
                signals.append(
                    PatternSignal(
                        pattern="channel",
                        direction="sell",
                        pair=pair,
                        timeframe=tf,
                        confirm_idx=k,
                        confirm_ts=pd.Timestamp(idx[k]),
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        neckline=u,
                        pattern_high=u,
                        pattern_low=l,
                        swing_indices=[p.idx for p in window],
                        meta={
                            "quality": min(100.0, 40 + 10 * len(tops) + 10 * len(bots) + 20 * min(1.0, w / (2 * atr[k]))),
                            "channel_width_atr": w / atr[k],
                            "position_in_channel": pos,
                            "slope_norm": slope_norm,
                            "touch_upper": float(len(tops)),
                            "touch_lower": float(len(bots)),
                            "channel_age_bars": float(k - window[0].idx),
                            "peak_equality_atr": 0.0,
                            "depth_atr": w / atr[k],
                            "time_symmetry": 1.0,
                            "prior_pctile": 0.5,
                            "pattern_r2": 0.7,
                            "pattern_height_atr": w / atr[k],
                            "pattern_bars": float(k - window[0].idx),
                            "dist_to_neckline_atr": abs(entry - u) / atr[k],
                            "neckline_slope_atr": slope_norm,
                            "head_dominance_atr": 0.0,
                            "line_slope": hs,
                            "line_intercept": hi,
                            "atr_ref": float(atr[k]),
                        },
                    )
                )
                break

    # dedupe by confirm bar
    by_bar = {}
    for s in signals:
        key = (s.confirm_idx, s.direction)
        if key not in by_bar or s.meta.get("quality", 0) > by_bar[key].meta.get("quality", 0):
            by_bar[key] = s
    return sorted(by_bar.values(), key=lambda x: x.confirm_idx)
