"""Bull / bear flag detector — impulse pole then tight counter-drift, then breakout.

Structure:
  1. Pole: a fast directional move of >= FLAG_MIN_POLE_ATR with high path efficiency.
  2. Flag: a short consolidation that retraces only part of the pole and stays tight.
  3. Break: close beyond the flag's extreme in the pole's direction.
"""

# Vendored from Mo_Dash strategies/chart_patterns/flags.py - keep in sync manually.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import (
    FLAG_MAX_BARS,
    FLAG_MAX_RANGE_ATR,
    FLAG_MAX_RETRACE,
    FLAG_MIN_BARS,
    FLAG_MIN_POLE_ATR,
    FLAG_MIN_POLE_EFFICIENCY,
    FLAG_MIN_QUALITY,
    FLAG_POLE_MAX_BARS,
    FLAG_POLE_MIN_BARS,
    TP_R,
)
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.swing_points import compute_atr


def _pole(
    closes: np.ndarray,
    atr: np.ndarray,
    start: int,
    end: int,
    *,
    up: bool,
) -> Optional[Tuple[float, float]]:
    """Return (net_atr, efficiency) when the leg qualifies as an impulse pole."""
    atr_ref = atr[end]
    if np.isnan(atr_ref) or atr_ref <= 0:
        return None
    seg = closes[start : end + 1]
    net = float(seg[-1] - seg[0])
    net_atr = net / atr_ref
    if up and net_atr < FLAG_MIN_POLE_ATR:
        return None
    if (not up) and net_atr > -FLAG_MIN_POLE_ATR:
        return None
    path = float(np.abs(np.diff(seg)).sum())
    eff = abs(net) / path if path > 1e-12 else 0.0
    if eff < FLAG_MIN_POLE_EFFICIENCY:
        return None
    return net_atr, eff


def _quality(pole_atr: float, eff: float, tight: float, retrace: float, bars: int) -> float:
    pole_s = min(1.0, abs(pole_atr) / 6.0)
    eff_s = min(1.0, max(0.0, (eff - FLAG_MIN_POLE_EFFICIENCY) / (1.0 - FLAG_MIN_POLE_EFFICIENCY)))
    tight_s = 1.0 - min(1.0, tight / FLAG_MAX_RANGE_ATR)
    retr_s = 1.0 - min(1.0, retrace / FLAG_MAX_RETRACE)
    bars_s = min(1.0, bars / float(FLAG_MAX_BARS))
    return float(np.clip(25 * pole_s + 25 * eff_s + 20 * tight_s + 20 * retr_s + 10 * bars_s, 0, 100))


def detect_flags(ohlc: pd.DataFrame, pair: str, tf: str, *, bull: bool) -> List[PatternSignal]:
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    n = len(closes)
    pattern = "bull_flag" if bull else "bear_flag"
    direction = "buy" if bull else "sell"
    idx = ohlc.index

    signals: List[PatternSignal] = []
    last_confirm = -10_000
    step = 2

    for pole_end in range(FLAG_POLE_MAX_BARS + 20, n - 2, step):
        atr_ref = atr[pole_end]
        if np.isnan(atr_ref) or atr_ref <= 0:
            continue

        best = None
        for pole_len in range(FLAG_POLE_MIN_BARS, FLAG_POLE_MAX_BARS + 1, 4):
            res = _pole(closes, atr, pole_end - pole_len, pole_end, up=bull)
            if res is None:
                continue
            net_atr, eff = res
            if best is None or abs(net_atr) > abs(best[0]):
                best = (net_atr, eff, pole_len)
        if best is None:
            continue
        pole_atr, pole_eff, pole_len = best
        pole_start = pole_end - pole_len
        pole_top = float(highs[pole_start : pole_end + 1].max())
        pole_bot = float(lows[pole_start : pole_end + 1].min())
        pole_height = pole_top - pole_bot
        if pole_height <= 0:
            continue

        # Grow the flag bar by bar and take the first valid breakout. The flag
        # extremes are measured strictly before k so the break test stays causal.
        for flag_len in range(FLAG_MIN_BARS, FLAG_MAX_BARS + 1):
            k = pole_end + flag_len + 1
            if k >= n - 1:
                break
            f_hi = float(highs[pole_end + 1 : k].max())
            f_lo = float(lows[pole_end + 1 : k].min())
            atr_k = atr[k]
            if np.isnan(atr_k) or atr_k <= 0:
                break
            rng = (f_hi - f_lo) / atr_k
            if rng > FLAG_MAX_RANGE_ATR:
                break  # consolidation too loose - not a flag

            if bull:
                retrace = (pole_top - f_lo) / pole_height
                if retrace > FLAG_MAX_RETRACE:
                    break
                if closes[k] <= f_hi:
                    continue
                entry = float(closes[k])
                sl = f_lo - 0.5 * atr_k
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + TP_R * risk
                neckline = f_hi
            else:
                retrace = (f_hi - pole_bot) / pole_height
                if retrace > FLAG_MAX_RETRACE:
                    break
                if closes[k] >= f_lo:
                    continue
                entry = float(closes[k])
                sl = f_hi + 0.5 * atr_k
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - TP_R * risk
                neckline = f_lo

            if k - last_confirm < 10:
                break
            qual = _quality(pole_atr, pole_eff, rng, retrace, flag_len)
            if qual < FLAG_MIN_QUALITY:
                break

            last_confirm = k
            signals.append(
                PatternSignal(
                    pattern=pattern,
                    direction=direction,
                    pair=pair,
                    timeframe=tf,
                    confirm_idx=k,
                    confirm_ts=pd.Timestamp(idx[k]),
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                    neckline=neckline,
                    pattern_high=f_hi,
                    pattern_low=f_lo,
                    swing_indices=[pole_start, pole_end, k],
                    meta={
                        "quality": qual,
                        "start_i": pole_start,
                        "pole_start_i": float(pole_start),
                        "pole_end_i": float(pole_end),
                        "pole_atr": pole_atr,
                        "pole_efficiency": pole_eff,
                        "pole_bars": float(pole_len),
                        "flag_bars": float(flag_len),
                        "flag_high": f_hi,
                        "flag_low": f_lo,
                        "flag_range_atr": rng,
                        "flag_retrace": retrace,
                        "peak_equality_atr": 0.0,
                        "depth_atr": risk / atr_k,
                        "time_symmetry": 1.0,
                        "prior_pctile": 0.9 if bull else 0.1,
                        "pattern_r2": pole_eff,
                        "pattern_height_atr": pole_height / atr_k,
                        "pattern_bars": float(k - pole_start),
                        "dist_to_neckline_atr": abs(entry - neckline) / atr_k,
                        "neckline_slope_atr": 0.0,
                        "head_dominance_atr": 0.0,
                        "line_slope": 0.0,
                        "line_intercept": neckline,
                        "atr_ref": float(atr_k),
                    },
                )
            )
            break

    return sorted(signals, key=lambda s: s.confirm_idx)
