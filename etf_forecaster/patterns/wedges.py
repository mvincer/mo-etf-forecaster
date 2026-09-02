"""Rising / falling wedge detector — converging rails sloping the same way.

Rising wedge  = both rails up, lower rises faster, width shrinks -> bearish break down.
Falling wedge = both rails down, upper falls faster, width shrinks -> bullish break up.

Causal: rails are fitted only from pivots already confirmed at the break bar.
"""

# Vendored from Mo_Dash strategies/chart_patterns/wedges.py - keep in sync manually.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import (
    PIVOT_ORDER,
    TP_R,
    WEDGE_MAX_BARS,
    WEDGE_MIN_CONVERGENCE,
    WEDGE_MIN_HEIGHT_ATR,
    WEDGE_MIN_QUALITY,
    WEDGE_MIN_TOUCHES,
    WEDGE_SLOPE_RATIO_MIN,
)
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.pivots import Pivot, alternating_pivots, extract_pivots
from etf_forecaster.patterns.swing_points import compute_atr


def _fit(idxs: List[int], prices: List[float]) -> Optional[Tuple[float, float, float]]:
    """Least-squares line plus R^2."""
    if len(idxs) < 2:
        return None
    x = np.asarray(idxs, dtype=float)
    y = np.asarray(prices, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def _wedge_meta(
    kind: str,
    hs: float,
    hi: float,
    ls: float,
    li: float,
    tops: List[Pivot],
    bots: List[Pivot],
    start_i: int,
    k: int,
    atr_k: float,
    width_start: float,
    width_now: float,
    r2: float,
    risk: float,
) -> dict:
    conv = width_start / width_now if width_now > 1e-12 else 99.0
    return {
        "upper_slope": hs,
        "upper_intercept": hi,
        "lower_slope": ls,
        "lower_intercept": li,
        "rail_start_i": float(start_i),
        "rail_end_i": float(k),
        "wedge_convergence": conv,
        "wedge_width_start_atr": width_start / atr_k,
        "wedge_width_break_atr": width_now / atr_k,
        "touch_upper": float(len(tops)),
        "touch_lower": float(len(bots)),
        "start_i": start_i,
        # shared ML geometry keys
        "peak_equality_atr": 0.0,
        "depth_atr": risk / atr_k,
        "time_symmetry": 1.0,
        "prior_pctile": 0.5,
        "pattern_r2": r2,
        "pattern_height_atr": width_start / atr_k,
        "pattern_bars": float(k - start_i),
        "dist_to_neckline_atr": 0.0,
        "neckline_slope_atr": ((hs + ls) / 2.0) / atr_k,
        "head_dominance_atr": 0.0,
    }


def _quality(conv: float, r2: float, n_touch: int, height_atr: float) -> float:
    conv_s = min(1.0, (conv - 1.0) / 1.5)
    r2_s = max(0.0, min(1.0, r2))
    touch_s = min(1.0, (n_touch - 4) / 4.0) if n_touch >= 4 else 0.0
    height_s = min(1.0, height_atr / 3.0)
    return float(np.clip(30 * conv_s + 25 * r2_s + 25 * touch_s + 20 * height_s, 0, 100))


def detect_wedges(
    ohlc: pd.DataFrame,
    pair: str,
    tf: str,
    *,
    rising: bool,
) -> List[PatternSignal]:
    """Rising wedge -> sell on lower-rail break. Falling wedge -> buy on upper-rail break."""
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    order = PIVOT_ORDER.get(tf, 6)
    pivots = alternating_pivots(extract_pivots(closes, order))
    if len(pivots) < WEDGE_MIN_TOUCHES * 2:
        return []

    pattern = "rising_wedge" if rising else "falling_wedge"
    direction = "sell" if rising else "buy"
    idx = ohlc.index
    signals: List[PatternSignal] = []
    seen: set = set()

    for end in range(WEDGE_MIN_TOUCHES * 2, len(pivots) + 1):
        window = pivots[max(0, end - 8) : end]
        tops = [p for p in window if p.kind == 1]
        bots = [p for p in window if p.kind == -1]
        if len(tops) < WEDGE_MIN_TOUCHES or len(bots) < WEDGE_MIN_TOUCHES:
            continue

        hi_fit = _fit([p.idx for p in tops], [p.price for p in tops])
        lo_fit = _fit([p.idx for p in bots], [p.price for p in bots])
        if hi_fit is None or lo_fit is None:
            continue
        hs, hint, hr2 = hi_fit
        ls, lint, lr2 = lo_fit

        start_i = min(p.idx for p in window)
        confirm_i = max(window[-1].confirm_i, window[-1].idx + order)
        if confirm_i >= len(closes) - 2:
            continue
        if confirm_i - start_i > WEDGE_MAX_BARS.get(tf, 250):
            continue
        atr_ref = atr[confirm_i]
        if np.isnan(atr_ref) or atr_ref <= 0:
            continue

        # Both rails must slope the same direction as the wedge type.
        if rising and not (hs > 0 and ls > 0):
            continue
        if (not rising) and not (hs < 0 and ls < 0):
            continue

        # Converging: for a rising wedge the floor must climb faster than the ceiling.
        if rising:
            if ls <= hs * WEDGE_SLOPE_RATIO_MIN:
                continue
        else:
            if hs >= ls * WEDGE_SLOPE_RATIO_MIN:
                continue

        width_start = (hs * start_i + hint) - (ls * start_i + lint)
        width_now = (hs * confirm_i + hint) - (ls * confirm_i + lint)
        if width_start <= 0 or width_now <= 0:
            continue
        if width_start / width_now < WEDGE_MIN_CONVERGENCE:
            continue
        if width_start < WEDGE_MIN_HEIGHT_ATR * atr_ref:
            continue

        r2 = (hr2 + lr2) / 2.0
        qual = _quality(width_start / width_now, r2, len(tops) + len(bots), width_start / atr_ref)
        if qual < WEDGE_MIN_QUALITY:
            continue

        # Walk forward for the rail break that confirms the wedge.
        for k in range(confirm_i + 1, min(len(closes), confirm_i + 60)):
            atr_k = atr[k]
            if np.isnan(atr_k) or atr_k <= 0:
                continue
            upper = hs * k + hint
            lower = ls * k + lint
            if upper <= lower:
                break  # rails crossed - wedge has expired

            key = (round(hs, 10), round(ls, 10), k)
            if key in seen:
                continue

            if rising:
                if closes[k] >= lower:
                    continue
                entry = float(closes[k])
                sl = float(max(highs[max(0, k - 10) : k + 1].max(), upper)) + 0.5 * atr_k
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - TP_R * risk
            else:
                if closes[k] <= upper:
                    continue
                entry = float(closes[k])
                sl = float(min(lows[max(0, k - 10) : k + 1].min(), lower)) - 0.5 * atr_k
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + TP_R * risk

            seen.add(key)
            meta = _wedge_meta(
                pattern, hs, hint, ls, lint, tops, bots, start_i, k, atr_k,
                width_start, width_now, r2, risk,
            )
            meta["quality"] = qual
            # The broken rail is the one price retests.
            meta["line_slope"] = ls if rising else hs
            meta["line_intercept"] = lint if rising else hint
            meta["atr_ref"] = float(atr_k)
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
                    neckline=lower if rising else upper,
                    pattern_high=float(hs * start_i + hint),
                    pattern_low=float(ls * start_i + lint),
                    swing_indices=[p.idx for p in window],
                    meta=meta,
                )
            )
            break

    by_bar = {}
    for s in signals:
        if s.confirm_idx not in by_bar or s.meta["quality"] > by_bar[s.confirm_idx].meta["quality"]:
            by_bar[s.confirm_idx] = s
    return sorted(by_bar.values(), key=lambda x: x.confirm_idx)
