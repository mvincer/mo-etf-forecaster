"""Uptrend / downtrend structure detector with pullback entry.

An uptrend needs a run of higher highs *and* higher lows. The trade is the
continuation entry: price pulls back onto the rising trendline drawn through
the swing lows and closes back above it.
"""

# Vendored from Mo_Dash strategies/chart_patterns/trends.py - keep in sync manually.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import (
    PIVOT_ORDER,
    TP_R,
    TREND_MIN_NET_ATR,
    TREND_MIN_QUALITY,
    TREND_MIN_R2,
    TREND_MIN_SWINGS,
    TREND_TOUCH_TOL_ATR,
)
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.pivots import Pivot, alternating_pivots, extract_pivots
from etf_forecaster.patterns.swing_points import compute_atr


def _fit(idxs: List[int], prices: List[float]) -> Optional[Tuple[float, float, float]]:
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


def _monotone(vals: List[float], increasing: bool) -> float:
    """Fraction of consecutive steps that go the required way."""
    if len(vals) < 2:
        return 0.0
    steps = np.diff(np.asarray(vals, dtype=float))
    good = np.sum(steps > 0) if increasing else np.sum(steps < 0)
    return float(good) / len(steps)


def _quality(hh: float, hl: float, r2: float, net_atr: float) -> float:
    struct = 0.5 * hh + 0.5 * hl
    r2_s = max(0.0, min(1.0, r2))
    net_s = min(1.0, abs(net_atr) / 8.0)
    return float(np.clip(45 * struct + 30 * r2_s + 25 * net_s, 0, 100))


def detect_trends(ohlc: pd.DataFrame, pair: str, tf: str, *, up: bool) -> List[PatternSignal]:
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    order = PIVOT_ORDER.get(tf, 6)
    pivots = alternating_pivots(extract_pivots(closes, order))
    if len(pivots) < TREND_MIN_SWINGS * 2:
        return []

    pattern = "uptrend" if up else "downtrend"
    direction = "buy" if up else "sell"
    idx = ohlc.index
    signals: List[PatternSignal] = []
    last_confirm = -10_000

    for end in range(TREND_MIN_SWINGS * 2, len(pivots) + 1):
        window = pivots[max(0, end - 8) : end]
        tops = [p for p in window if p.kind == 1]
        bots = [p for p in window if p.kind == -1]
        if len(tops) < TREND_MIN_SWINGS or len(bots) < TREND_MIN_SWINGS:
            continue

        hh = _monotone([p.price for p in tops], increasing=up)
        hl = _monotone([p.price for p in bots], increasing=up)
        if hh < 1.0 or hl < 1.0:
            continue  # require strictly stair-stepping structure

        # The rail we trade off: swing lows in an uptrend, swing highs in a downtrend.
        anchor = bots if up else tops
        fit = _fit([p.idx for p in anchor], [p.price for p in anchor])
        if fit is None:
            continue
        slope, intercept, r2 = fit
        if r2 < TREND_MIN_R2:
            continue
        if up and slope <= 0:
            continue
        if (not up) and slope >= 0:
            continue

        start_i = min(p.idx for p in window)
        confirm_i = max(window[-1].confirm_i, window[-1].idx + order)
        if confirm_i >= len(closes) - 2:
            continue
        atr_ref = atr[confirm_i]
        if np.isnan(atr_ref) or atr_ref <= 0:
            continue

        net_atr = (closes[confirm_i] - closes[start_i]) / atr_ref
        if up and net_atr < TREND_MIN_NET_ATR:
            continue
        if (not up) and net_atr > -TREND_MIN_NET_ATR:
            continue

        qual = _quality(hh, hl, r2, net_atr)
        if qual < TREND_MIN_QUALITY:
            continue

        # Pullback entry: tag the rail, then close back on the trend side.
        for k in range(confirm_i + 1, min(len(closes), confirm_i + 50)):
            atr_k = atr[k]
            if np.isnan(atr_k) or atr_k <= 0:
                continue
            rail = slope * k + intercept
            tol = TREND_TOUCH_TOL_ATR * atr_k
            if k - last_confirm < 8:
                continue

            if up:
                if not (lows[k] <= rail + tol and closes[k] > rail):
                    continue
                if closes[k] <= ohlc["Open"].values[k]:
                    continue  # want a bullish rejection candle
                entry = float(closes[k])
                sl = float(min(lows[k], rail)) - 0.5 * atr_k
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + TP_R * risk
            else:
                if not (highs[k] >= rail - tol and closes[k] < rail):
                    continue
                if closes[k] >= ohlc["Open"].values[k]:
                    continue
                entry = float(closes[k])
                sl = float(max(highs[k], rail)) + 0.5 * atr_k
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - TP_R * risk

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
                    neckline=float(rail),
                    pattern_high=float(max(p.price for p in tops)),
                    pattern_low=float(min(p.price for p in bots)),
                    swing_indices=[p.idx for p in window],
                    meta={
                        "quality": qual,
                        "start_i": start_i,
                        "trend_slope": slope,
                        "trend_intercept": intercept,
                        "rail_start_i": float(start_i),
                        "rail_end_i": float(k),
                        "hh_frac": hh,
                        "hl_frac": hl,
                        "trend_net_atr": net_atr,
                        "touch_upper": float(len(tops)),
                        "touch_lower": float(len(bots)),
                        "peak_equality_atr": 0.0,
                        "depth_atr": risk / atr_k,
                        "time_symmetry": 1.0,
                        "prior_pctile": 0.9 if up else 0.1,
                        "pattern_r2": r2,
                        "pattern_height_atr": abs(net_atr),
                        "pattern_bars": float(k - start_i),
                        "dist_to_neckline_atr": abs(entry - rail) / atr_k,
                        "neckline_slope_atr": slope / atr_k,
                        "head_dominance_atr": 0.0,
                        "line_slope": slope,
                        "line_intercept": intercept,
                        "atr_ref": float(atr_k),
                    },
                )
            )
            break

    return sorted(signals, key=lambda s: s.confirm_idx)


def detect_trend_breaks(ohlc: pd.DataFrame, pair: str, tf: str, *, up: bool) -> List[PatternSignal]:
    """Broken trendline flip.

    A rising support line that price closes *below* becomes resistance, so the
    trade is a sell on the pullback back up to it. A falling resistance line that
    price closes *above* becomes support -> buy the pullback down to it.
    """
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    order = PIVOT_ORDER.get(tf, 6)
    pivots = alternating_pivots(extract_pivots(closes, order))
    if len(pivots) < TREND_MIN_SWINGS * 2:
        return []

    pattern = "uptrend_break" if up else "downtrend_break"
    direction = "sell" if up else "buy"
    idx = ohlc.index
    signals: List[PatternSignal] = []
    last_confirm = -10_000

    for end in range(TREND_MIN_SWINGS * 2, len(pivots) + 1):
        window = pivots[max(0, end - 8) : end]
        tops = [p for p in window if p.kind == 1]
        bots = [p for p in window if p.kind == -1]
        if len(tops) < TREND_MIN_SWINGS or len(bots) < TREND_MIN_SWINGS:
            continue

        hh = _monotone([p.price for p in tops], increasing=up)
        hl = _monotone([p.price for p in bots], increasing=up)
        if hh < 1.0 or hl < 1.0:
            continue

        anchor = bots if up else tops
        fit = _fit([p.idx for p in anchor], [p.price for p in anchor])
        if fit is None:
            continue
        slope, intercept, r2 = fit
        if r2 < TREND_MIN_R2:
            continue
        if up and slope <= 0:
            continue
        if (not up) and slope >= 0:
            continue

        start_i = min(p.idx for p in window)
        confirm_i = max(window[-1].confirm_i, window[-1].idx + order)
        if confirm_i >= len(closes) - 2:
            continue
        atr_ref = atr[confirm_i]
        if np.isnan(atr_ref) or atr_ref <= 0:
            continue

        net_atr = (closes[confirm_i] - closes[start_i]) / atr_ref
        if up and net_atr < TREND_MIN_NET_ATR:
            continue
        if (not up) and net_atr > -TREND_MIN_NET_ATR:
            continue

        qual = _quality(hh, hl, r2, net_atr)
        if qual < TREND_MIN_QUALITY:
            continue

        # Find the bar that decisively closes through the rail.
        for k in range(confirm_i + 1, min(len(closes), confirm_i + 60)):
            atr_k = atr[k]
            if np.isnan(atr_k) or atr_k <= 0:
                continue
            rail = slope * k + intercept
            broke = closes[k] < rail - TREND_TOUCH_TOL_ATR * atr_k if up else \
                closes[k] > rail + TREND_TOUCH_TOL_ATR * atr_k
            if not broke:
                continue
            if k - last_confirm < 8:
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
                    entry_price=float(closes[k]),
                    sl=float(rail + atr_k) if up else float(rail - atr_k),
                    tp=float(closes[k] - 2 * atr_k) if up else float(closes[k] + 2 * atr_k),
                    neckline=float(rail),
                    pattern_high=float(max(p.price for p in tops)),
                    pattern_low=float(min(p.price for p in bots)),
                    swing_indices=[p.idx for p in window],
                    meta={
                        "quality": qual,
                        "start_i": start_i,
                        "trend_slope": slope,
                        "trend_intercept": intercept,
                        "rail_start_i": float(start_i),
                        "rail_end_i": float(k),
                        "hh_frac": hh,
                        "hl_frac": hl,
                        "trend_net_atr": net_atr,
                        "touch_upper": float(len(tops)),
                        "touch_lower": float(len(bots)),
                        "peak_equality_atr": 0.0,
                        "depth_atr": 1.0,
                        "time_symmetry": 1.0,
                        "prior_pctile": 0.9 if up else 0.1,
                        "pattern_r2": r2,
                        "pattern_height_atr": abs(net_atr),
                        "pattern_bars": float(k - start_i),
                        "dist_to_neckline_atr": abs(closes[k] - rail) / atr_k,
                        "neckline_slope_atr": slope / atr_k,
                        "head_dominance_atr": 0.0,
                        "line_slope": slope,
                        "line_intercept": intercept,
                        "atr_ref": float(atr_k),
                    },
                )
            )
            break

    return sorted(signals, key=lambda s: s.confirm_idx)
