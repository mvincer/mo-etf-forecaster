"""Unified n-peak pattern detector — causal, continuous features.

Covers double/triple tops & bottoms and head & shoulders / inverse H&S
as one family (Lo/Mamaysky/Wang + neurotrader balance/symmetry rules).

All features timestamped at the confirmation (neckline break) bar.
"""

# Vendored from Mo_Dash strategies/chart_patterns/npeak.py - keep in sync manually.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_forecaster.patterns.pattern_config import (
    MIN_DEPTH_ATR,
    MIN_HEAD_DOMINANCE_ATR,
    MIN_PATTERN_R2,
    MIN_PRIOR_EFFICIENCY,
    MIN_PRIOR_MOVE_ATR,
    MIN_PRIOR_PCTILE,
    MIN_PRIOR_STRUCTURE,
    MIN_QUALITY,
    PEAK_EQ_ATR,
    PRIOR_LOOKBACK,
    SHOULDER_EQ_ATR,
    TIME_SYM_HI,
    TIME_SYM_LO,
    TP_R,
    PIVOT_ORDER,
)
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.pivots import Pivot, alternating_pivots, extract_pivots
from etf_forecaster.patterns.swing_points import compute_atr


@dataclass
class PatternHit:
    pattern: str
    direction: str
    start_i: int
    break_i: int
    break_p: float
    neckline: float
    sl: float
    tp: float
    pattern_high: float
    pattern_low: float
    swing_indices: List[int]
    features: Dict[str, float] = field(default_factory=dict)
    quality: float = 0.0


def _atr_at(atr: np.ndarray, i: int) -> float:
    a = atr[i]
    if np.isnan(a) or a <= 0:
        return np.nan
    return float(a)


def _prior_trend_pctile(closes: np.ndarray, t: int, lookback: int, top: bool) -> float:
    """Percentile rank of price at t within trailing lookback (0–1)."""
    if t < lookback:
        return 0.0
    window = closes[t - lookback : t + 1]
    rank = float(np.sum(window <= closes[t]) / len(window))
    return rank if top else (1.0 - rank)


def _prior_swing_structure(
    pivots: List[Pivot],
    first_idx: int,
    *,
    up: bool,
    need: int = 3,
) -> Tuple[bool, float]:
    """Were the swings *before* the pattern stair-stepping with the trend?

    A top is only a reversal if an uptrend preceded it, which textbook-wise means
    higher highs and higher lows into the pattern.
    """
    prior = [p for p in pivots if p.idx <= first_idx]
    tops = [p.price for p in prior if p.kind == 1][-need:]
    bots = [p.price for p in prior if p.kind == -1][-need:]
    if len(tops) < 2 or len(bots) < 2:
        return False, 0.0

    def frac(vals: List[float]) -> float:
        d = np.diff(np.asarray(vals, dtype=float))
        good = int(np.sum(d > 0)) if up else int(np.sum(d < 0))
        return good / len(d)

    score = 0.5 * frac(tops) + 0.5 * frac(bots)
    return score >= MIN_PRIOR_STRUCTURE, float(score)


def _prior_extended_trend(
    closes: np.ndarray,
    atr: np.ndarray,
    t: int,
    *,
    up: bool,
    lookback: int = PRIOR_LOOKBACK,
) -> Tuple[bool, float, float, float]:
    """Require an *extended* directional move into the pattern (not a sideways box).

    Returns (ok, pctile, net_atr, efficiency).
    """
    pctile = _prior_trend_pctile(closes, t, lookback, top=up)
    if t < lookback:
        return False, pctile, 0.0, 0.0
    atr_ref = _atr_at(atr, t)
    if np.isnan(atr_ref):
        return False, pctile, 0.0, 0.0
    window = closes[t - lookback : t + 1].astype(float)
    net = float(window[-1] - window[0])
    net_atr = net / atr_ref
    path = float(np.abs(np.diff(window)).sum())
    eff = abs(net) / path if path > 1e-12 else 0.0
    if pctile < MIN_PRIOR_PCTILE:
        return False, pctile, net_atr, eff
    if up and net_atr < MIN_PRIOR_MOVE_ATR:
        return False, pctile, net_atr, eff
    if (not up) and net_atr > -MIN_PRIOR_MOVE_ATR:
        return False, pctile, net_atr, eff
    if eff < MIN_PRIOR_EFFICIENCY:
        return False, pctile, net_atr, eff
    return True, pctile, net_atr, eff


def _neckline_at(l_i: int, l_p: float, r_i: int, r_p: float, x: int) -> float:
    if r_i == l_i:
        return l_p
    slope = (r_p - l_p) / (r_i - l_i)
    return l_p + (x - l_i) * slope


def _quality_score(feats: Dict[str, float], kind: str) -> float:
    """0–100 composite; higher = clearer textbook pattern."""
    eq = 1.0 - min(1.0, feats.get("peak_equality_atr", 9) / max(PEAK_EQ_ATR, 1e-9))
    depth = min(1.0, feats.get("depth_atr", 0) / 3.0)
    tsym = feats.get("time_symmetry", 0)
    # map [TIME_SYM_LO, 1] → 1, outside → 0
    if tsym <= 0:
        tsym_s = 0.0
    else:
        # closer to 1.0 is better
        tsym_s = max(0.0, 1.0 - abs(np.log(tsym)) / np.log(TIME_SYM_HI))
    prior = feats.get("prior_pctile", 0)
    r2 = max(0.0, min(1.0, feats.get("pattern_r2", 0)))
    head = min(1.0, feats.get("head_dominance_atr", 0) / 2.0) if "head" in kind or kind.endswith("shoulders") else 0.5

    if kind in ("head_shoulders", "inverse_head_shoulders"):
        q = 25 * eq + 15 * depth + 15 * tsym_s + 20 * prior + 15 * r2 + 10 * head
    else:
        q = 30 * eq + 25 * depth + 20 * tsym_s + 15 * prior + 10 * r2
    return float(np.clip(q, 0, 100))


def _iter_triplets(alt: List[Pivot], bearish: bool):
    """Yield all H-L-H (bearish) or L-H-L (bullish) triplets in alternating pivots."""
    for i in range(2, len(alt)):
        a, b, c = alt[i - 2], alt[i - 1], alt[i]
        if bearish and a.kind == 1 and b.kind == -1 and c.kind == 1:
            yield a, b, c
        if (not bearish) and a.kind == -1 and b.kind == 1 and c.kind == -1:
            yield a, b, c


def _check_double_at(
    a: Pivot, b: Pivot, c: Pivot,
    closes: np.ndarray,
    atr: np.ndarray,
    i: int,
    *,
    bearish: bool,
    pivots: List[Pivot],
) -> Optional[PatternHit]:
    """Double top/bottom for a specific triplet, confirmed at bar i."""
    if i <= c.confirm_i:
        return None
    atr_ref = _atr_at(atr, i)
    if np.isnan(atr_ref):
        return None

    eq = abs(c.price - a.price) / atr_ref
    if eq > PEAK_EQ_ATR:
        return None

    t1 = b.idx - a.idx
    t2 = c.idx - b.idx
    if t1 <= 0 or t2 <= 0:
        return None
    tsym = t2 / t1
    if tsym < TIME_SYM_LO or tsym > TIME_SYM_HI:
        return None

    if bearish:
        depth = (min(a.price, c.price) - b.price) / atr_ref
        if depth < MIN_DEPTH_ATR:
            return None
        neck = b.price
        if closes[i] >= neck:
            return None
        # first break only — previous bar still above/at neck
        if i > 0 and closes[i - 1] < neck:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, a.idx, up=True)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, a.idx, up=True)
        if not ok_st:
            return None
        pattern, direction = "double_top", "sell"
        peak, trough = max(a.price, c.price), b.price
        sl = peak + 0.5 * atr_ref
        entry = float(closes[i])
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - TP_R * risk
    else:
        depth = (b.price - max(a.price, c.price)) / atr_ref
        if depth < MIN_DEPTH_ATR:
            return None
        neck = b.price
        if closes[i] <= neck:
            return None
        if i > 0 and closes[i - 1] > neck:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, a.idx, up=False)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, a.idx, up=False)
        if not ok_st:
            return None
        pattern, direction = "double_bottom", "buy"
        peak, trough = b.price, min(a.price, c.price)
        sl = trough - 0.5 * atr_ref
        entry = float(closes[i])
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + TP_R * risk

    seg = closes[a.idx : i + 1]
    if len(seg) < 5:
        return None
    model = np.linspace(a.price, c.price, len(seg))
    ss_res = float(np.sum((seg - model) ** 2))
    ss_tot = float(np.sum((seg - seg.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    feats = {
        "peak_equality_atr": eq,
        "depth_atr": depth,
        "time_symmetry": tsym,
        "prior_pctile": prior,
        "prior_net_atr": abs(prior_net),
        "prior_efficiency": prior_eff,
        "prior_structure": struct,
        "pattern_r2": r2,
        "pattern_height_atr": abs(peak - trough) / atr_ref,
        "pattern_bars": float(i - a.idx),
        "dist_to_neckline_atr": abs(entry - neck) / atr_ref,
        "neckline_slope_atr": 0.0,
        "head_dominance_atr": 0.0,
        # A double top/bottom has a single middle pivot, so the neckline is flat.
        "line_slope": 0.0,
        "line_intercept": neck,
        "atr_ref": atr_ref,
    }
    q = _quality_score(feats, pattern)
    feats["quality"] = q
    if q < MIN_QUALITY:
        return None

    return PatternHit(
        pattern=pattern, direction=direction, start_i=a.idx, break_i=i, break_p=entry,
        neckline=neck, sl=sl, tp=tp, pattern_high=peak, pattern_low=trough,
        swing_indices=[a.idx, b.idx, c.idx], features=feats, quality=q,
    )


def _check_double(
    pivots: List[Pivot],
    closes: np.ndarray,
    atr: np.ndarray,
    i: int,
    *,
    bearish: bool,
) -> Optional[PatternHit]:
    """Scan recent triplets (last 8 swings) for a fresh neckline break at i."""
    recent = pivots[-10:] if len(pivots) > 10 else pivots
    best: Optional[PatternHit] = None
    for a, b, c in _iter_triplets(recent, bearish):
        h = _check_double_at(a, b, c, closes, atr, i, bearish=bearish, pivots=pivots)
        if h is not None and (best is None or h.quality > best.quality):
            best = h
    return best


def _check_triple(
    pivots: List[Pivot],
    closes: np.ndarray,
    atr: np.ndarray,
    i: int,
    *,
    bearish: bool,
) -> Optional[PatternHit]:
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    kinds = [p.kind for p in seq]
    if bearish and kinds != [1, -1, 1, -1, 1]:
        return None
    if not bearish and kinds != [-1, 1, -1, 1, -1]:
        return None

    atr_ref = _atr_at(atr, i)
    if np.isnan(atr_ref):
        return None

    p0, v0, p1, v1, p2 = seq
    eq01 = abs(p1.price - p0.price) / atr_ref
    eq12 = abs(p2.price - p1.price) / atr_ref
    eq = max(eq01, eq12)
    if eq > PEAK_EQ_ATR:
        return None

    # Neckline is the line joining the two middle pivots (the middle lows of a
    # triple top, the middle highs of a triple bottom), projected to bar i.
    neck_slope = (v1.price - v0.price) / max(1, v1.idx - v0.idx)
    neck_intercept = v0.price - neck_slope * v0.idx
    neck_at_i = neck_slope * i + neck_intercept
    neck_mid = 0.5 * (v0.price + v1.price)

    if bearish:
        neck = neck_at_i
        depth = (min(p0.price, p1.price, p2.price) - neck_mid) / atr_ref
        if depth < MIN_DEPTH_ATR:
            return None
        if closes[i] >= neck:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, p0.idx, up=True)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, p0.idx, up=True)
        if not ok_st:
            return None
        pattern, direction = "triple_top", "sell"
        peak = max(p0.price, p1.price, p2.price)
        trough = neck
        sl = peak + 0.5 * atr_ref
        entry = float(closes[i])
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - TP_R * risk
    else:
        neck = neck_at_i
        depth = (neck_mid - max(p0.price, p1.price, p2.price)) / atr_ref
        if depth < MIN_DEPTH_ATR:
            return None
        if closes[i] <= neck:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, p0.idx, up=False)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, p0.idx, up=False)
        if not ok_st:
            return None
        pattern, direction = "triple_bottom", "buy"
        peak = neck
        trough = min(p0.price, p1.price, p2.price)
        sl = trough - 0.5 * atr_ref
        entry = float(closes[i])
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + TP_R * risk

    tsym = (p2.idx - p1.idx) / max(1, p1.idx - p0.idx)
    if tsym < TIME_SYM_LO or tsym > TIME_SYM_HI:
        return None

    feats = {
        "peak_equality_atr": eq,
        "depth_atr": depth,
        "time_symmetry": tsym,
        "prior_pctile": prior,
        "prior_net_atr": abs(prior_net),
        "prior_efficiency": prior_eff,
        "prior_structure": struct,
        "pattern_r2": 0.5,
        "pattern_height_atr": abs(peak - trough) / atr_ref,
        "pattern_bars": float(i - p0.idx),
        "dist_to_neckline_atr": abs(entry - neck) / atr_ref,
        "neckline_slope_atr": neck_slope / atr_ref,
        "head_dominance_atr": 0.0,
        "line_slope": neck_slope,
        "line_intercept": neck_intercept,
        "atr_ref": atr_ref,
    }
    q = _quality_score(feats, pattern)
    feats["quality"] = q
    if q < MIN_QUALITY:
        return None

    return PatternHit(
        pattern=pattern,
        direction=direction,
        start_i=p0.idx,
        break_i=i,
        break_p=entry,
        neckline=neck,
        sl=sl,
        tp=tp,
        pattern_high=peak,
        pattern_low=trough,
        swing_indices=[p.idx for p in seq],
        features=feats,
        quality=q,
    )


def _check_hs(
    pivots: List[Pivot],
    closes: np.ndarray,
    atr: np.ndarray,
    i: int,
    *,
    inverted: bool,
) -> Optional[PatternHit]:
    """Head & shoulders (neurotrader balance + symmetry + neckline break)."""
    if len(pivots) < 5:
        return None
    # For H&S top: need H L H L H confirmed; right shoulder may still be forming
    # Use last confirmed alternating: LS, LA, Head, RA — find RS as extremum since RA
    if inverted:
        # IHS checked after a confirmed top; extrema L H L H, then find RS as min since RA
        # Take pivots ending with armpit (top)
        if pivots[-1].kind != 1:
            return None
        if len(pivots) < 4:
            return None
        ls, la, head, ra = pivots[-4:]
        if [ls.kind, la.kind, head.kind, ra.kind] != [-1, 1, -1, 1]:
            return None
        if i - ra.idx < 2:
            return None
        rs_rel = int(np.argmin(closes[ra.idx + 1 : i + 1]))
        rs_idx = ra.idx + 1 + rs_rel
        rs_price = float(closes[rs_idx])
        if head.price >= min(ls.price, rs_price):
            return None
        # balance
        r_mid = 0.5 * (rs_price + ra.price)
        l_mid = 0.5 * (ls.price + la.price)
        if ls.price > r_mid or rs_price > l_mid:
            return None
    else:
        if pivots[-1].kind != -1:
            return None
        if len(pivots) < 4:
            return None
        ls, la, head, ra = pivots[-4:]
        if [ls.kind, la.kind, head.kind, ra.kind] != [1, -1, 1, -1]:
            return None
        if i - ra.idx < 2:
            return None
        rs_rel = int(np.argmax(closes[ra.idx + 1 : i + 1]))
        rs_idx = ra.idx + 1 + rs_rel
        rs_price = float(closes[rs_idx])
        if head.price <= max(ls.price, rs_price):
            return None
        r_mid = 0.5 * (rs_price + ra.price)
        l_mid = 0.5 * (ls.price + la.price)
        if ls.price < r_mid or rs_price < l_mid:
            return None

    atr_ref = _atr_at(atr, i)
    if np.isnan(atr_ref):
        return None

    # time symmetry
    r_to_h = rs_idx - head.idx
    l_to_h = head.idx - ls.idx
    if r_to_h <= 0 or l_to_h <= 0:
        return None
    if r_to_h > 2.5 * l_to_h or l_to_h > 2.5 * r_to_h:
        return None
    tsym = r_to_h / l_to_h

    shoulder_eq = abs(ls.price - rs_price) / atr_ref
    if shoulder_eq > SHOULDER_EQ_ATR:
        return None

    if inverted:
        dominance = (min(ls.price, rs_price) - head.price) / atr_ref
    else:
        dominance = (head.price - max(ls.price, rs_price)) / atr_ref
    if dominance < MIN_HEAD_DOMINANCE_ATR:
        return None

    neck_at_i = _neckline_at(la.idx, la.price, ra.idx, ra.price, i)
    neck_slope = (ra.price - la.price) / max(1, ra.idx - la.idx)
    neck_slope_atr = (neck_slope / atr_ref)

    if inverted:
        if closes[i] <= neck_at_i:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, ls.idx, up=False)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, ls.idx, up=False)
        if not ok_st:
            return None
        pattern, direction = "inverse_head_shoulders", "buy"
        entry = float(closes[i])
        sl = head.price - 0.5 * atr_ref
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + TP_R * risk
        peak = max(la.price, ra.price, neck_at_i)
        trough = head.price
    else:
        if closes[i] >= neck_at_i:
            return None
        ok_tr, prior, prior_net, prior_eff = _prior_extended_trend(closes, atr, ls.idx, up=True)
        if not ok_tr:
            return None
        ok_st, struct = _prior_swing_structure(pivots, ls.idx, up=True)
        if not ok_st:
            return None
        pattern, direction = "head_shoulders", "sell"
        entry = float(closes[i])
        sl = head.price + 0.5 * atr_ref
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - TP_R * risk
        peak = head.price
        trough = min(la.price, ra.price, neck_at_i)

    # pattern R² (neurotrader-style polyline fit)
    start_i = ls.idx
    seg = closes[start_i : i + 1]
    if len(seg) < 8:
        return None
    # simple: compare to piecewise linear through pivots
    xs = np.array([ls.idx, la.idx, head.idx, ra.idx, rs_idx, i], dtype=float) - start_i
    ys = np.array([ls.price, la.price, head.price, ra.price, rs_price, entry], dtype=float)
    model = np.interp(np.arange(len(seg)), xs, ys)
    ss_res = float(np.sum((seg - model) ** 2))
    ss_tot = float(np.sum((seg - seg.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    if r2 < MIN_PATTERN_R2:
        return None

    depth = abs(peak - trough) / atr_ref
    feats = {
        "peak_equality_atr": shoulder_eq,
        "depth_atr": depth,
        "time_symmetry": tsym,
        "prior_pctile": prior,
        "prior_net_atr": abs(prior_net),
        "prior_efficiency": prior_eff,
        "prior_structure": struct,
        "pattern_r2": r2,
        "pattern_height_atr": depth,
        "pattern_bars": float(i - start_i),
        "dist_to_neckline_atr": abs(entry - neck_at_i) / atr_ref,
        "neckline_slope_atr": neck_slope_atr,
        "head_dominance_atr": dominance,
        "line_slope": neck_slope,
        "line_intercept": la.price - neck_slope * la.idx,
        "atr_ref": atr_ref,
    }
    q = _quality_score(feats, pattern)
    feats["quality"] = q
    if q < MIN_QUALITY:
        return None

    return PatternHit(
        pattern=pattern,
        direction=direction,
        start_i=start_i,
        break_i=i,
        break_p=entry,
        neckline=neck_at_i,
        sl=sl,
        tp=tp,
        pattern_high=peak,
        pattern_low=trough,
        swing_indices=[ls.idx, la.idx, head.idx, ra.idx, rs_idx],
        features=feats,
        quality=q,
    )


def detect_npeak_patterns(
    ohlc: pd.DataFrame,
    pair: str,
    tf: str,
    patterns: Optional[List[str]] = None,
) -> List[PatternSignal]:
    closes = ohlc["Close"].values.astype(float)
    highs = ohlc["High"].values.astype(float)
    lows = ohlc["Low"].values.astype(float)
    atr = compute_atr(highs, lows, closes)
    order = PIVOT_ORDER.get(tf, 6)
    want = set(patterns) if patterns else {
        "double_top", "double_bottom", "triple_top", "triple_bottom",
        "head_shoulders", "inverse_head_shoulders",
    }

    raw = extract_pivots(closes, order)
    confirmed: List[Pivot] = []
    hits: List[PatternHit] = []
    # lock per (pattern, first_swing_idx) to avoid re-firing same structure
    seen_keys: set = set()

    pivot_i = 0
    # Only evaluate near potential breaks: every bar after we have enough pivots,
    # but skip if no new pivot and no price near a recent neckline candidate.
    # No check looks further back than the last 10 pivots, so cap the trailing
    # window - otherwise rebuilding the alternating list each bar is O(n^2).
    PIVOT_WINDOW = 30
    for i in range(order * 2 + 1, len(closes)):
        while pivot_i < len(raw) and raw[pivot_i].confirm_i <= i:
            confirmed.append(raw[pivot_i])
            pivot_i += 1
        if len(confirmed) > PIVOT_WINDOW:
            del confirmed[:-PIVOT_WINDOW]
        if len(confirmed) < 3:
            continue
        # cheap throttle: only check when a pivot confirmed today OR every 3rd bar
        new_pivot = pivot_i > 0 and raw[pivot_i - 1].confirm_i == i
        if not new_pivot and (i % 2 != 0):
            continue

        alt = alternating_pivots(confirmed)
        if len(alt) < 3:
            continue

        candidates: List[PatternHit] = []
        if "double_top" in want:
            h = _check_double(alt, closes, atr, i, bearish=True)
            if h:
                candidates.append(h)
        if "double_bottom" in want:
            h = _check_double(alt, closes, atr, i, bearish=False)
            if h:
                candidates.append(h)
        if "triple_top" in want:
            h = _check_triple(alt, closes, atr, i, bearish=True)
            if h:
                candidates.append(h)
        if "triple_bottom" in want:
            h = _check_triple(alt, closes, atr, i, bearish=False)
            if h:
                candidates.append(h)
        if "head_shoulders" in want:
            h = _check_hs(alt, closes, atr, i, inverted=False)
            if h:
                candidates.append(h)
        if "inverse_head_shoulders" in want:
            h = _check_hs(alt, closes, atr, i, inverted=True)
            if h:
                candidates.append(h)

        for h in candidates:
            key = (h.pattern, h.start_i, tuple(h.swing_indices[:3]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(h)

    idx = ohlc.index
    signals: List[PatternSignal] = []
    for h in hits:
        signals.append(
            PatternSignal(
                pattern=h.pattern,
                direction=h.direction,
                pair=pair,
                timeframe=tf,
                confirm_idx=h.break_i,
                confirm_ts=pd.Timestamp(idx[h.break_i]),
                entry_price=h.break_p,
                sl=h.sl,
                tp=h.tp,
                neckline=h.neckline,
                pattern_high=h.pattern_high,
                pattern_low=h.pattern_low,
                swing_indices=h.swing_indices,
                meta={**h.features, "quality": h.quality, "start_i": h.start_i},
            )
        )
    return signals
