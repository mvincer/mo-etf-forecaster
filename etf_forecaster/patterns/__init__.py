"""Pattern detector registry (vendored from Mo_Dash strategies/chart_patterns).

Causal n-peak reversals + channels + S/R + wedges + flags + trendlines, all
ATR-normalized OHLC geometry. `symbol` is a label only; nothing here is FX-specific.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from etf_forecaster.patterns.channels import detect_channels
from etf_forecaster.patterns.flags import detect_flags
from etf_forecaster.patterns.npeak import detect_npeak_patterns
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.sr_levels import detect_sr
from etf_forecaster.patterns.trends import detect_trend_breaks, detect_trends
from etf_forecaster.patterns.wedges import detect_wedges

REVERSAL = {
    "double_top", "triple_top", "double_bottom", "triple_bottom",
    "head_shoulders", "inverse_head_shoulders",
}

ALL_PATTERNS = [
    "double_top", "triple_top", "double_bottom", "triple_bottom",
    "head_shoulders", "inverse_head_shoulders",
    "channel", "support_resistance",
    "rising_wedge", "falling_wedge", "bull_flag", "bear_flag",
    "uptrend", "downtrend", "uptrend_break", "downtrend_break",
]

# Families used for per-bar featurization (direction comes from the signal itself)
FAMILIES = {
    "reversal": ["double_top", "triple_top", "double_bottom", "triple_bottom",
                 "head_shoulders", "inverse_head_shoulders"],
    "channel": ["channel"],
    "sr": ["support_resistance"],
    "wedge": ["rising_wedge", "falling_wedge"],
    "flag": ["bull_flag", "bear_flag"],
    "trend": ["uptrend", "downtrend"],
    "trend_break": ["uptrend_break", "downtrend_break"],
}


def detect_pattern(pattern: str, ohlc: pd.DataFrame, symbol: str, tf: str) -> List[PatternSignal]:
    if pattern in REVERSAL:
        return detect_npeak_patterns(ohlc, symbol, tf, patterns=[pattern])
    if pattern == "channel":
        return detect_channels(ohlc, symbol, tf)
    if pattern == "support_resistance":
        return detect_sr(ohlc, symbol, tf)
    if pattern == "rising_wedge":
        return detect_wedges(ohlc, symbol, tf, rising=True)
    if pattern == "falling_wedge":
        return detect_wedges(ohlc, symbol, tf, rising=False)
    if pattern == "bull_flag":
        return detect_flags(ohlc, symbol, tf, bull=True)
    if pattern == "bear_flag":
        return detect_flags(ohlc, symbol, tf, bull=False)
    if pattern == "uptrend":
        return detect_trends(ohlc, symbol, tf, up=True)
    if pattern == "downtrend":
        return detect_trends(ohlc, symbol, tf, up=False)
    if pattern == "uptrend_break":
        return detect_trend_breaks(ohlc, symbol, tf, up=True)
    if pattern == "downtrend_break":
        return detect_trend_breaks(ohlc, symbol, tf, up=False)
    raise ValueError(f"Unknown pattern: {pattern}")


def detect_all(ohlc: pd.DataFrame, symbol: str, tf: str,
               patterns: List[str] | None = None) -> List[PatternSignal]:
    out: List[PatternSignal] = []
    for name in (patterns or ALL_PATTERNS):
        try:
            out.extend(detect_pattern(name, ohlc, symbol, tf))
        except Exception:  # noqa: BLE001 - one bad detector must not kill the batch
            continue
    return sorted(out, key=lambda s: s.confirm_idx)
