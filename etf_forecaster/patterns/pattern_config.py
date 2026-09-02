"""Geometry configuration for the vendored chart-pattern detectors.

Provenance: vendored from Mo_Dash strategies/chart_patterns/pattern_config.py with the
FX/OTA cost and instrument layers stripped, and W1 (weekly) / M1 (monthly) timeframe
entries added for higher-timeframe detection on resampled daily bars.
"""

from __future__ import annotations

TARGET_TIMEFRAMES = ["D1", "W1", "M1"]

PATTERN_NAMES = [
    "double_top",
    "triple_top",
    "double_bottom",
    "triple_bottom",
    "head_shoulders",
    "inverse_head_shoulders",
    "channel",
    "support_resistance",
    "rising_wedge",
    "falling_wedge",
    "bull_flag",
    "bear_flag",
    "uptrend",
    "downtrend",
    "uptrend_break",
    "downtrend_break",
]

# Causal pivot order (confirmation lag = order bars)
PIVOT_ORDER: dict[str, int] = {
    "m5": 8, "m15": 7, "H1": 6, "H4": 5, "D1": 4,
    "W1": 3, "M1": 2,
}
SWING_ORDER = PIVOT_ORDER

MIN_PATTERN_BARS: dict[str, int] = {
    "m5": 20, "m15": 15, "H1": 12, "H4": 10, "D1": 8,
    "W1": 6, "M1": 5,
}
MAX_PATTERN_BARS: dict[str, int] = {
    "m5": 200, "m15": 150, "H1": 120, "H4": 80, "D1": 60,
    "W1": 52, "M1": 36,
}

TP_R = 2.0
SL_ATR_BUFFER = 0.5

PEAK_TOLERANCE_ATR = 0.50
NECKLINE_BUFFER_ATR = 0.05

# Strict geometry for clear (non-wishy-washy) reversal patterns
PEAK_EQ_ATR = 0.50
SHOULDER_EQ_ATR = 0.60
MIN_DEPTH_ATR = 1.25
MIN_HEAD_DOMINANCE_ATR = 0.70
TIME_SYM_LO = 0.50
TIME_SYM_HI = 2.0
MIN_PRIOR_PCTILE = 0.80
MIN_PRIOR_MOVE_ATR = 2.5
PRIOR_LOOKBACK = 40
MIN_PRIOR_EFFICIENCY = 0.18
MIN_PRIOR_STRUCTURE = 0.60
MIN_PATTERN_R2 = 0.30
MIN_QUALITY = 58.0

# Channel / S/R
CHANNEL_MIN_TOUCHES = 3
CHANNEL_MIN_WIDTH_ATR = 1.5
CHANNEL_SLOPE_TOL = 0.20
CHANNEL_MIN_QUALITY = 75.0
SR_MIN_TOUCHES = 3
SR_CLUSTER_ATR = 0.35

# Wedges
WEDGE_MIN_TOUCHES = 2
WEDGE_MIN_CONVERGENCE = 1.5
WEDGE_SLOPE_RATIO_MIN = 1.3
WEDGE_MIN_HEIGHT_ATR = 1.5
WEDGE_MIN_QUALITY = 55.0
WEDGE_MAX_BARS: dict[str, int] = {
    "m5": 250, "m15": 200, "H1": 160, "H4": 120, "D1": 90,
    "W1": 60, "M1": 36,
}

# Flags
FLAG_POLE_MIN_BARS = 8
FLAG_POLE_MAX_BARS = 40
FLAG_MIN_POLE_ATR = 3.0
FLAG_MIN_POLE_EFFICIENCY = 0.55
FLAG_MIN_BARS = 5
FLAG_MAX_BARS = 30
FLAG_MAX_RANGE_ATR = 1.8
FLAG_MAX_RETRACE = 0.50
FLAG_MIN_QUALITY = 55.0

# Trend continuation / breaks
TREND_MIN_SWINGS = 3
TREND_MIN_R2 = 0.70
TREND_MIN_NET_ATR = 3.0
TREND_TOUCH_TOL_ATR = 0.25
TREND_MIN_QUALITY = 60.0

ANN_FACTOR = {
    "m5": (252 * 24 * 12) ** 0.5,
    "m15": (252 * 24 * 4) ** 0.5,
    "H1": (252 * 24) ** 0.5,
    "H4": (252 * 6) ** 0.5,
    "D1": 252 ** 0.5,
    "W1": 52 ** 0.5,
    "M1": 12 ** 0.5,
}


def ann_factor(tf: str) -> float:
    return ANN_FACTOR.get(tf, 252 ** 0.5)
