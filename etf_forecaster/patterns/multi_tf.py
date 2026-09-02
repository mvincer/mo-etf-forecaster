"""Resample daily bars to weekly/monthly and run the pattern detectors on each timeframe."""

from __future__ import annotations

import pandas as pd

from etf_forecaster.patterns import ALL_PATTERNS, detect_all
from etf_forecaster.patterns.pattern_signal import PatternSignal

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def to_detector_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """lowercase lake bars -> capitalized OHLCV frame the detectors expect."""
    out = pd.DataFrame({
        "Open": bars["open"], "High": bars["high"], "Low": bars["low"],
        "Close": bars["close"], "Volume": bars.get("volume", 0.0),
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def resample_htf(d1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Completed-bar weekly (W-FRI) or monthly frame; the trailing partial bar is dropped
    so a pattern can never confirm on a bar that has not actually closed yet."""
    rule = {"W1": "W-FRI", "M1": "ME"}[tf]
    res = d1.resample(rule).agg(_AGG).dropna(subset=["Close"])
    if len(res) == 0:
        return res
    last_daily = d1.index.max()
    # the final resampled bar is partial unless the daily history reaches its period end
    if tf == "W1" and last_daily.dayofweek != 4:
        res = res.iloc[:-1]
    elif tf == "M1" and (last_daily + pd.Timedelta(days=1)).month == last_daily.month:
        res = res.iloc[:-1]
    # keep the actual end-of-period data date for as-of alignment
    ends = d1.index.to_series().resample(rule).max().dropna()
    res["bar_end"] = ends.reindex(res.index)
    return res.dropna(subset=["bar_end"])


def detect_multi_tf(bars: pd.DataFrame, symbol: str,
                    timeframes: tuple[str, ...] = ("D1", "W1", "M1"),
                    patterns: list[str] | None = None) -> dict[str, list[PatternSignal]]:
    """Run detect_all per timeframe. Each signal gains meta['bar_end'] = the actual
    last data date inside its confirming bar (used for leak-free daily alignment)."""
    d1 = to_detector_frame(bars)
    out: dict[str, list[PatternSignal]] = {}
    for tf in timeframes:
        if tf == "D1":
            frame = d1
            bar_end = pd.Series(frame.index, index=frame.index)
        else:
            frame = resample_htf(d1, tf)
            if len(frame) < 30:
                out[tf] = []
                continue
            bar_end = frame["bar_end"]
            frame = frame.drop(columns=["bar_end"])
        signals = detect_all(frame, symbol, tf, patterns=patterns or ALL_PATTERNS)
        for sig in signals:
            sig.meta["bar_end"] = pd.Timestamp(bar_end.iloc[sig.confirm_idx])
        out[tf] = signals
    return out
