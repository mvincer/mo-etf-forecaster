"""Turn sparse PatternSignal events into dense per-daily-bar ML features.

As-of alignment: a D1 pattern confirming at day t's close is visible at t (forecasts are
made after the close). A W1/M1 pattern confirming on the bar whose last data date is e is
visible only from the NEXT daily bar after e - one full higher-timeframe bar of caution is
already embedded because detection only runs on completed bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_forecaster.patterns import FAMILIES
from etf_forecaster.patterns.pattern_signal import PatternSignal
from etf_forecaster.patterns.swing_points import compute_atr

# how long (in daily bars) an event keeps influencing features
_DECAY = {"D1": 21, "W1": 63, "M1": 126}


def _visible_from(sig: PatternSignal, tf: str, daily_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    end = pd.Timestamp(sig.meta.get("bar_end", sig.confirm_ts)).normalize()
    if tf == "D1":
        pos = daily_index.searchsorted(end, side="left")
        if pos >= len(daily_index):
            return None
        return daily_index[pos]
    pos = daily_index.searchsorted(end, side="right")   # strictly after the HTF bar end
    if pos >= len(daily_index):
        return None
    return daily_index[pos]


def featurize(bars: pd.DataFrame, signals_by_tf: dict[str, list[PatternSignal]]) -> pd.DataFrame:
    """Dense feature frame indexed like `bars` (daily). Column names carry no block
    prefix; assemble.py maps D1 -> patd_, W1/M1 + agreement -> path_."""
    idx = bars.index
    n = len(idx)
    atr = compute_atr(bars["high"].values, bars["low"].values, bars["close"].values)
    atr = pd.Series(atr, index=idx).ffill()
    close = bars["close"]

    cols: dict[str, np.ndarray] = {}
    tf_net_dir: dict[str, np.ndarray] = {}

    for tf, signals in signals_by_tf.items():
        window = _DECAY[tf]
        fam_of = {p: fam for fam, pats in FAMILIES.items() for p in pats}
        # collect events per family: (pos, dir, quality, neckline, tp, sl)
        events: dict[str, list[tuple[int, float, float, float, float, float]]] = {
            fam: [] for fam in FAMILIES
        }
        for sig in signals:
            vis = _visible_from(sig, tf, idx)
            if vis is None:
                continue
            pos = int(idx.get_indexer([vis])[0])
            if pos < 0:
                continue
            direction = 1.0 if sig.direction == "buy" else -1.0
            quality = float(sig.meta.get("quality", 60.0)) / 100.0
            events[fam_of.get(sig.pattern, "sr")].append(
                (pos, direction, quality, float(sig.neckline), float(sig.tp), float(sig.sl))
            )

        net_dir = np.zeros(n)
        for fam, evs in events.items():
            dir_a = np.zeros(n)
            qual_a = np.zeros(n)
            age_a = np.ones(n)          # normalized age; 1.0 = expired/none
            trig_a = np.zeros(n)
            neck_a = np.full(n, np.nan)
            tp_a = np.full(n, np.nan)
            sl_a = np.full(n, np.nan)
            for pos, direction, quality, neck, tp, sl in sorted(evs):
                end = min(pos + window, n)
                ages = np.arange(0, end - pos)
                decay = np.maximum(0.0, 1.0 - ages / window)
                dir_a[pos:end] = direction * decay
                qual_a[pos:end] = quality
                age_a[pos:end] = ages / window
                trig_a[pos] = direction
                neck_a[pos:end] = neck
                tp_a[pos:end] = tp
                sl_a[pos:end] = sl
            key = f"{tf.lower()}_{fam}"
            cols[f"{key}_dir"] = dir_a
            cols[f"{key}_quality"] = qual_a
            cols[f"{key}_age"] = age_a
            cols[f"{key}_trigger"] = trig_a
            with np.errstate(all="ignore"):
                cols[f"{key}_dist_neck_atr"] = np.clip(
                    (close.values - neck_a) / atr.values, -10, 10)
                cols[f"{key}_dist_tp_atr"] = np.clip((tp_a - close.values) / atr.values, -20, 20)
                cols[f"{key}_dist_sl_atr"] = np.clip((close.values - sl_a) / atr.values, -20, 20)
            net_dir += dir_a

        # pattern-pressure counts over a rolling window of visible events
        trig_all = np.zeros(n)
        for evs in events.values():
            for pos, direction, *_ in evs:
                trig_all[pos] += direction
        trig_ser = pd.Series(trig_all, index=idx)
        pos_cnt = trig_ser.clip(lower=0).rolling(window, min_periods=1).sum()
        neg_cnt = (-trig_ser.clip(upper=0)).rolling(window, min_periods=1).sum()
        cols[f"{tf.lower()}_bull_count"] = pos_cnt.values
        cols[f"{tf.lower()}_bear_count"] = neg_cnt.values
        cols[f"{tf.lower()}_net_count"] = (pos_cnt - neg_cnt).values
        tf_net_dir[tf] = net_dir

    # cross-timeframe agreement
    d1 = tf_net_dir.get("D1")
    w1 = tf_net_dir.get("W1")
    m1 = tf_net_dir.get("M1")
    if d1 is not None and w1 is not None:
        cols["d1_w1_agree"] = np.sign(d1) * (np.sign(d1) == np.sign(w1)) * (d1 != 0) * (w1 != 0)
    if d1 is not None and w1 is not None and m1 is not None:
        same = (np.sign(d1) == np.sign(w1)) & (np.sign(w1) == np.sign(m1)) & (d1 != 0) & (m1 != 0)
        cols["all_tf_agree"] = np.sign(d1) * same

    out = pd.DataFrame(cols, index=idx)
    # NaN distances where no event is active are legitimate missing values (trees handle them)
    return out
