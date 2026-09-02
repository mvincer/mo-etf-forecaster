"""SEASON block (cal_): day-of-week/month effects, turn-of-month, OPEX/quad-witching,
FOMC and macro release days, holiday proximity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonality_features(index: pd.DatetimeIndex,
                         events: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    dow = index.dayofweek
    for d, name in enumerate(("mon", "tue", "wed", "thu")):
        out[f"cal_{name}"] = (dow == d).astype(float)
    out["cal_month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    out["cal_month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)

    # position within the month's trading days
    ser = pd.Series(index, index=index)
    month_key = index.to_period("M")
    rank_in_month = ser.groupby(month_key).cumcount()
    month_len = ser.groupby(month_key).transform("size")
    out["cal_month_pos"] = (rank_in_month / (month_len - 1).clip(lower=1)).values

    # turn of month: last 2 + first 3 trading days
    from_end = (month_len - 1 - rank_in_month).values
    out["cal_turn_of_month"] = ((rank_in_month.values <= 2) | (from_end <= 1)).astype(float)

    # holiday proximity: calendar-day gap to the next trading day
    next_gap = pd.Series(index).diff().shift(-1).dt.days.fillna(1).values
    out["cal_pre_holiday"] = (next_gap > 3).astype(float)

    if events is not None and len(events):
        ev = events.reindex(index).fillna(0)
        for col in ("fomc", "opex", "quadwitch", "rel_cpi", "rel_employment",
                    "month_end", "quarter_end"):
            if col in ev:
                out[f"cal_{col}"] = ev[col].astype(float)
        if "fomc" in ev:
            f = ev["fomc"].astype(float)
            out["cal_fomc_next"] = f.shift(-1).fillna(0)   # decision tomorrow (scheduled, known)
            out["cal_fomc_prev"] = f.shift(1).fillna(0)
        if "opex" in ev:
            opex_pos = np.flatnonzero(ev["opex"].values)
            days_to = np.full(len(index), 30.0)
            if len(opex_pos):
                nxt = np.searchsorted(opex_pos, np.arange(len(index)), side="left")
                nxt = np.clip(nxt, 0, len(opex_pos) - 1)
                days_to = np.abs(opex_pos[nxt] - np.arange(len(index))).astype(float)
            out["cal_opex_week"] = (days_to <= 4).astype(float)
    return out
