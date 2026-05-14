"""Pairwise Engle–Granger cointegration on log prices and spread analytics."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import coint

from data.etf_asset_groups import is_near_duplicate_tracker, same_underlying

logger = logging.getLogger(__name__)

# Minimum overlapping 5m bars for regression / cointegration (≳ multiple sessions).
MIN_ALIGNED_BARS = 1500

# Require completed p90→p70 and p10→p30 episodes on the scan window.
MIN_UP_EXTREME_CROSSINGS = 3
MIN_DOWN_EXTREME_CROSSINGS = 3

# Default calendar window for scans (must fit enough 5m bars + crossings).
DEFAULT_SCAN_LOOKBACK_DAYS = 60


@dataclass
class PairResult:
    etf_y: str
    etf_x: str
    coint_pvalue: float
    hedge_beta: float
    hedge_const: float
    spread: pd.Series
    half_life_hours: float | None
    avg_hours_p90_to_p70: float | None
    avg_hours_p10_to_p30: float | None
    current_spread: float
    current_percentile: float
    n_up_extreme_crossings: int
    n_down_extreme_crossings: int


def _align_log_closes(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    y_sym: str,
    x_sym: str,
) -> tuple[pd.Series, pd.Series] | None:
    if df_y.empty or df_x.empty or "close" not in df_y.columns or "close" not in df_x.columns:
        return None
    y = df_y["close"].astype(float)
    x = df_x["close"].astype(float)
    joined = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if len(joined) < MIN_ALIGNED_BARS:
        return None
    ly = np.log(joined["y"])
    lx = np.log(joined["x"])
    return ly, lx


def _spread_from_ols(ly: pd.Series, lx: pd.Series) -> tuple[np.ndarray, float, float, pd.DatetimeIndex]:
    X = sm.add_constant(lx.values)
    model = sm.OLS(ly.values, X).fit()
    const, beta = float(model.params[0]), float(model.params[1])
    fitted = model.fittedvalues
    spread = ly.values - fitted
    return spread, const, beta, ly.index


def ols_log_spread(ly: pd.Series, lx: pd.Series) -> tuple[pd.Series, float, float]:
    """Residual spread from OLS log(y) ~ log(x); index aligned to ly/lx."""
    spread_arr, const, beta, spread_idx = _spread_from_ols(ly, lx)
    return pd.Series(spread_arr, index=spread_idx), float(const), float(beta)


def estimate_half_life_hours(spread: pd.Series) -> float | None:
    s = spread.dropna()
    if len(s) < 10:
        return None
    return _half_life_hours(s.values.astype(float), s.index)


def _half_life_hours(spread: np.ndarray, index: pd.DatetimeIndex) -> float | None:
    z = spread - np.nanmean(spread)
    if len(z) < 10:
        return None
    z1 = z[:-1]
    z2 = z[1:]
    if np.allclose(z1, z2):
        return None
    X = z1.reshape(-1, 1)
    try:
        rho = np.linalg.lstsq(X, z2, rcond=None)[0][0]
    except Exception:
        return None
    if rho <= 0 or rho >= 1:
        return None
    periods = -np.log(2) / np.log(rho)
    if len(index) < 2:
        return None
    dt = pd.Series(index).diff().median()
    if pd.isna(dt) or dt.total_seconds() <= 0:
        return None
    hours_per_step = dt.total_seconds() / 3600.0
    return float(periods * hours_per_step)


def _high_episode_durations_hours(spread: pd.Series) -> list[float]:
    """Completed excursions: enter >=p90 → first touch <=p70 (upside / stretched)."""
    s = spread.dropna()
    if len(s) < 20:
        return []
    idx = s.index
    vals = s.values.astype(float)
    p90, p70 = float(np.percentile(vals, 90)), float(np.percentile(vals, 70))
    n = len(vals)
    out: list[float] = []
    in_ep = False
    start_ts = None
    for i in range(n):
        v = vals[i]
        prev = vals[i - 1] if i > 0 else np.nan
        if not in_ep:
            crossed = v >= p90 and (i == 0 or prev < p90)
            if crossed:
                in_ep = True
                start_ts = idx[i]
        else:
            if v <= p70:
                out.append((idx[i] - start_ts).total_seconds() / 3600.0)
                in_ep = False
                start_ts = None
    return out


def _low_episode_durations_hours(spread: pd.Series) -> list[float]:
    """Completed excursions: enter <=p10 → first touch >=p30 (downside / stretched)."""
    s = spread.dropna()
    if len(s) < 20:
        return []
    idx = s.index
    vals = s.values.astype(float)
    p30, p10 = float(np.percentile(vals, 30)), float(np.percentile(vals, 10))
    n = len(vals)
    out: list[float] = []
    in_ep = False
    start_ts = None
    for i in range(n):
        v = vals[i]
        prev = vals[i - 1] if i > 0 else np.nan
        if not in_ep:
            crossed = v <= p10 and (i == 0 or prev > p10)
            if crossed:
                in_ep = True
                start_ts = idx[i]
        else:
            if v >= p30:
                out.append((idx[i] - start_ts).total_seconds() / 3600.0)
                in_ep = False
                start_ts = None
    return out


def count_extreme_reversions(spread: pd.Series) -> tuple[int, int]:
    hi = _high_episode_durations_hours(spread)
    lo = _low_episode_durations_hours(spread)
    return len(hi), len(lo)


def _avg_reversion_hours(spread: pd.Series) -> tuple[float | None, float | None]:
    hi = _high_episode_durations_hours(spread)
    lo = _low_episode_durations_hours(spread)
    return (
        float(np.mean(hi)) if hi else None,
        float(np.mean(lo)) if lo else None,
    )


def _current_percentile(spread: pd.Series, current: float) -> float:
    v = spread.dropna().values.astype(float)
    if len(v) == 0:
        return float("nan")
    return float(stats.percentileofscore(v, current, kind="rank"))


def scan_pairs(
    etf_frames: dict[str, pd.DataFrame],
    lookback_days: int = DEFAULT_SCAN_LOOKBACK_DAYS,
    max_pairs: int = 200,
    pvalue_cutoff: float = 0.05,
) -> list[PairResult]:
    from data.yahoo import last_n_calendar_days

    syms = list(etf_frames.keys())
    trimmed: dict[str, pd.DataFrame] = {}
    for s in syms:
        df = etf_frames[s]
        if df.empty:
            continue
        w = last_n_calendar_days(df, lookback_days)
        if len(w) < MIN_ALIGNED_BARS:
            continue
        trimmed[s] = w

    syms = sorted(trimmed.keys())
    results: list[PairResult] = []

    for y_sym, x_sym in itertools.combinations(syms, 2):
        aligned = _align_log_closes(trimmed[y_sym], trimmed[x_sym], y_sym, x_sym)
        if aligned is None:
            continue
        ly, lx = aligned
        if same_underlying(y_sym, x_sym):
            continue

        spread_arr, const, beta, spread_idx = _spread_from_ols(ly, lx)
        if is_near_duplicate_tracker(beta, ly, lx):
            continue

        try:
            _, pvalue, _ = coint(ly, lx)
        except Exception as e:
            logger.debug("coint failed %s %s: %s", y_sym, x_sym, e)
            continue
        if pvalue > pvalue_cutoff or np.isnan(pvalue):
            continue
        spread_series = pd.Series(spread_arr, index=spread_idx)
        n_up, n_down = count_extreme_reversions(spread_series)
        if n_up < MIN_UP_EXTREME_CROSSINGS or n_down < MIN_DOWN_EXTREME_CROSSINGS:
            continue
        hl = _half_life_hours(spread_arr, spread_idx)
        p90_to_p70, p10_to_p30 = _avg_reversion_hours(spread_series)
        cur_spread = float(spread_arr[-1])
        cur_pct = _current_percentile(spread_series, cur_spread)

        results.append(
            PairResult(
                etf_y=y_sym,
                etf_x=x_sym,
                coint_pvalue=float(pvalue),
                hedge_beta=beta,
                hedge_const=const,
                spread=spread_series,
                half_life_hours=hl,
                avg_hours_p90_to_p70=p90_to_p70,
                avg_hours_p10_to_p30=p10_to_p30,
                current_spread=cur_spread,
                current_percentile=cur_pct,
                n_up_extreme_crossings=n_up,
                n_down_extreme_crossings=n_down,
            )
        )

    results.sort(key=lambda r: r.coint_pvalue)
    return results[:max_pairs]


def results_to_records(rows: list[PairResult]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "etf_y": r.etf_y,
                "etf_x": r.etf_x,
                "cointegration_pvalue": r.coint_pvalue,
                "hedge_ratio_beta": r.hedge_beta,
                "hedge_intercept_log_space": r.hedge_const,
                "half_life_hours_ar1": r.half_life_hours,
                "avg_hours_p90_to_p70": r.avg_hours_p90_to_p70,
                "avg_hours_p10_to_p30": r.avg_hours_p10_to_p30,
                "latest_spread": r.current_spread,
                "latest_spread_percentile_scan": r.current_percentile,
                "n_up_extreme_crossings": r.n_up_extreme_crossings,
                "n_down_extreme_crossings": r.n_down_extreme_crossings,
            }
        )
    return out
