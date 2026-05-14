"""Phase 2–3: preset pairs, cointegration, Granger (leader→follower), half-life in days, Z-score signal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, grangercausalitytests

from analysis.cointegration import ols_log_spread
from data.etf_asset_groups import underlying_group
from data.preset_pairs import PRESET_PAIRS, PresetPair
from data.timeframes import bar_duration_hours

logger = logging.getLogger(__name__)

COINT_CUTOFF = 0.05
GRANGER_CUTOFF = 0.05
HALF_LIFE_DAYS_MIN = 0.5
HALF_LIFE_DAYS_MAX = 2.0
SCREEN_BARS = 150
Z_ENTRY = 2.0
Z_EXIT = 0.5


def _coerce_df(x: object) -> pd.DataFrame:
    """Avoid `df or pd.DataFrame()` — bool(DataFrame) is ambiguous in pandas."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _top100_representative(ticker: str, top100: list[str]) -> str | None:
    """
    Map a preset ticker to the symbol actually present in the ranked top-100 list.
    The list keeps at most one fund per underlying (e.g. VOO may appear instead of SPY).
    """
    if ticker in top100:
        return ticker
    g = underlying_group(ticker)
    for s in top100:
        if underlying_group(s) == g:
            return s
    return None


@dataclass
class ScreenedPair:
    category: str
    etf_y: str
    etf_x: str
    preset_sym_a: str
    preset_sym_b: str
    leader: str
    follower: str
    coint_pvalue: float
    coint_pass: bool
    granger_min_pvalue: float | None
    granger_pass: bool
    half_life_days: float | None
    half_life_pass: bool
    hedge_beta: float
    hedge_alpha: float
    zscore: float
    phase3_signal: str
    all_phase2_pass: bool


def _align_log_closes(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    *,
    max_bars: int | None = None,
) -> tuple[pd.Series, pd.Series] | None:
    if df_y.empty or df_x.empty or "close" not in df_y.columns or "close" not in df_x.columns:
        return None
    y = df_y["close"].astype(float)
    x = df_x["close"].astype(float)
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if max_bars is not None and len(j) > max_bars:
        j = j.iloc[-max_bars:]
    if len(j) < 30:
        return None
    return np.log(j["y"]), np.log(j["x"])


def _half_life_periods(spread: np.ndarray) -> float | None:
    z = spread - np.nanmean(spread)
    if len(z) < 10:
        return None
    z1, z2 = z[:-1], z[1:]
    if np.allclose(z1, z2):
        return None
    X = z1.reshape(-1, 1)
    try:
        rho = np.linalg.lstsq(X, z2, rcond=None)[0][0]
    except Exception:
        return None
    if rho <= 0 or rho >= 1:
        return None
    return float(-np.log(2) / np.log(rho))


def _half_life_days_from_spread(spread: pd.Series, bar_interval: str) -> float | None:
    s = spread.dropna()
    if len(s) < 10:
        return None
    periods = _half_life_periods(s.values.astype(float))
    if periods is None:
        return None
    h_per_bar = bar_duration_hours(bar_interval)
    hours = periods * h_per_bar
    return float(hours / 24.0)


def _granger_min_p(
    ret_follower: pd.Series,
    ret_leader: pd.Series,
    maxlag: int = 5,
) -> float | None:
    j = pd.concat([ret_follower.rename("f"), ret_leader.rename("l")], axis=1, join="inner").dropna()
    if len(j) < maxlag + 15:
        return None
    j = j.iloc[-SCREEN_BARS:] if len(j) > SCREEN_BARS else j
    if len(j) < maxlag + 10:
        return None
    # Column 0 = caused (follower); column 1 = causing (leader)
    data = j[["f", "l"]].values.astype(float)
    lag = min(maxlag, max(1, len(j) // 10))
    try:
        with np.errstate(invalid="ignore"):
            res = grangercausalitytests(data, maxlag=lag, verbose=False)
    except Exception as e:
        logger.debug("Granger failed: %s", e)
        return None
    pvals: list[float] = []
    for k in range(1, lag + 1):
        block = res.get(k)
        if not block:
            continue
        d0 = block[0]
        for key in ("ssr_ftest", "params_ftest", "lrtest"):
            ft = d0.get(key)
            if ft is not None and len(ft) > 1:
                pvals.append(float(ft[1]))
                break
    if not pvals:
        return None
    return float(min(pvals))


def _phase3_z_and_signal(spread_full: pd.Series, etf_y: str, etf_x: str) -> tuple[float, str]:
    s = spread_full.dropna()
    if len(s) < 20:
        return float("nan"), "Insufficient history for Z-score"
    mu = float(s.mean())
    sig = float(s.std())
    if sig < 1e-12 or np.isnan(sig):
        return float("nan"), "Spread std ~ 0"
    z = float((s.iloc[-1] - mu) / sig)
    # Spread = log(Y) - α - β·log(X); high Z ⇒ Y rich vs hedge ⇒ short Y, long X per unit β
    if z > Z_ENTRY:
        return (
            z,
            f"Z>{Z_ENTRY}: short `{etf_y}`, long `{etf_x}` (vs β-hedge); exit when |Z|≤{Z_EXIT} (or 0)",
        )
    if z < -Z_ENTRY:
        return (
            z,
            f"Z<-{Z_ENTRY}: long `{etf_y}`, short `{etf_x}` (vs β-hedge); exit when |Z|≤{Z_EXIT} (or 0)",
        )
    return z, f"No entry (|Z|≤{Z_ENTRY}); when in a trade, exit toward mean (|Z|≤{Z_EXIT})"


def screen_preset_pair(
    p: PresetPair,
    etf_raw: dict[str, pd.DataFrame],
    top100: list[str],
    bar_interval: str,
) -> ScreenedPair | None:
    ra = _top100_representative(p.sym_a, top100)
    rb = _top100_representative(p.sym_b, top100)
    if ra is None or rb is None or ra == rb:
        return None

    leader_r = _top100_representative(p.leader, top100)
    follower_r = _top100_representative(p.follower, top100)
    if leader_r is None or follower_r is None:
        return None

    da = _coerce_df(etf_raw.get(ra))
    db = _coerce_df(etf_raw.get(rb))
    if da.empty or db.empty:
        return None

    aligned_full = _align_log_closes(da, db, max_bars=None)
    if aligned_full is None:
        return None
    ly_f, lx_f = aligned_full
    spread_full, alpha, beta = ols_log_spread(ly_f, lx_f)
    spread_full = spread_full.dropna()

    z, sig3 = _phase3_z_and_signal(spread_full, ra, rb)

    aligned_150 = _align_log_closes(da, db, max_bars=SCREEN_BARS)
    if aligned_150 is None:
        return None
    ly, lx = aligned_150
    try:
        _, coint_p, _ = coint(ly, lx)
    except Exception as e:
        logger.debug("coint %s %s: %s", ra, rb, e)
        coint_p = 1.0
    coint_p = float(coint_p) if not np.isnan(coint_p) else 1.0
    coint_ok = coint_p < COINT_CUTOFF

    spread_150, _, _ = ols_log_spread(ly, lx)
    spread_150 = spread_150.dropna()
    hl_days = _half_life_days_from_spread(spread_150, bar_interval)
    hl_ok = hl_days is not None and HALF_LIFE_DAYS_MIN <= hl_days <= HALF_LIFE_DAYS_MAX

    # Returns on aligned closes (simple %) for Granger: leader → follower
    cl = pd.concat(
        [da["close"].astype(float).rename("a"), db["close"].astype(float).rename("b")],
        axis=1,
        join="inner",
    ).dropna()
    if len(cl) > SCREEN_BARS:
        cl = cl.iloc[-SCREEN_BARS:]
    ra = cl["a"].pct_change()
    rb = cl["b"].pct_change()
    jret = pd.concat([ra.rename("a"), rb.rename("b")], axis=1).dropna()
    follower_ret = jret["a"] if follower_r == ra else jret["b"]
    leader_ret = jret["a"] if leader_r == ra else jret["b"]

    gp = _granger_min_p(follower_ret, leader_ret, maxlag=5)
    gr_ok = gp is not None and gp < GRANGER_CUTOFF

    all2 = coint_ok and gr_ok and hl_ok

    return ScreenedPair(
        category=p.category,
        etf_y=ra,
        etf_x=rb,
        preset_sym_a=p.sym_a,
        preset_sym_b=p.sym_b,
        leader=leader_r,
        follower=follower_r,
        coint_pvalue=coint_p,
        coint_pass=coint_ok,
        granger_min_pvalue=gp,
        granger_pass=gr_ok,
        half_life_days=hl_days,
        half_life_pass=hl_ok,
        hedge_beta=beta,
        hedge_alpha=alpha,
        zscore=z,
        phase3_signal=sig3,
        all_phase2_pass=all2,
    )


def run_preset_screening(
    etf_raw: dict[str, pd.DataFrame],
    top100: list[str],
    bar_interval: str,
) -> list[ScreenedPair]:
    out: list[ScreenedPair] = []
    for p in PRESET_PAIRS:
        r = screen_preset_pair(p, etf_raw, top100, bar_interval)
        if r is not None:
            out.append(r)
    out.sort(key=lambda x: (not x.all_phase2_pass, x.coint_pvalue))
    return out


def screened_to_records(rows: list[ScreenedPair]) -> list[dict[str, Any]]:
    recs = []
    for r in rows:
        recs.append(
            {
                "category": r.category,
                "etf_y": r.etf_y,
                "etf_x": r.etf_x,
                "preset_sym_a": r.preset_sym_a,
                "preset_sym_b": r.preset_sym_b,
                "leader": r.leader,
                "follower": r.follower,
                "cointegration_pvalue": r.coint_pvalue,
                "coint_pass": r.coint_pass,
                "granger_min_p": r.granger_min_pvalue,
                "granger_pass": r.granger_pass,
                "half_life_days": r.half_life_days,
                "half_life_pass": r.half_life_pass,
                "phase2_all_pass": r.all_phase2_pass,
                "hedge_beta": r.hedge_beta,
                "hedge_alpha_log": r.hedge_alpha,
                "zscore_1000bar": r.zscore,
                "phase3_signal": r.phase3_signal,
            }
        )
    return recs
