"""Forecast quality metrics + multiple-testing corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def hit_rate(y: np.ndarray, p: np.ndarray) -> float:
    pred = (np.asarray(p) > 0.5).astype(float)
    return float(np.mean(pred == np.asarray(y)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), NaN if one class absent."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = np.clip((df["p"] * n_bins).astype(int), 0, n_bins - 1)
    g = df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))
    return g.reset_index()


def signal_pnl(y_ret: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> float:
    """Cumulative return of trading sign(p-0.5), unit notional."""
    side = np.where(np.asarray(p) > threshold, 1.0, -1.0)
    return float(np.nansum(side * np.asarray(y_ret, dtype=float)))


def direction_pvalue(hits: int, n: int, p0: float = 0.5) -> float:
    """One-sided binomial test that accuracy exceeds p0."""
    if n == 0:
        return 1.0
    return float(stats.binomtest(hits, n, p0, alternative="greater").pvalue)


def benjamini_hochberg(pvals: pd.Series, alpha: float = 0.10) -> pd.Series:
    """BH-FDR: boolean survives-correction flag per test."""
    p = pvals.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(False, index=pvals.index)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p.values <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    cutoff = p.iloc[k - 1] if k > 0 else -1.0
    return pvals <= cutoff


def deflated_edge(hit: float, n: int, n_trials: int) -> float:
    """Haircut a hit rate for selection over n_trials: expected max of n_trials
    N(0.5, 0.25/n) draws (Bailey/Lopez de Prado-style expected-maximum adjustment)."""
    if n <= 0 or n_trials <= 1:
        return hit
    se = 0.5 / np.sqrt(n)
    # expected maximum of n_trials standard normals
    emax = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + \
        np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(hit - se * emax)
