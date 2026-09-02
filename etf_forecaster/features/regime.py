"""Regime block (reg_): trending vs ranging state, market structure, statistical
trend tests, MA posture, vol regime, and a leak-free Markov-switching probability."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from etf_forecaster.patterns.swing_points import find_swing_highs, find_swing_lows

log = logging.getLogger(__name__)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _variance_ratio(logret: pd.Series, k: int, window: int = 252) -> pd.Series:
    """VR(k) = Var(k-period returns) / (k * Var(1-period)); >1 trending, <1 mean-reverting."""
    rk = logret.rolling(k).sum()
    var1 = logret.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1)


def _market_structure(high: pd.Series, low: pd.Series, order: int = 5) -> pd.Series:
    """+1 HH/HL, -1 LH/LL, 0 mixed - computed causally (swings confirm `order` bars late)."""
    h = high.values
    l = low.values
    n = len(h)
    state = np.zeros(n)
    sh = find_swing_highs(h, order)
    sl = find_swing_lows(l, order)
    # swing at index i is confirmed at i + order
    events: list[tuple[int, str, float]] = [(i + order, "h", v) for i, v in sh] + \
        [(i + order, "l", v) for i, v in sl]
    events.sort()
    last_h = prev_h = last_l = prev_l = np.nan
    cur = 0.0
    ei = 0
    for t in range(n):
        while ei < len(events) and events[ei][0] <= t:
            _, kind, val = events[ei]
            if kind == "h":
                prev_h, last_h = last_h, val
            else:
                prev_l, last_l = last_l, val
            ei += 1
        if np.isfinite(last_h) and np.isfinite(prev_h) and np.isfinite(last_l) and np.isfinite(prev_l):
            hh = last_h > prev_h
            hl = last_l > prev_l
            cur = 1.0 if (hh and hl) else (-1.0 if (not hh and not hl) else 0.0)
        state[t] = cur
    return pd.Series(state, index=high.index)


def _markov_prob(logret: pd.Series, *, refit_every: int = 504, min_train: int = 1000) -> pd.Series:
    """P(low-vol regime), filtered with parameters estimated on strictly past data."""
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:  # pragma: no cover
        return pd.Series(np.nan, index=logret.index)

    r = (logret * 100).dropna()
    out = pd.Series(np.nan, index=r.index)
    starts = range(min_train, len(r), refit_every)
    for s in starts:
        try:
            fit = MarkovRegression(r.iloc[:s], k_regimes=2, trend="c",
                                   switching_variance=True).fit(disp=False, maxiter=100)
            # filter the chunk [s, s+refit_every) with params fitted on [0, s)
            mod = MarkovRegression(r.iloc[: min(s + refit_every, len(r))], k_regimes=2,
                                   trend="c", switching_variance=True)
            filt = mod.filter(fit.params)
            probs = filt.filtered_marginal_probabilities
            # regime 0/1 ordering is arbitrary: pick the LOW-variance regime
            variances = fit.params[-2:]
            low = int(np.argmin(variances))
            chunk = probs[low].iloc[s: s + refit_every]
            out.loc[chunk.index] = chunk.values
        except Exception as exc:  # noqa: BLE001
            log.debug("markov chunk failed at %s: %s", s, exc)
            continue
    return out.reindex(logret.index)


def regime_features(bars: pd.DataFrame, *, markov: bool = True) -> pd.DataFrame:
    """All columns prefixed reg_."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    logret = np.log(close / close.shift())
    out = pd.DataFrame(index=bars.index)

    adx = _adx(high, low, close)
    out["reg_adx"] = adx
    trending = (adx > 25).astype(float)
    out["reg_trending"] = trending
    # persistence: bars since the trending state last changed
    change = trending.ne(trending.shift()).cumsum()
    out["reg_state_age"] = trending.groupby(change).cumcount().clip(upper=126) / 126.0

    out["reg_vr5"] = _variance_ratio(logret, 5)
    out["reg_vr20"] = _variance_ratio(logret, 20)

    ms = _market_structure(high, low)
    out["reg_structure"] = ms
    ms_change = ms.ne(ms.shift()).cumsum()
    out["reg_structure_age"] = ms.groupby(ms_change).cumcount().clip(upper=126) / 126.0

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["reg_dist_sma50"] = close / sma50 - 1
    out["reg_dist_sma200"] = close / sma200 - 1
    out["reg_sma50_slope"] = sma50.pct_change(21)
    out["reg_sma200_slope"] = sma200.pct_change(63)
    out["reg_golden"] = (sma50 > sma200).astype(float)

    rv20 = logret.rolling(20).std() * np.sqrt(252)
    rv_rank = rv20.rolling(756, min_periods=252).rank(pct=True)
    out["reg_rv20_rank"] = rv_rank
    out["reg_rv_tercile"] = pd.cut(rv_rank, [0, 1 / 3, 2 / 3, 1.0], labels=False).astype(float)

    if markov:
        out["reg_markov_lowvol_p"] = _markov_prob(logret)
    return out
