"""Black-Scholes / Black-76 pricing and implied vol.

Primary engine is pyvolr (Rust core, vectorized, Jackel 'Let's Be Rational'); a pure-numpy
Newton-with-bisection solver is the fallback so the pipeline never hard-depends on it.
ETF options are American, but IV solved on the parity-implied forward (Black-76) is the
standard, accurate treatment for the near-ATM short-dated strikes used here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

try:
    from pyvolr import black76 as _b76
    _HAVE_PYVOLR = True
except ImportError:  # pragma: no cover
    _b76 = None
    _HAVE_PYVOLR = False


def black76_price(flag: np.ndarray | str, F, K, T, r, sigma) -> np.ndarray:
    """Discounted Black-76 premium."""
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    is_call = _is_call(flag, K)
    srt = np.maximum(sigma * np.sqrt(T), 1e-12)
    d1 = (np.log(F / K) + 0.5 * srt**2) / srt
    d2 = d1 - srt
    disc = np.exp(-r * T)
    call = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def black76_delta(flag, F, K, T, r, sigma) -> np.ndarray:
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    is_call = _is_call(flag, K)
    srt = np.maximum(sigma * np.sqrt(T), 1e-12)
    d1 = (np.log(F / K) + 0.5 * srt**2) / srt
    disc = np.exp(-r * T)
    return np.where(is_call, disc * norm.cdf(d1), -disc * norm.cdf(-d1))


def black76_gamma(F, K, T, r, sigma) -> np.ndarray:
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    srt = np.maximum(sigma * np.sqrt(T), 1e-12)
    d1 = (np.log(F / K) + 0.5 * srt**2) / srt
    return np.exp(-r * T) * norm.pdf(d1) / (F * srt)


def _is_call(flag, like) -> np.ndarray:
    if isinstance(flag, str):
        return np.full(np.shape(like), flag.lower().startswith("c"))
    arr = np.asarray(flag)
    return np.char.lower(arr.astype(str)).astype("U1") == "c"


def implied_vol(price, flag, F, K, T, r) -> np.ndarray:
    """Vectorized Black-76 implied vol; NaN where unsolvable."""
    price, F, K, T, r = (np.asarray(x, dtype=float) for x in (price, F, K, T, r))
    if _HAVE_PYVOLR:
        try:
            with np.errstate(all="ignore"):
                iv = np.asarray(_b76.implied_vol(price, flag, F, K, T, r, on_error="nan"))
            return iv
        except Exception:  # noqa: BLE001 - fall through to numpy solver
            pass
    return _iv_newton(price, flag, F, K, T, r)


def _iv_newton(price, flag, F, K, T, r, *, tol: float = 1e-7, max_iter: int = 60) -> np.ndarray:
    is_call = _is_call(flag, K)
    disc = np.exp(-r * T)
    intrinsic = disc * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = disc * np.where(is_call, F, K)
    ok = (price > intrinsic + 1e-10) & (price < upper) & (T > 0)

    sigma = np.full_like(price, 0.3)
    lo = np.full_like(price, 1e-4)
    hi = np.full_like(price, 5.0)
    for _ in range(max_iter):
        with np.errstate(all="ignore"):
            model = black76_price(flag, F, K, T, r, sigma)
            srt = np.maximum(sigma * np.sqrt(T), 1e-12)
            d1 = (np.log(F / K) + 0.5 * srt**2) / srt
            vega = disc * F * norm.pdf(d1) * np.sqrt(T)
        diff = model - price
        hi = np.where(diff > 0, np.minimum(hi, sigma), hi)
        lo = np.where(diff < 0, np.maximum(lo, sigma), lo)
        step = np.where(vega > 1e-10, diff / np.maximum(vega, 1e-10), 0.0)
        newton = sigma - step
        inside = (newton > lo) & (newton < hi)
        sigma = np.where(inside, newton, 0.5 * (lo + hi))
        if np.nanmax(np.where(ok, np.abs(diff), 0.0)) < tol:
            break
    return np.where(ok, sigma, np.nan)


def parity_forward(strikes, call_mid, put_mid, T, r, spot) -> float:
    """Forward from put-call parity, medianed over strikes closest to spot."""
    strikes = np.asarray(strikes, dtype=float)
    call_mid = np.asarray(call_mid, dtype=float)
    put_mid = np.asarray(put_mid, dtype=float)
    valid = np.isfinite(call_mid) & np.isfinite(put_mid) & (call_mid > 0) & (put_mid > 0)
    if valid.sum() < 2:
        return float(spot * np.exp(r * T))
    strikes, call_mid, put_mid = strikes[valid], call_mid[valid], put_mid[valid]
    order = np.argsort(np.abs(strikes - spot))[: min(7, len(strikes))]
    F = strikes[order] + np.exp(r * T) * (call_mid[order] - put_mid[order])
    F = F[(F > 0.5 * spot) & (F < 2.0 * spot)]
    if len(F) == 0:
        return float(spot * np.exp(r * T))
    return float(np.median(F))
