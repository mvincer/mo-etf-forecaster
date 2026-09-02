"""VOLOPT block (vol_): CBOE vol complex with term structure, per-ticker IBKR implied
vol, variance risk premium, HAR-RV components, EWMA and (block-refit, leak-free) GARCH."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _z(s: pd.Series, window: int = 252) -> pd.Series:
    return (s - s.rolling(window, min_periods=60).mean()) / \
        s.rolling(window, min_periods=60).std().replace(0, np.nan)


def vix_complex_features(vol_closes: dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Market-wide vol features shared by every ticker (computed once)."""
    def _get(name: str) -> pd.Series | None:
        s = vol_closes.get(name)
        return s.reindex(index).ffill(limit=5) if s is not None else None

    out = pd.DataFrame(index=index)
    vix = _get("^VIX")
    if vix is None:
        return out
    out["vol_vix"] = vix
    out["vol_vix_log"] = np.log(vix)
    out["vol_vix_z"] = _z(vix)
    out["vol_vix_roc5"] = vix.pct_change(5)
    out["vol_vix_roc21"] = vix.pct_change(21)

    vix9d, vix3m, vix6m = _get("^VIX9D"), _get("^VIX3M"), _get("^VIX6M")
    if vix9d is not None:
        out["vol_ts_9d_1m"] = vix9d / vix
    if vix3m is not None:
        out["vol_ts_1m_3m"] = vix / vix3m
        out["vol_contango"] = vix3m - vix
    if vix3m is not None and vix6m is not None:
        out["vol_ts_3m_6m"] = vix3m / vix6m

    vvix = _get("^VVIX")
    if vvix is not None:
        out["vol_vvix"] = vvix
        out["vol_vvix_vix"] = vvix / vix
        out["vol_vvix_z"] = _z(vvix)

    skew = _get("^SKEW")
    if skew is not None:
        out["vol_skew"] = skew
        out["vol_skew_z"] = _z(skew)
        out["vol_skew_chg21"] = skew.diff(21)

    for name, col in (("^VXN", "vxn"), ("^VXD", "vxd"), ("^OVX", "ovx"), ("^GVZ", "gvz")):
        s = _get(name)
        if s is not None:
            out[f"vol_{col}_z"] = _z(s)
    if (vxn := _get("^VXN")) is not None:
        out["vol_vxn_vix"] = vxn / vix
    return out


def _garch_sigma(logret: pd.Series, *, refit_every: int = 63, min_train: int = 750) -> pd.Series:
    """One-step-ahead GARCH(1,1) conditional vol, parameters fitted on strictly past data."""
    try:
        from arch import arch_model
    except ImportError:  # pragma: no cover
        return pd.Series(np.nan, index=logret.index)
    r = (logret.dropna() * 100).clip(-20, 20)
    out = pd.Series(np.nan, index=r.index)
    for s in range(min_train, len(r), refit_every):
        try:
            fit = arch_model(r.iloc[:s], vol="GARCH", p=1, q=1, rescale=False) \
                .fit(disp="off", show_warning=False)
            end = min(s + refit_every, len(r))
            fixed = arch_model(r.iloc[:end], vol="GARCH", p=1, q=1, rescale=False) \
                .fix(fit.params.values)
            sigma = pd.Series(np.sqrt(fixed.conditional_volatility**2), index=r.index[:end]) \
                if not isinstance(fixed.conditional_volatility, pd.Series) \
                else fixed.conditional_volatility
            out.iloc[s:end] = np.asarray(sigma)[s:end]
        except Exception as exc:  # noqa: BLE001
            log.debug("garch chunk failed: %s", exc)
    return (out / 100 * np.sqrt(252)).reindex(logret.index)


def ticker_vol_features(bars: pd.DataFrame, *, iv: pd.DataFrame | None = None,
                        hv: pd.DataFrame | None = None, vix: pd.Series | None = None,
                        garch: bool = True) -> pd.DataFrame:
    """Per-ticker realized/implied vol features (vol_ prefix)."""
    out = pd.DataFrame(index=bars.index)
    logret = np.log(bars["close"] / bars["close"].shift())

    rv5 = logret.rolling(5).std() * np.sqrt(252)
    rv20 = logret.rolling(20).std() * np.sqrt(252)
    rv60 = logret.rolling(60).std() * np.sqrt(252)
    out["vol_rv5"] = rv5
    out["vol_rv20"] = rv20
    out["vol_rv60"] = rv60
    # HAR components (daily proxy = |r|)
    out["vol_har_d"] = logret.abs() * np.sqrt(252)
    out["vol_har_w"] = rv5
    out["vol_har_m"] = rv20
    out["vol_rv_ratio_5_20"] = rv5 / rv20.replace(0, np.nan)

    lam = 0.94
    ewma_var = (logret**2).ewm(alpha=1 - lam, adjust=False).mean()
    out["vol_ewma94"] = np.sqrt(ewma_var * 252)

    if garch:
        out["vol_garch"] = _garch_sigma(logret)

    if iv is not None and not iv.empty:
        iv_s = iv["value"].reindex(bars.index).ffill(limit=5)
        out["vol_iv"] = iv_s
        out["vol_iv_rank"] = iv_s.rolling(252, min_periods=126).rank(pct=True)
        out["vol_vrp"] = iv_s**2 - rv20**2
        out["vol_iv_chg5"] = iv_s.diff(5)
    if hv is not None and not hv.empty:
        hv_s = hv["value"].reindex(bars.index).ffill(limit=5)
        out["vol_ibkr_hv"] = hv_s
        if "vol_iv" in out:
            out["vol_iv_hv_spread"] = out["vol_iv"] - hv_s
    if vix is not None and "vol_iv" not in out:
        # no per-ticker IV: index-level VRP as a fallback
        vix_a = vix.reindex(bars.index).ffill(limit=5) / 100.0
        out["vol_vrp"] = vix_a**2 - rv20**2
    return out
