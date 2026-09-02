"""Turn one day's option-chain snapshot into a stable skew/surface feature row.

Pipeline per snapshot: clean quotes -> parity forward per expiry -> Black-76 IV per strike
(OTM side only) -> smile interpolated in delta space -> 25d risk reversal / butterfly / ATM
per expiry -> tenor interpolation (total variance) to 30/60/90d -> flow stats + GEX proxy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from etf_forecaster.options import pricing

log = logging.getLogger(__name__)

_MIN_QUOTES_PER_EXPIRY = 6


def _clean(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    for c in ("bid", "ask", "strike", "volume", "openInterest"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    ok = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
        & ((df["ask"] - df["bid"]) / df["mid"].clip(lower=1e-9) < 0.75)
    )
    return df[ok & df["mid"].notna()].copy()


def _expiry_smile(exp_df: pd.DataFrame, spot: float, T: float, r: float) -> dict | None:
    calls = exp_df[exp_df["right"] == "c"].set_index("strike")["mid"]
    puts = exp_df[exp_df["right"] == "p"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    F = pricing.parity_forward(both.values, calls.loc[both].values, puts.loc[both].values,
                               T, r, spot) if len(both) >= 2 else spot * np.exp(r * T)

    # OTM side only: puts below the forward, calls above
    otm = exp_df[((exp_df["right"] == "p") & (exp_df["strike"] <= F))
                 | ((exp_df["right"] == "c") & (exp_df["strike"] > F))].copy()
    if len(otm) < _MIN_QUOTES_PER_EXPIRY:
        return None
    iv = pricing.implied_vol(otm["mid"].values, otm["right"].values,
                             F, otm["strike"].values, T, r)
    otm["iv"] = iv
    otm = otm[np.isfinite(otm["iv"]) & (otm["iv"] > 0.01) & (otm["iv"] < 4.0)]
    if len(otm) < _MIN_QUOTES_PER_EXPIRY:
        return None
    otm["delta"] = pricing.black76_delta(otm["right"].values, F, otm["strike"].values,
                                         T, r, otm["iv"].values)

    # ATM: interpolate IV vs log-moneyness at 0
    otm["logm"] = np.log(otm["strike"] / F)
    smile = otm.sort_values("logm")
    atm_iv = float(np.interp(0.0, smile["logm"].values, smile["iv"].values))

    def _iv_at_delta(side: str, target: float) -> float | None:
        sub = smile[smile["right"] == side]
        if len(sub) < 3:
            return None
        d = sub["delta"].values
        v = sub["iv"].values
        order = np.argsort(d)
        d, v = d[order], v[order]
        if not (d.min() <= target <= d.max()):
            return None
        return float(np.interp(target, d, v))

    ivc25 = _iv_at_delta("c", 0.25)
    ivp25 = _iv_at_delta("p", -0.25)
    rr25 = (ivp25 - ivc25) if (ivc25 is not None and ivp25 is not None) else None
    bf25 = (0.5 * (ivp25 + ivc25) - atm_iv) if (ivc25 is not None and ivp25 is not None) else None
    return {"T": T, "F": F, "atm_iv": atm_iv, "rr25": rr25, "bf25": bf25, "n_quotes": len(otm)}


def _interp_tenor(rows: list[dict], key: str, days: float) -> float | None:
    pts = [(r["T"], r[key]) for r in rows if r.get(key) is not None]
    if len(pts) < 2:
        return pts[0][1] if pts else None
    pts.sort()
    T = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    t = days / 365.0
    if key == "atm_iv":  # interpolate in total variance
        tv = v**2 * T
        target_tv = float(np.interp(t, T, tv))
        return float(np.sqrt(max(target_tv, 1e-12) / t))
    return float(np.interp(t, T, v))


def surface_features(snap: pd.DataFrame, *, spot: float, r: float) -> dict:
    """One feature row from one day's snapshot. Keys are un-prefixed; caller adds skw_."""
    out: dict[str, float | None] = {}
    df = _clean(snap)
    asof = pd.Timestamp(snap["asof"].iloc[0])
    rows: list[dict] = []
    for expiry, exp_df in df.groupby("expiry"):
        T = max((pd.Timestamp(expiry) - asof).days, 1) / 365.0
        try:
            row = _expiry_smile(exp_df, spot, T, r)
        except Exception as exc:  # noqa: BLE001
            log.debug("smile failed for %s: %s", expiry, exc)
            row = None
        if row:
            rows.append(row)

    for days in (30, 60, 90):
        out[f"atm_iv_{days}d"] = _interp_tenor(rows, "atm_iv", days)
        out[f"rr25_{days}d"] = _interp_tenor(rows, "rr25", days)
        out[f"bf25_{days}d"] = _interp_tenor(rows, "bf25", days)
    a30, a90 = out.get("atm_iv_30d"), out.get("atm_iv_90d")
    out["term_slope_30_90"] = (a90 - a30) if (a30 is not None and a90 is not None) else None

    # Flow stats over all cleaned quotes
    vol_c = df.loc[df["right"] == "c", "volume"].sum()
    vol_p = df.loc[df["right"] == "p", "volume"].sum()
    oi_c = df.loc[df["right"] == "c", "openInterest"].sum()
    oi_p = df.loc[df["right"] == "p", "openInterest"].sum()
    out["pc_volume_ratio"] = float(vol_p / vol_c) if vol_c > 0 else None
    out["pc_oi_ratio"] = float(oi_p / oi_c) if oi_c > 0 else None

    # Dealer gamma exposure proxy (long calls / short puts convention)
    dfx = df.dropna(subset=["openInterest"])
    if len(dfx) and rows:
        T_arr = (pd.to_datetime(dfx["expiry"]) - asof).dt.days.clip(lower=1).values / 365.0
        ivs = pricing.implied_vol(dfx["mid"].values, dfx["right"].values,
                                  spot * np.exp(r * T_arr), dfx["strike"].values, T_arr, r)
        with np.errstate(all="ignore"):
            gamma = pricing.black76_gamma(spot * np.exp(r * T_arr), dfx["strike"].values,
                                          T_arr, r, np.nan_to_num(ivs, nan=0.3))
        sign = np.where(dfx["right"].values == "c", 1.0, -1.0)
        gex = sign * gamma * dfx["openInterest"].values * 100.0 * spot**2 * 0.01
        out["gex_total"] = float(np.nansum(gex) / 1e9)  # $bn per 1% move
        # flip level: strike where cumulative (sorted by strike) GEX crosses zero
        order = np.argsort(dfx["strike"].values)
        cum = np.cumsum(gex[order])
        strikes_sorted = dfx["strike"].values[order]
        crossing = np.where(np.diff(np.sign(cum)) != 0)[0]
        if len(crossing):
            flips = strikes_sorted[crossing + 1]
            flip = flips[np.argmin(np.abs(flips - spot))]  # crossing nearest spot
            out["gex_flip_dist"] = float((spot - flip) / spot)
        else:
            out["gex_flip_dist"] = None
    else:
        out["gex_total"] = None
        out["gex_flip_dist"] = None
    out["n_expiries"] = float(len(rows))
    return out


def build_skew_history(ticker: str, *, rate_series: pd.Series | None = None) -> pd.DataFrame:
    """(Re)build the derived skew feature history from all stored snapshots."""
    from etf_forecaster.data import chains, lake, yahoo

    bars = yahoo.load_bars(ticker)
    rows: dict[pd.Timestamp, dict] = {}
    for asof in chains.list_snapshots(ticker):
        snap = chains.load_snapshot(ticker, asof)
        if snap is None or bars is None:
            continue
        spot_ser = bars["close_raw"].reindex(bars.index[bars.index <= asof])
        if spot_ser.empty:
            continue
        spot = float(spot_ser.iloc[-1])
        r = 0.04
        if rate_series is not None and len(rate_series):
            r_ser = rate_series[rate_series.index <= asof]
            if len(r_ser):
                r = float(r_ser.iloc[-1]) / 100.0
        rows[asof] = surface_features(snap, spot=spot, r=r)
    hist = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    hist.index.name = "date"
    if len(hist):
        lake.write_df(hist, lake.skew_path(ticker), manifest_key=f"skew/{lake.sym_key(ticker)}")
    return hist
