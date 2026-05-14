"""
Technical features — aligned with **Currencies** ``technical_indicators.compute_technical_features``.

Daily OHLCV → pandas-ta metrics; weekly bars (W-FRI) use the same functions with ``*_wk_`` prefixes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    lc = {str(c).strip().lower(): c for c in df.columns}

    def col(*names: str) -> pd.Series:
        for n in names:
            if n in lc:
                return df[lc[n]].astype(float)
        raise ValueError(f"Need column one of {names}, have {list(df.columns)}")

    out = pd.DataFrame(
        {
            "Open": col("open"),
            "High": col("high"),
            "Low": col("low"),
            "Close": col("close"),
        },
        index=df.index,
    )
    if "volume" in lc:
        out["Volume"] = df[lc["volume"]].astype(float)
    elif "vol" in lc:
        out["Volume"] = df[lc["vol"]].astype(float)
    else:
        out["Volume"] = 0.0
    return out


def ohlc_from_close(close: pd.Series, *, name: str = "asset") -> pd.DataFrame:
    """When only Close exists (some macro downloads), approximate OHLC = Close for indicators."""
    c = pd.to_numeric(close, errors="coerce").dropna()
    if c.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0.0},
        index=c.index,
    )


def _fib_nearest_distance_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 55) -> pd.Series:
    hh = high.rolling(window, min_periods=max(2, window // 2)).max()
    ll = low.rolling(window, min_periods=max(2, window // 2)).min()
    rng = (hh - ll).replace(0, np.nan)
    levels = np.array([0.236, 0.382, 0.500, 0.618])
    fib_stack = np.stack([hh.values - rng.values * f for f in levels], axis=1)
    c = close.values.reshape(-1, 1)
    dmat = np.abs(fib_stack - c)
    dmat[~np.isfinite(dmat)] = np.inf
    ix = np.argmin(dmat, axis=1)
    nearest = fib_stack[np.arange(len(close)), ix]
    return pd.Series((close.values - nearest) / (rng.values + 1e-12), index=close.index)


def _pivot_distance_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    pivot = (prev_h + prev_l + prev_c) / 3.0
    return (close - pivot) / close.replace(0, np.nan)


def compute_technical_features(ohlcv: pd.DataFrame, *, prefix: str = "tech_") -> pd.DataFrame:
    try:
        import pandas_ta as ta
    except ImportError as e:
        raise ImportError("Install pandas-ta: pip install pandas-ta") from e

    d = _ensure_ohlcv(ohlcv)
    x = d.rename(columns=str.lower).copy()

    out = pd.DataFrame(index=x.index)
    c, h, l, o, v = x["close"], x["high"], x["low"], x["open"], x["volume"]

    rsi = ta.rsi(c, length=14)
    out[f"{prefix}rsi14"] = rsi

    macd = ta.macd(c)
    if macd is not None and not macd.empty:
        out[f"{prefix}macd"] = macd.iloc[:, 0]
        out[f"{prefix}macd_signal"] = macd.iloc[:, 1]
        out[f"{prefix}macd_hist"] = macd.iloc[:, 2]

    bb = ta.bbands(c, length=20)
    if bb is not None and not bb.empty:
        lowbb = bb.iloc[:, 0]
        hibb = bb.iloc[:, 2]
        out[f"{prefix}bb_pctb"] = (c - lowbb) / (hibb - lowbb).replace(0, np.nan)

    for n in (5, 20, 50, 100, 200):
        sma = ta.sma(c, length=n)
        out[f"{prefix}dist_sma{n}"] = (c - sma) / c.replace(0, np.nan)

    for n in (12, 26, 50):
        ema = ta.ema(c, length=n)
        out[f"{prefix}dist_ema{n}"] = (c - ema) / c.replace(0, np.nan)

    ich = ta.ichimoku(h, l, c)
    if ich is not None and isinstance(ich, tuple) and len(ich) > 0 and ich[0] is not None:
        ichi_df = ich[0]
        if "ISA_9" in ichi_df.columns:
            out[f"{prefix}ichi_conv_dist"] = (c - ichi_df["ISA_9"]) / c.replace(0, np.nan)
        if "ISB_26" in ichi_df.columns:
            out[f"{prefix}ichi_base_dist"] = (c - ichi_df["ISB_26"]) / c.replace(0, np.nan)

    psar = ta.psar(h, l, c)
    if psar is not None and not psar.empty:
        pcols = [xx for xx in psar.columns if "PSAR" in str(xx).upper()]
        if pcols:
            out[f"{prefix}psar_dist"] = (c - psar[pcols[0]]) / c.replace(0, np.nan)

    adx = ta.adx(h, l, c, length=14)
    if adx is not None and not adx.empty:
        acol = [z for z in adx.columns if str(z).upper().startswith("ADX")]
        if acol:
            out[f"{prefix}adx"] = adx[acol[0]]

    st = ta.supertrend(h, l, c, length=7, multiplier=3)
    if st is not None and not st.empty:
        scol = [z for z in st.columns if str(z).upper().startswith("SUPERT")]
        if scol:
            out[f"{prefix}supert_dist"] = (c - st[scol[0]]) / c.replace(0, np.nan)

    stoch = ta.stoch(h, l, c)
    if stoch is not None and not stoch.empty:
        kcol = [z for z in stoch.columns if "STOCHk" in str(z).upper() or str(z).endswith("k")]
        if kcol:
            out[f"{prefix}stoch_k"] = stoch[kcol[0]]

    out[f"{prefix}cci20"] = ta.cci(h, l, c, length=20)
    out[f"{prefix}willr14"] = ta.willr(h, l, c, length=14)
    out[f"{prefix}mfi14"] = ta.mfi(h, l, c, volume=v, length=14)
    out[f"{prefix}atr14"] = ta.atr(h, l, c, length=14)
    out[f"{prefix}atr14_pct"] = out[f"{prefix}atr14"] / c.replace(0, np.nan)

    kc = ta.kc(h, l, c, length=20)
    if kc is not None and not kc.empty:
        kcl = kc.iloc[:, 0]
        kch = kc.iloc[:, 2]
        out[f"{prefix}kc_pctb"] = (c - kcl) / (kch - kcl).replace(0, np.nan)

    out[f"{prefix}pivot_pct"] = _pivot_distance_pct(h, l, c)
    out[f"{prefix}fib_dist_pct"] = _fib_nearest_distance_pct(h, l, c, window=55)

    try:
        out[f"{prefix}vwap_dist"] = (c - ta.vwap(h, l, c, volume=v)) / c.replace(0, np.nan)
    except Exception:
        tp = (h + l + c) / 3.0
        cs = (tp * v.replace(0, np.nan)).cumsum()
        vs = v.replace(0, np.nan).cumsum()
        vwap = cs / vs.replace(0, np.nan)
        out[f"{prefix}vwap_dist"] = (c - vwap) / c.replace(0, np.nan)

    return out.astype(float)


def weekly_technicals_ffill_to_daily(
    daily_ohlc: pd.DataFrame,
    *,
    prefix: str,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Resample to W-FRI, compute features with ``prefix`` (should include ``_wk_``), forward-fill to daily."""
    if daily_ohlc.empty or len(daily_ohlc) < 5:
        return pd.DataFrame(index=daily_index)
    d = daily_ohlc.copy()
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index, utc=True)).tz_localize(None).normalize()
    w = d.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        },
    )
    w = w.dropna(how="all")
    if w.empty:
        return pd.DataFrame(index=daily_index)
    try:
        tech_w = compute_technical_features(w, prefix=prefix)
        tech_w = tech_w.reindex(daily_index).ffill()
        return tech_w
    except Exception as e:
        logger.warning("Weekly technicals failed prefix=%s: %s", prefix, e)
        return pd.DataFrame(index=daily_index)
