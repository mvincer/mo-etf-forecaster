"""TECH block (tech_): the top-20 technical indicator families via pandas-ta-classic,
plus Ulcer Index and Choppiness. Every indicator is wrapped so a single failure
cannot take down the panel build."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import pandas_ta_classic as pta
except ImportError:  # pragma: no cover
    import pandas_ta as pta  # type: ignore[no-redef]

log = logging.getLogger(__name__)


def technical_features(bars: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    v = bars.get("volume", pd.Series(0.0, index=bars.index))
    out = pd.DataFrame(index=bars.index)

    def _try(name: str, fn) -> None:
        try:
            res = fn()
            if res is None:
                return
            if isinstance(res, pd.Series):
                out[f"tech_{name}"] = res
            else:
                for col, series in res.items():
                    out[f"tech_{name}_{col}"] = series
        except Exception as exc:  # noqa: BLE001
            log.debug("indicator %s failed: %s", name, exc)

    _try("rsi14", lambda: pta.rsi(c, length=14))
    _try("macd", lambda: _rename(pta.macd(c), {0: "line", 1: "hist", 2: "signal"}))
    _try("stoch", lambda: _rename(pta.stoch(h, l, c), {0: "k", 1: "d"}))
    _try("cci20", lambda: pta.cci(h, l, c, length=20))
    _try("willr14", lambda: pta.willr(h, l, c, length=14))
    _try("roc10", lambda: pta.roc(c, length=10))
    _try("mfi14", lambda: pta.mfi(h, l, c, v, length=14) if v.sum() > 0 else None)
    _try("adx", lambda: _rename(pta.adx(h, l, c), {0: "adx", 1: "dmp", 2: "dmn"}))
    _try("aroon", lambda: _rename(pta.aroon(h, l, length=25), {0: "dn", 1: "up", 2: "osc"}))

    def _atr_pct():
        atr = pta.atr(h, l, c, length=14)
        return atr / c * 100.0
    _try("atr14_pct", _atr_pct)

    def _bbands():
        bb = pta.bbands(c, length=20, std=2)
        if bb is None:
            return None
        cols = list(bb.columns)
        lower = bb[cols[0]]
        upper = bb[cols[2]]
        width = (upper - lower)
        pctb = (c - lower) / width.replace(0, np.nan)
        return pd.DataFrame({"pctb": pctb, "bw": width / c})
    _try("bb", _bbands)

    def _kc():
        kc = pta.kc(h, l, c, length=20)
        if kc is None:
            return None
        cols = list(kc.columns)
        lower, upper = kc[cols[0]], kc[cols[2]]
        return (c - lower) / (upper - lower).replace(0, np.nan)
    _try("kc_pctb", _kc)

    def _donchian(w: int):
        hi = h.rolling(w).max()
        lo = l.rolling(w).min()
        return (c - lo) / (hi - lo).replace(0, np.nan)
    _try("donch20", lambda: _donchian(20))
    _try("donch55", lambda: _donchian(55))

    for w in (20, 50, 200):
        _try(f"sma{w}_dist", lambda w=w: c / c.rolling(w).mean() - 1)
    for w in (12, 26, 50):
        _try(f"ema{w}_dist", lambda w=w: c / c.ewm(span=w, adjust=False).mean() - 1)

    def _psar():
        ps = pta.psar(h, l, c)
        if ps is None:
            return None
        cols = [x for x in ps.columns if x.startswith("PSARl") or x.startswith("PSARs")]
        sar = ps[cols].bfill(axis=1).iloc[:, 0] if cols else None
        return (c - sar) / c if sar is not None else None
    _try("psar_dist", _psar)

    def _supertrend():
        st = pta.supertrend(h, l, c, length=10, multiplier=3.0)
        if st is None:
            return None
        line = st[[x for x in st.columns if x.startswith("SUPERT_")][0]]
        return (c - line) / c
    _try("supertrend_dist", _supertrend)

    def _ichimoku():
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return pd.DataFrame({
            "tenkan_dist": c / tenkan - 1,
            "kijun_dist": c / kijun - 1,
            "above_cloud": (c > cloud_top).astype(float) - (c < cloud_bot).astype(float),
        })
    _try("ichi", _ichimoku)

    def _obv_slope():
        if v.sum() <= 0:
            return None
        obv = pta.obv(c, v)
        return obv.diff(20) / obv.rolling(63).std().replace(0, np.nan)
    _try("obv_slope", _obv_slope)

    _try("cmf20", lambda: pta.cmf(h, l, c, v, length=20) if v.sum() > 0 else None)

    def _vwap_dist():
        if v.sum() <= 0:
            return None
        tp = (h + l + c) / 3
        vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        return c / vwap - 1
    _try("vwap20_dist", _vwap_dist)

    def _ulcer():
        roll_max = c.rolling(14).max()
        dd = (c / roll_max - 1) * 100
        return np.sqrt((dd**2).rolling(14).mean())
    _try("ulcer14", _ulcer)

    def _chop():
        atr1 = pta.atr(h, l, c, length=1)
        rng = h.rolling(14).max() - l.rolling(14).min()
        return 100 * np.log10(atr1.rolling(14).sum() / rng.replace(0, np.nan)) / np.log10(14)
    _try("chop14", _chop)

    _try("tsi", lambda: _first_col(pta.tsi(c)))
    return out


def _rename(df: pd.DataFrame | None, names: dict[int, str]) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df.columns = [names.get(i, str(col)) for i, col in enumerate(df.columns)]
    return df[[cname for i, cname in enumerate(df.columns) if i in names or cname in names.values()]]


def _first_col(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]
