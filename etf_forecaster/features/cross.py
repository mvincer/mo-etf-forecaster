"""CROSS block (x_): session-aligned cross-asset lead-lag, relative strength,
rolling beta, and cross-sectional momentum ranks.

Session alignment: a source ticker whose session closes AFTER the target's close on the
same calendar day must be lagged one day (its day-t close is not known at the target's
day-t close). Korean names close ~02:00 ET so they are same-day usable for US targets;
the reverse direction gets lagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_forecaster.config import all_tickers, close_order, sym_key, ticker_group, universe_cfg


def cross_features(target: str, bars_by_ticker: dict[str, pd.DataFrame],
                   cross_closes: dict[str, pd.Series]) -> pd.DataFrame:
    tgt_bars = bars_by_ticker[target]
    index = tgt_bars.index
    tgt_order = close_order(target)
    out = pd.DataFrame(index=index)

    logret_cache: dict[str, pd.Series] = {}

    def _aligned_logret(src: str) -> pd.Series | None:
        if src not in logret_cache:
            b = bars_by_ticker.get(src)
            if b is None or b.empty:
                logret_cache[src] = None  # type: ignore[assignment]
            else:
                lr = np.log(b["close"] / b["close"].shift())
                lag = 1 if close_order(src) > tgt_order else 0
                logret_cache[src] = lr.shift(lag).reindex(index, method="ffill") \
                    if lag else lr.reindex(index).ffill(limit=5)
        return logret_cache[src]

    # lead-lag returns of every other universe member (1d and 5d)
    for src in all_tickers():
        if src == target:
            continue
        lr = _aligned_logret(src)
        if lr is None:
            continue
        key = sym_key(src).lower()
        out[f"x_{key}_r1"] = lr
        out[f"x_{key}_r5"] = lr.rolling(5).sum()

    # relative strength vs SPY + rolling beta
    tgt_lr = np.log(tgt_bars["close"] / tgt_bars["close"].shift())
    spy_lr = _aligned_logret("SPY") if target != "SPY" else tgt_lr
    if spy_lr is not None:
        out["x_rel_spy_21"] = tgt_lr.rolling(21).sum() - spy_lr.rolling(21).sum()
        out["x_rel_spy_63"] = tgt_lr.rolling(63).sum() - spy_lr.rolling(63).sum()
        cov = tgt_lr.rolling(63).cov(spy_lr)
        var = spy_lr.rolling(63).var()
        out["x_beta_spy"] = cov / var.replace(0, np.nan)

    # cross-sectional 21d momentum rank within the target's group
    group = ticker_group(target)
    members = universe_cfg()["groups"].get(group, [])
    if len(members) > 2:
        moms = {}
        for m in members:
            lr = tgt_lr if m == target else _aligned_logret(m)
            if lr is not None:
                moms[m] = lr.rolling(21).sum()
        if len(moms) > 2:
            mom_df = pd.DataFrame(moms)
            out["x_group_mom_rank"] = mom_df.rank(axis=1, pct=True)[target].reindex(index)

    # derived risk-appetite ratios from the cross-series closes
    def _ratio_z(a: str, b: str, name: str) -> None:
        sa, sb = cross_closes.get(a), cross_closes.get(b)
        if sa is None or sb is None:
            return
        ratio = np.log(sa.reindex(index).ffill(limit=5) / sb.reindex(index).ffill(limit=5))
        out[f"x_{name}_z"] = (ratio - ratio.rolling(252, min_periods=60).mean()) / \
            ratio.rolling(252, min_periods=60).std().replace(0, np.nan)
        out[f"x_{name}_chg21"] = ratio.diff(21)

    _ratio_z("HYG", "LQD", "hyg_lqd")
    # breadth and risk-appetite ratios built from universe bars
    spy_close = bars_by_ticker.get("SPY", pd.DataFrame()).get("close")
    rsp = cross_closes.get("RSP")
    if rsp is not None and spy_close is not None:
        ratio = np.log(rsp.reindex(index).ffill(limit=5) / spy_close.reindex(index).ffill(limit=5))
        out["x_rsp_spy_z"] = (ratio - ratio.rolling(252, min_periods=60).mean()) / \
            ratio.rolling(252, min_periods=60).std().replace(0, np.nan)
        out["x_rsp_spy_chg21"] = ratio.diff(21)
    xly = bars_by_ticker.get("XLY", pd.DataFrame()).get("close")
    xlp = bars_by_ticker.get("XLP", pd.DataFrame()).get("close")
    if xly is not None and xlp is not None:
        ratio = np.log(xly.reindex(index).ffill(limit=5) / xlp.reindex(index).ffill(limit=5))
        out["x_xly_xlp_z"] = (ratio - ratio.rolling(252, min_periods=60).mean()) / \
            ratio.rolling(252, min_periods=60).std().replace(0, np.nan)
        out["x_xly_xlp_chg21"] = ratio.diff(21)

    # cross-asset levels: 21d change of rates, dollar, commodities
    for src, name in (("^TNX", "tnx"), ("^TYX", "tyx"), ("DX-Y.NYB", "dxy"),
                      ("CL=F", "wti"), ("GC=F", "gold"), ("HG=F", "copper"), ("TLT", "tlt")):
        s = cross_closes.get(src)
        if s is None:
            continue
        s = s.reindex(index).ffill(limit=5)
        out[f"x_{name}_chg21"] = s.pct_change(21)
        out[f"x_{name}_chg5"] = s.pct_change(5)
    return out
