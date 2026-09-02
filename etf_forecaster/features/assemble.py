"""Assemble per-ticker feature panels. Every column carries its block prefix
(configs/blocks.yaml); the ablation search toggles whole blocks by prefix."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from etf_forecaster.config import all_tickers, ticker_meta
from etf_forecaster.data import ibkr_vol, lake, yahoo
from etf_forecaster.data.calendar import build_calendar, load_calendar
from etf_forecaster.data.fred import load_fast_panel
from etf_forecaster.data.shiller import load_shiller
from etf_forecaster.features import cross as cross_mod
from etf_forecaster.features import macro as macro_mod
from etf_forecaster.features import price as price_mod
from etf_forecaster.features import regime as regime_mod
from etf_forecaster.features import seasonality as season_mod
from etf_forecaster.features import technical as tech_mod
from etf_forecaster.features import vol as vol_mod
from etf_forecaster.patterns import featurize as pat_featurize
from etf_forecaster.patterns import multi_tf

log = logging.getLogger(__name__)


@dataclass
class SharedData:
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    vol_closes: dict[str, pd.Series] = field(default_factory=dict)
    cross_closes: dict[str, pd.Series] = field(default_factory=dict)
    fred_fast: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    shiller: pd.DataFrame | None = None
    ms_ref: pd.DataFrame | None = None      # slow macro on the reference index
    earn_ref: pd.DataFrame | None = None
    ref_index: pd.DatetimeIndex | None = None


def load_shared(*, slow_macro: bool = True) -> SharedData:
    sh = SharedData()
    for t in all_tickers():
        b = yahoo.load_bars(t)
        if b is not None and len(b) > 100:
            sh.bars[t] = b
    from etf_forecaster.config import cross_series_tickers, vol_index_tickers
    for t in vol_index_tickers():
        df = yahoo.load_vol_index(t)
        if df is not None:
            sh.vol_closes[t] = df["close"]
    for t in cross_series_tickers():
        df = yahoo.load_cross(t)
        if df is not None:
            sh.cross_closes[t] = df["close"]
    sh.fred_fast = load_fast_panel()
    sh.shiller = load_shiller()
    sh.ref_index = sh.bars["SPY"].index if "SPY" in sh.bars else None

    sh.events = load_calendar()
    if sh.events is None and sh.ref_index is not None:
        try:
            sh.events = build_calendar(sh.ref_index)
        except Exception as exc:  # noqa: BLE001
            log.warning("calendar build failed: %s", exc)

    if slow_macro and sh.ref_index is not None:
        log.info("building point-in-time slow macro panel (one-off, cached in memory)")
        sh.ms_ref = macro_mod.slow_macro_features(sh.ref_index)
        real10 = sh.fred_fast["real_10y"] if (sh.fred_fast is not None
                                              and "real_10y" in sh.fred_fast) else None
        sh.earn_ref = macro_mod.earnings_features(sh.ref_index, shiller=sh.shiller,
                                                  real_10y=real10)
    return sh


def assemble_ticker(ticker: str, shared: SharedData, *, markov: bool = True,
                    garch: bool = True, write: bool = True) -> pd.DataFrame:
    bars = shared.bars[ticker]
    index = bars.index
    meta = ticker_meta(ticker)
    is_vol_target = meta.get("target_type") == "vol"
    parts: list[pd.DataFrame] = []

    # PRICE + TECH
    parts.append(price_mod.price_features(bars, log_level=is_vol_target))
    parts.append(tech_mod.technical_features(bars))

    # PATTERN_D1 / PATTERN_HTF
    try:
        sigs = multi_tf.detect_multi_tf(bars, ticker)
        pat = pat_featurize.featurize(bars, sigs)
        d1_cols = [c for c in pat.columns if c.startswith("d1_")]
        htf_cols = [c for c in pat.columns if not c.startswith("d1_")]
        parts.append(pat[d1_cols].rename(columns=lambda c: f"patd_{c}"))
        parts.append(pat[htf_cols].rename(columns=lambda c: f"path_{c}"))
    except Exception as exc:  # noqa: BLE001
        log.warning("pattern features failed for %s: %s", ticker, exc)

    # REGIME
    parts.append(regime_mod.regime_features(bars, markov=markov))

    # VOLOPT: market-wide complex + per-ticker
    parts.append(vol_mod.vix_complex_features(shared.vol_closes, index))
    iv = ibkr_vol.load_ibkr_vol(ticker, "iv") if meta.get("ibkr_iv") else None
    hv = ibkr_vol.load_ibkr_vol(ticker, "hv") if meta.get("ibkr_iv") else None
    vix = shared.vol_closes.get("^VIX")
    parts.append(vol_mod.ticker_vol_features(bars, iv=iv, hv=hv, vix=vix, garch=garch))

    # SKEW (chain-derived; short history, NaN before first snapshot)
    skew = lake.read_df(lake.skew_path(ticker))
    if skew is not None and len(skew):
        skew = skew.apply(pd.to_numeric, errors="coerce")
        parts.append(skew.reindex(index).rename(columns=lambda c: f"skw_{c}"))

    # MACRO_FAST / MACRO_SLOW / EARNINGS (shared panels, reindexed as-of)
    if shared.fred_fast is not None:
        parts.append(macro_mod.fast_macro_features(shared.fred_fast, index))
    if shared.ms_ref is not None and len(shared.ms_ref):
        parts.append(shared.ms_ref.reindex(index, method="ffill"))
    if shared.earn_ref is not None and len(shared.earn_ref):
        parts.append(shared.earn_ref.reindex(index, method="ffill"))

    # CROSS + SEASON
    parts.append(cross_mod.cross_features(ticker, shared.bars, shared.cross_closes))
    parts.append(season_mod.seasonality_features(index, shared.events))

    panel = pd.concat(parts, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel = panel.replace([float("inf"), float("-inf")], pd.NA).astype("float64", errors="ignore")
    if write:
        lake.write_df(panel, lake.features_path(ticker),
                      manifest_key=f"features/{lake.sym_key(ticker)}")
    return panel


def load_features(ticker: str) -> pd.DataFrame | None:
    return lake.read_df(lake.features_path(ticker))
