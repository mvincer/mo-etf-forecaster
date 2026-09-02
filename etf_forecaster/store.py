"""The output contract Mo_Dash reads: five artifacts in data/outputs/.

- forecasts_latest.parquet    one row per (ticker, horizon) with p_up, quantiles, meta
- predictions_history.parquet every OOS prediction ever made + realized outcome
- scorecard.parquet           rolling metrics per (ticker, horizon)
- ablation_results.parquet    every ablation/model-search cell with its trial count
- champions.json              selected config per (ticker, horizon)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etf_forecaster.config import OUTPUTS_DIR
from etf_forecaster.validation import metrics

FORECASTS = OUTPUTS_DIR / "forecasts_latest.parquet"
HISTORY = OUTPUTS_DIR / "predictions_history.parquet"
SCORECARD = OUTPUTS_DIR / "scorecard.parquet"
ABLATION = OUTPUTS_DIR / "ablation_results.parquet"
CHAMPIONS = OUTPUTS_DIR / "champions.json"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def read(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ------------------------------------------------------------------ history

def append_history(preds: pd.DataFrame) -> None:
    """Append predictions (date, ticker, horizon, p_up, expected_return, q05..q95,
    model, blocks). Existing (date, ticker, horizon, model) rows are replaced."""
    old = read(HISTORY)
    if old is not None and len(old):
        key = ["date", "ticker", "horizon", "model"]
        merged = pd.concat([old, preds], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
    else:
        merged = preds
    _write(merged.sort_values(["ticker", "horizon", "date"]), HISTORY)


def backfill_outcomes(bars_by_ticker: dict[str, pd.DataFrame]) -> int:
    """Fill realized y/ret for matured predictions. Returns rows updated."""
    hist = read(HISTORY)
    if hist is None or not len(hist):
        return 0
    hist["date"] = pd.to_datetime(hist["date"])
    need = hist["y"].isna() if "y" in hist else pd.Series(True, index=hist.index)
    if "y" not in hist:
        hist["y"] = np.nan
        hist["ret_realized"] = np.nan
    updated = 0
    for (ticker, horizon), grp in hist[need].groupby(["ticker", "horizon"]):
        bars = bars_by_ticker.get(ticker)
        if bars is None:
            continue
        logc = np.log(bars["close"])
        idx = bars.index
        for i, row in grp.iterrows():
            pos = idx.searchsorted(row["date"])
            if pos >= len(idx) or idx[pos] != row["date"]:
                continue
            tgt = pos + int(horizon)
            if tgt >= len(idx):
                continue  # not matured yet
            fwd = float(logc.iloc[tgt] - logc.iloc[pos])
            hist.loc[i, "ret_realized"] = fwd
            hist.loc[i, "y"] = float(fwd > 0)
            updated += 1
    if updated:
        _write(hist, HISTORY)
    return updated


# ------------------------------------------------------------------ scorecard

def rebuild_scorecard(*, windows: tuple[int, ...] = (63, 252, 100000)) -> pd.DataFrame:
    hist = read(HISTORY)
    if hist is None or not len(hist):
        return pd.DataFrame()
    hist = hist.dropna(subset=["y", "p_up"])
    rows: list[dict] = []
    for (ticker, horizon, model), g in hist.groupby(["ticker", "horizon", "model"]):
        g = g.sort_values("date")
        for w in windows:
            tail = g.tail(w)
            if len(tail) < 30:
                continue
            y, p = tail["y"].values, tail["p_up"].values
            hits = int(((p > 0.5) == (y > 0.5)).sum())
            rows.append({
                "ticker": ticker, "horizon": int(horizon), "model": model,
                "window": ("all" if w > 99000 else str(w)),
                "n": len(tail),
                "hit_rate": metrics.hit_rate(y, p),
                "log_loss": metrics.log_loss(y, p),
                "brier": metrics.brier(y, p),
                "auc": metrics.auc(y, p),
                "pnl": metrics.signal_pnl(tail["ret_realized"].values, p),
                "p_value": metrics.direction_pvalue(hits, len(tail)),
                "last_date": tail["date"].max(),
            })
    sc = pd.DataFrame(rows)
    if len(sc):
        for w in sc["window"].unique():
            m = sc["window"] == w
            sc.loc[m, "fdr_pass"] = metrics.benjamini_hochberg(sc.loc[m, "p_value"])
        _write(sc, SCORECARD)
    return sc


# ------------------------------------------------------------------ misc artifacts

def write_forecasts(df: pd.DataFrame) -> None:
    _write(df, FORECASTS)


def append_ablation(records: list[dict]) -> None:
    new = pd.DataFrame(records)
    old = read(ABLATION)
    merged = pd.concat([old, new], ignore_index=True) if old is not None else new
    # resumed runs re-append cached cells; keep the latest evaluation of each cell
    key = [c for c in ("group", "horizon", "stage", "blocks", "model", "params", "window")
           if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    _write(merged, ABLATION)


def write_champions(champions: dict) -> None:
    CHAMPIONS.parent.mkdir(parents=True, exist_ok=True)
    CHAMPIONS.write_text(json.dumps(champions, indent=2, default=str), encoding="utf-8")


def read_champions() -> dict:
    if CHAMPIONS.is_file():
        return json.loads(CHAMPIONS.read_text(encoding="utf-8"))
    return {}
