"""Walk-forward execution engine.

Works on a Dataset (single ticker or pooled group panel). Folds are date-based so purging
and embargo remain correct when several tickers share a date. All training/calibration
data for a fold ends `horizon` bars before the first forecast date of that fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_forecaster.models.registry import make_classifier
from etf_forecaster.models.sequence import SequenceClassifier
from etf_forecaster.validation import metrics

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    X: np.ndarray                 # float32 (n, k)
    y: np.ndarray                 # float (n,) direction labels for one horizon
    ret: np.ndarray               # float (n,) forward log return (same horizon)
    dates: np.ndarray             # datetime64[ns] (n,) sorted
    tickers: np.ndarray           # str (n,)
    feature_names: list[str]

    def subset_columns(self, cols: list[str]) -> "Dataset":
        idx = [self.feature_names.index(c) for c in cols]
        return Dataset(self.X[:, idx], self.y, self.ret, self.dates, self.tickers, list(cols))


def build_dataset(features: pd.DataFrame, labels: pd.DataFrame, horizon: int,
                  ticker: str) -> Dataset | None:
    y = labels[f"y_dir_{horizon}"]
    ret = labels[f"y_ret_{horizon}"]
    warm = features.filter(regex=r"^px_ret_252$")
    start_mask = warm.iloc[:, 0].notna() if warm.shape[1] else pd.Series(True, index=features.index)
    mask = start_mask & y.notna()
    if mask.sum() < 300:
        return None
    f = features.loc[mask]
    return Dataset(
        X=f.to_numpy(dtype=np.float32),
        y=y.loc[mask].to_numpy(dtype=float),
        ret=ret.loc[mask].to_numpy(dtype=float),
        dates=f.index.to_numpy(),
        tickers=np.full(mask.sum(), ticker),
        feature_names=list(f.columns),
    )


def pool_datasets(parts: list[Dataset]) -> Dataset | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    common = set(parts[0].feature_names)
    for p in parts[1:]:
        common &= set(p.feature_names)
    cols = [c for c in parts[0].feature_names if c in common]
    parts = [p.subset_columns(cols) for p in parts]
    X = np.vstack([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    ret = np.concatenate([p.ret for p in parts])
    dates = np.concatenate([p.dates for p in parts])
    tickers = np.concatenate([p.tickers for p in parts])
    order = np.argsort(dates, kind="stable")
    return Dataset(X[order], y[order], ret[order], dates[order], tickers[order], cols)


def _fit_predict(model_name: str, params: dict, X_tr, y_tr, X_te,
                 feature_names: list[str], n_jobs: int) -> np.ndarray:
    model = make_classifier(model_name, params, n_jobs=n_jobs)
    if isinstance(model, SequenceClassifier):
        model.fit(X_tr, y_tr, feature_names=feature_names)
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run_walkforward(ds: Dataset, model_name: str, params: dict | None = None, *,
                    min_train: int = 1000, refit_every: int = 21, horizon: int = 1,
                    start_date=None, end_date=None, n_jobs: int = -1,
                    calibrate: bool = False, calib_frac: float = 0.15) -> pd.DataFrame:
    """OOS predictions for one (model, config). Returns columns:
    date, ticker, p_up, y, ret."""
    params = params or {}
    udates = np.unique(ds.dates)
    n = len(udates)
    lo = np.searchsorted(udates, np.datetime64(start_date)) if start_date is not None else min_train
    lo = max(lo, min_train)
    hi = np.searchsorted(udates, np.datetime64(end_date)) if end_date is not None else n

    rows: list[pd.DataFrame] = []
    for s in range(lo, hi, refit_every):
        train_end_d = udates[s - horizon]           # exclusive; purge + embargo = horizon
        test_d = udates[s: min(s + refit_every, hi)]
        tr = ds.dates < train_end_d
        te = np.isin(ds.dates, test_d)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        try:
            if calibrate:
                cut = int(tr.sum() * (1 - calib_frac))
                tr_idx = np.flatnonzero(tr)
                fit_idx, cal_idx = tr_idx[:cut], tr_idx[cut:]
                model = make_classifier(model_name, params, n_jobs=n_jobs)
                if isinstance(model, SequenceClassifier):
                    model.fit(ds.X[fit_idx], ds.y[fit_idx], feature_names=ds.feature_names)
                else:
                    model.fit(ds.X[fit_idx], ds.y[fit_idx])
                from etf_forecaster.models.ensemble import IsotonicCalibrator
                p_cal = model.predict_proba(ds.X[cal_idx])[:, 1]
                p = model.predict_proba(ds.X[te])[:, 1]
                if len(np.unique(ds.y[cal_idx])) > 1 and len(cal_idx) >= 100:
                    p = IsotonicCalibrator().fit(p_cal, ds.y[cal_idx]).transform(p)
            else:
                p = _fit_predict(model_name, params, ds.X[tr], ds.y[tr], ds.X[te],
                                 ds.feature_names, n_jobs)
        except Exception as exc:  # noqa: BLE001
            log.warning("fold fit failed (%s @ %s): %s", model_name, str(train_end_d)[:10], exc)
            continue
        rows.append(pd.DataFrame({
            "date": ds.dates[te], "ticker": ds.tickers[te],
            "p_up": p, "y": ds.y[te], "ret": ds.ret[te],
        }))
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "p_up", "y", "ret"])
    return pd.concat(rows, ignore_index=True)


def score_predictions(preds: pd.DataFrame) -> dict[str, float]:
    if len(preds) < 50:
        return {"n": len(preds), "log_loss": np.nan, "brier": np.nan,
                "auc": np.nan, "hit_rate": np.nan}
    y, p = preds["y"].values, preds["p_up"].values
    return {
        "n": int(len(preds)),
        "log_loss": metrics.log_loss(y, p),
        "brier": metrics.brier(y, p),
        "auc": metrics.auc(y, p),
        "hit_rate": metrics.hit_rate(y, p),
        "pnl": metrics.signal_pnl(preds["ret"].values, p),
    }
