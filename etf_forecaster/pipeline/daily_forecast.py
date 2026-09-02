"""Daily inference: fit each champion on all available history, forecast today's close
forward 1-5 days, refresh the output artifacts, and backfill matured outcomes.

Run:  python -m etf_forecaster.pipeline.daily_forecast [--jobs 8]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from etf_forecaster import store
from etf_forecaster.config import search_cfg, ticker_meta
from etf_forecaster.models.ensemble import IsotonicCalibrator
from etf_forecaster.models.magnitude import QuantileMagnitude
from etf_forecaster.models.registry import make_classifier
from etf_forecaster.models.sequence import SequenceClassifier
from etf_forecaster.pipeline.train_search import _load_ticker_data
from etf_forecaster.search.blocks import select_columns

log = logging.getLogger(__name__)


def _forecast_cell(ticker: str, horizon: int, meta: dict, cfg: dict,
                   n_jobs: int = 2) -> dict | None:
    data = _load_ticker_data(ticker, [horizon])
    if not data or horizon not in data[ticker]:
        return None
    ds = data[ticker][horizon]
    config = meta.get("champion_config") or (meta.get("configs") or [None])[0]
    if not config:
        return None
    cols = select_columns(ds.feature_names, config["blocks"], window=config.get("window", 20))
    sub = ds.subset_columns(cols)

    # training rows: labels for the last `horizon` rows are not yet known -> excluded by
    # build_dataset already (y is NaN there). But we still need TODAY's feature row, which
    # build_dataset dropped. Reload it from the features panel directly.
    from etf_forecaster.features.assemble import load_features
    feats = load_features(ticker)
    if feats is None or not len(feats):
        return None
    x_today = feats.iloc[[-1]][cols].to_numpy(dtype=np.float32)
    asof = feats.index[-1]

    n = len(sub.X)
    cut = int(n * 0.85)
    try:
        model = make_classifier(config["model"], config.get("params", {}), n_jobs=n_jobs)
        if isinstance(model, SequenceClassifier):
            model.fit(sub.X[:cut], sub.y[:cut], feature_names=sub.feature_names)
        else:
            model.fit(sub.X[:cut], sub.y[:cut])
        p_raw = model.predict_proba(x_today)[0, 1]
        p_cal = model.predict_proba(sub.X[cut:])[:, 1]
        if len(np.unique(sub.y[cut:])) > 1 and (n - cut) >= 100:
            p_up = float(IsotonicCalibrator().fit(p_cal, sub.y[cut:]).transform([p_raw])[0])
        else:
            p_up = float(p_raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("forecast fit failed %s h=%d: %s", ticker, horizon, exc)
        return None

    qs = tuple(cfg["magnitude"]["quantiles"])
    quants = {}
    try:
        qm = QuantileMagnitude(quantiles=qs, n_jobs=n_jobs, params={"n_estimators": 150})
        qm.fit(sub.X[:cut], sub.ret[:cut], X_calib=sub.X[cut:], y_calib=sub.ret[cut:],
               alpha=cfg["magnitude"]["conformal_alpha"])
        pred = qm.predict(x_today)
        quants = {f"q{int(q * 100):02d}": float(pred[q][0]) for q in qs}
    except Exception as exc:  # noqa: BLE001
        log.warning("magnitude failed %s h=%d: %s", ticker, horizon, exc)

    return {
        "ticker": ticker, "asof": asof, "horizon": horizon,
        "p_up": p_up, "expected_return": quants.get("q50", np.nan), **quants,
        "champion_model": config["model"],
        "active_blocks": ",".join(config["blocks"]),
        "window": config.get("window", 20),
        "target_type": ticker_meta(ticker).get("target_type", "etf"),
    }


def run(jobs: int = 8) -> pd.DataFrame:
    from joblib import Parallel, delayed

    cfg = search_cfg()
    champions = store.read_champions()
    if not champions:
        raise RuntimeError("champions.json missing - run train_search first")
    cells = []
    for key, meta in champions.items():
        ticker, h = key.rsplit("|", 1)
        cells.append((ticker, int(h), meta))
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_forecast_cell)(t, h, m, cfg) for t, h, m in cells)
    rows = [r for r in rows if r]
    latest = pd.DataFrame(rows)

    # attach trailing OOS hit rate from the scorecard (champion series, 252-day window)
    sc = store.read(store.SCORECARD)
    if sc is not None and len(sc) and len(latest):
        champ = sc[(sc["model"] == "champion") & (sc["window"] == "252")]
        latest = latest.merge(
            champ[["ticker", "horizon", "hit_rate", "log_loss", "auc", "n"]].rename(
                columns={"hit_rate": "trailing_hit_rate", "log_loss": "trailing_log_loss",
                         "auc": "trailing_auc", "n": "trailing_n"}),
            on=["ticker", "horizon"], how="left")
    store.write_forecasts(latest)

    # append today's live predictions into the history (outcomes backfilled later)
    if len(latest):
        hist_rows = latest.rename(columns={"asof": "date"})[
            ["date", "ticker", "horizon", "p_up", "expected_return"]
            + [c for c in latest.columns if c.startswith("q")]].copy()
        hist_rows["model"] = "champion"
        hist_rows["y"] = np.nan
        hist_rows["ret_realized"] = np.nan
        store.append_history(hist_rows)

    # backfill matured outcomes + refresh scorecard
    from etf_forecaster.data import yahoo
    bars = {t: yahoo.load_bars(t) for t in {c[0] for c in cells}}
    n_filled = store.backfill_outcomes({k: v for k, v in bars.items() if v is not None})
    store.rebuild_scorecard()
    log.info("daily forecast: %d cells, %d outcomes backfilled", len(latest), n_filled)
    return latest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(jobs=args.jobs)


if __name__ == "__main__":
    main()
