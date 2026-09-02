"""End-to-end smoke test of the search/walk-forward machinery on one ticker (VIXY).
Small budgets; verifies ablation, model search, outer walk-forward, magnitude,
store artifacts and daily-forecast cell. Run with the app venv python."""

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from etf_forecaster.pipeline.train_search import (  # noqa: E402
    _load_ticker_data, magnitude_quantiles, outer_walkforward,
)
from etf_forecaster.config import search_cfg  # noqa: E402
from etf_forecaster.search.ablation import AblationRun  # noqa: E402
from etf_forecaster.search.tuning import model_search  # noqa: E402

TICKER = "VIXY"
H = 1

t0 = time.time()
data = _load_ticker_data(TICKER, [H])
ds = data[H]
print(f"dataset: {ds.X.shape}, dates {ds.dates[0]} .. {ds.dates[-1]} ({time.time()-t0:.0f}s)")

ud = np.unique(ds.dates)
dev_end = ud[int(len(ud) * 0.6)]
geometry = {"min_train": 500, "refit_every": 252, "end_date": dev_end}

ab = AblationRun(ds, horizon=H, geometry=geometry, screen_model="lightgbm",
                 screen_params={"n_estimators": 120}, group="smoke", n_jobs=4)
res = ab.run(["TECH", "VOLOPT", "REGIME", "SEASON"], exhaustive_max=3)
print("ablation result:", {k: v for k, v in res.items() if k != "records"})

cfg = search_cfg()
top, recs = model_search(ds, horizon=H, geometry=geometry, block_sets=res["top_sets"],
                         roster=["baseline_climatology", "logistic_l2", "lightgbm"],
                         window_grid=[10, 20], n_trials=0, group="smoke", n_jobs=4)
print("top configs:", [(c["model"], c["blocks"], round(c["loss"], 4)) for c in top])

preds, meta = outer_walkforward(TICKER, H, ds, top[:2], dev_end, cfg, fast=True, n_jobs=4)
print("outer preds:", preds.shape, "models:", preds['model'].unique().tolist())
print("final champion:", meta["final_champion"])
print("oos scores:", meta["oos_scores"])

champ_cfg = top[0]
q = magnitude_quantiles(TICKER, H, ds, champ_cfg, dev_end, cfg, fast=True, n_jobs=4)
print("quantiles:", q.shape, q.columns.tolist())
cov = ((q.merge(preds[preds.model == 'champion'][['date', 'ret']], on='date'))
       .assign(inside=lambda d: (d['ret'] >= d['q10']) & (d['ret'] <= d['q90'])))
print("q10-q90 empirical coverage:", round(cov['inside'].mean(), 3), "target 0.80")
print(f"TOTAL {time.time()-t0:.0f}s")
