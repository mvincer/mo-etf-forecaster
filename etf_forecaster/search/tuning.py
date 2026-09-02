"""Stage 4: Optuna model/hyperparameter search over the winning block sets,
including the lag-window X. Nested on the development slice only."""

from __future__ import annotations

import json
import logging

import numpy as np

from etf_forecaster.config import models_cfg
from etf_forecaster.search.blocks import select_columns
from etf_forecaster.validation.walkforward import Dataset, run_walkforward, score_predictions

log = logging.getLogger(__name__)


def _suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "loguniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        else:
            out[name] = trial.suggest_float(name, spec["low"], spec["high"])
    return out


def model_search(ds: Dataset, *, horizon: int, geometry: dict, block_sets: list[list[str]],
                 roster: list[str], window_grid: list[int], n_trials: int,
                 group: str = "", n_jobs: int = -1) -> tuple[list[dict], list[dict]]:
    """Returns (top_configs, records). Each config: {model, params, blocks, window, loss}.
    n_trials == 0 -> defaults-only sweep of roster x block_sets (fast mode)."""
    defaults = models_cfg()["defaults"]
    spaces = models_cfg()["search_spaces"]
    records: list[dict] = []
    evals: list[dict] = []
    trial_counter = 0

    def _eval(model: str, params: dict, blocks: list[str], window: int) -> float:
        nonlocal trial_counter
        cols = select_columns(ds.feature_names, blocks, window=window)
        sub = ds.subset_columns(cols)
        preds = run_walkforward(sub, model, params, horizon=horizon, n_jobs=n_jobs, **geometry)
        score = score_predictions(preds)
        trial_counter += 1
        loss = score["log_loss"] if np.isfinite(score.get("log_loss", np.nan)) else 10.0
        rec = {"group": group, "horizon": horizon, "stage": "s4_model",
               "blocks": ",".join(sorted(blocks)), "model": model,
               "params": json.dumps(params), "window": window,
               "n_trials_so_far": trial_counter, **score}
        records.append(rec)
        evals.append({"model": model, "params": params, "blocks": sorted(blocks),
                      "window": window, "loss": loss, **score})
        log.info("[%s h=%d s4] %s w=%d loss=%.4f", group, horizon, model, window, loss)
        return loss

    if n_trials <= 0:
        for blocks in block_sets:
            for model in roster:
                params = dict(defaults.get(model, {}))
                _eval(model, params, blocks, 20)
        # sweep windows for the best defaults config
        evals.sort(key=lambda e: e["loss"])
        best = evals[0]
        for w in window_grid:
            if w != best["window"]:
                _eval(best["model"], best["params"], best["blocks"], w)
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            model = trial.suggest_categorical("model", roster)
            bi = trial.suggest_int("block_set", 0, len(block_sets) - 1)
            window = trial.suggest_categorical("window", window_grid)
            space = spaces.get(model)
            params = _suggest(trial, space) if space else dict(defaults.get(model, {}))
            return _eval(model, params, block_sets[bi], window)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    evals.sort(key=lambda e: e["loss"])
    # top configs, deduped by (model, blocks)
    top: list[dict] = []
    seen = set()
    for e in evals:
        key = (e["model"], ",".join(e["blocks"]))
        if key in seen:
            continue
        seen.add(key)
        top.append(e)
        if len(top) >= 3:
            break
    return top, records
