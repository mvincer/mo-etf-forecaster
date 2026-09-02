"""Staged include/exclude ablation over feature blocks.

Stage 1: PRICE floor, then PRICE + each candidate block alone (screening).
Stage 2: greedy forward addition then backward pruning.
Stage 3: exhaustive 2^k over the surviving shortlist - the literal include/exclude grid,
         restricted to blocks that could plausibly matter.

All evaluations are nested walk-forwards on the development slice only; every cell is
recorded with its running trial count for multiple-testing accounting.
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np

from etf_forecaster.search.blocks import select_columns
from etf_forecaster.validation.walkforward import Dataset, run_walkforward, score_predictions

log = logging.getLogger(__name__)


class AblationRun:
    def __init__(self, ds: Dataset, *, horizon: int, geometry: dict, screen_model: str,
                 screen_params: dict | None = None, window: int = 20,
                 group: str = "", n_jobs: int = -1):
        self.ds = ds
        self.horizon = horizon
        self.geometry = geometry          # min_train, refit_every, start/end dates
        self.screen_model = screen_model
        self.screen_params = screen_params or {}
        self.window = window
        self.group = group
        self.n_jobs = n_jobs
        self.records: list[dict] = []
        self._cache: dict[frozenset, float] = {}
        self.n_trials = 0

    # ------------------------------------------------------------- evaluation
    def evaluate(self, blocks: set[str], stage: str) -> float:
        key = frozenset(blocks)
        if key in self._cache:
            # still record the cell under this stage so ablation_results is complete
            prior = next((r for r in self.records
                          if r["blocks"] == (",".join(sorted(blocks)) or "PRICE_only")), None)
            if prior is not None and prior["stage"] != stage:
                self.records.append({**prior, "stage": stage})
            return self._cache[key]
        cols = select_columns(self.ds.feature_names, blocks, window=self.window)
        sub = self.ds.subset_columns(cols)
        preds = run_walkforward(
            sub, self.screen_model, self.screen_params, horizon=self.horizon,
            n_jobs=self.n_jobs, **self.geometry)
        score = score_predictions(preds)
        self.n_trials += 1
        loss = score["log_loss"] if np.isfinite(score.get("log_loss", np.nan)) else 10.0
        self.records.append({
            "group": self.group, "horizon": self.horizon, "stage": stage,
            "blocks": ",".join(sorted(blocks)) or "PRICE_only",
            "model": self.screen_model, "params": json.dumps(self.screen_params),
            "window": self.window, "n_trials_so_far": self.n_trials, **score,
        })
        self._cache[key] = loss
        log.info("[%s h=%d %s] {%s} log_loss=%.4f hit=%.3f n=%s",
                 self.group, self.horizon, stage, ",".join(sorted(blocks)) or "-",
                 loss, score.get("hit_rate", float("nan")), score.get("n"))
        return loss

    # ------------------------------------------------------------- stages
    def run(self, candidate_blocks: list[str], *, greedy_min_gain: float = 5e-4,
            exhaustive_max: int = 6) -> dict:
        base = self.evaluate(set(), "s1_screen")

        singles: dict[str, float] = {}
        for b in candidate_blocks:
            singles[b] = self.evaluate({b}, "s1_screen")
        ranked = sorted(singles, key=singles.get)

        # Stage 2: greedy forward from the best single block
        current: set[str] = {ranked[0]} if singles[ranked[0]] < base else set()
        best = min(singles[ranked[0]], base)
        improved = True
        while improved:
            improved = False
            gains = {}
            for b in candidate_blocks:
                if b in current:
                    continue
                loss = self.evaluate(current | {b}, "s2_greedy")
                gains[b] = best - loss
            if gains:
                top = max(gains, key=gains.get)
                if gains[top] > greedy_min_gain:
                    current.add(top)
                    best -= gains[top]
                    improved = True
        # backward pass
        for b in list(current):
            loss = self.evaluate(current - {b}, "s2_greedy")
            if loss <= best + greedy_min_gain / 2:
                current.discard(b)
                best = min(best, loss)

        # Stage 3: exhaustive over shortlist = greedy set + next best screened blocks
        shortlist = list(current)
        for b in ranked:
            if len(shortlist) >= exhaustive_max:
                break
            if b not in shortlist and singles[b] < base:
                shortlist.append(b)
        shortlist = shortlist[:exhaustive_max]
        best_set, best_loss = set(current), best
        for r in range(len(shortlist) + 1):
            for combo in itertools.combinations(shortlist, r):
                loss = self.evaluate(set(combo), "s3_exhaustive")
                if loss < best_loss:
                    best_set, best_loss = set(combo), loss

        cells = [r for r in self.records if r["stage"] == "s3_exhaustive"]
        cells.sort(key=lambda r: r["log_loss"])
        top_sets = []
        seen = set()
        for r in cells:
            bs = r["blocks"] if r["blocks"] != "PRICE_only" else ""
            if bs not in seen:
                seen.add(bs)
                top_sets.append(sorted(b for b in bs.split(",") if b))
            if len(top_sets) >= 3:
                break
        return {
            "group": self.group, "horizon": self.horizon,
            "best_blocks": sorted(best_set), "best_loss": best_loss,
            "base_loss": base, "top_sets": top_sets or [sorted(best_set)],
            "n_trials": self.n_trials,
        }
