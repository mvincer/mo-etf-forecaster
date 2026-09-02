"""Magnitude model: LightGBM quantile regression + split-conformal (CQR) band correction."""

from __future__ import annotations

import numpy as np

_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class QuantileMagnitude:
    """One LightGBM quantile model per requested quantile, with CQR widening of the
    outer bands fitted on a held-out calibration slice."""

    def __init__(self, quantiles: tuple[float, ...] = _QUANTILES,
                 params: dict | None = None, n_jobs: int = -1, seed: int = 7):
        self.quantiles = quantiles
        self.params = params or {}
        self.n_jobs = n_jobs
        self.seed = seed
        self.offsets_: dict[tuple[float, float], float] = {}

    def fit(self, X, y, *, X_calib=None, y_calib=None, alpha: float = 0.20):
        from lightgbm import LGBMRegressor
        self.models_ = {}
        for q in self.quantiles:
            m = LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=int(self.params.get("n_estimators", 300)),
                learning_rate=float(self.params.get("learning_rate", 0.03)),
                num_leaves=int(self.params.get("num_leaves", 31)),
                min_child_samples=int(self.params.get("min_child_samples", 40)),
                random_state=self.seed, n_jobs=self.n_jobs, verbosity=-1)
            m.fit(X, y)
            self.models_[q] = m

        # CQR on the outer symmetric pairs
        if X_calib is not None and len(X_calib) >= 50:
            for lo, hi in ((0.10, 0.90), (0.05, 0.95)):
                if lo in self.models_ and hi in self.models_:
                    ql = self.models_[lo].predict(X_calib)
                    qh = self.models_[hi].predict(X_calib)
                    e = np.maximum(ql - y_calib, y_calib - qh)
                    cov = 1 - (hi - lo)          # miscoverage target for this pair
                    n = len(e)
                    k = min(n - 1, int(np.ceil((1 - cov) * (n + 1))) - 1)
                    self.offsets_[(lo, hi)] = float(np.sort(e)[max(k, 0)])
        return self

    def predict(self, X) -> dict[float, np.ndarray]:
        out = {q: m.predict(X) for q, m in self.models_.items()}
        for (lo, hi), off in self.offsets_.items():
            out[lo] = out[lo] - off
            out[hi] = out[hi] + off
        # enforce monotone quantiles
        qs = sorted(out)
        stacked = np.sort(np.stack([out[q] for q in qs], axis=1), axis=1)
        return {q: stacked[:, i] for i, q in enumerate(qs)}
