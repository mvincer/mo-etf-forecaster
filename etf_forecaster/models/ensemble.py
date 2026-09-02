"""Stacked ensemble of the top-k configs with isotonic probability calibration."""

from __future__ import annotations

import numpy as np


class IsotonicCalibrator:
    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        from sklearn.isotonic import IsotonicRegression
        self.iso_ = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        self.iso_.fit(np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.iso_.transform(np.asarray(p_raw, dtype=float))


class StackedEnsemble:
    """Average member probabilities (equal weight), then isotonic-calibrate on a
    held-out slice. Members are already-fitted classifiers."""

    def __init__(self, members: list):
        self.members = members
        self.calib: IsotonicCalibrator | None = None

    def raw_proba(self, X, feature_sets: list | None = None) -> np.ndarray:
        ps = []
        for i, m in enumerate(self.members):
            Xi = X if feature_sets is None else feature_sets[i]
            ps.append(m.predict_proba(Xi)[:, 1])
        return np.mean(np.column_stack(ps), axis=1)

    def calibrate(self, X_calib, y_calib, feature_sets: list | None = None):
        p = self.raw_proba(X_calib, feature_sets)
        if len(np.unique(np.asarray(y_calib))) > 1 and len(y_calib) >= 50:
            self.calib = IsotonicCalibrator().fit(p, y_calib)
        return self

    def predict_proba(self, X, feature_sets: list | None = None) -> np.ndarray:
        p = self.raw_proba(X, feature_sets)
        if self.calib is not None:
            p = self.calib.transform(p)
        return np.column_stack([1 - p, p])
