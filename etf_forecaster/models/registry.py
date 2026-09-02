"""Model factory: every entry returns an object with fit(X, y) and predict_proba(X)
that tolerates NaNs (natively or via an imputing pipeline)."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

# These fire thousands of times inside walk-forward loops and drown real errors in logs.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

log = logging.getLogger(__name__)

SEED = 7


class Climatology:
    """Baseline: the training base rate."""

    def fit(self, X, y):
        self.p_ = float(np.clip(np.mean(y), 0.05, 0.95))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_), np.full(n, self.p_)])


class MomentumBaseline:
    """Baseline: logistic on the single 21d-return column (index passed at build)."""

    def __init__(self, col_idx: int = 0):
        self.col_idx = col_idx

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        x = np.nan_to_num(np.asarray(X, dtype=float)[:, [self.col_idx]])
        self.m_ = LogisticRegression(max_iter=500).fit(x, y)
        return self

    def predict_proba(self, X):
        x = np.nan_to_num(np.asarray(X, dtype=float)[:, [self.col_idx]])
        return self.m_.predict_proba(x)


def _linear_pipeline(estimator):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _impute_pipeline(estimator):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def make_classifier(name: str, params: dict[str, Any] | None = None, *,
                    n_jobs: int = -1, seed: int = SEED):
    p = dict(params or {})
    if name == "baseline_climatology":
        return Climatology()
    if name == "baseline_momentum":
        return MomentumBaseline(col_idx=int(p.get("col_idx", 0)))
    if name in ("logistic_l1", "logistic_l2"):
        from sklearn.linear_model import LogisticRegression
        return _linear_pipeline(LogisticRegression(
            penalty="l1" if name == "logistic_l1" else "l2",
            C=float(p.get("C", 1.0)), solver="liblinear" if name == "logistic_l1" else "lbfgs",
            max_iter=2000, random_state=seed))
    if name == "elastic_net":
        from sklearn.linear_model import SGDClassifier
        return _linear_pipeline(SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=float(p.get("alpha", 1e-3)), l1_ratio=float(p.get("l1_ratio", 0.5)),
            max_iter=2000, random_state=seed))
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            num_leaves=int(p.get("num_leaves", 31)),
            min_child_samples=int(p.get("min_child_samples", 40)),
            subsample=float(p.get("subsample", 0.8)), subsample_freq=1,
            colsample_bytree=float(p.get("colsample_bytree", 0.7)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            random_state=seed, n_jobs=n_jobs, verbosity=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            max_depth=int(p.get("max_depth", 4)),
            min_child_weight=float(p.get("min_child_weight", 20)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.7)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            random_state=seed, n_jobs=n_jobs, verbosity=0, eval_metric="logloss")
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=int(p.get("iterations", 300)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            depth=int(p.get("depth", 5)),
            l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
            random_seed=seed, verbose=False, allow_writing_files=False,
            thread_count=n_jobs if n_jobs > 0 else None)
    if name in ("random_forest", "extra_trees"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        return _impute_pipeline(cls(
            n_estimators=int(p.get("n_estimators", 400)),
            max_depth=int(p.get("max_depth", 6)),
            min_samples_leaf=int(p.get("min_samples_leaf", 30)),
            max_features=float(p.get("max_features", 0.4)),
            random_state=seed, n_jobs=n_jobs))
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 300)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            max_depth=int(p.get("max_depth", 4)),
            min_samples_leaf=int(p.get("min_samples_leaf", 40)),
            random_state=seed)
    if name == "svm_rbf":
        from sklearn.svm import SVC
        return _linear_pipeline(SVC(C=float(p.get("C", 1.0)), kernel="rbf",
                                    probability=True, random_state=seed))
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return _linear_pipeline(MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden", (64, 32))),
            alpha=float(p.get("alpha", 1e-3)), max_iter=400,
            early_stopping=True, random_state=seed))
    if name in ("lstm", "tcn"):
        from etf_forecaster.models.sequence import SequenceClassifier
        return SequenceClassifier(arch=name, **p)
    raise ValueError(f"unknown model: {name}")
