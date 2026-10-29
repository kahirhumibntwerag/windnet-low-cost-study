"""Fixed, data-independent estimators for the continuation protocol.

This module never loads study files or trains on import. Every call to
``build_model`` returns a fresh estimator. The finite grid is five ridge
penalties and four shallow histogram-gradient-boosted tree configurations.
Feature construction and any logarithmic transforms belong to the caller.

Sample weights, when supplied, affect the learner's training loss only.
Imputation and standardization are fitted without weights on the fitting
matrix, matching the original compact-model preprocessing convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
TREE_LEAF_COUNTS = (7, 15)
TREE_MIN_LEAF_SIZES = (30, 60)


@dataclass(frozen=True)
class CandidateSpec:
    """Immutable candidate identity with JSON-ready, copied parameters."""

    name: str
    family: str
    alpha: float | None = None
    max_leaf_nodes: int | None = None
    min_samples_leaf: int | None = None

    @property
    def params(self) -> dict[str, Any]:
        """Return a new dict, so callers cannot mutate the registered grid."""
        if self.family == "ridge":
            return {"alpha": self.alpha, "fit_intercept": True}
        if self.family == "hist_gradient_boosting":
            return {
                "loss": "squared_error",
                "max_iter": 150,
                "learning_rate": 0.05,
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": 10.0,
                "early_stopping": False,
                "random_state": 2026,
            }
        raise ValueError(f"Unknown candidate family: {self.family!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "params": self.params,
            "preprocessing": (
                "mean imputation with missing indicators and retained empty "
                "features; standard scaling"
                if self.family == "ridge"
                else "native missing-value handling; no scaling"
            ),
            "sample_weight_scope": "learner loss only",
        }


_CANDIDATES = tuple(
    CandidateSpec(
        name=f"ridge_alpha_{alpha:g}", family="ridge", alpha=alpha
    )
    for alpha in RIDGE_ALPHAS
) + tuple(
    CandidateSpec(
        name=f"hgb_leaves_{leaves}_minleaf_{leaf_size}",
        family="hist_gradient_boosting",
        max_leaf_nodes=leaves,
        min_samples_leaf=leaf_size,
    )
    for leaves in TREE_LEAF_COUNTS
    for leaf_size in TREE_MIN_LEAF_SIZES
)
_BY_NAME = {spec.name: spec for spec in _CANDIDATES}


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the nine fixed protocol candidates, in deterministic order."""
    return _CANDIDATES


def _canonical_spec(spec: CandidateSpec | str) -> CandidateSpec:
    if isinstance(spec, str):
        try:
            return _BY_NAME[spec]
        except KeyError as error:
            raise ValueError(f"Unknown fixed candidate: {spec!r}") from error
    if not isinstance(spec, CandidateSpec):
        raise TypeError("spec must be a CandidateSpec or a registered name")
    registered = _BY_NAME.get(spec.name)
    if registered is None or registered != spec:
        raise ValueError("Candidate is outside the fixed protocol grid")
    return registered


def _fresh_estimator(spec: CandidateSpec) -> BaseEstimator:
    if spec.family == "ridge":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="mean",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(**spec.params)),
            ]
        )
    return HistGradientBoostingRegressor(**spec.params)


def _feature_matrix(X: Any) -> np.ndarray:
    """Accept NaNs for the prescribed missing-data treatment, never infinity."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if np.isinf(array).any():
        raise ValueError("X contains infinite values; handle them upstream")
    return array


def _target_vector(y: Any, n_rows: int) -> np.ndarray:
    array = np.asarray(y, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError("y must contain exactly one target per fitting row")
    if not np.isfinite(array).all():
        raise ValueError("Fitting targets must all be finite")
    return array


def _weights(sample_weight: Any, n_rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != n_rows:
        raise ValueError("sample_weight must have one value per fitting row")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weight must be finite and nonnegative")
    if not (weights > 0).any():
        raise ValueError("At least one fitting weight must be positive")
    return weights


class FiniteRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible wrapper with explicit weights and finite predictions.

    Invalid predictions raise an error instead of being silently clipped or
    replaced using validation/verification values. The caller owns all sample
    selection, source-fold masking, inner validation and model comparison.
    """

    def __init__(self, spec: CandidateSpec | str):
        self.spec = spec

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> FiniteRegressor:
        spec = _canonical_spec(self.spec)
        features = _feature_matrix(X)
        target = _target_vector(y, features.shape[0])
        weights = _weights(sample_weight, features.shape[0])
        estimator = _fresh_estimator(spec)
        if isinstance(estimator, Pipeline):
            kwargs = (
                {} if weights is None else {"regressor__sample_weight": weights}
            )
        else:
            kwargs = {} if weights is None else {"sample_weight": weights}
        estimator.fit(features, target, **kwargs)
        self.estimator_ = estimator
        self.n_features_in_ = features.shape[1]
        self.candidate_name_ = spec.name
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["estimator_", "n_features_in_"])
        features = _feature_matrix(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features differ from fitting features")
        prediction = np.asarray(self.estimator_.predict(features), dtype=np.float64)
        if prediction.ndim != 1 or len(prediction) != features.shape[0]:
            raise ValueError("Estimator returned an unexpected prediction shape")
        if not np.isfinite(prediction).all():
            raise ValueError("Estimator returned nonfinite predictions")
        return prediction


def build_model(spec: CandidateSpec | str) -> FiniteRegressor:
    """Construct an unfitted, independent estimator from the fixed grid."""
    return FiniteRegressor(_canonical_spec(spec))
