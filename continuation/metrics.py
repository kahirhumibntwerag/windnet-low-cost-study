"""Forecast verification without data loading or model execution.

Speeds are daily means in km/s. Primary fast wind is observed speed >= 500;
>= 600 is secondary. Thresholds are fixed before evaluation. Bias is forecast
minus observation. Undefined quantities are ``None``, never invented zeros.

Day-level contingency scores are not event-episode scores. TSS/HSS require
meaningful negative cases and should not be applied to a list of matched
episodes. All functions are pure calculations on the arrays supplied by the
caller: this module opens no files and imports no research/training modules.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FAST_THRESHOLDS = (500.0, 600.0)


def _arrays(*values: Any) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Forecast verification requires one-dimensional arrays.")
    if arrays and any(len(array) != len(arrays[0]) for array in arrays[1:]):
        raise ValueError("Observations and forecasts must have identical lengths.")
    return arrays


def _number(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return _number(numerator / denominator) if denominator != 0 else None


def _threshold(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("A speed threshold must be positive and finite.")
    return value


def _continuous_finite(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    if n == 0:
        return {"n": 0, "mse": None, "rmse": None, "mae": None,
                "bias": None, "r": None}
    error = p - y
    mse = np.mean(np.square(error))
    r = None
    if n >= 2 and np.std(y) > 0 and np.std(p) > 0:
        r = _number(np.corrcoef(y, p)[0, 1])
    return {"n": n, "mse": _number(mse), "rmse": _number(np.sqrt(mse)),
            "mae": _number(np.mean(np.abs(error))),
            "bias": _number(np.mean(error)), "r": r}


def continuous_metrics(observed: Any, forecast: Any) -> dict:
    """RMSE, MAE, signed bias, and Pearson r on finite observation/forecast pairs.

    MSE has units (km/s)^2; RMSE/MAE/bias have units km/s. Correlation is
    undefined with fewer than two cases or a constant observation/forecast.
    """
    y, p = _arrays(observed, forecast)
    finite = np.isfinite(y) & np.isfinite(p)
    result = _continuous_finite(y[finite], p[finite])
    result["n_excluded_nonfinite"] = int(len(y) - np.sum(finite))
    return result


def detection_metrics(observed: Any, forecast: Any, threshold: float = 500.0) -> dict:
    """Daily exceedance detection at a fixed inclusive speed threshold.

    POD = TP/(TP+FN); FAR = FP/(TP+FP), a false-alarm *ratio*;
    FPR = FP/(FP+TN), the false-positive rate; CSI = TP/(TP+FP+FN);
    TSS = POD-FPR; frequency bias = (TP+FP)/(TP+FN).
    HSS = 2(TP*TN-FP*FN) /
          [(TP+FN)(FN+TN)+(TP+FP)(FP+TN)].

    All metrics use the same finite pairs. No positive/negative class is
    manufactured when absent; every zero denominator returns None.
    """
    limit = _threshold(threshold)
    y, p = _arrays(observed, forecast)
    finite = np.isfinite(y) & np.isfinite(p)
    event = y[finite] >= limit
    predicted = p[finite] >= limit
    tp = int(np.sum(event & predicted))
    fp = int(np.sum(~event & predicted))
    fn = int(np.sum(event & ~predicted))
    tn = int(np.sum(~event & ~predicted))
    pod = _ratio(tp, tp + fn)
    fpr = _ratio(fp, fp + tn)
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "threshold_km_s": limit, "comparison": ">=",
        "n": int(np.sum(finite)),
        "n_excluded_nonfinite": int(len(y) - np.sum(finite)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "observed_fast_days": tp + fn,
        "forecast_fast_days": tp + fp,
        "pod": pod, "precision": _ratio(tp, tp + fp),
        "far": _ratio(fp, tp + fp), "fpr": fpr,
        "csi": _ratio(tp, tp + fp + fn),
        "tss": None if pod is None or fpr is None else _number(pod - fpr),
        "hss": _ratio(2 * (tp * tn - fp * fn), hss_denominator),
        "frequency_bias": _ratio(tp + fp, tp + fn),
    }


def fast_wind_metrics(observed: Any, forecast: Any, threshold: float = 500.0) -> dict:
    """Detection and continuous error on OBSERVED fast-wind cases.

    Conditional RMSE/MAE/bias are selected by observed >= threshold, not by
    forecast exceedance. They therefore retain misses rather than rewarding
    a model that rarely predicts fast wind.
    """
    limit = _threshold(threshold)
    y, p = _arrays(observed, forecast)
    finite = np.isfinite(y) & np.isfinite(p)
    fast = finite & (y >= limit)
    return {"threshold_km_s": limit,
            "condition": "observed >= threshold on finite forecast pairs",
            "continuous_observed_fast": _continuous_finite(y[fast], p[fast]),
            "detection": detection_metrics(y, p, limit)}


def persistence_skill(observed: Any, forecast: Any, persistence: Any,
                      threshold: float | None = None) -> dict:
    """Paired continuous skill against a supplied persistence forecast.

    All three arrays share one common finite mask. If a threshold is supplied,
    the SAME observed fast-wind cases are used for model and reference.
    MSESS = 1 - MSE_model/MSE_reference; delta_RMSE = RMSE_model-RMSE_reference.
    Positive MSESS and negative delta_RMSE favor the candidate. A perfect
    reference has zero MSE, so normalized MSESS is undefined even if both
    forecasts are perfect. Delta RMSE remains defined when cases exist.

    The caller must construct the causal reference on the same issue/target
    pairs (for example latest available speed or 27-day persistence). This
    function does not retrieve, shift, interpolate, or repair a reference.
    """
    limit = None if threshold is None else _threshold(threshold)
    y, p, reference = _arrays(observed, forecast, persistence)
    common = np.isfinite(y) & np.isfinite(p) & np.isfinite(reference)
    selected = common.copy()
    if limit is not None:
        selected &= y >= limit
    model = _continuous_finite(y[selected], p[selected])
    baseline = _continuous_finite(y[selected], reference[selected])
    mse_skill = None
    delta = None
    if model["n"]:
        if baseline["mse"] is not None and baseline["mse"] != 0:
            mse_skill = _number(1 - model["mse"] / baseline["mse"])
        if model["rmse"] is not None and baseline["rmse"] is not None:
            delta = _number(model["rmse"] - baseline["rmse"])
    return {
        "threshold_km_s": limit,
        "condition": "all common finite cases" if limit is None else "observed >= threshold on common finite cases",
        "n_input": len(y), "n_common_finite": int(np.sum(common)),
        "n_excluded_nonfinite": int(len(y) - np.sum(common)),
        "n": model["n"], "model": model, "reference": baseline,
        "mse_skill_score": mse_skill, "delta_rmse": delta,
    }


def evaluate_forecast(observed: Any, forecast: Any,
                      persistence: Mapping[str, Any] | None = None,
                      thresholds: Sequence[float] = FAST_THRESHOLDS) -> dict:
    """Return overall, fixed-threshold fast-wind, and paired reference metrics.

    ``persistence`` maps descriptive reference names to already aligned arrays.
    Reference comparisons can have different available-case counts; each one
    explicitly reports its common mask size. Never compare unpaired RMSEs as
    though their cases were identical.
    """
    y, p = _arrays(observed, forecast)
    limits = tuple(_threshold(value) for value in thresholds)
    if len(set(limits)) != len(limits):
        raise ValueError("Speed thresholds must be unique.")
    result = {"units": {"speed": "km/s", "mse": "(km/s)^2"},
              "overall": continuous_metrics(y, p),
              "fast_wind": {f"ge_{limit:g}": fast_wind_metrics(y, p, limit) for limit in limits},
              "persistence_skill": {}}
    for name, reference in (persistence or {}).items():
        result["persistence_skill"][name] = {
            "all": persistence_skill(y, p, reference),
            **{f"ge_{limit:g}": persistence_skill(y, p, reference, limit) for limit in limits},
        }
    return result


def deduplicate_issue_target_pairs(issue: Sequence[Any], target: Sequence[Any],
                                   observed: Any,
                                   forecasts: Mapping[str, Any]) -> dict:
    """Optional, explicit fixed aggregation for duplicate issue/target pairs.

    Group ONLY identical issue AND target identifiers. Preserve first-seen
    group order, require agreeing finite observations, and average each
    forecast's finite values with equal weights. A wholly missing group stays
    NaN and is later excluded by the metric's common finite mask. This helper
    is not applied automatically and must not silently replace a same-row
    historical comparison or change the originally declared evaluation unit.
    """
    y, = _arrays(observed)
    if len(issue) != len(y) or len(target) != len(y):
        raise ValueError("Issue, target, and observation arrays must align.")
    predictions = {name: _arrays(value)[0] for name, value in forecasts.items()}
    if any(len(value) != len(y) for value in predictions.values()):
        raise ValueError("All forecast columns must align with observations.")
    groups: dict[tuple[Any, Any], list[int]] = {}
    for index, pair in enumerate(zip(issue, target)):
        try:
            groups.setdefault(pair, []).append(index)
        except TypeError as error:
            raise ValueError("Issue/target identifiers must be hashable scalars.") from error
    pairs = list(groups)
    observations = []
    means = {name: [] for name in predictions}
    counts = []
    for indexes in groups.values():
        values = y[indexes]
        known = values[np.isfinite(values)]
        if len(known) and not np.allclose(known, known[0], rtol=0, atol=1e-8):
            raise ValueError("Duplicate issue/target pairs have inconsistent observations.")
        observations.append(float(known[0]) if len(known) else np.nan)
        counts.append(len(indexes))
        for name, column in predictions.items():
            values = column[indexes]
            known = values[np.isfinite(values)]
            means[name].append(float(np.mean(known)) if len(known) else np.nan)
    return {
        "issue": [pair[0] for pair in pairs], "target": [pair[1] for pair in pairs],
        "observed": np.asarray(observations, dtype=float),
        "forecasts": {name: np.asarray(values, dtype=float) for name, values in means.items()},
        "duplicate_counts": np.asarray(counts, dtype=int),
        "n_input": len(y), "n_unique_pairs": len(pairs),
        "aggregation": "equal-weight mean of finite forecasts within identical issue-target pairs",
    }
