"""Pure paired conditional bootstrap for a predeclared frozen final panel.

This module opens no files, fits no models and performs no model selection.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


ROLES = ("starting_wind", "h9_fast_specialist", "P0")
REFERENCES = ("released_windnet", "old_frozen_selected")
CONDITIONS = (("all", None), ("ge_500", 500.0), ("ge_600", 600.0))
ANCHOR = "2011-01-01"
BIN_DAYS = 54
RESAMPLES = 5000
SEED = 20261029


def paired_final_uncertainty(observed, predictions: Mapping, dates, folds) -> dict:
    """Paired pooled delta-RMSE intervals with original window multiplicity.

    A common resample of occupied UTC bins is reused for every model,
    reference and threshold WITHIN each scope. The all-row and Fold-5 scopes
    each resample their occupied bins with replacement. Thresholds and finite
    masks change grouped error sums/counts, never the bootstrap choices.
    """
    y = np.asarray(observed, dtype=float)
    d = np.asarray(dates, dtype="datetime64[D]")
    f = np.asarray(folds)
    if y.ndim != 1 or not len(y) or d.shape != y.shape or f.shape != y.shape or np.isnat(d).any():
        raise ValueError("Aligned nonempty observations, UTC dates and folds are required")
    if not np.isin(f, (1, 2, 3, 4, 5)).all():
        raise ValueError("Only original fold identifiers 1..5 are allowed")
    needed = set(ROLES) | set(REFERENCES)
    if not needed.issubset(predictions):
        raise ValueError("Missing a preregistered model role or frozen comparator")
    p = {name: np.asarray(predictions[name], dtype=float) for name in needed}
    if any(values.shape != y.shape for values in p.values()):
        raise ValueError("Forecast columns must align exactly to original windows")
    bins = ((d - np.datetime64(ANCHOR)) / np.timedelta64(1, "D")).astype(np.int64) // BIN_DAYS
    rng = np.random.default_rng(SEED)
    scopes = {}
    comparisons = [(model, reference, condition, threshold)
                   for model in ROLES for reference in REFERENCES for condition, threshold in CONDITIONS]
    for scope, scope_mask in (("all_original_windows", np.ones(len(y), dtype=bool)), ("original_fold5", f == 5)):
        occupied = np.unique(bins[scope_mask])
        k = len(occupied)
        if not k:
            raise ValueError("Each declared uncertainty scope requires at least one original window")
        cluster = np.searchsorted(occupied, bins[scope_mask])
        # Sampling choices depend only on the scope's occupied bins, not model
        # availability, prediction, observed threshold or result magnitude.
        choices = rng.integers(0, k, size=(RESAMPLES, k))
        multiplicities = np.zeros((RESAMPLES, k), dtype=np.int32)
        np.add.at(multiplicities, (np.arange(RESAMPLES)[:, None], choices), 1)
        counts = np.zeros((k, len(comparisons)))
        error_model = np.zeros_like(counts)
        error_reference = np.zeros_like(counts)
        ys = y[scope_mask]
        for column, (model, reference, _, threshold) in enumerate(comparisons):
            a, b = p[model][scope_mask], p[reference][scope_mask]
            common = np.isfinite(ys) & np.isfinite(a) & np.isfinite(b)
            if threshold is not None:
                common &= ys >= threshold
            counts[:, column] = np.bincount(cluster[common], minlength=k)
            error_model[:, column] = np.bincount(cluster[common], weights=(ys[common] - a[common]) ** 2, minlength=k)
            error_reference[:, column] = np.bincount(cluster[common], weights=(ys[common] - b[common]) ** 2, minlength=k)
        sampled_counts = multiplicities @ counts
        sampled_model = multiplicities @ error_model
        sampled_reference = multiplicities @ error_reference
        values = np.full(sampled_counts.shape, np.nan)
        defined = sampled_counts > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            values[defined] = np.sqrt(sampled_model[defined] / sampled_counts[defined]) - np.sqrt(sampled_reference[defined] / sampled_counts[defined])
        scope_reports = {}
        for column, (model, reference, condition, threshold) in enumerate(comparisons):
            finite = values[:, column][np.isfinite(values[:, column])]
            total = int(counts[:, column].sum())
            delta = (float(np.sqrt(error_model[:, column].sum() / total) - np.sqrt(error_reference[:, column].sum() / total)) if total else None)
            scope_reports.setdefault(model, {}).setdefault(reference, {})[condition] = {
                "threshold_km_s": threshold, "n_paired_original_windows": total,
                "bins_with_paired_cases": int((counts[:, column] > 0).sum()),
                "delta_rmse_km_s": delta,
                "percentile95_km_s": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
                "resamples": RESAMPLES, "defined_draws": int(len(finite)), "undefined_draws": int(RESAMPLES - len(finite)),
                "undefined_rule": "No common finite cases satisfying the observed threshold gives undefined, never zero",
                "difference_definition": "model pooled RMSE minus reference pooled RMSE on identical finite cases; negative favors model"}
        scopes[scope] = {"original_windows": int(scope_mask.sum()), "occupied_utc_bins": k,
                         "occupied_bin_ids": occupied.tolist(), "comparisons": scope_reports}
    return {"method": "paired percentile bootstrap of fixed 54-day UTC bins", "bin_anchor_utc": ANCHOR,
            "bin_days": BIN_DAYS, "resamples": RESAMPLES, "seed": SEED,
            "sampling": "Within each scope sample its occupied UTC bins with replacement, drawing the same number of bins; reuse every draw across all models, references and thresholds",
            "window_weighting": "All original windows retained, including duplicate issue/target rows and their original multiplicity",
            "finite_cases": "Each comparison uses one common finite observation/model/reference mask before thresholding on observed speed",
            "scope_draw_order": ["all_original_windows", "original_fold5"], "scopes": scopes,
            "interpretation": "Conditional uncertainty for a historical benchmark after adaptive model design; not independent confirmation and not used for promotion, tuning or selection"}
