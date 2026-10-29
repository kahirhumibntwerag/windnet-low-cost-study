"""H10 mean-one fast-wind loss experiment, restricted to training Folds 1-4.

Imports are data-independent. Only explicit run() calls load verified training
bundles or fit models. H8 and H9 source/output artifacts are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    from . import fast_loss as h9
    from .train import FOLDS, feature_matrices, forecast_report, load_eve_bundle, load_training_bundle, sha256
    from .models import build_model
    from .metrics import deduplicate_issue_target_pairs
except ImportError:
    import fast_loss as h9
    from train import FOLDS, feature_matrices, forecast_report, load_eve_bundle, load_training_bundle, sha256
    from models import build_model
    from metrics import deduplicate_issue_target_pairs


H9_REFERENCE = "h9_reference"
WIND_REFERENCE = "starting_wind_reference"
REGISTERED_WIND_RMSE = 80.72360614459454
REGISTERED_H9_FAST_RMSE = 123.90929011714888


def normalized_weights(observed: np.ndarray, multiplier: int) -> tuple[np.ndarray, dict]:
    """Literal high-speed weights divided by their fitting-fold mean."""
    y = np.asarray(observed, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or multiplier not in h9.LOSS_WEIGHTS:
        raise ValueError("Invalid fitting targets or unregistered loss multiplier")
    raw = 1.0 + (multiplier - 1.0) * (y >= 500.0)
    mean = float(raw.mean())
    weights = raw / mean
    return weights, {"multiplier": multiplier, "n_fit": len(y), "n_fast500": int((y >= 500).sum()),
                     "raw_weight_mean": mean, "normalized_weight_mean": float(weights.mean()),
                     "normalized_weight_sum": float(weights.sum()),
                     "normalized_weight_min": float(weights.min()), "normalized_weight_max": float(weights.max())}


def load_prior(root: Path, stage: str, bundle: dict, versions: dict) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Verify only training-only development outputs before using references."""
    results_path, predictions_path = root / f"{stage}_results.json", root / f"{stage}_predictions.npz"
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("experiment") != stage.upper() or result.get("status") != "completed":
        raise ValueError(f"Expected completed {stage.upper()} training-only outputs")
    previous = result.get("environment", {})
    if any(previous.get(key) != versions[key] for key in ("python", "packages")):
        raise ValueError(f"{stage.upper()} environment differs from H10")
    if result.get("prediction_file_sha256") != sha256(predictions_path):
        raise ValueError(f"{stage.upper()} predictions differ from their recorded hash")
    for name in ("train_only.npz", "train_only.manifest.json"):
        if result.get("input_hashes", {}).get(name) != sha256(root / name):
            raise ValueError(f"{stage.upper()} used a different verified training bundle")
    if result.get("eve", {}).get("feature_bundle_sha256") != sha256(root / "eve_features.npz"):
        raise ValueError(f"{stage.upper()} used different EVE features")
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"rowid", "fold", "block", "offset", "date", "observed", "model_names", "predictions", "selected"}
        if not required.issubset(archive.files):
            raise ValueError(f"Incomplete {stage.upper()} prediction schema")
        if not np.isin(archive["fold"], FOLDS).all() or set(np.unique(archive["fold"]).tolist()) != set(FOLDS):
            raise ValueError(f"{stage.upper()} has excluded or missing folds")
        for name, target in (("rowid", "rowid"), ("fold", "fold"), ("block", "block"),
                             ("offset", "offset"), ("date", "date"), ("observed", "y")):
            if not np.array_equal(archive[name], bundle[target]):
                raise ValueError(f"{stage.upper()} training rows/targets differ: {name}")
        names = [str(name) for name in archive["model_names"]]
        values = archive["predictions"]
        if len(set(names)) != len(names) or values.shape != (len(bundle["y"]), len(names)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {stage.upper()} forecast matrix")
        selected = str(archive["selected"].item())
        if selected != result.get("selected") or selected not in names:
            raise ValueError(f"{stage.upper()} selected identifiers disagree")
        predictions = {name: values[:, i].astype(float).copy() for i, name in enumerate(names)}
    return result, predictions, {"results_sha256": sha256(results_path),
                                 "predictions_sha256": sha256(predictions_path),
                                 "selected_source_id": selected, "environment": previous}


def matched_metrics(normalized: dict, literal: dict) -> dict:
    """Matched-window score changes; negative error deltas favor normalization."""
    result = {}
    for label in ("all", "ge_500", "ge_600"):
        if label == "all":
            a, b = normalized["metrics"]["overall"], literal["metrics"]["overall"]
        else:
            a = normalized["metrics"]["fast_wind"][label]["continuous_observed_fast"]
            b = literal["metrics"]["fast_wind"][label]["continuous_observed_fast"]
        keys = ("n", "mse", "rmse", "mae", "bias", "r")
        result[label] = {"normalized": {key: a.get(key) for key in keys},
                         "literal": {key: b.get(key) for key in keys},
                         "delta_normalized_minus_literal": {
                             key: None if a.get(key) is None or b.get(key) is None else a[key] - b[key]
                             for key in ("mse", "rmse", "mae", "bias", "r")}}
        if label != "all":
            da, db = normalized["metrics"]["fast_wind"][label]["detection"], literal["metrics"]["fast_wind"][label]["detection"]
            result[label]["detection_delta_normalized_minus_literal"] = {
                key: None if da.get(key) is None or db.get(key) is None else da[key] - db[key]
                for key in ("pod", "far", "fpr", "csi", "tss", "hss", "frequency_bias")}
    return result


def select_candidate(reports: dict, metadata: dict, candidate_ids: list[str]) -> tuple[str, dict, dict]:
    wind = reports[WIND_REFERENCE]["metrics"]["overall"]
    fast = reports[H9_REFERENCE]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
    anchors = {"wind_overall_rmse_km_s": wind["rmse"], "wind_overall_mse_km2_s2": wind["mse"],
               "h9_fast500_rmse_km_s": fast["rmse"], "h9_fast500_mse_km2_s2": fast["mse"],
               "overall_limit_km_s": None if wind["rmse"] is None else 1.01 * wind["rmse"],
               "fast500_limit_km_s": None if fast["rmse"] is None else 1.02 * fast["rmse"]}
    checks = {}
    for name in candidate_ids:
        overall = reports[name]["metrics"]["overall"]
        high = reports[name]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
        defined = wind["mse"] is not None and wind["mse"] > 0 and fast["mse"] is not None and fast["mse"] > 0 and high["mse"] is not None
        overall_ok = overall["rmse"] is not None and anchors["overall_limit_km_s"] is not None and overall["rmse"] <= anchors["overall_limit_km_s"]
        fast_ok = high["rmse"] is not None and anchors["fast500_limit_km_s"] is not None and high["rmse"] <= anchors["fast500_limit_km_s"]
        objective = 0.5 * overall["mse"] / wind["mse"] + 0.5 * high["mse"] / fast["mse"] if defined else None
        checks[name] = {"eligible": bool(defined and overall_ok and fast_ok), "overall_guard_passed": bool(overall_ok),
                        "fast_guard_passed": bool(fast_ok), "J": objective}
    eligible = [name for name in candidate_ids if checks[name]["eligible"]]
    def key(name: str) -> tuple:
        spec = metadata[name]
        return (checks[name]["J"], spec["feature_columns"], 2 if "members" in spec else 1, name)
    return (min(eligible, key=key) if eligible else H9_REFERENCE), checks, anchors


def run(bundle_directory: str | Path, output_directory: str | Path, stage: str = "h10") -> dict:
    started = time.perf_counter()
    if stage != "h10":
        raise ValueError("Only preregistered H10 is supported")
    root, output = Path(bundle_directory).resolve(), Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    if any((output / name).exists() for name in ("h10_results.json", "h10_predictions.npz")):
        raise ValueError("Refusing to overwrite H10 outputs")
    bundle, versions = load_training_bundle(root), h9.environment()
    eve = load_eve_bundle(root, bundle)
    h8_result, h8_predictions, h8_provenance = load_prior(root, "h8", bundle, versions)
    h9_result, literal_predictions, h9_provenance = load_prior(root, "h9", bundle, versions)
    if set(np.unique(bundle["fold"]).tolist()) != set(FOLDS):
        raise ValueError("All four allowed training folds are required")
    source_wind, source_h9 = h8_result.get("incumbent"), h9_result.get("selected")
    if source_wind not in h8_predictions or source_h9 not in literal_predictions:
        raise ValueError("Registered anchor forecasts are missing")
    wind_reference, h9_reference = h8_predictions[source_wind], literal_predictions[source_h9]
    h8_selected = h8_result["selected"]
    required_members = ["eve_multi_ridge__ridge_alpha_1000", "eve_multi_tree__hgb_leaves_7_minleaf_30"]
    selected_spec = h8_result.get("candidate_specs", {}).get(h8_selected, {})
    if selected_spec.get("members") != required_members or selected_spec.get("weights") != [0.25, 0.75]:
        raise ValueError("H8 selected pair differs from the preregistered reproduction target")
    metadata = h9.definitions(eve["fields"])
    candidate_ids = list(metadata)
    for name, spec in metadata.items():
        if h9_result.get("candidate_specs", {}).get(name) != spec or name not in literal_predictions:
            raise ValueError("H9 literal reference recipe differs from fixed H10 recipe")
    references = {"latest_persistence": bundle["short"][:, -1].copy(), "persistence_27day": bundle["recurrence"][:, 21].copy()}
    initial_wind = forecast_report(bundle["y"], wind_reference, references)
    initial_h9 = forecast_report(bundle["y"], h9_reference, references)
    actual_wind = initial_wind["metrics"]["overall"]["rmse"]
    actual_fast = initial_h9["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]["rmse"]
    if actual_fast is None or not np.isclose(actual_wind, REGISTERED_WIND_RMSE, rtol=0, atol=1e-8) or not np.isclose(actual_fast, REGISTERED_H9_FAST_RMSE, rtol=0, atol=1e-8):
        raise ValueError("Training-only reference scores differ from preregistered H10 anchors")
    n = len(bundle["y"])
    predictions = {h9.pair_id(group, multiplier): np.full(n, np.nan) for group in h9.GROUPS for multiplier in h9.LOSS_WEIGHTS}
    split_audit = []
    for fold in FOLDS:
        val, other = bundle["fold"] == fold, bundle["fold"] != fold
        validation_dates = np.unique(bundle["date"][val])
        purged = np.unique(bundle["block"][other & np.isin(bundle["date"], validation_dates)])
        fit = other & bundle["fit_eligible"].astype(bool) & ~np.isin(bundle["block"], purged)
        if not fit.any() or not val.any() or np.intersect1d(bundle["date"][fit], validation_dates).size:
            raise ValueError("Invalid training-only purged fold")
        full_x, full_v, masks = feature_matrices(bundle, eve, "multi", fit, val, fold, True)
        weights_by_multiplier = {multiplier: normalized_weights(bundle["y"][fit], multiplier) for multiplier in h9.LOSS_WEIGHTS}
        for group in h9.GROUPS:
            indexes = np.r_[np.arange(78), 78 + h9.feature_indexes(eve["fields"], group)]
            x, v = full_x[:, indexes], full_v[:, indexes]
            for multiplier in h9.LOSS_WEIGHTS:
                weights, _ = weights_by_multiplier[multiplier]
                sample_weight = None if multiplier == 1 else weights
                ridge, tree = h9.build_model(h9.RIDGE), h9.build_model(h9.TREE)
                ridge.fit(x, bundle["y"][fit], sample_weight=sample_weight)
                tree.fit(x, bundle["y"][fit], sample_weight=sample_weight)
                p = 0.25 * ridge.predict(v) + 0.75 * tree.predict(v)
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite H10 forecasts")
                predictions[h9.pair_id(group, multiplier)][val] = p
        split_audit.append({"validation_fold": fold, "fit_rows": int(fit.sum()), "validation_rows": int(val.sum()),
                            "purged_blocks": purged.astype(int).tolist(), "feature_masks_all_eve": masks,
                            "normalization_weights": {str(m): audit for m, (_, audit) in weights_by_multiplier.items()}})
        print(json.dumps({"progress": "inner_fold_complete", "fold": fold, "fixed_pair_fits": 12, "estimator_fits": 24}), flush=True)
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Incomplete H10 OOF forecasts")
    reproduction_id = h9.pair_id("wind_all_eve", 1)
    reproduction_error = float(np.max(np.abs(predictions[reproduction_id] - h8_predictions[h8_selected])))
    if reproduction_error > 1e-8:
        raise ValueError(f"H8 unweighted reproduction exceeds tolerance: {reproduction_error:g} km/s")
    for name, spec in metadata.items():
        if "members" in spec:
            left, right = spec["members"]
            a, b = spec["weights"]
            predictions[name] = a * predictions[left] + b * predictions[right]
        spec["loss_weight_normalization"] = "raw weights divided by their fitting-fold mean"
        spec["sample_weight_scope"] = "weights passed to learner; explicit ridge imputer/scaler remain unweighted; HGB also uses weights in internal histogram binning"
    if set(predictions) != set(candidate_ids):
        raise AssertionError("H10 finite grid changed")
    predictions[H9_REFERENCE], predictions[WIND_REFERENCE] = h9_reference, wind_reference
    metadata[H9_REFERENCE] = {"id": H9_REFERENCE, "family": "h9_reference", "uses_eve": h9_result["candidate_specs"][source_h9]["uses_eve"],
                              "source_id": source_h9, "source_stage": "H9", "reference_label": "unbalanced fast-wind specialist",
                              "feature_columns": h9_result["candidate_specs"][source_h9]["feature_columns"]}
    metadata[WIND_REFERENCE] = {"id": WIND_REFERENCE, "family": "starting_wind_reference", "uses_eve": False,
                                "source_id": source_wind, "source_stage": "H8", "feature_columns": h8_result["candidate_specs"][source_wind].get("feature_columns", 78)}
    reports = {name: forecast_report(bundle["y"], p, references) for name, p in predictions.items()}
    selected, eligibility, anchors = select_candidate(reports, metadata, candidate_ids)
    literal_reports = {name: forecast_report(bundle["y"], literal_predictions[name], references) for name in candidate_ids}
    matched = {name: matched_metrics(reports[name], literal_reports[name]) for name in candidate_ids}
    dates = bundle["date"].astype("datetime64[D]")
    issues = (dates - np.timedelta64(3, "D")).astype(str)
    unique = deduplicate_issue_target_pairs(issues.tolist(), dates.astype(str).tolist(), bundle["y"], {**predictions, **references})
    unique_refs = {name: unique["forecasts"][name] for name in references}
    for name in reports:
        reports[name]["unique_issue_target_day"] = forecast_report(unique["observed"], unique["forecasts"][name], unique_refs)
    by_fold = {name: {str(fold): forecast_report(bundle["y"][bundle["fold"] == fold], p[bundle["fold"] == fold],
                {key: ref[bundle["fold"] == fold] for key, ref in references.items()}) for fold in FOLDS} for name, p in predictions.items()}
    best_by_family = {group: min([name for name in candidate_ids if metadata[name]["family"] == group],
                                 key=lambda name: (reports[name]["metrics"]["overall"]["rmse"], name)) for group in h9.GROUPS}
    source_names = ("normalized_loss.py", "fast_loss.py", "train.py", "models.py", "metrics.py", "prepare_training.py", "fetch_eve.py")
    input_names = ("train_only.npz", "train_only.manifest.json", "eve_features.npz", "h8_results.json", "h8_predictions.npz", "h9_results.json", "h9_predictions.npz")
    result = {"experiment": "H10", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_scope": "Training Folds 1-4 only; selected CV scores are optimistic development estimates, not final-test evidence.",
              "selection_unit": "pooled original validation windows; all fourteen offsets retained",
              "hypothesis": "Mean-one fast-wind weights improve the overall/fast-wind tradeoff while keeping total training loss scale fixed.",
              "incumbent": H9_REFERENCE, "selected": selected, "improvement_retained": selected != H9_REFERENCE,
              "selected_label": "unbalanced fast-wind specialist retained" if selected == H9_REFERENCE else "eligible normalized-loss tradeoff candidate",
              "eve_retained": metadata[selected]["uses_eve"], "candidate_count": len(reports), "new_candidate_count": len(candidate_ids),
              "single_candidate_count": 12, "candidate_specs": metadata, "candidate_reports": reports,
              "best_by_family": best_by_family, "best_models_by_fold": by_fold, "candidate_eligibility": eligibility, "eve_eligibility": eligibility,
              "normalized_vs_literal": matched,
              "feature_ablations_unweighted": {group: {"candidate": h9.pair_id(group, 1), "loss_multiplier": 1, "model_weights": [0.25, 0.75]} for group in h9.GROUPS},
              "thresholds": {"primary_km_s": 500, "secondary_km_s": 600, "secondary_used_for_selection": False},
              "selection_rule": {"overall_RMSE_ratio_max_vs_starting_wind": 1.01, "fast500_RMSE_ratio_max_vs_H9": 1.02,
                                 "objective": "0.5*MSE_all/MSE_all_starting_wind + 0.5*MSE_fast500/MSE_fast500_H9",
                                 "tie_break": "fewer feature columns, then fewer model pairs, then candidate identifier",
                                 "fallback": "retain H9 reference and label unbalanced fast-wind specialist"},
              "actual_anchors": anchors, "registered_anchors": {"wind_overall_rmse_km_s": REGISTERED_WIND_RMSE, "h9_fast500_rmse_km_s": REGISTERED_H9_FAST_RMSE},
              "h8_provenance": h8_provenance, "h9_provenance": h9_provenance,
              "h8_reproduction": {"candidate": reproduction_id, "max_absolute_difference_km_s": reproduction_error, "tolerance_km_s": 1e-8, "passed": True},
              "split_audit": split_audit,
              "secondary_deduplication": {"n_input": unique["n_input"], "n_unique_pairs": unique["n_unique_pairs"], "aggregation": unique["aggregation"], "issue_date": "nominal target minus three calendar days"},
              "persistence_definition": h9_result["persistence_definition"], "eve": h9_result["eve"],
              "loss_weighting": {"observed_threshold_km_s": 500, "multipliers": list(h9.LOSS_WEIGHTS), "weighted_pair_fractions": list(h9.MIX_WEIGHTS),
                                 "weight_normalization": "raw = 1+(multiplier-1)*(y_fit>=500); weight=raw/raw.mean() within each fitting fold",
                                 "sample_weight_scope": "weights passed to learner; explicit ridge imputer/scaler and evaluation remain unweighted",
                                 "histogram_tree_details": "HGB receives sample weights in internal histogram binning as well as gradients/Hessians; mean-one normalization preserves relative weights but changes gradient/Hessian scale relative to fixed L2"},
              "environment": versions, "input_hashes": {name: sha256(root / name) for name in input_names},
              "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in source_names if Path(__file__).with_name(name).exists()},
              "seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    names = list(predictions)
    prediction_path = output / "h10_predictions.npz"
    np.savez_compressed(prediction_path, rowid=bundle["rowid"], fold=bundle["fold"], block=bundle["block"], offset=bundle["offset"], date=bundle["date"],
                        issue=issues, observed=bundle["y"], model_names=np.asarray(names), predictions=np.column_stack([predictions[name] for name in names]),
                        latest_persistence=references["latest_persistence"], persistence_27day=references["persistence_27day"],
                        incumbent=np.asarray(H9_REFERENCE), selected=np.asarray(selected))
    result["prediction_file_sha256"] = sha256(prediction_path)
    temporary = output / "h10_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "h10_results.json")
    print(json.dumps({"progress": "complete", "new_candidate_count": len(candidate_ids), "outputs": ["h10_results.json", "h10_predictions.npz"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Verified training-only bundle, EVE features and frozen H8/H9 outputs")
    parser.add_argument("--output", type=Path, required=True, help="Directory without existing H10 outputs; may equal bundle directory")
    parser.add_argument("--stage", choices=("h10",), default="h10")
    args = parser.parse_args()
    run(args.bundle, args.output, args.stage)


if __name__ == "__main__":
    main()
