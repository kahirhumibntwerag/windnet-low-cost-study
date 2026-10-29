"""H9 training-only, preregistered fast-wind loss and feature ablations.

No files are read and no estimators fit on import. Scientific execution is
restricted to verified training-only bundles and their H8 development outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

try:
    from .train import (FOLDS, feature_matrices, forecast_report,
                        load_eve_bundle, load_training_bundle, sha256)
    from .models import build_model
    from .metrics import deduplicate_issue_target_pairs
except ImportError:
    from train import (FOLDS, feature_matrices, forecast_report,
                       load_eve_bundle, load_training_bundle, sha256)
    from models import build_model
    from metrics import deduplicate_issue_target_pairs


GROUPS = ("wind_only", "wind_esp", "wind_lines", "wind_all_eve")
LOSS_WEIGHTS = (1, 2, 4)
MIX_WEIGHTS = (0.25, 0.5, 0.75)
RIDGE = "ridge_alpha_1000"
TREE = "hgb_leaves_7_minleaf_30"
INCUMBENT_ID = "h8_incumbent"


def feature_indexes(fields: list[str], group: str) -> np.ndarray:
    """Indices in the H8 62-column EVE construction; wind columns added later."""
    if group not in GROUPS:
        raise ValueError("Unknown registered feature ablation")
    count = len(fields)
    if group == "wind_only":
        return np.asarray([], dtype=int)
    keep = [i for i, name in enumerate(fields)
            if group == "wind_all_eve"
            or (group == "wind_esp" and name.startswith("diode"))
            or (group == "wind_lines" and not name.startswith("diode"))]
    columns = [lag * count + i for lag in range(4) for i in keep]
    columns.extend(4 * count + i for i in keep)
    columns.extend(5 * count + i for i in keep)
    if group in ("wind_lines", "wind_all_eve"):
        columns.extend(range(6 * count, 6 * count + 8))
    expected = {"wind_esp": 18, "wind_lines": 44, "wind_all_eve": 62}[group]
    if len(columns) != expected:
        raise ValueError("H9 requires the same nine-channel EVE bundle as H8")
    return np.asarray(columns, dtype=int)


def pair_id(group: str, multiplier: int) -> str:
    return f"{group}__loss_weight_{multiplier}"


def definitions(fields: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for group in GROUPS:
        indexes = feature_indexes(fields, group)
        for multiplier in LOSS_WEIGHTS:
            name = pair_id(group, multiplier)
            result[name] = {"id": name, "family": group, "feature_group": group,
                            "uses_eve": group != "wind_only", "feature_columns": 78 + len(indexes),
                            "eve_feature_indexes": indexes.tolist(),
                            "loss_weight_observed_ge_500": multiplier,
                            "loss_weight_other": 1,
                            "ridge": RIDGE, "tree": TREE, "model_weights": [0.25, 0.75],
                            "sample_weight_scope": "learner loss only; preprocessing remains unweighted"}
        for multiplier in (2, 4):
            for weight in MIX_WEIGHTS:
                name = f"{group}__weighted_{multiplier}_mix_{weight:g}"
                result[name] = {"id": name, "family": group, "feature_group": group,
                                "uses_eve": group != "wind_only", "feature_columns": 78 + len(indexes),
                                "members": [pair_id(group, multiplier), pair_id(group, 1)],
                                "weights": [weight, 1.0 - weight],
                                "weighted_pair_fraction": weight, "weighted_loss_multiplier": multiplier}
    if len(result) != 36:
        raise AssertionError("Registered H9 grid changed")
    return result


def environment() -> dict:
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "packages": packages, "neural_models_used": False}


def load_h8(root: Path, bundle: dict, current_environment: dict) -> tuple[np.ndarray, dict]:
    """Require row identity, allowed folds, and the exact H8 selected OOF column."""
    results_path, predictions_path = root / "h8_results.json", root / "h8_predictions.npz"
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("experiment") != "H8" or result.get("status") != "completed":
        raise ValueError("H9 requires completed H8 training-only outputs")
    prior_environment = result.get("environment", {})
    if any(prior_environment.get(key) != current_environment[key] for key in ("python", "packages")):
        raise ValueError("H8/H9 Python and numerical package versions must match for reproduction")
    if result.get("prediction_file_sha256") != sha256(predictions_path):
        raise ValueError("H8 predictions differ from their recorded hash")
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"rowid", "fold", "block", "offset", "date", "observed", "model_names", "predictions", "selected"}
        if not required.issubset(archive.files):
            raise ValueError("Incomplete H8 training-only prediction schema")
        if not np.isin(archive["fold"], FOLDS).all() or set(np.unique(archive["fold"]).tolist()) != set(FOLDS):
            raise ValueError("H8 predictions contain an excluded or missing fold")
        for name, target in (("rowid", "rowid"), ("fold", "fold"), ("block", "block"),
                             ("offset", "offset"), ("date", "date"), ("observed", "y")):
            if not np.array_equal(archive[name], bundle[target]):
                raise ValueError(f"H8 rows/targets differ from the isolated training bundle: {name}")
        selected = str(archive["selected"].item())
        if selected != result.get("selected"):
            raise ValueError("H8 selected identifiers disagree")
        names = [str(name) for name in archive["model_names"]]
        if len(set(names)) != len(names) or selected not in names:
            raise ValueError("H8 selected forecast is absent or ambiguous")
        predictions = archive["predictions"]
        if predictions.shape != (len(bundle["y"]), len(names)):
            raise ValueError("H8 forecast dimensions differ")
        incumbent = predictions[:, names.index(selected)].astype(float).copy()
    if not np.isfinite(incumbent).all():
        raise ValueError("H8 selected OOF forecast is incomplete")
    spec = result.get("candidate_specs", {}).get(selected, {})
    members = ["eve_multi_ridge__ridge_alpha_1000", "eve_multi_tree__hgb_leaves_7_minleaf_30"]
    if spec.get("members") != members or spec.get("weights") != [0.25, 0.75]:
        raise ValueError("H8 incumbent differs from the preregistered fixed 0.25 ridge/0.75 tree pair")
    return incumbent, {"selected_source_id": selected, "results_sha256": sha256(results_path),
                       "predictions_sha256": sha256(predictions_path), "environment": prior_environment}


def select_candidate(reports: dict, candidate_ids: list[str]) -> tuple[str, dict]:
    incumbent_all = reports[INCUMBENT_ID]["metrics"]["overall"]
    incumbent_fast = reports[INCUMBENT_ID]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
    checks = {}
    for name in candidate_ids:
        overall = reports[name]["metrics"]["overall"]
        fast = reports[name]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
        defined = (incumbent_all["mse"] is not None and incumbent_all["mse"] > 0
                   and incumbent_fast["mse"] is not None and incumbent_fast["mse"] > 0
                   and fast["mse"] is not None and overall["mse"] is not None)
        overall_ok = overall["rmse"] is not None and overall["rmse"] <= 1.01 * incumbent_all["rmse"]
        fast_ok = (fast["rmse"] is not None and incumbent_fast["rmse"] is not None
                   and fast["rmse"] <= 0.98 * incumbent_fast["rmse"])
        objective = (0.5 * overall["mse"] / incumbent_all["mse"]
                     + 0.5 * fast["mse"] / incumbent_fast["mse"] if defined else None)
        checks[name] = {"eligible": bool(defined and overall_ok and fast_ok),
                        "overall_guard_passed": bool(overall_ok), "fast_guard_passed": bool(fast_ok), "J": objective}
    eligible = [name for name in candidate_ids if checks[name]["eligible"]]
    # Exact objective ties prefer fewer solar features and fewer fitted pairs.
    # This order is fixed before execution, not chosen from score differences.
    def tie_order(name):
        group = name.split('__')[0]
        feature_complexity = GROUPS.index(group)
        mixture_complexity = int('_mix_' in name)
        return (checks[name]['J'], feature_complexity, mixture_complexity, name)
    selected = min(eligible, key=tie_order) if eligible else INCUMBENT_ID
    return selected, checks


def run(bundle_directory: str | Path, output_directory: str | Path, stage: str = "h9") -> dict:
    started = time.perf_counter()
    if stage != "h9":
        raise ValueError("This runner implements only preregistered H9")
    root, output = Path(bundle_directory).resolve(), Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    if any((output / name).exists() for name in ("h9_results.json", "h9_predictions.npz")):
        raise ValueError("Refusing to overwrite existing H9 outputs")
    bundle = load_training_bundle(root)
    eve = load_eve_bundle(root, bundle)
    versions = environment()
    incumbent, h8 = load_h8(root, bundle, versions)
    if set(np.unique(bundle["fold"]).tolist()) != set(FOLDS):
        raise ValueError("H9 requires all four allowed training folds")
    metadata = definitions(eve["fields"])
    n = len(bundle["y"])
    predictions = {pair_id(group, multiplier): np.full(n, np.nan)
                   for group in GROUPS for multiplier in LOSS_WEIGHTS}
    split_audit = []
    for fold in FOLDS:
        val = bundle["fold"] == fold
        other = ~val
        dates = np.unique(bundle["date"][val])
        purged_blocks = np.unique(bundle["block"][other & np.isin(bundle["date"], dates)])
        fit = other & bundle["fit_eligible"].astype(bool) & ~np.isin(bundle["block"], purged_blocks)
        if not fit.any() or not val.any() or np.intersect1d(bundle["date"][fit], dates).size:
            raise ValueError("Invalid purged training-only fold")
        full_x, full_v, masks = feature_matrices(bundle, eve, "multi", fit, val, fold, True)
        counts = {}
        for group in GROUPS:
            extra = feature_indexes(eve["fields"], group)
            indexes = np.r_[np.arange(78), 78 + extra]
            x, v = full_x[:, indexes], full_v[:, indexes]
            for multiplier in LOSS_WEIGHTS:
                weights = np.where(bundle["y"][fit] >= 500.0, float(multiplier), 1.0)
                # Omit all-one weights to reproduce H8's unweighted solver path exactly.
                sample_weight = None if multiplier == 1 else weights
                ridge, tree = build_model(RIDGE), build_model(TREE)
                ridge.fit(x, bundle["y"][fit], sample_weight=sample_weight)
                tree.fit(x, bundle["y"][fit], sample_weight=sample_weight)
                prediction = 0.25 * ridge.predict(v) + 0.75 * tree.predict(v)
                if not np.isfinite(prediction).all():
                    raise ValueError("A fixed H9 pair produced nonfinite forecasts")
                predictions[pair_id(group, multiplier)][val] = prediction
                counts[str(multiplier)] = {"fit_fast500": int((bundle["y"][fit] >= 500).sum()),
                                            "weight_sum": float(weights.sum())}
        split_audit.append({"validation_fold": fold, "fit_rows": int(fit.sum()),
                            "validation_rows": int(val.sum()), "purged_blocks": purged_blocks.astype(int).tolist(),
                            "feature_masks_all_eve": masks, "fitting_loss_weights": counts})
        print(json.dumps({"progress": "inner_fold_complete", "fold": fold,
                          "fixed_pair_fits": 12, "estimator_fits": 24}), flush=True)
    reproduction_id = pair_id("wind_all_eve", 1)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Incomplete H9 OOF coverage")
    difference = float(np.max(np.abs(predictions[reproduction_id] - incumbent)))
    if difference > 1e-8:
        raise ValueError(f"H8 incumbent reproduction failed: max absolute difference {difference:g} > 1e-8 km/s")
    for name, spec in metadata.items():
        if "members" in spec:
            left, right = spec["members"]
            a, b = spec["weights"]
            predictions[name] = a * predictions[left] + b * predictions[right]
    candidate_ids = list(metadata)
    if set(predictions) != set(candidate_ids):
        raise AssertionError("H9 forecast and candidate grids differ")
    predictions[INCUMBENT_ID] = incumbent
    metadata[INCUMBENT_ID] = {"id": INCUMBENT_ID, "family": "h8_reference", "uses_eve": True,
                              "feature_columns": 140, "source_id": h8["selected_source_id"],
                              "model_weights": [0.25, 0.75], "ridge": RIDGE, "tree": TREE}
    references = {"latest_persistence": bundle["short"][:, -1].copy(),
                  "persistence_27day": bundle["recurrence"][:, 21].copy()}
    reports = {name: forecast_report(bundle["y"], values, references) for name, values in predictions.items()}
    selected, eligibility = select_candidate(reports, candidate_ids)
    dates = bundle["date"].astype("datetime64[D]")
    issues = (dates - np.timedelta64(3, "D")).astype(str)
    unique = deduplicate_issue_target_pairs(issues.tolist(), dates.astype(str).tolist(), bundle["y"], {**predictions, **references})
    unique_refs = {name: unique["forecasts"][name] for name in references}
    for name in reports:
        reports[name]["unique_issue_target_day"] = forecast_report(unique["observed"], unique["forecasts"][name], unique_refs)
    by_fold = {name: {str(fold): forecast_report(bundle["y"][bundle["fold"] == fold], values[bundle["fold"] == fold],
                {key: ref[bundle["fold"] == fold] for key, ref in references.items()}) for fold in FOLDS}
               for name, values in predictions.items()}
    best_by_family = {group: min([name for name in candidate_ids if metadata[name]["family"] == group],
                                 key=lambda name: (reports[name]["metrics"]["overall"]["rmse"], name)) for group in GROUPS}
    feature_ablation_ids = {group: pair_id(group, 1) for group in GROUPS}
    ablations = {}
    control = reports[pair_id("wind_only", 1)]["metrics"]
    for group, name in feature_ablation_ids.items():
        scores = reports[name]["metrics"]
        deltas = {"all_rmse": scores["overall"]["rmse"] - control["overall"]["rmse"],
                  "all_mae": scores["overall"]["mae"] - control["overall"]["mae"]}
        for threshold in (500, 600):
            current = scores["fast_wind"][f"ge_{threshold}"]["continuous_observed_fast"]["rmse"]
            baseline = control["fast_wind"][f"ge_{threshold}"]["continuous_observed_fast"]["rmse"]
            deltas[f"fast{threshold}_rmse"] = None if current is None or baseline is None else current - baseline
        ablations[group] = {"candidate": name, "delta_from_unweighted_wind_only_km_s": deltas,
                            "loss_multiplier": 1, "model_weights": [0.25, 0.75]}
    code_names = ("fast_loss.py", "train.py", "models.py", "metrics.py", "prepare_training.py", "fetch_eve.py")
    result = {"experiment": "H9", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_scope": "Training Folds 1-4 only; selected CV scores are optimistic development estimates, not final-test evidence.",
              "selection_unit": "pooled original validation windows; all fourteen offsets retained",
              "incumbent": INCUMBENT_ID, "selected": selected, "improvement_retained": selected != INCUMBENT_ID,
              "eve_retained": metadata[selected]["uses_eve"], "candidate_count": len(reports),
              "new_candidate_count": len(candidate_ids), "single_candidate_count": 12,
              "candidate_specs": metadata, "candidate_reports": reports,
              "best_by_family": best_by_family, "best_models_by_fold": by_fold,
              "candidate_eligibility": eligibility, "eve_eligibility": eligibility,
              "feature_ablations_unweighted": ablations, "split_audit": split_audit,
              "thresholds": {"primary_km_s": 500, "secondary_km_s": 600, "secondary_used_for_selection": False},
              "selection_rule": {"overall_RMSE_ratio_max": 1.01, "fast500_RMSE_ratio_max": 0.98,
                                 "objective": "0.5*MSE_all/MSE_all_H8 + 0.5*MSE_fast500/MSE_fast500_H8",
                                 "fallback": "retain exact H8 selected forecast if no candidate is eligible"},
              "h8_provenance": h8,
              "h8_reproduction": {"candidate": reproduction_id, "max_absolute_difference_km_s": difference,
                                   "tolerance_km_s": 1e-8, "passed": True},
              "secondary_deduplication": {"n_input": unique["n_input"], "n_unique_pairs": unique["n_unique_pairs"],
                                           "aggregation": unique["aggregation"], "issue_date": "nominal target minus three calendar days"},
              "persistence_definition": {"latest_persistence": "last of four original chronological history entries; gaps not repaired",
                                         "persistence_27day": "recurrence column 21: 4 original + 14 recent + index3 in 24..31 day band"},
              "eve": {"features": len(eve["names"]), "names": eve["names"], "fields": eve["fields"],
                      "lag_calendar_days": [5, 6, 7, 8], "feature_bundle_sha256": eve["sha256"], "manifest": eve["manifest"]},
              "loss_weighting": {"observed_threshold_km_s": 500, "multipliers": list(LOSS_WEIGHTS),
                                 "weighted_pair_fractions": list(MIX_WEIGHTS), "weight_normalization": "none",
                                 "sample_weight_scope": "training learner loss only, not preprocessing or evaluation"},
              "environment": versions,
              "input_hashes": {name: sha256(root / name) for name in ("train_only.npz", "train_only.manifest.json", "eve_features.npz", "h8_results.json", "h8_predictions.npz")},
              "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in code_names if Path(__file__).with_name(name).exists()},
              "seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    names = list(predictions)
    predictions_path = output / "h9_predictions.npz"
    np.savez_compressed(predictions_path, rowid=bundle["rowid"], fold=bundle["fold"], block=bundle["block"],
                        offset=bundle["offset"], date=bundle["date"], issue=issues, observed=bundle["y"],
                        model_names=np.asarray(names), predictions=np.column_stack([predictions[name] for name in names]),
                        latest_persistence=references["latest_persistence"], persistence_27day=references["persistence_27day"],
                        incumbent=np.asarray(INCUMBENT_ID), selected=np.asarray(selected))
    result["prediction_file_sha256"] = sha256(predictions_path)
    temporary = output / "h9_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "h9_results.json")
    print(json.dumps({"progress": "complete", "new_candidate_count": len(candidate_ids),
                      "outputs": ["h9_results.json", "h9_predictions.npz"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Verified train-only bundle directory, EVE and H8 outputs")
    parser.add_argument("--output", type=Path, required=True, help="Directory without existing H9 outputs; H8 files may coexist")
    parser.add_argument("--stage", choices=("h9",), default="h9")
    args = parser.parse_args()
    run(args.bundle, args.output, args.stage)


if __name__ == "__main__":
    main()
