"""H11 fixed prior-champion feature and diversity comparisons, training only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

try:
    from .train import FOLDS, feature_matrices, forecast_report, load_eve_bundle, load_training_bundle, sha256
    from .metrics import deduplicate_issue_target_pairs
    from .fast_loss import GROUPS, definitions as h9_definitions, feature_indexes
    from .prior_champion import fit_prior_champion, recipe_metadata
except ImportError:
    from train import FOLDS, feature_matrices, forecast_report, load_eve_bundle, load_training_bundle, sha256
    from metrics import deduplicate_issue_target_pairs
    from fast_loss import GROUPS, definitions as h9_definitions, feature_indexes
    from prior_champion import fit_prior_champion, recipe_metadata


H9_REFERENCE = "h9_reference"
WIND_REFERENCE = "starting_wind_reference"
WIND_ANCHOR_RMSE = 80.72360614459454
PREREGISTRATION = "https://fixrelay.vercel.app/experiments/f978fd1e-c27c-4cf9-ae86-faaac54dbc0c"


def current_environment() -> dict:
    import torch
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "neural_models_used": True}


def load_frozen(root: Path, stage: str, bundle: dict, versions: dict) -> tuple[dict, dict, dict]:
    """Only frozen training-fold controls; their environments may differ."""
    json_path, npz_path = root / f"{stage}_results.json", root / f"{stage}_predictions.npz"
    result = json.loads(json_path.read_text(encoding="utf-8"))
    if result.get("experiment") != stage.upper() or result.get("status") != "completed":
        raise ValueError("Expected completed frozen development controls")
    if result.get("prediction_file_sha256") != sha256(npz_path):
        raise ValueError("Frozen control predictions differ from their recorded hash")
    for name in ("train_only.npz", "train_only.manifest.json"):
        if result.get("input_hashes", {}).get(name) != sha256(root / name):
            raise ValueError("Frozen control used a different isolated training bundle")
    if result.get("eve", {}).get("feature_bundle_sha256") != sha256(root / "eve_features.npz"):
        raise ValueError("Frozen control used a different EVE feature bundle")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"rowid", "fold", "block", "offset", "date", "observed", "model_names", "predictions", "selected", "incumbent"}
        if not required.issubset(archive.files):
            raise ValueError("Incomplete frozen prediction schema")
        if not np.isin(archive["fold"], FOLDS).all() or set(np.unique(archive["fold"]).tolist()) != set(FOLDS):
            raise ValueError("Frozen control includes an excluded or missing fold")
        for name, target in (("rowid", "rowid"), ("fold", "fold"), ("block", "block"),
                             ("offset", "offset"), ("date", "date"), ("observed", "y")):
            if not np.array_equal(archive[name], bundle[target]):
                raise ValueError(f"Frozen control row/target identity differs: {name}")
        names = [str(name) for name in archive["model_names"]]
        values = archive["predictions"]
        if len(set(names)) != len(names) or values.shape != (len(bundle["y"]), len(names)) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen forecast matrix")
        for name in ("selected", "incumbent"):
            identifier = str(archive[name].item())
            if identifier != result.get(name) or identifier not in names or identifier not in result.get("candidate_specs", {}):
                raise ValueError("Frozen control candidate identifiers disagree")
        predictions = {name: values[:, i].astype(float).copy() for i, name in enumerate(names)}
    previous = result.get("environment", {})
    return result, predictions, {"results_sha256": sha256(json_path), "predictions_sha256": sha256(npz_path),
                                 "environment": previous, "current_environment": versions,
                                 "environment_matches_current": {"python": previous.get("python") == versions["python"],
                                      "numerical_packages": all(previous.get("packages", {}).get(key) == versions["packages"].get(key)
                                                                 for key in ("numpy", "scipy", "scikit-learn"))},
                                 "interpretation": "frozen matched-row forecasts used as controls; not retrained in the H11 environment"}


def definitions(fields: list[str], h9_source: str, h9_spec: dict) -> dict:
    result = {}
    for i, group in enumerate(GROUPS):
        name = f"P{i}"
        result[name] = {"id": name, "family": "fixed_prior_champion", "feature_group": group,
                        "uses_eve": i > 0, "feature_columns": 78 + len(feature_indexes(fields, group)),
                        "fit_count": 5, "recipe": recipe_metadata()}
    h9_fit_count = 4 if "members" in h9_spec else 2
    for i, weight in enumerate((0.25, 0.5, 0.75), start=4):
        name = f"P{i}"
        result[name] = {"id": name, "family": "fixed_diversity_blend", "feature_group": "wind_plus_frozen_h9",
                        "uses_eve": bool(h9_spec["uses_eve"]),
                        "feature_columns": max(78, h9_spec["feature_columns"]), "fit_count": 5 + h9_fit_count,
                        "members": ["P0", H9_REFERENCE], "weights": [1.0 - weight, weight],
                        "h9_fraction": weight, "h9_source_id": h9_source}
    return result


def select_candidate(reports: dict, metadata: dict) -> tuple[str, dict, dict]:
    base_all = reports["P0"]["metrics"]["overall"]
    base_fast = reports["P0"]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
    wind = reports[WIND_REFERENCE]["metrics"]["overall"]
    anchors = {"P0_overall_rmse_km_s": base_all["rmse"], "P0_fast500_rmse_km_s": base_fast["rmse"],
               "starting_wind_rmse_km_s": wind["rmse"], "registered_starting_wind_rmse_km_s": WIND_ANCHOR_RMSE,
               "overall_improvement_limit_km_s": None if base_all["rmse"] is None else 0.99 * base_all["rmse"],
               "fast500_limit_km_s": None if base_fast["rmse"] is None else 1.01 * base_fast["rmse"],
               "cumulative_overall_limit_km_s": 1.01 * WIND_ANCHOR_RMSE}
    checks = {}
    for name in (f"P{i}" for i in range(1, 7)):
        overall = reports[name]["metrics"]["overall"]
        fast = reports[name]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
        defined = (base_all["mse"] is not None and base_all["mse"] > 0 and base_fast["mse"] is not None
                   and base_fast["mse"] > 0 and overall["mse"] is not None and fast["mse"] is not None)
        improved = overall["rmse"] is not None and anchors["overall_improvement_limit_km_s"] is not None and overall["rmse"] <= anchors["overall_improvement_limit_km_s"]
        fast_ok = fast["rmse"] is not None and anchors["fast500_limit_km_s"] is not None and fast["rmse"] <= anchors["fast500_limit_km_s"]
        cumulative = overall["rmse"] is not None and overall["rmse"] <= anchors["cumulative_overall_limit_km_s"]
        objective = 0.5 * overall["mse"] / base_all["mse"] + 0.5 * fast["mse"] / base_fast["mse"] if defined else None
        checks[name] = {"eligible": bool(defined and improved and fast_ok and cumulative),
                        "overall_improvement_guard_passed": bool(improved), "fast_guard_passed": bool(fast_ok),
                        "cumulative_overall_guard_passed": bool(cumulative), "J": objective}
    eligible = [name for name, check in checks.items() if check["eligible"]]
    selected = min(eligible, key=lambda name: (checks[name]["J"], metadata[name]["feature_columns"], metadata[name]["fit_count"], name)) if eligible else "P0"
    return selected, checks, anchors


def run(bundle_directory: str | Path, output_directory: str | Path, device: str = "auto", stage: str = "h11") -> dict:
    started = time.perf_counter()
    if stage != "h11":
        raise ValueError("Only preregistered H11 is supported")
    root, output = Path(bundle_directory).resolve(), Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    if any((output / name).exists() for name in ("h11_results.json", "h11_predictions.npz")):
        raise ValueError("Refusing to overwrite H11 outputs")
    bundle, versions = load_training_bundle(root), current_environment()
    eve = load_eve_bundle(root, bundle)
    h8_result, h8_predictions, h8_provenance = load_frozen(root, "h8", bundle, versions)
    h9_result, h9_predictions, h9_provenance = load_frozen(root, "h9", bundle, versions)
    if set(np.unique(bundle["fold"]).tolist()) != set(FOLDS):
        raise ValueError("H11 requires all four allowed training folds")
    expected = h9_definitions(eve["fields"])
    for name, spec in expected.items():
        if h9_result.get("candidate_specs", {}).get(name) != spec or name not in h9_predictions:
            raise ValueError("Frozen H9 recipe identity differs from its fixed protocol")
    h9_source, wind_source = h9_result["selected"], h8_result["incumbent"]
    h9_spec = h9_result["candidate_specs"][h9_source]
    if h8_result["candidate_specs"][wind_source].get("uses_eve"):
        raise ValueError("The starting wind anchor must be wind-only")
    metadata = definitions(eve["fields"], h9_source, h9_spec)
    n = len(bundle["y"])
    predictions = {f"P{i}": np.full(n, np.nan) for i in range(4)}
    component_predictions = {f"P{i}_{component}": np.full(n, np.nan)
                             for i in range(4) for component in ("ridge", "tree", "residual_neural", "neural_seed_19", "neural_seed_43", "neural_seed_101")}
    split_audit = []
    for fold in FOLDS:
        val, other = bundle["fold"] == fold, bundle["fold"] != fold
        dates = np.unique(bundle["date"][val])
        purged = np.unique(bundle["block"][other & np.isin(bundle["date"], dates)])
        fit = other & bundle["fit_eligible"].astype(bool) & ~np.isin(bundle["block"], purged)
        if not fit.any() or not val.any() or np.intersect1d(bundle["date"][fit], dates).size:
            raise ValueError("Invalid purged training-only split")
        full_x, full_v, masks = feature_matrices(bundle, eve, "multi", fit, val, fold, True)
        runtimes = {}
        for i, group in enumerate(GROUPS):
            indexes = np.r_[np.arange(78), 78 + feature_indexes(eve["fields"], group)]
            model = fit_prior_champion(full_x[:, indexes], bundle["y"][fit], device=device)
            components = model.predict_components(full_v[:, indexes])
            name = f"P{i}"
            predictions[name][val] = components["prediction"]
            for component in components:
                if component != "prediction":
                    component_predictions[f"{name}_{component}"][val] = components[component]
            runtimes[name] = model.runtime_
        split_audit.append({"validation_fold": fold, "fit_rows": int(fit.sum()), "validation_rows": int(val.sum()),
                            "purged_blocks": purged.astype(int).tolist(), "feature_masks_all_eve": masks,
                            "prior_champion_runtime": runtimes})
        print(json.dumps({"progress": "inner_fold_complete", "fold": fold, "fixed_feature_recipes_fit": 4}), flush=True)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Incomplete fixed prior-champion OOF coverage")
    predictions[H9_REFERENCE] = h9_predictions[h9_source]
    predictions[WIND_REFERENCE] = h8_predictions[wind_source]
    for i in range(4, 7):
        a, b = metadata[f"P{i}"]["weights"]
        predictions[f"P{i}"] = a * predictions["P0"] + b * predictions[H9_REFERENCE]
    metadata[H9_REFERENCE] = {"id": H9_REFERENCE, "family": "frozen_h9_control", "source_id": h9_source,
                              "feature_columns": h9_spec["feature_columns"], "uses_eve": h9_spec["uses_eve"],
                              "fit_count": 4 if "members" in h9_spec else 2}
    metadata[WIND_REFERENCE] = {"id": WIND_REFERENCE, "family": "starting_wind_control", "source_id": wind_source,
                                "feature_columns": 78, "uses_eve": False, "fit_count": 2}
    references = {"latest_persistence": bundle["short"][:, -1].copy(), "persistence_27day": bundle["recurrence"][:, 21].copy()}
    reports = {name: forecast_report(bundle["y"], values, references) for name, values in predictions.items()}
    if not np.isclose(reports[WIND_REFERENCE]["metrics"]["overall"]["rmse"], WIND_ANCHOR_RMSE, rtol=0, atol=1e-8):
        raise ValueError("Frozen starting wind anchor differs from preregistration")
    selected, checks, anchors = select_candidate(reports, metadata)
    dates = bundle["date"].astype("datetime64[D]")
    issues = (dates - np.timedelta64(3, "D")).astype(str)
    unique = deduplicate_issue_target_pairs(issues.tolist(), dates.astype(str).tolist(), bundle["y"], {**predictions, **references})
    unique_refs = {name: unique["forecasts"][name] for name in references}
    for name in reports:
        reports[name]["unique_issue_target_day"] = forecast_report(unique["observed"], unique["forecasts"][name], unique_refs)
    by_fold = {name: {str(fold): forecast_report(bundle["y"][bundle["fold"] == fold], values[bundle["fold"] == fold],
                {key: ref[bundle["fold"] == fold] for key, ref in references.items()}) for fold in FOLDS} for name, values in predictions.items()}
    component_reports = {name: forecast_report(bundle["y"], values, references) for name, values in component_predictions.items()}
    controls = {}
    selected_metrics = reports[selected]["metrics"]
    for name in (H9_REFERENCE, WIND_REFERENCE):
        control = reports[name]["metrics"]
        controls[name] = {"selected_minus_control_overall_rmse_km_s": selected_metrics["overall"]["rmse"] - control["overall"]["rmse"],
                          "selected_overall_rmse_lower": selected_metrics["overall"]["rmse"] < control["overall"]["rmse"],
                          "interpretation": "P0-relative adoption does not establish a new global best"}
    code_names = ("champion_continuation.py", "prior_champion.py", "train.py", "models.py", "metrics.py", "prepare_training.py", "fetch_eve.py", "fast_loss.py")
    input_names = ("train_only.npz", "train_only.manifest.json", "eve_features.npz", "h8_results.json", "h8_predictions.npz", "h9_results.json", "h9_predictions.npz")
    result = {"experiment": "H11", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "preregistration_url": PREREGISTRATION,
              "evaluation_scope": "Training Folds 1-4 only; selected CV scores are optimistic development estimates, not final-test evidence.",
              "selection_unit": "pooled original validation windows; all fourteen offsets retained",
              "incumbent": "P0", "selected": selected, "adoption": selected != "P0", "improvement_retained": selected != "P0",
              "selected_label": "eligible P0-relative continuation" if selected != "P0" else "P0 family retained; no support for registered continuation",
              "eve_retained": metadata[selected]["uses_eve"], "candidate_count": len(reports), "new_candidate_count": 7,
              "single_candidate_count": 4, "candidate_specs": metadata, "candidate_reports": reports,
              "candidate_eligibility": checks, "eve_eligibility": checks, "best_models_by_fold": by_fold,
              "best_by_family": {"registered_selection": selected}, "component_reports": component_reports,
              "control_comparisons": controls, "actual_anchors": anchors,
              "selection_rule": {"overall_ratio_max_vs_P0": 0.99, "fast500_ratio_max_vs_P0": 1.01,
                                 "overall_ratio_max_vs_starting_wind": 1.01,
                                 "objective": "0.5*MSE_all/MSE_all_P0 + 0.5*MSE_fast500/MSE_fast500_P0",
                                 "tie_break": "fewer feature columns, then fewer deployed estimator fits, then identifier",
                                 "fallback": "P0 family retained; controls remain separately reported"},
              "thresholds": {"primary_km_s": 500, "secondary_km_s": 600, "secondary_used_for_selection": False},
              "prior_champion_recipe": recipe_metadata(), "h8_provenance": h8_provenance, "h9_provenance": h9_provenance,
              "h10_status": "cancelled before fitting because prior neural champion was omitted from earlier SDO comparisons",
              "split_audit": split_audit, "secondary_deduplication": {"n_input": unique["n_input"], "n_unique_pairs": unique["n_unique_pairs"],
                        "aggregation": unique["aggregation"], "issue_date": "nominal target minus three calendar days"},
              "persistence_definition": h9_result["persistence_definition"], "eve": h9_result["eve"],
              "environment": versions, "input_hashes": {name: sha256(root / name) for name in input_names},
              "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in code_names if Path(__file__).with_name(name).exists()},
              "seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    names, component_names = list(predictions), list(component_predictions)
    prediction_path = output / "h11_predictions.npz"
    np.savez_compressed(prediction_path, rowid=bundle["rowid"], fold=bundle["fold"], block=bundle["block"], offset=bundle["offset"], date=bundle["date"], issue=issues,
                        observed=bundle["y"], model_names=np.asarray(names), predictions=np.column_stack([predictions[name] for name in names]),
                        component_names=np.asarray(component_names), component_predictions=np.column_stack([component_predictions[name] for name in component_names]),
                        latest_persistence=references["latest_persistence"], persistence_27day=references["persistence_27day"], incumbent=np.asarray("P0"), selected=np.asarray(selected))
    result["prediction_file_sha256"] = sha256(prediction_path)
    temporary = output / "h11_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "h11_results.json")
    print(json.dumps({"progress": "complete", "candidate_count": len(reports), "outputs": ["h11_results.json", "h11_predictions.npz"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another supported torch device")
    parser.add_argument("--stage", choices=("h11",), default="h11")
    args = parser.parse_args()
    run(args.bundle, args.output, args.device, args.stage)


if __name__ == "__main__":
    main()
