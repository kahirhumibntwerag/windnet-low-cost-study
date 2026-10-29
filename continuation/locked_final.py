"""Locked E13 same-holdout final panel: fixed recipes, no test selection.

Execution requires an exact portfolio-freeze manifest and preregistration.
Importing this module opens no data and fits no estimator.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import tarfile
import time

import numpy as np

try:
    from .prepare_locked_final import validate_freeze, fixed_panel_recipe, recipe_sha256, sha256, CODE_NAMES
    from .train import feature_matrices, recompute_eve_features, reference_detection_deltas
    from .fast_loss import GROUPS, feature_indexes
    from .models import build_model
    from .prior_champion import fit_prior_champion
    from .metrics import evaluate_forecast, deduplicate_issue_target_pairs
    from .final_uncertainty import paired_final_uncertainty
except ImportError:
    from prepare_locked_final import validate_freeze, fixed_panel_recipe, recipe_sha256, sha256, CODE_NAMES
    from train import feature_matrices, recompute_eve_features, reference_detection_deltas
    from fast_loss import GROUPS, feature_indexes
    from models import build_model
    from prior_champion import fit_prior_champion
    from metrics import evaluate_forecast, deduplicate_issue_target_pairs
    from final_uncertainty import paired_final_uncertainty


FOLDS = (1, 2, 3, 4, 5)
MODEL_IDS = tuple(f"P{i}" for i in range(7)) + ("h9_fast_specialist", "starting_wind", "h8_sdo",
            "matched_classical_wind", "h9_unweighted_lines", "h9_weight4_lines", "old_frozen_selected", "released_windnet")


def load_final_bundle(root: Path, frozen: dict, freeze_hash: str) -> tuple[dict, dict, dict]:
    manifest = json.loads((root / "final_bundle.manifest.json").read_text(encoding="utf-8"))
    required = {"schema": "locked_final_bundle_v1", "scope": "final_only", "experiment": "E13",
                "freeze_manifest_sha256": freeze_hash, "protocol_url": frozen["protocol_url"],
                "recipes_sha256": recipe_sha256(), "selection_from_final_test": False,
                "source_sha256": frozen["source_sha256"]}
    for name, value in required.items():
        if manifest.get(name) != value:
            raise ValueError(f"Rejected final-only bundle manifest: {name}")
    if manifest.get("preparation_code_sha256") != frozen["code_sha256"]["prepare_locked_final.py"]:
        raise ValueError("Bundle preparation code differs from the frozen implementation")
    path = root / "final_bundle.npz"
    if sha256(path) != manifest.get("export_sha256"):
        raise ValueError("Final bundle changed after mechanical preparation")
    expected = {"y", "fold", "block", "offset", "date", "short", "recurrence", "multi", "rowid", "fit_eligible",
                "source_fold_short", "source_fold_recurrence", "source_fold_multi", "old_frozen_selected", "released_windnet",
                "eve_lag_values", "eve_source_fold", "eve_features", "eve_names", "eve_fields"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Unexpected final-only bundle schema")
        b = {name: archive[name] for name in archive.files}
    n = 2198
    if not np.array_equal(b["rowid"], np.arange(n)) or set(np.unique(b["fold"]).tolist()) != set(FOLDS):
        raise ValueError("Final target rows or original folds differ")
    for name in ("y", "fold", "block", "offset", "date", "rowid", "fit_eligible", "old_frozen_selected", "released_windnet"):
        if b[name].shape != (n,):
            raise ValueError("Malformed final row column")
    if not np.isfinite(b["y"]).all() or not np.isin(b["offset"], np.arange(6, 20)).all() or not np.array_equal(b["fit_eligible"], b["offset"] < 19):
        raise ValueError("Final target/offset identity differs")
    for kind, width in (("short", 4), ("recurrence", 34), ("multi", 78)):
        x, source = b[kind], b[f"source_fold_{kind}"]
        if x.shape != (n, width) or source.shape != x.shape or np.isinf(x).any() or not np.isin(source, np.arange(6)).all():
            raise ValueError("Malformed final wind feature/provenance matrix")
        if not np.array_equal(source[:, :4], np.repeat(b["fold"][:, None], 4, axis=1)):
            raise ValueError("Original chronological history provenance was not corrected")
    fields = [str(name) for name in b["eve_fields"]]
    if len(fields) != 9 or b["eve_lag_values"].shape != (n, 4, 9) or b["eve_source_fold"].shape != (n, 4):
        raise ValueError("Malformed final EVE values or provenance")
    if not np.isin(b["eve_source_fold"], np.arange(6)).all():
        raise ValueError("Unknown EVE source fold")
    expected_sources = np.column_stack([b["source_fold_recurrence"][:, 4 + lag - 3] for lag in (5, 6, 7, 8)])
    if not np.array_equal(b["eve_source_fold"], expected_sources):
        raise ValueError("EVE calendar lag source provenance differs")
    features, names = recompute_eve_features(b["eve_lag_values"], fields)
    if names != [str(name) for name in b["eve_names"]] or not np.allclose(features, b["eve_features"], rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError("Final EVE features differ from their raw logged lag construction")
    # feature_indexes verifies registered ESP/line group sizes before fitting.
    for group in GROUPS:
        feature_indexes(fields, group)
    eve = {"lag_values": b["eve_lag_values"], "source_fold": b["eve_source_fold"],
           "fields": fields, "features": features, "names": names}
    return b, eve, manifest


def fit_pair(x: np.ndarray, y: np.ndarray, validation: np.ndarray,
             ridge_weight: float = 0.25, raw_fast_multiplier: int = 1,
             return_components: bool = False, checkpoint_path: Path | None = None) -> np.ndarray | dict:
    ridge, tree = build_model("ridge_alpha_1000"), build_model("hgb_leaves_7_minleaf_30")
    weights = None if raw_fast_multiplier == 1 else 1.0 + (raw_fast_multiplier - 1.0) * (y >= 500.0)
    ridge.fit(x, y, sample_weight=weights)
    tree.fit(x, y, sample_weight=weights)
    if checkpoint_path is not None:
        import joblib
        if checkpoint_path.exists():
            raise ValueError("Refusing to overwrite a fitted-pair checkpoint")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"ridge": ridge, "tree": tree, "ridge_weight": ridge_weight,
                     "raw_fast_multiplier": raw_fast_multiplier,
                     "prediction_rule": "ridge_weight*ridge.predict(X)+(1-ridge_weight)*tree.predict(X)"},
                    checkpoint_path, compress=3)
    ridge_prediction, tree_prediction = ridge.predict(validation), tree.predict(validation)
    result = ridge_weight * ridge_prediction + (1.0 - ridge_weight) * tree_prediction
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite fixed ridge/tree pair predictions")
    return {"ridge": ridge_prediction, "tree": tree_prediction, "prediction": result} if return_components else result


def save_prior_checkpoint(model, path: Path) -> dict:
    """Serialize a CPU copy of fitted state, without retaining fitting targets."""
    import copy
    import joblib
    import torch
    if path.exists():
        raise ValueError("Refusing to overwrite a fitted prior-champion checkpoint")
    portable = copy.deepcopy(model)
    portable.networks_ = [network.cpu() for network in portable.networks_]
    portable.device_ = torch.device("cpu")
    portable.runtime_ = {**portable.runtime_, "checkpoint_device": "cpu"}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(portable, path, compress=3)
    return {"path": str(path.name), "sha256": sha256(path), "bytes": path.stat().st_size,
            "serialized_device": "cpu"}


def final_report(y: np.ndarray, prediction: np.ndarray, references: dict) -> dict:
    result = evaluate_forecast(y, prediction, persistence=references)
    result["reference_skill"] = result.pop("persistence_skill")
    result["reference_detection_deltas"] = reference_detection_deltas(y, prediction, references)
    return result


def fold_summary(reports: list[dict]) -> dict:
    rmse = np.asarray([row["metrics"]["overall"]["rmse"] for row in reports], dtype=float)
    mae = np.asarray([row["metrics"]["overall"]["mae"] for row in reports], dtype=float)
    r = [row["metrics"]["overall"]["r"] for row in reports if row["metrics"]["overall"]["r"] is not None]
    finite_rmse, finite_mae = rmse[np.isfinite(rmse)], mae[np.isfinite(mae)]
    return {"mean_rmse": float(finite_rmse.mean()) if len(finite_rmse) else None,
            "rmse_se": float(finite_rmse.std(ddof=1) / np.sqrt(len(finite_rmse))) if len(finite_rmse) > 1 else None,
            "mean_mae": float(finite_mae.mean()) if len(finite_mae) else None,
            "fisher_mean_r": float(np.tanh(np.mean(np.arctanh(np.clip(r, -0.999999999999, 0.999999999999))))) if r else None,
            "finite_rmse_folds": int(len(finite_rmse)), "finite_correlation_folds": len(r),
            "aggregation": "unweighted mean across original folds; RMSE SE=sample SD/sqrt(number of finite folds); Fisher-z correlation mean",
            "uncertainty_interpretation": "descriptive fold variability, not a confidence interval or protection against adaptive study design"}


def run(bundle_directory: str | Path, output_directory: str | Path, freeze_manifest: str | Path,
        freeze_hash: str, protocol_url: str, device: str = "auto") -> dict:
    started = time.perf_counter()
    frozen = validate_freeze(freeze_manifest, freeze_hash, protocol_url)
    root, output = Path(bundle_directory).resolve(), Path(output_directory).resolve()
    bundle_execution_marker = root / f"execution_{freeze_hash}.started.json"
    if bundle_execution_marker.exists():
        raise ValueError("This frozen bundle has already started final execution; changing output directories cannot authorize a second pass")
    outputs = ("final_results.json", "predictions.npz", "comparison.csv", "execution_started.json", "checkpoints", "checkpoints.tar.gz")
    if output.exists() and not output.is_dir():
        raise ValueError("Final output path must be a directory")
    if any((output / name).exists() for name in outputs):
        raise ValueError("Final execution already started or outputs exist; reruns require an explicit separately preregistered protocol")
    b, eve, bundle_manifest = load_final_bundle(root, frozen, freeze_hash)
    import torch
    versions = {"python": platform.python_version(), "packages": {}, "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_request": device}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "h5py", "astropy"):
        try:
            versions["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions["packages"][name] = None
    output.mkdir(parents=True, exist_ok=True)
    # A persistent start marker blocks accidental second passes, including after errors.
    marker = {"experiment": "E13", "freeze_manifest_sha256": freeze_hash, "protocol_url": protocol_url,
              "started_utc": datetime.now(timezone.utc).isoformat(), "selection_from_final_test": False}
    # Bundle-scoped exclusive creation prevents a second run merely choosing
    # another output directory. Keep both markers even if a later step fails.
    with bundle_execution_marker.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
        stream.write("\n")
    with (output / "execution_started.json").open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
        stream.write("\n")
    predictions = {name: np.full(2198, np.nan) for name in MODEL_IDS if name not in ("old_frozen_selected", "released_windnet")}
    audits, checkpoints = [], {}
    for fold in FOLDS:
        test, other = b["fold"] == fold, b["fold"] != fold
        test_dates = np.unique(b["date"][test])
        purged = np.unique(b["block"][other & np.isin(b["date"], test_dates)])
        fit = other & b["fit_eligible"].astype(bool) & ~np.isin(b["block"], purged)
        if not fit.any() or not test.any() or np.intersect1d(b["date"][fit], test_dates).size:
            raise ValueError("Invalid final outer-fold purging")
        full_x, full_test, masks = feature_matrices(b, eve, "multi", fit, test, fold, True)
        matrices, runtime = {}, {}
        checkpoint_directory = output / "checkpoints" / f"fold{fold}"
        fold_checkpoints = {}
        for i, group in enumerate(GROUPS):
            indexes = np.r_[np.arange(78), 78 + feature_indexes(eve["fields"], group)]
            x, v = full_x[:, indexes], full_test[:, indexes]
            matrices[group] = (x, v)
            model = fit_prior_champion(x, b["y"][fit], device=device)
            fold_checkpoints[f"P{i}"] = save_prior_checkpoint(model, checkpoint_directory / f"P{i}.joblib")
            predictions[f"P{i}"][test] = model.predict(v)
            runtime[f"P{i}"] = model.runtime_
        x, v = matrices["wind_only"]
        wind_pair = fit_pair(x, b["y"][fit], v, ridge_weight=0.75, return_components=True,
                             checkpoint_path=checkpoint_directory / "wind_pair.joblib")
        predictions["starting_wind"][test] = wind_pair["prediction"]
        predictions["matched_classical_wind"][test] = 0.25 * wind_pair["ridge"] + 0.75 * wind_pair["tree"]
        x, v = matrices["wind_lines"]
        unweighted = fit_pair(x, b["y"][fit], v, checkpoint_path=checkpoint_directory / "unweighted_lines.joblib")
        raw_weight4 = fit_pair(x, b["y"][fit], v, raw_fast_multiplier=4,
                               checkpoint_path=checkpoint_directory / "raw4_lines.joblib")
        predictions["h9_unweighted_lines"][test] = unweighted
        predictions["h9_weight4_lines"][test] = raw_weight4
        predictions["h9_fast_specialist"][test] = 0.75 * unweighted + 0.25 * raw_weight4
        x, v = matrices["wind_all_eve"]
        predictions["h8_sdo"][test] = fit_pair(x, b["y"][fit], v, checkpoint_path=checkpoint_directory / "all_eve_pair.joblib")
        for checkpoint_name in ("wind_pair", "unweighted_lines", "raw4_lines", "all_eve_pair"):
            path = checkpoint_directory / f"{checkpoint_name}.joblib"
            fold_checkpoints[checkpoint_name] = {"path": str(path.name), "sha256": sha256(path), "bytes": path.stat().st_size,
                                                  "serialized_device": "cpu"}
        checkpoints[str(fold)] = fold_checkpoints
        for name, weights in fixed_panel_recipe()["diversity"].items():
            predictions[name][test] = weights["P0"] * predictions["P0"][test] + weights["h9_fast_specialist"] * predictions["h9_fast_specialist"][test]
        audits.append({"heldout_fold": fold, "fit_rows": int(fit.sum()), "test_rows": int(test.sum()),
                       "purged_blocks": purged.astype(int).tolist(), "feature_masks_fit_only": masks,
                       "test_feature_masking": "none; all already-causal historical observations retained", "runtime": runtime})
    if any(not np.isfinite(prediction).all() for prediction in predictions.values()):
        raise ValueError("Fixed final panel has incomplete forecasts")
    checkpoint_archive = output / "checkpoints.tar.gz"
    with tarfile.open(checkpoint_archive, "w:gz") as archive:
        archive.add(output / "checkpoints", arcname="checkpoints")
    for name in ("old_frozen_selected", "released_windnet"):
        predictions[name] = b[name].astype(float).copy()
    references = {"released_windnet": predictions["released_windnet"],
                  "old_frozen_selected": predictions["old_frozen_selected"],
                  "latest_persistence": b["short"][:, -1].copy(), "persistence_27day": b["recurrence"][:, 21].copy()}
    metrics = {}
    for name in MODEL_IDS:
        p = predictions[name]
        folds = [{"fold": fold, "metrics": final_report(b["y"][b["fold"] == fold], p[b["fold"] == fold],
                  {key: ref[b["fold"] == fold] for key, ref in references.items()})} for fold in FOLDS]
        metrics[name] = {"pooled": final_report(b["y"], p, references), "folds": folds, **fold_summary(folds)}
    dates = b["date"].astype("datetime64[D]")
    issues = (dates - np.timedelta64(3, "D")).astype(str)
    # Same-column WindNet is also a reference; avoid key collision harmlessly.
    unique = deduplicate_issue_target_pairs(issues.tolist(), dates.astype(str).tolist(), b["y"], {**predictions, **references})
    unique_refs = {name: unique["forecasts"][name] for name in references}
    for name in MODEL_IDS:
        metrics[name]["unique_issue_target_day"] = final_report(unique["observed"], unique["forecasts"][name], unique_refs)
    uncertainty = paired_final_uncertainty(b["y"], predictions, b["date"], b["fold"])
    result = {"experiment": "E13", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_url": protocol_url, "freeze_manifest_sha256": freeze_hash, "recipes_sha256": recipe_sha256(),
              "one_shot_markers": {"bundle": bundle_execution_marker.name, "output": "execution_started.json"},
              "scope": "locked same-original-holdout final comparison", "selection_from_final_test": False,
              "primary": "starting_wind", "fast_wind_primary": "h9_fast_specialist", "fixed_panel": fixed_panel_recipe(),
              "model_count": len(MODEL_IDS), "model_ids": list(MODEL_IDS), "metrics": metrics,
              "paired_final_uncertainty": uncertainty,
              "split_audit": audits, "environment": versions, "bundle_manifest": bundle_manifest,
              "fitted_checkpoints": {"format": "joblib compress=3; prior neural state copied to CPU", "by_outer_fold": checkpoints,
                                      "archive": "checkpoints.tar.gz", "archive_sha256": sha256(checkpoint_archive),
                                      "archive_bytes": checkpoint_archive.stat().st_size,
                                      "resume_note": "Actual fitted models are retained with source/environment fingerprints; fixed mixture recipes reconstruct derived forecasts. No training target arrays are added to checkpoint payloads."},
              "input_hashes": {"final_bundle.npz": sha256(root / "final_bundle.npz"), "final_bundle.manifest.json": sha256(root / "final_bundle.manifest.json")},
              "source_sha256": frozen["source_sha256"], "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in CODE_NAMES},
              "secondary_deduplication": {"n_input": unique["n_input"], "n_unique_pairs": unique["n_unique_pairs"],
                                           "aggregation": unique["aggregation"], "issue_date": "nominal target minus three calendar days"},
              "caveats": ["Folds 1-4 informed prior model design; all-fold averages are exploratory.",
                          "Original Fold 5 has already been opened in the earlier campaign; this is a locked comparison, not a newly untouched test.",
                          "Fixed earlier-champion recipe retraining differs from the separately retained old frozen selected pipeline.",
                          "Released WindNet forecasts are frozen references and are not retrained or selected here.",
                          "Present-day calibrated EVE measurements do not establish as-issued operational availability.",
                          "No tuning, adoption, ranking-based selection or new model design is performed from this final panel."],
              "seconds": time.perf_counter() - started}
    path = output / "predictions.npz"
    np.savez_compressed(path, rowid=b["rowid"], fold=b["fold"], block=b["block"], offset=b["offset"], date=b["date"], issue=issues,
                        observed=b["y"], model_names=np.asarray(MODEL_IDS), predictions=np.column_stack([predictions[name] for name in MODEL_IDS]),
                        latest_persistence=references["latest_persistence"], persistence_27day=references["persistence_27day"],
                        primary=np.asarray("starting_wind"), fast_wind_primary=np.asarray("h9_fast_specialist"))
    result["prediction_file_sha256"] = sha256(path)
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "mean_fold_rmse_km_s", "se_fold_rmse_km_s", "mean_fold_mae_km_s", "fisher_mean_r", "pooled_rmse_km_s",
                         "fast500_rmse_km_s", "fast600_rmse_km_s", "paired_windnet_mse_skill", "paired_windnet_delta_rmse_km_s", "paired_windnet_n"])
        for name in MODEL_IDS:
            row = metrics[name]
            pooled = row["pooled"]
            paired = pooled["reference_skill"]["released_windnet"]["all"]
            writer.writerow([name, row["mean_rmse"], row["rmse_se"], row["mean_mae"], row["fisher_mean_r"], pooled["overall"]["rmse"],
                             pooled["fast_wind"]["ge_500"]["continuous_observed_fast"]["rmse"],
                             pooled["fast_wind"]["ge_600"]["continuous_observed_fast"]["rmse"],
                             paired["mse_skill_score"], paired["delta_rmse"], paired["n"]])
    result["comparison_csv_sha256"] = sha256(output / "comparison.csv")
    temporary = output / "final_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "final_results.json")
    print(json.dumps({"status": "complete", "experiment": "E13", "outputs": ["final_results.json", "predictions.npz", "comparison.csv"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--protocol-url", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.bundle, args.output, args.freeze_manifest, args.freeze_sha256, args.protocol_url, args.device)


if __name__ == "__main__":
    main()
