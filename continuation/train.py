"""Training-only cross-validation with fixed EVE and wind candidate grids.

Only ``load_training_bundle`` can supply solar-wind targets. Original study
modules and excluded targets are never imported or opened. Scores here are
development scores after selection, not independent test estimates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

try:
    from .prepare_training import load_training_bundle
    from .models import CandidateSpec, build_model, candidate_specs
    from .metrics import deduplicate_issue_target_pairs, detection_metrics, evaluate_forecast
except ImportError:
    from prepare_training import load_training_bundle
    from models import CandidateSpec, build_model, candidate_specs
    from metrics import deduplicate_issue_target_pairs, detection_metrics, evaluate_forecast


FOLDS = (1, 2, 3, 4)
EVE_LAGS = (5, 6, 7, 8)
RATIO_FIELDS = (("Fe_XII_20", "Fe_XIV_21"), ("Fe_IX_17", "Fe_XVI_34"))
BLEND_WEIGHTS = (0.25, 0.5, 0.75)
H8_FIELDS = {
    "Fe_IX_17", "Fe_XII_20", "Fe_XIV_21", "He_II_30", "Fe_XVI_34", "He_I_58",
    "diode171", "diode257", "diode304",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_eve_features(
    lag_values: np.ndarray, fields: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Rebuild 6*channel_count+8 columns after masking, without fitting.

    Inputs are natural-log irradiances: [rows, lags 5/6/7/8, channels].
    Means propagate any missing lag; trends use the two endpoints and have
    units log-irradiance/day. Ratios are differences of the two channel logs.
    """
    values = np.asarray(lag_values, dtype=float)
    if values.ndim != 3 or values.shape[1] != 4 or values.shape[2] != len(fields):
        raise ValueError("EVE lag_values must have shape [rows,4,channel_count]")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("EVE requires uniquely named scalar channels")
    if np.isinf(values).any():
        raise ValueError("EVE logged irradiances may be missing, never infinite")
    names = [f"log_{field}_lag{lag}" for lag in EVE_LAGS for field in fields]
    columns = [values.reshape(len(values), 4 * len(fields))]
    columns.append(np.mean(values, axis=1))
    names.extend(f"log_{field}_mean_lags5to8" for field in fields)
    columns.append((values[:, 0, :] - values[:, -1, :]) / 3.0)
    names.extend(f"log_{field}_trend_lags5to8" for field in fields)
    ratios = []
    for numerator, denominator in RATIO_FIELDS:
        if numerator not in fields or denominator not in fields:
            raise ValueError("Required EVE ratio channel is missing")
        a, b = fields.index(numerator), fields.index(denominator)
        for position, lag in enumerate(EVE_LAGS):
            ratios.append(values[:, position, a] - values[:, position, b])
            names.append(f"logratio_{numerator}_{denominator}_lag{lag}")
    columns.append(np.column_stack(ratios))
    result = np.column_stack(columns)
    if result.shape != (len(values), 6 * len(fields) + 8):
        raise AssertionError("EVE feature construction changed its fixed width")
    return result, names


def load_eve_bundle(root: Path, bundle: dict[str, np.ndarray]) -> dict[str, Any]:
    """Read only EVE values aligned to the already isolated training rows."""
    path = root / "eve_features.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {"rowid", "eve", "lag_values", "lag_source_fold", "names", "fields"}
        if set(archive.files) != required:
            raise ValueError("Unexpected EVE feature bundle schema")
        eve = {name: archive[name] for name in archive.files}
    n = len(bundle["y"])
    if not np.array_equal(eve["rowid"], bundle["rowid"]):
        raise ValueError("EVE row identifiers do not match the training-only bundle")
    if eve["lag_source_fold"].shape != (n, 4):
        raise ValueError("Invalid EVE lag provenance shape")
    if not np.isin(eve["lag_source_fold"], np.arange(6)).all():
        raise ValueError("Unknown EVE source-fold identifier")
    fields = [str(x) for x in eve["fields"]]
    if set(fields) != H8_FIELDS or len(fields) != len(H8_FIELDS):
        raise ValueError("H8 requires its nine fixed channels; diode366 and diode1216 are excluded before execution")
    if eve["lag_values"].shape != (n, 4, len(fields)):
        raise ValueError("EVE values do not align to the training rows")
    if not np.isnan(eve["lag_values"][eve["lag_source_fold"] == 5]).all():
        raise ValueError("Rejected EVE bundle: excluded-source values remain")
    rebuilt, names = recompute_eve_features(eve["lag_values"], fields)
    if eve["eve"].shape != rebuilt.shape or not np.allclose(
        eve["eve"], rebuilt, rtol=1e-10, atol=1e-12, equal_nan=True
    ):
        raise ValueError("Cached EVE features disagree with raw logged lag values")
    if names != [str(x) for x in eve["names"]]:
        raise ValueError("EVE column names/order differ from the fixed construction")
    manifest_path = root / "eve_features.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None and manifest.get("feature_bundle_sha256") != sha256(path):
        raise ValueError("EVE feature bundle differs from its manifest hash")
    return {"lag_values": eve["lag_values"], "source_fold": eve["lag_source_fold"],
            "fields": fields, "features": rebuilt, "names": names,
            "sha256": sha256(path), "manifest": manifest}


def definitions(eve_feature_count: int = 62) -> list[dict[str, Any]]:
    """The finite grid: fourteen wind-only and eighteen EVE candidates."""
    result = []
    for spec in candidate_specs():
        for kind in ("recurrence", "multi"):
            if kind == "recurrence" and spec.family != "ridge":
                continue
            family = f"wind_{kind}_{'ridge' if spec.family == 'ridge' else 'tree'}"
            result.append({"id": f"{family}__{spec.name}", "family": family,
                           "wind_kind": kind, "uses_eve": False,
                           "feature_columns": 34 if kind == "recurrence" else 78,
                           "spec": spec})
    for kind in ("recurrence", "multi"):
        for spec in candidate_specs():
            family = f"eve_{kind}_{'ridge' if spec.family == 'ridge' else 'tree'}"
            result.append({"id": f"{family}__{spec.name}", "family": family,
                           "wind_kind": kind, "uses_eve": True,
                           "feature_columns": (34 if kind == "recurrence" else 78) + eve_feature_count,
                           "spec": spec})
    if len(result) != 32:
        raise AssertionError("Registered candidate grid changed")
    return result


def feature_matrices(
    bundle: dict[str, np.ndarray], eve: dict[str, Any], kind: str,
    fit: np.ndarray, validation: np.ndarray, heldout_fold: int, uses_eve: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train = bundle[kind][fit].copy()
    source = bundle[f"source_fold_{kind}"][fit]
    masked = source == heldout_fold
    train[masked] = np.nan
    val = bundle[kind][validation].copy()
    counts = {"wind_values_masked": int(masked.sum()), "eve_lag_channels_masked": 0}
    if uses_eve:
        lag = eve["lag_values"][fit].copy()
        eve_mask = eve["source_fold"][fit] == heldout_fold
        lag[eve_mask] = np.nan
        extra, _ = recompute_eve_features(lag, eve["fields"])
        train = np.column_stack((train, extra))
        val = np.column_stack((val, eve["features"][validation]))
        counts["eve_lag_channels_masked"] = int(eve_mask.sum() * len(eve["fields"]))
    return train, val, counts


def reference_detection_deltas(y: np.ndarray, p: np.ndarray, references: dict) -> dict:
    result = {}
    for name, ref in references.items():
        common = np.isfinite(y) & np.isfinite(p) & np.isfinite(ref)
        result[name] = {}
        for threshold in (500.0, 600.0):
            model = detection_metrics(y[common], p[common], threshold)
            baseline = detection_metrics(y[common], ref[common], threshold)
            result[name][f"ge_{threshold:g}"] = {
                "n_common": int(common.sum()), "model": model, "reference": baseline,
                "delta": {key: None if model[key] is None or baseline[key] is None
                          else float(model[key] - baseline[key])
                          for key in ("pod", "far", "fpr", "csi", "tss", "hss", "frequency_bias")},
            }
    return result


def forecast_report(y: np.ndarray, p: np.ndarray, references: dict) -> dict:
    return {"metrics": evaluate_forecast(y, p, persistence=references),
            "persistence_detection_deltas": reference_detection_deltas(y, p, references)}


def _rmse(report: dict) -> float:
    return float(report["metrics"]["overall"]["rmse"])


def _metadata(item: dict) -> dict:
    return {key: value.to_dict() if isinstance(value, CandidateSpec) else value
            for key, value in item.items()}


def run(bundle_directory: str | Path, output_directory: str | Path, stage: str = "h8") -> dict:
    started = time.perf_counter()
    if stage != "h8":
        raise ValueError("H7 was cancelled before execution; only registered replacement H8 is executable here")
    root, output = Path(bundle_directory).resolve(), Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Output path must be a directory")
    existing = [output / f"{stage}_results.json", output / f"{stage}_predictions.npz"]
    if any(path.exists() for path in existing):
        raise ValueError("Refusing to overwrite existing continuation results or predictions")
    b = load_training_bundle(root)
    eve = load_eve_bundle(root, b)
    if set(np.unique(b["fold"]).tolist()) != set(FOLDS):
        raise ValueError("Continuation requires all four allowed training folds")
    n = len(b["y"])
    references = {"latest_persistence": b["short"][:, -1].copy(),
                  "persistence_27day": b["recurrence"][:, 21].copy()}
    candidates = definitions(len(eve["names"]))
    predictions = {item["id"]: np.full(n, np.nan) for item in candidates}
    split_audit = []
    for fold in FOLDS:
        validation = b["fold"] == fold
        other = ~validation
        validation_dates = np.unique(b["date"][validation])
        offending_rows = other & np.isin(b["date"], validation_dates)
        excluded_blocks = np.unique(b["block"][offending_rows])
        fit = other & b["fit_eligible"].astype(bool) & ~np.isin(b["block"], excluded_blocks)
        if not fit.any() or not validation.any():
            raise ValueError("A purged inner split has no fitting or validation rows")
        if np.intersect1d(b["date"][fit], validation_dates).size:
            raise AssertionError("Duplicate target date survived block purging")
        audit = {"validation_fold": fold, "fit_rows": int(fit.sum()),
                 "validation_rows": int(validation.sum()),
                 "purged_blocks": [int(x) for x in excluded_blocks],
                 "purged_rows_before_fitting_eligibility": int((other & np.isin(b["block"], excluded_blocks)).sum()),
                 "feature_masks": {}}
        matrices = {}
        for item in candidates:
            key = (item["wind_kind"], item["uses_eve"])
            if key not in matrices:
                matrices[key] = feature_matrices(b, eve, key[0], fit, validation, fold, key[1])
                audit["feature_masks"][f"{key[0]}{'_eve' if key[1] else ''}"] = matrices[key][2]
            x, v, _ = matrices[key]
            model = build_model(item["spec"])
            model.fit(x, b["y"][fit])
            p = model.predict(v)
            if not np.isfinite(p).all():
                raise ValueError("A fixed candidate failed to provide finite OOF forecasts")
            predictions[item["id"]][validation] = p
        split_audit.append(audit)
        print(json.dumps({"progress": "inner_fold_complete", "fold": fold,
                          "candidate_fits": len(candidates)}), flush=True)
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("An OOF candidate has incomplete forecast coverage")
    reports = {name: forecast_report(b["y"], p, references) for name, p in predictions.items()}
    metadata = {item["id"]: _metadata(item) for item in candidates}
    best_by_family = {}
    for family in sorted({item["family"] for item in candidates}):
        names = [item["id"] for item in candidates if item["family"] == family]
        best_by_family[family] = min(names, key=lambda name: (_rmse(reports[name]), name))
    for prefix in ("wind", "eve"):
        families = [f"{prefix}_recurrence_ridge", f"{prefix}_multi_ridge", f"{prefix}_multi_tree"]
        members = [best_by_family[family] for family in families]
        for left, right in itertools.combinations(members, 2):
            for weight in BLEND_WEIGHTS:
                name = f"{prefix}_blend__{left}__{right}__w{weight:g}"
                p = weight * predictions[left] + (1.0 - weight) * predictions[right]
                predictions[name] = p
                metadata[name] = {"id": name, "family": f"{prefix}_blend", "uses_eve": prefix == "eve",
                                  "members": [left, right], "weights": [weight, 1.0 - weight]}
                reports[name] = forecast_report(b["y"], p, references)
    baseline_names = [name for name, item in metadata.items() if not item["uses_eve"]]
    incumbent = min(baseline_names, key=lambda name: (_rmse(reports[name]), name))
    inc_all = reports[incumbent]["metrics"]["overall"]
    inc_fast = reports[incumbent]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
    eligibility = {}
    for name, item in metadata.items():
        if not item["uses_eve"]:
            continue
        overall = reports[name]["metrics"]["overall"]
        fast = reports[name]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]
        denominators = (inc_all["mse"] is not None and inc_all["mse"] > 0
                        and inc_fast["mse"] is not None and inc_fast["mse"] > 0)
        overall_ok = overall["rmse"] <= 1.01 * inc_all["rmse"]
        fast_ok = fast["rmse"] is not None and inc_fast["rmse"] is not None and fast["rmse"] <= 0.98 * inc_fast["rmse"]
        objective = (0.5 * overall["mse"] / inc_all["mse"] + 0.5 * fast["mse"] / inc_fast["mse"]
                     if denominators and fast["mse"] is not None else None)
        eligibility[name] = {"eligible": bool(denominators and overall_ok and fast_ok),
                             "overall_guard_passed": bool(overall_ok), "fast_guard_passed": bool(fast_ok),
                             "J": objective}
    eligible = [name for name, values in eligibility.items() if values["eligible"]]
    selected = min(eligible, key=lambda name: (eligibility[name]["J"], _rmse(reports[name]), name)) if eligible else incumbent
    dates = b["date"].astype("datetime64[D]")
    issues = (dates - np.timedelta64(3, "D")).astype(str)
    unique = deduplicate_issue_target_pairs(issues.tolist(), dates.astype(str).tolist(), b["y"], {**predictions, **references})
    unique_refs = {name: unique["forecasts"][name] for name in references}
    for name in reports:
        reports[name]["unique_issue_target_day"] = forecast_report(unique["observed"], unique["forecasts"][name], unique_refs)
    report_names = sorted(set(best_by_family.values()) | {incumbent, selected})
    by_fold = {name: {str(fold): forecast_report(b["y"][b["fold"] == fold], predictions[name][b["fold"] == fold],
                     {key: ref[b["fold"] == fold] for key, ref in references.items()}) for fold in FOLDS} for name in report_names}
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                   ("train.py", "models.py", "metrics.py", "prepare_training.py", "fetch_eve.py")
                   if Path(__file__).with_name(name).exists()}
    result = {"experiment": stage.upper(), "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_scope": "Training Folds 1-4 only; selected cross-validation scores are optimistic development estimates, not final-test evidence.",
              "selection_unit": "pooled original validation windows; all fourteen evaluation offsets retained",
              "secondary_deduplication": {"n_input": unique["n_input"], "n_unique_pairs": unique["n_unique_pairs"],
                                           "aggregation": unique["aggregation"], "issue_date": "nominal target minus three calendar days"},
              "thresholds": {"primary_km_s": 500, "secondary_km_s": 600, "secondary_used_for_selection": False},
              "incumbent": incumbent, "selected": selected, "eve_retained": bool(eligible),
              "selection_rule": {"overall_RMSE_ratio_max": 1.01, "fast500_RMSE_ratio_max": 0.98,
                                 "objective": "0.5*MSE_all/MSE_all_incumbent + 0.5*MSE_fast500/MSE_fast500_incumbent",
                                 "fallback": "retain incumbent if no eligible EVE candidate or undefined normalization"},
              "candidate_count": len(reports), "single_candidate_count": len(candidates),
              "best_by_family": best_by_family, "candidate_specs": metadata, "candidate_reports": reports,
              "eve_eligibility": eligibility, "best_models_by_fold": by_fold, "split_audit": split_audit,
              "persistence_definition": {"latest_persistence": "last of the four original chronologically ordered history entries; no calendar-gap repair",
                                         "persistence_27day": "recurrence column 21: 4 original + 14 recent + index3 in the 24..31 calendar-day band"},
              "eve": {"features": len(eve["names"]), "names": eve["names"], "fields": eve["fields"],
                      "lag_calendar_days": list(EVE_LAGS), "feature_bundle_sha256": eve["sha256"], "manifest": eve["manifest"]},
              "pre_execution_quality_exclusions": {"diode1216": "Fourier-filter one-sidedness undocumented",
                                                   "diode366": "Unstable shunt channel per official documentation",
                                                   "cancelled_protocol": "H7, before model execution"},
              "environment": {"python": platform.python_version(), "packages": versions, "neural_models_used": False},
              "input_hashes": {"train_only.npz": sha256(root / "train_only.npz"),
                               "train_only.manifest.json": sha256(root / "train_only.manifest.json")},
              "code_sha256": code_hashes, "seconds": time.perf_counter() - started}
    output.mkdir(parents=True, exist_ok=True)
    names = list(predictions)
    prediction_file = output / f"{stage}_predictions.npz"
    np.savez_compressed(prediction_file, rowid=b["rowid"], fold=b["fold"], block=b["block"],
                        offset=b["offset"], date=b["date"], issue=issues, observed=b["y"],
                        model_names=np.asarray(names), predictions=np.column_stack([predictions[name] for name in names]),
                        latest_persistence=references["latest_persistence"], persistence_27day=references["persistence_27day"],
                        incumbent=np.asarray(incumbent), selected=np.asarray(selected))
    result["prediction_file_sha256"] = sha256(prediction_file)
    temporary = output / f"{stage}_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / f"{stage}_results.json")
    print(json.dumps({"progress": "complete", "candidate_count": len(reports),
                      "outputs": [f"{stage}_results.json", f"{stage}_predictions.npz"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Verified training-only bundle directory including eve_features.npz")
    parser.add_argument("--output", required=True, type=Path, help="Directory without existing results/prediction outputs; may equal the bundle directory")
    parser.add_argument("--stage", choices=("h8",), default="h8", help="Registered replacement protocol; H7 was cancelled before execution")
    args = parser.parse_args()
    run(args.bundle, args.output, stage=args.stage)


if __name__ == "__main__":
    main()
