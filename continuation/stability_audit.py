"""H12 conditional stability audit of fixed H11 training-only forecasts.

Reads exactly h11_results.json and h11_predictions.npz from the requested
results directory. There is no model fitting, selection, or source-data access.
The registered winner is retained regardless of the resampling intervals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

try:
    from .metrics import deduplicate_issue_target_pairs, detection_metrics, evaluate_forecast
except ImportError:
    from metrics import deduplicate_issue_target_pairs, detection_metrics, evaluate_forecast


FOLDS = (1, 2, 3, 4)
BOOTSTRAP_DRAWS = 5000
SEED = 20261029
CALENDAR_WIDTH_DAYS = 54
CALENDAR_ORIGIN = "2011-01-01"
PREREGISTRATION = "https://fixrelay.vercel.app/experiments/509276ed-b66e-4869-8cb8-f6b6c250a1e5"
MODEL_IDS = {*(f"P{i}" for i in range(7)), "h9_reference", "starting_wind_reference"}
COMPONENT_IDS = {f"P{i}_{component}" for i in range(4) for component in
                 ("ridge", "tree", "residual_neural", "neural_seed_19", "neural_seed_43", "neural_seed_101")}
THRESHOLDS = (("all", None), ("observed_ge_500", 500.0), ("observed_ge_600", 600.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError("H12 artifact integrity: " + message)


def _report(y: np.ndarray, prediction: np.ndarray, references: dict) -> dict:
    deltas = {}
    for name, reference in references.items():
        common = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(reference)
        deltas[name] = {}
        for threshold in (500.0, 600.0):
            model = detection_metrics(y[common], prediction[common], threshold)
            baseline = detection_metrics(y[common], reference[common], threshold)
            deltas[name][f"ge_{threshold:g}"] = {
                "n_common": int(common.sum()), "model": model, "reference": baseline,
                "delta": {key: None if model[key] is None or baseline[key] is None
                          else float(model[key] - baseline[key])
                          for key in ("pod", "far", "fpr", "csi", "tss", "hss", "frequency_bias")},
            }
    return {"metrics": evaluate_forecast(y, prediction, persistence=references),
            "persistence_detection_deltas": deltas}


def _compare(actual: Any, recorded: Any, path: str, audit: dict) -> None:
    """Compare all recomputed report fields, including undefined metrics."""
    if isinstance(actual, dict):
        _require(isinstance(recorded, dict), path + " is not an object")
        for key, value in actual.items():
            _require(key in recorded, path + "/" + key + " is missing")
            _compare(value, recorded[key], path + "/" + key, audit)
    elif actual is None or isinstance(actual, (str, bool)):
        _require(actual == recorded, path + " differs")
        audit["exact_fields_checked"] += 1
    elif isinstance(actual, (int, np.integer)):
        _require(isinstance(recorded, (int, float)) and not isinstance(recorded, bool)
                 and actual == recorded, path + " count differs")
        audit["exact_fields_checked"] += 1
    else:
        _require(isinstance(recorded, (int, float)) and not isinstance(recorded, bool)
                 and np.isfinite(recorded), path + " is not a finite recorded number")
        difference = abs(float(actual) - float(recorded))
        _require(np.isclose(actual, recorded, rtol=1e-10, atol=1e-8), path + " differs")
        audit["floating_fields_checked"] += 1
        audit["max_absolute_float_difference"] = max(audit["max_absolute_float_difference"], difference)


def _identity_hash(data: dict) -> str:
    digest = hashlib.sha256()
    for name in ("rowid", "fold", "block", "offset", "date", "issue", "observed"):
        array = np.ascontiguousarray(data[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_and_verify(results_directory: str | Path) -> tuple[dict, dict, dict]:
    root = Path(results_directory).resolve()
    result_path, prediction_path = root / "h11_results.json", root / "h11_predictions.npz"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    _require(result.get("experiment") == "H11" and result.get("status") == "completed", "expected completed H11")
    prediction_bytes = prediction_path.read_bytes()
    prediction_hash = hashlib.sha256(prediction_bytes).hexdigest()
    _require(result.get("prediction_file_sha256") == prediction_hash, "prediction file hash differs")
    with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
        required = {"rowid", "fold", "block", "offset", "date", "issue", "observed", "model_names",
                    "predictions", "component_names", "component_predictions", "latest_persistence",
                    "persistence_27day", "selected", "incumbent"}
        _require(required == set(archive.files), "prediction schema differs from the fixed H11 export")
        data = {name: archive[name].copy() for name in required}
    y = data["observed"]
    _require(y.ndim == 1 and len(y) > 0 and np.isfinite(y).all(), "targets are malformed or nonfinite")
    n = len(y)
    for name in ("rowid", "fold", "block", "offset", "date", "issue", "latest_persistence", "persistence_27day"):
        _require(data[name].shape == (n,), "row field shape differs: " + name)
    for name in ("rowid", "fold", "block", "offset"):
        _require(data[name].dtype.kind in "iu", "row identifier is not integer: " + name)
    _require(len(np.unique(data["rowid"])) == n and (data["rowid"] >= 0).all(), "row IDs are not unique nonnegative integers")
    _require(set(np.unique(data["fold"]).tolist()) == set(FOLDS), "target folds differ from allowed Folds 1-4")
    _require((data["block"] >= 0).all(), "negative block identifier")
    _require(np.isin(data["offset"], np.arange(6, 20)).all(), "offset is outside original fourteen-window evaluation")
    for block in np.unique(data["block"]):
        rows = data["block"] == block
        _require(len(np.unique(data["fold"][rows])) == 1, "a block has multiple original fold IDs")
        _require(np.array_equal(np.sort(data["offset"][rows]), np.arange(6, 20)), "a block lacks unique offsets 6 through 19")
    try:
        dates = data["date"].astype("datetime64[D]")
        issues = data["issue"].astype("datetime64[D]")
    except (TypeError, ValueError) as error:
        raise ValueError("H12 artifact integrity: invalid calendar date") from error
    _require(not np.isnat(dates).any() and not np.isnat(issues).any(), "missing calendar date")
    _require(np.array_equal(issues, dates - np.timedelta64(3, "D")), "nominal issue day differs from target minus three days")
    data["date"], data["issue"] = dates, issues
    for names_key, values_key, expected in (("model_names", "predictions", MODEL_IDS),
                                            ("component_names", "component_predictions", COMPONENT_IDS)):
        _require(data[names_key].ndim == 1, names_key + " is not a vector")
        names = [str(name) for name in data[names_key]]
        _require(len(set(names)) == len(names) and set(names) == expected, names_key + " differs from the fixed H11 panel")
        _require(data[values_key].shape == (n, len(names)) and np.isfinite(data[values_key]).all(), values_key + " has invalid dimensions or nonfinite values")
    for name in ("latest_persistence", "persistence_27day"):
        _require(not np.isinf(data[name]).any(), "reference contains infinity: " + name)
    names = [str(name) for name in data["model_names"]]
    forecasts = {name: data["predictions"][:, index] for index, name in enumerate(names)}
    components = {str(name): data["component_predictions"][:, index] for index, name in enumerate(data["component_names"])}
    _require(result.get("incumbent") == "P0" and result.get("selected") in {f"P{i}" for i in range(7)}, "selected/incumbent is outside the registered H11 family")
    for name in ("selected", "incumbent"):
        _require(data[name].shape == () and str(data[name].item()) == result[name], "JSON/NPZ identifier differs: " + name)
    _require(isinstance(result.get("adoption"), bool) and result["adoption"] == (result["selected"] != "P0"), "adoption flag disagrees with selected identity")
    _require(set(result.get("candidate_specs", {})) == MODEL_IDS and set(result.get("candidate_reports", {})) == MODEL_IDS, "candidate report/spec identities differ")
    _require(set(result.get("component_reports", {})) == COMPONENT_IDS, "component report identities differ")
    _require(set(result.get("best_models_by_fold", {})) == MODEL_IDS, "fold report identities differ")
    _require(result.get("candidate_count") == len(MODEL_IDS), "candidate count differs")
    references = {name: data[name] for name in ("latest_persistence", "persistence_27day")}
    audit = {"exact_fields_checked": 0, "floating_fields_checked": 0, "max_absolute_float_difference": 0.0,
             "pooled_forecasts_checked": 0, "component_forecasts_checked": 0, "fold_reports_checked": 0,
             "unique_pair_forecasts_checked": 0}
    for name, prediction in forecasts.items():
        _compare(_report(y, prediction, references), result["candidate_reports"][name], "candidate/" + name, audit)
        audit["pooled_forecasts_checked"] += 1
        _require(set(result["best_models_by_fold"][name]) == {str(fold) for fold in FOLDS}, "fold report is incomplete: " + name)
        for fold in FOLDS:
            rows = data["fold"] == fold
            _compare(_report(y[rows], prediction[rows], {key: value[rows] for key, value in references.items()}),
                     result["best_models_by_fold"][name][str(fold)], f"fold/{fold}/{name}", audit)
            audit["fold_reports_checked"] += 1
    for name, prediction in components.items():
        _compare(_report(y, prediction, references), result["component_reports"][name], "component/" + name, audit)
        audit["component_forecasts_checked"] += 1
    unique = deduplicate_issue_target_pairs(issues.astype(str).tolist(), dates.astype(str).tolist(), y, {**forecasts, **references})
    _require(result.get("secondary_deduplication", {}).get("n_input") == n
             and result["secondary_deduplication"].get("n_unique_pairs") == unique["n_unique_pairs"], "duplicate-pair counts differ")
    unique_references = {name: unique["forecasts"][name] for name in references}
    for name in forecasts:
        _compare(_report(unique["observed"], unique["forecasts"][name], unique_references),
                 result["candidate_reports"][name].get("unique_issue_target_day"), "unique/" + name, audit)
        audit["unique_pair_forecasts_checked"] += 1
    for i in range(4):
        _require(np.allclose(forecasts[f"P{i}"], .5 * components[f"P{i}_tree"] + .5 * components[f"P{i}_residual_neural"],
                            rtol=1e-12, atol=1e-8), "prior-champion component blend differs")
    for i, weight in enumerate((.25, .5, .75), start=4):
        _require(np.allclose(forecasts[f"P{i}"], (1 - weight) * forecasts["P0"] + weight * forecasts["h9_reference"],
                            rtol=1e-12, atol=1e-8), "fixed H9 diversity blend differs")
    audit.update({"passed": True, "rows": n, "row_identity_sha256": _identity_hash(data),
                  "fold_rows": {str(fold): int((data["fold"] == fold).sum()) for fold in FOLDS},
                  "all_forecast_values_finite": True, "unique_issue_target_pairs": unique["n_unique_pairs"],
                  "input_hashes": {"h11_results.json": hashlib.sha256(result_bytes).hexdigest(), "h11_predictions.npz": prediction_hash},
                  "scope": "H11 internal artifact consistency; no original data, held-out targets, or upstream result files reopened"})
    data["forecasts"] = forecasts
    return result, data, audit


def block_components(data: dict) -> tuple[np.ndarray, dict]:
    """Union actual block IDs connected by an identical issue-target pair."""
    blocks = [int(value) for value in np.unique(data["block"])]
    parent = {value: value for value in blocks}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    first = {}
    for block, issue, target in zip(data["block"], data["issue"].astype(str), data["date"].astype(str)):
        block = int(block)
        pair = (issue, target)
        if pair in first:
            left, right = find(first[pair]), find(block)
            if left != right:
                parent[max(left, right)] = min(left, right)
        else:
            first[pair] = block
    roots = sorted({find(block) for block in blocks})
    indexes = {root: index for index, root in enumerate(roots)}
    labels = np.asarray([indexes[find(int(block))] for block in data["block"]], dtype=int)
    components = [{"component": index, "block_ids": [block for block in blocks if find(block) == root],
                   "rows": int((labels == index).sum()),
                   "folds": [int(value) for value in np.unique(data["fold"][labels == index])]}
                  for index, root in enumerate(roots)]
    return labels, {"grouping": "union of actual block IDs sharing an identical nominal issue-target day pair",
                    "original_block_count": len(blocks), "cluster_count": len(roots), "components": components,
                    "duplicate_weighting": "all original windows retained, including repeated issue-target pairs"}


def calendar_clusters(data: dict) -> tuple[np.ndarray, dict]:
    day = data["date"].astype("datetime64[D]")
    bins = ((day - np.datetime64(CALENDAR_ORIGIN, "D")).astype(np.int64) // CALENDAR_WIDTH_DAYS)
    unique, labels = np.unique(bins, return_inverse=True)
    descriptions = []
    for index, value in enumerate(unique):
        start = np.datetime64(CALENDAR_ORIGIN, "D") + np.timedelta64(int(value) * CALENDAR_WIDTH_DAYS, "D")
        descriptions.append({"cluster": index, "calendar_bin": int(value), "start_utc_inclusive": str(start),
                             "stop_utc_exclusive": str(start + np.timedelta64(CALENDAR_WIDTH_DAYS, "D")),
                             "rows": int((labels == index).sum())})
    return labels, {"grouping": "nonoverlapping UTC target-day calendar bins", "width_days": CALENDAR_WIDTH_DAYS,
                    "origin_utc": CALENDAR_ORIGIN, "cluster_count": len(unique), "bins": descriptions,
                    "duplicate_weighting": "all original windows retained, including repeated issue-target pairs"}


def comparison_panel(selected: str) -> dict[str, tuple[str, str]]:
    return {"H11selected_vs_P0": (selected, "P0"),
            "H11selected_vs_startingwind": (selected, "starting_wind_reference"),
            "H11selected_vs_H9": (selected, "h9_reference"),
            "H9_vs_P0": ("h9_reference", "P0")}


def bootstrap(data: dict, labels: np.ndarray, grouping: dict, panel: dict,
              draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> dict:
    """Paired cluster bootstrap using sufficient statistics, without refitting."""
    y, forecasts = data["observed"], data["forecasts"]
    n_clusters = grouping["cluster_count"]
    _require(n_clusters > 0 and labels.shape == y.shape and set(np.unique(labels)) == set(range(n_clusters)), "invalid bootstrap grouping")
    ids = list(dict.fromkeys(name for pair in panel.values() for name in pair))
    indexes = {name: index for index, name in enumerate(ids)}
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=draws)
    comparisons = {name: {"challenger": pair[0], "reference": pair[1], "thresholds": {}} for name, pair in panel.items()}
    for label, threshold in THRESHOLDS:
        rows = np.ones(len(y), dtype=bool) if threshold is None else y >= threshold
        counts = np.bincount(labels[rows], minlength=n_clusters).astype(float)
        with np.errstate(over="ignore", invalid="ignore"):
            sums = np.column_stack([np.bincount(labels[rows], weights=np.square(forecasts[name][rows] - y[rows]), minlength=n_clusters) for name in ids]).astype(float, copy=False)
            sampled_counts = multiplicities @ counts
            sampled_sums = multiplicities @ sums
            sampled_rmse = np.sqrt(np.divide(sampled_sums, sampled_counts[:, None],
                                              out=np.full_like(sampled_sums, np.nan), where=sampled_counts[:, None] > 0))
        for name, (challenger, reference) in panel.items():
            a, b = indexes[challenger], indexes[reference]
            delta = sampled_rmse[:, a] - sampled_rmse[:, b]
            finite = delta[np.isfinite(delta)]
            n_defined = len(finite)
            point_delta = result_a = result_b = None
            if rows.any():
                with np.errstate(over="ignore", invalid="ignore"):
                    result_a = float(np.sqrt(np.mean(np.square(forecasts[challenger][rows] - y[rows]))))
                    result_b = float(np.sqrt(np.mean(np.square(forecasts[reference][rows] - y[rows]))))
                result_a = result_a if np.isfinite(result_a) else None
                result_b = result_b if np.isfinite(result_b) else None
                point_delta = None if result_a is None or result_b is None else result_a - result_b
            interval = np.percentile(finite, [2.5, 97.5], method="linear").tolist() if n_defined else None
            if not n_defined:
                annotation = "undefined: no finite resampling deltas"
            elif challenger == reference:
                annotation = "identical fixed forecast; no new comparison"
            elif interval[1] < 0:
                annotation = "negative conditional interval; resampling stability only"
            elif interval[0] > 0:
                annotation = "positive conditional interval; larger challenger error"
            else:
                annotation = "conditional interval includes zero; direction is not robust under this grouping"
            comparisons[name]["thresholds"][label] = {
                "observed_threshold_km_s": threshold, "point_n": int(rows.sum()),
                "challenger_rmse_km_s": result_a, "reference_rmse_km_s": result_b,
                "delta_rmse_km_s": point_delta, "percentile_95_interval_km_s": interval,
                "defined_replicates": n_defined, "undefined_replicates": draws - n_defined,
                "fraction_negative": float(np.mean(finite < 0)) if n_defined else None,
                "fraction_positive": float(np.mean(finite > 0)) if n_defined else None,
                "fraction_zero": float(np.mean(finite == 0)) if n_defined else None,
                "fraction_denominator": "defined replicates only", "annotation": annotation,
            }
    return {**grouping, "draws": draws, "seed": seed, "resampling": "sample cluster_count clusters uniformly with replacement; paired draws shared across models and thresholds",
            "percentile_method": "linear", "delta_sign": "challenger minus reference; negative favors challenger",
            "few_cluster_warning": n_clusters < 2, "comparisons": comparisons}


def registered_verdict(result: dict, schemes: dict) -> dict:
    p0_fast = result["candidate_reports"]["P0"]["metrics"]["fast_wind"]["ge_500"]["continuous_observed_fast"]["rmse"]
    limit = None if p0_fast is None else .01 * p0_fast
    checks = {}
    for name, scheme in schemes.items():
        values = scheme["comparisons"]["H11selected_vs_P0"]["thresholds"]
        overall, fast = values["all"]["percentile_95_interval_km_s"], values["observed_ge_500"]["percentile_95_interval_km_s"]
        checks[name] = {"overall_interval_upper_below_zero": None if overall is None else overall[1] < 0,
                        "fast500_interval_upper_within_one_percent_P0_RMSE": None if fast is None or limit is None else fast[1] <= limit}
    defined = all(value is not None for checks_for_scheme in checks.values() for value in checks_for_scheme.values())
    supported = defined and all(value for checks_for_scheme in checks.values() for value in checks_for_scheme.values())
    status = "inconclusive_no_H11_adoption" if not result["adoption"] else (
        "inconclusive_undefined_interval" if not defined else "supported_conditionally" if supported else "not_supported_conditionally")
    return {"status": status, "non_P0_adoption": result["adoption"], "checks": checks,
            "fast500_upper_delta_limit_km_s": limit,
            "rule": "If a non-P0 candidate was adopted, BOTH groupings require overall delta interval upper < 0 and observed-fast500 delta interval upper <= 0.01 * P0 fast500 RMSE; no adoption is inconclusive.",
            "selected_before_audit": result["selected"], "selected_after_audit": result["selected"],
            "model_promotion_performed": False,
            "freeze_action": "Freeze the portfolio after artifact integrity passes regardless of interval direction; no audit-driven refitting or reselection."}


def run(results_directory: str | Path, output_file: str | Path) -> dict:
    output = Path(output_file).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing H12 audit")
    result, data, integrity = load_and_verify(results_directory)
    panel = comparison_panel(result["selected"])
    labels, description = block_components(data)
    primary = bootstrap(data, labels, description, panel)
    labels, description = calendar_clusters(data)
    secondary = bootstrap(data, labels, description, panel)
    schemes = {"primary_block_components": primary, "secondary_calendar_54day": secondary}
    audit = {"experiment": "H12", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
             "preregistration_url": PREREGISTRATION,
             "scope": "Conditional stability of fixed, adaptively selected training-Folds-1-4 OOF forecasts; not fresh generalization evidence or a selection-adjusted confidence statement.",
             "integrity": integrity, "selected": result["selected"], "incumbent": result["incumbent"],
             "adoption": result["adoption"], "selection_performed": False, "model_refits": 0,
             "comparison_panel": {name: {"challenger": pair[0], "reference": pair[1]} for name, pair in panel.items()},
             "bootstrap_schemes": schemes, "registered_stability_hypothesis": registered_verdict(result, schemes),
             "fold_reports": result["best_models_by_fold"],
             "pooled_forecast_metrics": {name: report["metrics"] for name, report in result["candidate_reports"].items()},
             "duplicate_sensitivity": {"definition": result["secondary_deduplication"],
                                       "reports": {name: report["unique_issue_target_day"] for name, report in result["candidate_reports"].items()}},
             "limitations": ["Predictions, fitted models, candidate choices, and original-window duplicate weighting remain fixed.",
                             "Resampling does not account for adaptive hypothesis design, model-selection optimism, or fitting uncertainty.",
                             "Dependence beyond either prespecified grouping may remain; grouping sensitivity is reported without choosing the favorable result.",
                             "Intervals and sign fractions are conditional resampling diagnostics, not p-values or new model-promotion rules.",
                             "The >=600 km/s threshold remains secondary, including unfavorable or undefined results."],
             "input_hashes": integrity["input_hashes"],
             "code_sha256": {"stability_audit.py": sha256(Path(__file__)),
                             "metrics.py": sha256(Path(__file__).with_name("metrics.py"))},
             "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    encoded = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({"experiment": "H12", "integrity_passed": True,
                      "stability_status": audit["registered_stability_hypothesis"]["status"],
                      "selected_model_unchanged": result["selected"], "output": str(output)}), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/continuation"))
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON file; existing files are never overwritten")
    args = parser.parse_args()
    run(args.results, args.output)


if __name__ == "__main__":
    main()
