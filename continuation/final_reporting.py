"""Readable tables and figures from COMPLETED locked final outputs only.

No source data, original holdouts, intermediate experiments or checkpoints are
opened. No model fits, selection or recipe changes are performed here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

try:
    from .locked_final import MODEL_IDS
    from .train import sha256
except ImportError:
    from locked_final import MODEL_IDS
    from train import sha256


ROOT = Path(__file__).resolve().parents[1]
ROLE_IDS = ("released_windnet", "old_frozen_selected", "P0", "starting_wind", "h9_fast_specialist")
NN_IDS = ("P0", "P1", "P2", "P3")
CLASSICAL_IDS = ("matched_classical_wind", "h9_unweighted_lines", "h9_weight4_lines", "h9_fast_specialist")
REFERENCE_IDS = ("released_windnet", "old_frozen_selected")
LABELS = {"released_windnet": "Released WindNet", "old_frozen_selected": "Earlier frozen winner",
          "P0": "Neural/tree: wind only", "P1": "Neural/tree: + ESP", "P2": "Neural/tree: + EUV lines",
          "P3": "Neural/tree: + all EVE", "P4": "Neural/tree + 25% fast model",
          "P5": "Neural/tree + 50% fast model", "P6": "Neural/tree + 75% fast model",
          "starting_wind": "Starting wind model", "h9_fast_specialist": "Fast-wind model",
          "h8_sdo": "Classical: + all EVE", "matched_classical_wind": "Classical: wind only",
          "h9_unweighted_lines": "Classical: + EUV lines", "h9_weight4_lines": "EUV lines: fast weight 4"}


def summary_row(result: dict, identifier: str) -> dict:
    """One fixed-order, human-readable row using existing final metrics."""
    model = result["metrics"][identifier]
    pooled = model["pooled"]
    overall = pooled["overall"]
    row = {"model_id": identifier, "model": LABELS[identifier],
           "mean_fold_rmse_km_s": model["mean_rmse"], "se_fold_rmse_km_s": model["rmse_se"],
           "mean_fold_mae_km_s": model["mean_mae"], "fisher_mean_r": model["fisher_mean_r"],
           "pooled_n": overall["n"], "pooled_rmse_km_s": overall["rmse"],
           "pooled_mae_km_s": overall["mae"], "pooled_bias_km_s": overall["bias"], "pooled_r": overall["r"]}
    for threshold in (500, 600):
        fast = pooled["fast_wind"][f"ge_{threshold}"]
        continuous, detection = fast["continuous_observed_fast"], fast["detection"]
        row[f"fast{threshold}_n"] = continuous["n"]
        for field in ("rmse", "mae", "bias"):
            row[f"fast{threshold}_{field}_km_s"] = continuous[field]
        for field in ("tp", "fp", "fn", "tn", "pod", "far", "csi", "tss", "hss"):
            row[f"fast{threshold}_{field}"] = detection[field]
    return row


def reference_rows(result: dict) -> list[dict]:
    rows = []
    for identifier in MODEL_IDS:
        for reference in REFERENCE_IDS:
            for condition in ("all", "ge_500", "ge_600"):
                value = result["metrics"][identifier]["pooled"]["reference_skill"][reference][condition]
                rows.append({"model_id": identifier, "model": LABELS[identifier], "reference_id": reference,
                             "reference": LABELS[reference], "condition": condition,
                             "n_common_finite": value["n_common_finite"], "n_paired_condition": value["n"],
                             "model_rmse_km_s": value["model"]["rmse"], "reference_rmse_km_s": value["reference"]["rmse"],
                             "mse_skill_score": value["mse_skill_score"], "delta_rmse_km_s": value["delta_rmse"]})
    return rows


def detection_rows(result: dict) -> list[dict]:
    rows = []
    for identifier in MODEL_IDS:
        for threshold in (500, 600):
            detection = result["metrics"][identifier]["pooled"]["fast_wind"][f"ge_{threshold}"]["detection"]
            rows.append({"model_id": identifier, "model": LABELS[identifier], "threshold_km_s": threshold,
                         **{name: detection[name] for name in ("tp", "fp", "fn", "tn", "pod", "far", "fpr", "csi", "tss", "hss", "frequency_bias")}})
    return rows


def paired_detection_rows(result: dict) -> list[dict]:
    rows = []
    for identifier in MODEL_IDS:
        for reference in REFERENCE_IDS:
            for threshold in (500, 600):
                comparison = result["metrics"][identifier]["pooled"]["reference_detection_deltas"][reference][f"ge_{threshold}"]
                row = {"model_id": identifier, "model": LABELS[identifier], "reference_id": reference,
                       "reference": LABELS[reference], "threshold_km_s": threshold, "n_common_finite": comparison["n_common"]}
                for side in ("model", "reference"):
                    row.update({f"{side}_{name}": comparison[side][name] for name in ("tp", "fp", "fn", "tn", "pod", "far", "csi", "tss", "hss")})
                row.update({f"delta_{name}": comparison["delta"][name] for name in ("pod", "far", "fpr", "csi", "tss", "hss", "frequency_bias")})
                rows.append(row)
    return rows


def ablation_rows(result: dict, identifiers: tuple[str, ...]) -> list[dict]:
    rows = [summary_row(result, identifier) for identifier in identifiers]
    baseline = rows[0]
    for row in rows:
        row["ablation_control_id"] = baseline["model_id"]
        for key in ("mean_fold_rmse_km_s", "pooled_rmse_km_s", "fast500_rmse_km_s", "fast600_rmse_km_s"):
            row[f"delta_{key}_vs_control"] = None if row[key] is None or baseline[key] is None else row[key] - baseline[key]
    return rows


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figures(result: dict, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.dpi": 200})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(fig, name: str, caption: str) -> None:
        written = {}
        for extension in ("png", "svg"):
            path = figures / f"final_{name}.{extension}"
            if path.exists():
                raise ValueError("Refusing to overwrite a final reporting figure")
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written[extension] = str(path.resolve())
        paths[name] = {**written, "caption": caption}
        plt.close(fig)

    def errors(identifiers: tuple[str, ...], name: str, caption: str) -> None:
        rows = [summary_row(result, identifier) for identifier in identifiers]
        fig, axes = plt.subplots(1, 3, figsize=(12, max(3.3, 0.58 * len(rows) + 1.2)), sharey=True)
        positions = np.arange(len(rows))
        colors = ["#B96743" if identifier == "released_windnet" else "#8B939B" if identifier == "old_frozen_selected" else "#286386" for identifier in identifiers]
        for ax, column, title in zip(axes, ("mean_fold_rmse_km_s", "fast500_rmse_km_s", "fast600_rmse_km_s"),
                                     ("All speeds: fold mean ± SE", "Observed speed ≥ 500 km/s", "Observed speed ≥ 600 km/s")):
            values = np.asarray([row[column] if row[column] is not None else np.nan for row in rows])
            err = np.asarray([row["se_fold_rmse_km_s"] or 0.0 for row in rows]) if column == "mean_fold_rmse_km_s" else None
            ax.barh(positions, values, xerr=err, color=colors, height=0.58, capsize=3)
            finite = values[np.isfinite(values)]
            limits = values + err if err is not None else values
            finite_limits = limits[np.isfinite(limits)]
            maximum = max(float(finite_limits.max()) if len(finite_limits) else 1.0, 1.0)
            ax.set_xlim(0, maximum * 1.24)
            for index, value in enumerate(values):
                if np.isfinite(value):
                    ax.text(value + (err[index] if err is not None else 0.0) + maximum * 0.025, index, f"{value:.1f}", va="center", fontsize=9)
                else:
                    ax.text(maximum * 0.02, index, "Undefined", va="center", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("RMSE (km/s; lower is better)")
            ax.grid(axis="x", alpha=0.16)
            ax.set_axisbelow(True)
        axes[0].set_yticks(positions, [LABELS[identifier] for identifier in identifiers])
        axes[0].invert_yaxis()
        fig.tight_layout(w_pad=2)
        save(fig, name, caption)

    errors(ROLE_IDS, "fixed_roles", "Five roles were fixed before final evaluation. The first panel shows the unweighted mean of five original-fold RMSEs with descriptive standard error. Fast-wind panels pool original windows conditioned on observed speed; misses remain included. No ranking or promotion is performed.")
    errors(NN_IDS, "neural_feature_ablations", "Same fixed neural/tree recipe and seeds in every row: wind only, plus three ESP channels, plus six EUV lines, or plus both. Differences measure feature additions within this recipe; original windows retain duplicate weights.")
    errors(CLASSICAL_IDS, "classical_components", "Wind and EUV-line controls use the same 25% ridge/75% tree pair. The weighted row gives observed fast fitting cases raw weight four; H9 mixes 75% unweighted and 25% weighted line forecasts. Weights also affect HGB internal binning and gradient/Hessian scale relative to fixed L2.")

    matrix = np.asarray([[result["metrics"][identifier]["pooled"]["reference_skill"][reference][condition]["mse_skill_score"]
                          for reference in REFERENCE_IDS for condition in ("all", "ge_500", "ge_600")]
                         for identifier in ROLE_IDS], dtype=float)
    finite = matrix[np.isfinite(matrix)]
    limit = max(float(np.max(np.abs(finite))) if len(finite) else 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    display = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "Undefined" if not np.isfinite(value) else f"{value:+.3f}", ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(value) and abs(value) > 0.6 * limit else "#18222D")
    ax.set_yticks(range(len(ROLE_IDS)), [LABELS[identifier] for identifier in ROLE_IDS])
    ax.set_xticks(range(6), [f"{label}\n{condition}" for label in ("WindNet", "Earlier winner") for condition in ("All", "≥500", "≥600")])
    ax.set_title("Paired mean-squared-error skill: positive favors the model")
    fig.colorbar(display, ax=ax, label="MSE skill score", shrink=0.8)
    fig.tight_layout()
    save(fig, "paired_reference_skill", "MSE skill uses identical finite model/reference/observation cases for each cell. Zero is parity, positive favors the row model; a perfect reference makes normalized skill undefined. Conditions use observed speed in km/s.")

    counts = np.asarray([[result["metrics"][identifier]["pooled"]["fast_wind"][f"ge_{threshold}"]["detection"][field]
                          for threshold in (500, 600) for field in ("tp", "fp", "fn", "tn")] for identifier in ROLE_IDS])
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.imshow(counts, cmap="Blues", aspect="auto")
    maximum = max(float(counts.max()), 1.0)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, str(int(counts[i, j])), ha="center", va="center", color="white" if counts[i, j] > maximum * 0.6 else "#18222D")
    ax.set_yticks(range(len(ROLE_IDS)), [LABELS[identifier] for identifier in ROLE_IDS])
    ax.set_xticks(range(8), [f"≥{threshold}\n{label}" for threshold in (500, 600) for label in ("Hit", "False alarm", "Miss", "Correct quiet")])
    ax.set_title("Day/window detection counts at fixed observed-speed thresholds")
    fig.tight_layout()
    save(fig, "detection_counts", "Hits=TP, false alarms=FP, misses=FN, correct quiet=TN on finite forecast/observation pairs. These are day/window counts with original duplicate weights, not independent high-speed-stream events. Paired counts and skill against both frozen references are in the CSV tables.")
    return paths


def run(results_path: str | Path, predictions_path: str | Path, output_directory: str | Path) -> dict:
    results_path, predictions_path, output = Path(results_path), Path(predictions_path), Path(output_directory)
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("experiment") != "E13" or result.get("status") != "completed" or result.get("selection_from_final_test") is not False:
        raise ValueError("Reporting requires completed locked E13 outputs with no final-test selection")
    if result.get("model_ids") != list(MODEL_IDS) or set(result.get("metrics", {})) != set(MODEL_IDS):
        raise ValueError("Completed output differs from the fixed fifteen-model panel")
    if result.get("prediction_file_sha256") != sha256(predictions_path):
        raise ValueError("Completed prediction file differs from its recorded hash")
    with np.load(predictions_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["rowid"], np.arange(2198)) or set(np.unique(archive["fold"]).tolist()) != {1, 2, 3, 4, 5}:
            raise ValueError("Completed final prediction row/fold identity differs")
        if [str(name) for name in archive["model_names"]] != list(MODEL_IDS) or archive["predictions"].shape != (2198, 15):
            raise ValueError("Completed fixed-panel prediction dimensions differ")
    filenames = ("report_summary.json", "final_fixed_roles.csv", "final_all_models.csv", "final_neural_ablations.csv", "final_classical_ablations.csv",
                 "final_reference_skill.csv", "final_detection_counts.csv", "final_paired_detection.csv")
    if any((output / name).exists() for name in filenames):
        raise ValueError("Refusing to overwrite a completed final report")
    output.mkdir(parents=True, exist_ok=True)
    role_rows = [summary_row(result, identifier) for identifier in ROLE_IDS]
    all_rows = [summary_row(result, identifier) for identifier in MODEL_IDS]
    nn_rows, classical_rows = ablation_rows(result, NN_IDS), ablation_rows(result, CLASSICAL_IDS)
    paired_rows, counts, paired_counts = reference_rows(result), detection_rows(result), paired_detection_rows(result)
    tables = dict(zip(filenames[1:], (role_rows, all_rows, nn_rows, classical_rows, paired_rows, counts, paired_counts)))
    for filename, rows in tables.items():
        _csv(output / filename, rows)
    figures = _figures(result, output)
    summary = {"experiment": "E13", "status": "completed_reporting", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_results_sha256": sha256(results_path), "source_predictions_sha256": sha256(predictions_path),
               "reporting_code_sha256": sha256(Path(__file__)), "protocol_url": result["protocol_url"],
               "freeze_manifest_sha256": result["freeze_manifest_sha256"], "selection_from_final_test": False,
               "fixed_roles": role_rows, "all_models": all_rows, "neural_feature_ablations": nn_rows,
               "classical_component_ablations": classical_rows, "reference_skill": paired_rows,
               "detection_counts": counts, "paired_detection": paired_counts, "figures": figures,
               "tables": {name: {"path": str((output / name).resolve()), "sha256": sha256(output / name)} for name in tables},
               "paired_final_uncertainty": result["paired_final_uncertainty"], "caveats": result["caveats"],
               "reading_notes": ["All displayed roles, ablations and thresholds were fixed before final evaluation; tables preserve fixed order.",
                                 "Overall fold means and pooled fast-wind errors use different declared aggregations.",
                                 "Negative bias means underprediction; negative paired RMSE differences and positive MSE skill favor the row model.",
                                 "Mean-fold standard errors and bootstrap intervals describe this historical benchmark, not independent confirmation.",
                                 "All EVE data are retrospective calibrated scalar measurements; no direct SDO image data are used."]}
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report_summary": str((output / "report_summary.json").resolve()), "figure_count": len(figures)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Completed locked final_results.json only")
    parser.add_argument("--predictions", type=Path, required=True, help="Its matching completed predictions.npz only")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    run(args.results, args.predictions, args.output)


if __name__ == "__main__":
    main()
