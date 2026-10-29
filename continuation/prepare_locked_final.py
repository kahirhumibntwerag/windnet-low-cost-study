"""Mechanically prepare the one-shot final bundle ONLY after portfolio freeze.

Importing this module opens no scientific file. Both CLI execution and callers
must supply the exact frozen manifest hash and E13 preregistration URL.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

import numpy as np

try:
    from .train import recompute_eve_features, sha256
    from .prior_champion import recipe_metadata
    from .prepare_training import FROZEN_INPUT_SHA256, FROZEN_SOURCES_SHA256
except ImportError:
    from train import recompute_eve_features, sha256
    from prior_champion import recipe_metadata
    from prepare_training import FROZEN_INPUT_SHA256, FROZEN_SOURCES_SHA256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("study_inputs.npz", "source_folds.npz", "eve_source.ncdf", "matched_predictions.csv")
REFERENCE_COLUMNS = frozenset(("row", "fold", "block", "offset", "date", "y", "selected_model", "released_windnet"))
CODE_NAMES = ("prepare_locked_final.py", "locked_final.py", "final_uncertainty.py", "final_reporting.py", "prior_champion.py", "models.py", "metrics.py", "train.py", "prepare_training.py", "fast_loss.py")
LINES = (("Fe_IX_17", "Fe IX", 17.1073), ("Fe_XII_20", "Fe XII", 19.512),
         ("Fe_XIV_21", "Fe XIV", 21.1331), ("He_II_30", "He II", 30.3783),
         ("Fe_XVI_34", "Fe XVI", 33.541), ("He_I_58", "He I", 58.4334))


def locked_recipes() -> dict:
    return {"schema": "E13_fixed_panel_v1", "outer_folds": [1, 2, 3, 4, 5],
            "fit_offsets": list(range(6, 19)), "test_offsets": list(range(6, 20)),
            "feature_groups": {"P0": "wind78", "P1": "wind78+ESP18", "P2": "wind78+lines44", "P3": "wind78+allEVE62"},
            "prior_champion": recipe_metadata(),
            "h9_fast_specialist": {"features": "wind78+lines44", "pair": {"ridge_alpha": 1000, "ridge_weight": 0.25, "tree_weight": 0.75},
                                   "unweighted_pair_weight": 0.75, "raw_weight4_pair_weight": 0.25,
                                   "training_weight": "1+3*(y_fit>=500), no normalization; explicit imputer/scaler unweighted"},
            "starting_wind": {"features": "wind78", "ridge_alpha": 1000, "ridge_weight": 0.75, "tree_weight": 0.25, "sample_weight": None},
            "h8_sdo": {"features": "wind78+allEVE62", "ridge_alpha": 1000, "ridge_weight": 0.25, "tree_weight": 0.75, "sample_weight": None},
            "classical_component_ablations": {
                "matched_classical_wind": {"features": "wind78", "ridge_alpha": 1000, "ridge_weight": 0.25, "tree_weight": 0.75, "sample_weight": None},
                "h9_unweighted_lines": {"features": "wind78+lines44", "ridge_alpha": 1000, "ridge_weight": 0.25, "tree_weight": 0.75, "sample_weight": None},
                "h9_weight4_lines": {"features": "wind78+lines44", "ridge_alpha": 1000, "ridge_weight": 0.25, "tree_weight": 0.75,
                                     "sample_weight": "1+3*(y_fit>=500), no normalization; explicit imputer/scaler unweighted"}},
            "diversity": {"P4": {"P0": 0.75, "h9_fast_specialist": 0.25},
                          "P5": {"P0": 0.5, "h9_fast_specialist": 0.5},
                          "P6": {"P0": 0.25, "h9_fast_specialist": 0.75}},
            "tree": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 7,
                     "min_samples_leaf": 30, "l2_regularization": 10, "early_stopping": False, "random_state": 2026},
            "frozen_references": ["old_frozen_selected", "released_windnet"],
            "primary": "starting_wind", "fast_wind_primary": "h9_fast_specialist",
            "checkpoint_retention": "Each outer fold retains CPU joblib state for P0-P3 and four ridge/tree pairs; fixed mixes reconstruct derived models; archive checksummed before scoring",
            "thresholds_km_s": [500, 600], "selection_from_final_test": False,
            "paired_uncertainty": {"roles": ["starting_wind", "h9_fast_specialist", "P0"],
                                   "references": ["released_windnet", "old_frozen_selected"],
                                   "conditions": ["all", "observed>=500", "observed>=600"],
                                   "scopes": ["all_original_windows", "original_fold5"],
                                   "bin_days": 54, "bin_anchor_utc": "2011-01-01", "resamples": 5000, "seed": 20261029,
                                   "interval": "paired pooled RMSE difference, percentile95", "selection_use": False},
            "source_mask": "heldout outer-fold source values masked on FIT only; summaries recomputed; test causal history retained",
            "purge": "entire fitting block removed if any target date overlaps heldout target dates",
            "evaluation_unit": "original evaluation windows; issue=target-3 day deduplicated metrics secondary"}


def recipes_sha256() -> str:
    return hashlib.sha256(json.dumps(locked_recipes(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_freeze(path: str | Path, expected_hash: str, protocol_url: str) -> dict:
    """This gate must succeed BEFORE any original scientific source is opened."""
    manifest_path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or "") or sha256(manifest_path) != expected_hash:
        raise ValueError("Missing or mismatched exact freeze-manifest SHA256")
    if not re.fullmatch(r"https://fixrelay\.vercel\.app/experiments/[0-9a-f-]{36}", protocol_url or ""):
        raise ValueError("An explicit E13 public preregistration URL is required")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema": "locked_final_v1", "status": "frozen", "experiment": "E13", "protocol_url": protocol_url,
                "recipes_sha256": recipes_sha256(), "selection_from_final_test": False,
                "primary": "starting_wind", "fast_wind_primary": "h9_fast_specialist"}
    for name, value in required.items():
        if frozen.get(name) != value:
            raise ValueError(f"Unfrozen or inconsistent E13 manifest: {name}")
    hashes = frozen.get("source_sha256", {})
    if set(hashes) != set(SOURCE_NAMES) or any(not re.fullmatch(r"[0-9a-f]{64}", hashes[name]) for name in SOURCE_NAMES):
        raise ValueError("All four immutable full-source hashes must be frozen")
    if hashes["study_inputs.npz"] != FROZEN_INPUT_SHA256 or hashes["source_folds.npz"] != FROZEN_SOURCES_SHA256:
        raise ValueError("Original WindNet-matched source hashes differ")
    if hashes["eve_source.ncdf"] != "f150da94bec13f171f95e0ebbb7a368c8323f6292743180eaef27a597819a6f8":
        raise ValueError("EVE must be the previously frozen cached V8 product")
    code = frozen.get("code_sha256", {})
    if set(code) != set(CODE_NAMES):
        raise ValueError("All executable final-evaluation sources must be frozen")
    for name in CODE_NAMES:
        if sha256(Path(__file__).with_name(name)) != code[name]:
            raise ValueError(f"Final-evaluation source changed after freeze: {name}")
    return frozen


# Public pure aliases for freeze creation; these never open scientific files.
fixed_panel_recipe = locked_recipes
recipe_sha256 = recipes_sha256


def _text(value) -> str:
    value = np.asarray(value).reshape(-1)[0]
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def validate_reference_header(columns) -> None:
    if not REFERENCE_COLUMNS.issubset(columns or ()):
        raise ValueError("Frozen reference CSV requires row, fold, block, offset, date, y and both fixed forecasts")


def eve_from_cache(path: Path, target_dates: np.ndarray, source_recurrence: np.ndarray) -> dict:
    """Offline EVE metadata/time matching, without target or feature statistics."""
    import h5py
    from astropy.time import Time, TimeDelta
    fields, values, metadata = [], [], []
    with h5py.File(path, "r") as stream:
        if _text(stream.attrs["product_version"]) != "008":
            raise ValueError("Expected frozen EVE calibration V8")
        waves = stream["linesmeta_wave_center"][:]
        ions = [_text(value) for value in stream["linesmeta_name"][:]]
        for name, ion, wavelength in LINES:
            hits = [i for i, (label, wave) in enumerate(zip(ions, waves)) if label == ion and abs(wave - wavelength) < 0.002]
            if len(hits) != 1:
                raise ValueError("Ambiguous frozen EVE line metadata")
            index = hits[0]
            fields.append(name)
            values.append(stream["line_irradiance"][index, :])
            metadata.append({"name": name, "ion": ion, "wavelength_nm": float(waves[index]), "units": "W/m^2"})
        for name, index in (("diode171", 1), ("diode257", 2), ("diode304", 3)):
            fields.append(name)
            values.append(stream["diode_irradiance"][index, :])
            metadata.append({"name": name, "description": _text(stream["diodemeta_name"][index]), "units": "W/m^2"})
        tai = Time("1958-01-01T00:00:00", scale="tai") + TimeDelta(stream["time"][:], format="sec")
        utc_days = np.asarray(tai.utc.isot, dtype="datetime64[ms]").astype("datetime64[D]")
        year_days = stream["yyyydoy"][:].astype(int)
        independent_days = np.asarray([np.datetime64(date(int(value // 1000), 1, 1) + timedelta(days=int(value % 1000) - 1)) for value in year_days])
        if not np.array_equal(utc_days, independent_days):
            raise ValueError("Independent EVE TAI and year/day dates disagree")
    if len(np.unique(utc_days)) != len(utc_days):
        raise ValueError("Duplicate EVE daily dates")
    daily = np.column_stack(values).astype(float)
    daily[~np.isfinite(daily) | (daily <= 0)] = np.nan
    lookup = {str(day): i for i, day in enumerate(utc_days)}
    dates = target_dates.astype("datetime64[D]")
    lag_values = np.full((len(dates), 4, len(fields)), np.nan)
    lag_sources = np.column_stack([source_recurrence[:, 4 + lag - 3] for lag in (5, 6, 7, 8)])
    for position, lag in enumerate((5, 6, 7, 8)):
        for row, day in enumerate(dates - np.timedelta64(lag, "D")):
            index = lookup.get(str(day))
            if index is not None:
                lag_values[row, position] = np.log(daily[index])
    # No global Fold 5 masking: each final outer fit masks its own heldout fold.
    features, names = recompute_eve_features(lag_values, fields)
    return {"eve_lag_values": lag_values, "eve_source_fold": lag_sources, "eve_features": features,
            "eve_names": np.asarray(names), "eve_fields": np.asarray(fields), "channel_metadata": metadata}


def prepare(freeze_manifest: str | Path, freeze_hash: str, protocol_url: str,
            output_directory: str | Path = ROOT / "data" / "locked_final",
            eve_source: str | Path = ROOT / "data" / "continuation" / "eve_source" / "latest_EVE_L3_merged_1nm.ncdf") -> dict:
    frozen = validate_freeze(freeze_manifest, freeze_hash, protocol_url)
    output = Path(output_directory)
    paths = {"study_inputs.npz": ROOT / "data" / "study_inputs.npz", "source_folds.npz": ROOT / "data" / "source_folds.npz",
             "eve_source.ncdf": Path(eve_source), "matched_predictions.csv": ROOT / "results" / "matched_predictions.csv"}
    if any((output / name).exists() for name in ("final_bundle.npz", "final_bundle.manifest.json")):
        raise ValueError("Refusing to overwrite a locked final bundle")
    for name, path in paths.items():
        if sha256(path) != frozen["source_sha256"][name]:
            raise ValueError(f"Frozen full-source identity differs: {name}")
    with np.load(paths["study_inputs.npz"], allow_pickle=False) as original, np.load(paths["source_folds.npz"], allow_pickle=False) as source:
        payload = {name: original[name].copy() for name in ("y", "fold", "block", "offset", "date", "short", "recurrence", "multi")}
        if len(payload["y"]) != 2198 or set(np.unique(payload["fold"]).tolist()) != {1, 2, 3, 4, 5}:
            raise ValueError("Final rows differ from the fixed released evaluation panel")
        payload["rowid"] = np.arange(2198)
        payload["fit_eligible"] = payload["offset"] < 19
        for kind, width in (("short", 4), ("recurrence", 34), ("multi", 78)):
            sources = source[kind].copy()
            if sources.shape != (2198, width) or payload[kind].shape != sources.shape:
                raise ValueError("Frozen feature/provenance dimensions differ")
            sources[:, :4] = payload["fold"][:, None]
            payload[f"source_fold_{kind}"] = sources
    if not np.isfinite(payload["y"]).all() or not np.isin(payload["offset"], np.arange(6, 20)).all():
        raise ValueError("Malformed original evaluation targets or offsets")
    with paths["matched_predictions.csv"].open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        validate_reference_header(reader.fieldnames)
        rows = list(reader)
    if len(rows) != 2198 or len({int(row["row"]) for row in rows}) != 2198:
        raise ValueError("Frozen reference rows are incomplete or duplicated")
    old, windnet = np.full(2198, np.nan), np.full(2198, np.nan)
    for row in rows:
        index = int(row["row"])
        if not 0 <= index < 2198 or int(row["fold"]) != payload["fold"][index] or row["date"] != str(payload["date"][index]) or not np.isclose(float(row["y"]), payload["y"][index], rtol=0, atol=1e-8):
            raise ValueError("Frozen comparator row identity differs from the original bundle")
        for name in ("block", "offset"):
            if int(row[name]) != payload[name][index]:
                raise ValueError("Frozen comparator window identity differs")
        old[index], windnet[index] = float(row["selected_model"]), float(row["released_windnet"])
    if np.isinf(old).any() or np.isinf(windnet).any():
        raise ValueError("Infinite frozen reference forecast")
    payload["old_frozen_selected"], payload["released_windnet"] = old, windnet
    eve = eve_from_cache(paths["eve_source.ncdf"], payload["date"], payload["source_fold_recurrence"])
    channel_metadata = eve.pop("channel_metadata")
    payload.update(eve)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final_bundle.npz"
    np.savez_compressed(path, **payload)
    manifest = {"schema": "locked_final_bundle_v1", "scope": "final_only", "experiment": "E13", "freeze_manifest_sha256": freeze_hash,
                "protocol_url": protocol_url, "recipes_sha256": recipes_sha256(), "export_sha256": sha256(path),
                "source_sha256": frozen["source_sha256"], "channel_metadata": channel_metadata,
                "source_provenance_correction": "First four original history columns use own block fold; no global excluded-fold mask",
                "preparation_code_sha256": sha256(Path(__file__)), "selection_from_final_test": False}
    (output / "final_bundle.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--protocol-url", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "locked_final")
    parser.add_argument("--eve-source", type=Path, default=ROOT / "data" / "continuation" / "eve_source" / "latest_EVE_L3_merged_1nm.ncdf")
    args = parser.parse_args()
    result = prepare(args.freeze_manifest, args.freeze_sha256, args.protocol_url, args.output, args.eve_source)
    print(json.dumps({"status": "complete", "scope": "final_only", "export_sha256": result["export_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
