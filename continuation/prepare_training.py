"""Export and load the continuation's training-only data boundary.

Original Fold 5 is never an analysis input here. The preparation step performs
mechanical row selection from the frozen archive; it does not calculate or
display any statistic on excluded targets. No frozen study module is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
ORIGINAL_INPUTS = REPOSITORY / "data" / "study_inputs.npz"
ORIGINAL_SOURCES = REPOSITORY / "data" / "source_folds.npz"
EXPORT_DIRECTORY = REPOSITORY / "data" / "continuation"
EXPORT_NAME = "train_only.npz"
MANIFEST_NAME = "train_only.manifest.json"
FROZEN_INPUT_SHA256 = "f4723c671bb2f9a2f21dfa491c4c14e34626836f1fa82bf979c36a39d8ed273e"
FROZEN_SOURCES_SHA256 = "a3835b999a5a95ef3bb29e55fc2bbdf040939ffadb3ba7d525ecb6223d660d45"
ALLOWED_TARGET_FOLDS = (1, 2, 3, 4)
SEALED_TARGET_FOLD = 5
FEATURE_WIDTHS = {"short": 4, "recurrence": 34, "multi": 78}
EXPECTED_EXPORT_KEYS = {
    "short", "recurrence", "multi", "y", "fold", "block", "offset",
    "date", "rowid", "fit_eligible", "source_fold_short",
    "source_fold_recurrence", "source_fold_multi",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_arrays(data: dict[str, np.ndarray]) -> None:
    if set(data) != EXPECTED_EXPORT_KEYS:
        raise ValueError("The file is not the allowed training-only export schema.")
    folds = data["fold"]
    if folds.ndim != 1 or not len(folds):
        raise ValueError("Training fold identifiers are missing or malformed.")
    if not np.isin(folds, ALLOWED_TARGET_FOLDS).all():
        raise ValueError("Rejected export: target fold is outside allowed Folds 1-4.")
    n = len(folds)
    for key in ("y", "block", "offset", "date", "rowid", "fit_eligible"):
        if data[key].shape != (n,):
            raise ValueError(f"Training field {key!r} has an inconsistent shape.")
    if not np.isfinite(data["y"]).all():
        raise ValueError("Training targets contain missing or non-finite values.")
    if not np.isin(data["offset"], np.arange(6, 20)).all():
        raise ValueError("Training-side rows differ from the original evaluation offsets.")
    if not np.array_equal(data["fit_eligible"], data["offset"] < 19):
        raise ValueError("Original 13-window fitting eligibility was not preserved.")
    if len(np.unique(data["rowid"])) != n:
        raise ValueError("Original row identifiers must be unique.")
    for kind, width in FEATURE_WIDTHS.items():
        x = data[kind]
        source = data[f"source_fold_{kind}"]
        if x.shape != (n, width) or source.shape != x.shape:
            raise ValueError(f"Training feature/provenance shape mismatch for {kind}.")
        if not np.isin(source, np.arange(6)).all():
            raise ValueError("Unexpected source-fold identifier.")
        if not np.array_equal(source[:, :4], np.repeat(folds[:, None], 4, axis=1)):
            raise ValueError("Original four history values must originate in their own block's fold.")
        if not np.isnan(x[source == SEALED_TARGET_FOLD]).all():
            raise ValueError("Rejected export: an unmasked Fold 5 historical measurement remains.")
        if np.isinf(x).any():
            raise ValueError("Training features contain infinities.")


def load_training_bundle(directory: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load only a verified train_only export, including after copying to a worker.

    ``directory`` is a bundle directory, not an arbitrary NPZ path. A matching
    manifest, fixed upstream hashes, allowed schema, and allowed target folds
    are required. This prevents accidental routing of the full original data
    into the continuation trainer; it is not a cryptographic signature scheme.
    """
    root = Path(directory) if directory is not None else EXPORT_DIRECTORY
    if root.suffix.lower() == ".npz":
        raise ValueError("Pass a training bundle directory, not an arbitrary NPZ input.")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "allowed_target_folds": list(ALLOWED_TARGET_FOLDS),
        "excluded_target_folds": [SEALED_TARGET_FOLD],
        "original_inputs_sha256": FROZEN_INPUT_SHA256,
        "original_sources_sha256": FROZEN_SOURCES_SHA256,
        "target_values_from_excluded_folds_exported": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Rejected training manifest field: {key}.")
    path = root / EXPORT_NAME
    if sha256(path) != manifest.get("export_sha256"):
        raise ValueError("Training export hash differs from its manifest.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_arrays(arrays)
    if len(arrays["fold"]) != manifest.get("training_rows"):
        raise ValueError("Training row count differs from its manifest.")
    return arrays


def prepare() -> dict:
    if sha256(ORIGINAL_INPUTS) != FROZEN_INPUT_SHA256:
        raise ValueError("Original compact input archive differs from the frozen source hash.")
    if sha256(ORIGINAL_SOURCES) != FROZEN_SOURCES_SHA256:
        raise ValueError("Original provenance archive differs from the frozen source hash.")
    with np.load(ORIGINAL_INPUTS, allow_pickle=False) as original, np.load(
        ORIGINAL_SOURCES, allow_pickle=False
    ) as sources:
        fold_ids = original["fold"]
        keep = np.isin(fold_ids, ALLOWED_TARGET_FOLDS)
        selected_folds = fold_ids[keep].copy()
        # Mechanical selection only: no summary or inspection of excluded y.
        payload = {name: original[name][keep].copy()
                   for name in ("y", "block", "offset", "date")}
        payload["fold"] = selected_folds
        payload["rowid"] = np.flatnonzero(keep)
        payload["fit_eligible"] = payload["offset"] < 19
        for kind in FEATURE_WIDTHS:
            x = original[kind][keep].copy()
            source = sources[kind][keep].copy()
            # These four values are from the same original partition, even if
            # that partition has a calendar gap. They are not fixed-day lags.
            source[:, :4] = selected_folds[:, None]
            x[source == SEALED_TARGET_FOLD] = np.nan
            payload[kind] = x
            payload[f"source_fold_{kind}"] = source
    _validate_arrays(payload)
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIRECTORY / EXPORT_NAME
    with tempfile.NamedTemporaryFile(dir=EXPORT_DIRECTORY, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
    counts = {str(fold): int(np.sum(selected_folds == fold)) for fold in ALLOWED_TARGET_FOLDS}
    manifest = {
        "schema_version": 1,
        "allowed_target_folds": list(ALLOWED_TARGET_FOLDS),
        "excluded_target_folds": [SEALED_TARGET_FOLD],
        "target_values_from_excluded_folds_exported": False,
        "original_inputs_sha256": FROZEN_INPUT_SHA256,
        "original_sources_sha256": FROZEN_SOURCES_SHA256,
        "export_sha256": sha256(path),
        "preparation_code_sha256": sha256(Path(__file__)),
        "training_rows": len(selected_folds),
        "training_rows_by_fold": counts,
        "fitting_rows": int(np.sum(payload["fit_eligible"])),
        "fit_eligible_definition": "original offset < 19; offsets 6 through 18",
        "validation_offsets_retained": list(range(6, 20)),
        "source_fold_zero": "Unassigned historical date; retained as in the frozen source.",
        "source_provenance_correction": "First four original history columns originate in the row's own original block fold.",
        "feature_guard": "All fitting-candidate values originating in Fold 5 are NaN before export.",
        "inner_validation_requirement": "At each inner fit, additionally mask source folds withheld by that inner split before fitting preprocessing.",
        "checks": {"allowed_target_folds_only": True, "original_row_identifiers_unique": True,
                   "excluded_source_values_masked": True, "original_windows_retained": True},
    }
    target = EXPORT_DIRECTORY / MANIFEST_NAME
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=EXPORT_DIRECTORY,
                                     suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, target)
    load_training_bundle()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate an existing training-only export without reopening original data.")
    args = parser.parse_args()
    if args.check:
        data = load_training_bundle()
        manifest = json.loads((EXPORT_DIRECTORY / MANIFEST_NAME).read_text(encoding="utf-8"))
    else:
        manifest = prepare()
    print(json.dumps({"training_rows": manifest["training_rows"],
                      "training_rows_by_fold": manifest["training_rows_by_fold"],
                      "fitting_rows": manifest["fitting_rows"],
                      "export_sha256": manifest["export_sha256"],
                      "partition_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
