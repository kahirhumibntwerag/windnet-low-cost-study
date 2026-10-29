"""Rerun the frozen study into separate outputs, without provisioning compute."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parent
FROZEN_CODE_SHA256 = "8ec6bc81c39176c7186b3c8f0de9f52e414fbe77a32830656102eb300b332188"
RECORDED_DATA_SHA256 = "f4723c671bb2f9a2f21dfa491c4c14e34626836f1fa82bf979c36a39d8ed273e"


def sha256(path: Path, normalize_lines: bool = False) -> str:
    content = path.read_bytes()
    if normalize_lines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "requests", "gdown"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages}


def validate_inputs() -> dict:
    if (REPOSITORY / "study_inputs.npz").exists():
        raise RuntimeError(
            "A root-level study_inputs.npz would override data/ in the frozen script. "
            "Use the prepared data/ inputs and move the root-level copy to a separate directory."
        )
    required = [REPOSITORY / "data" / name for name in (
        "windnet211_labels.npz", "study_inputs.npz", "source_folds.npz"
    )]
    absent = [path.name for path in required if not path.is_file()]
    if absent:
        raise RuntimeError(
            "Missing prepared inputs: " + ", ".join(absent)
            + ". Run recover_folds.py, prepare_data.py, then verify_protocol.py."
        )
    code_hash = sha256(REPOSITORY / "study.py", normalize_lines=True)
    if code_hash != FROZEN_CODE_SHA256:
        raise RuntimeError("study.py differs from the frozen scientific source; use a clean published revision.")
    subprocess.run([sys.executable, str(REPOSITORY / "verify_protocol.py")], check=True, cwd=REPOSITORY)
    hashes = {path.name: sha256(path) for path in required}
    return {
        "study_sha256_lf": code_hash,
        "input_sha256": hashes,
        "recorded_data_byte_hash_match": hashes["study_inputs.npz"] == RECORDED_DATA_SHA256,
        "protocol_checks_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "development", "full", "ablations"), default="check")
    parser.add_argument("--output", type=Path, help="New or empty output directory; original results are preserved.")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    validation = validate_inputs()
    info = environment()

    if args.mode in ("check", "development", "full"):
        try:
            import torch
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["gpu"] = torch.cuda.get_device_name(0) if info["cuda_available"] else None
        except ImportError:
            info["cuda_available"] = False
            info["gpu"] = None
        if args.mode != "check" and not info["cuda_available"]:
            raise RuntimeError("This mode requires a CUDA-enabled PyTorch device. CPU ablations use --mode ablations.")

    if args.mode == "check":
        print(json.dumps({"mode": "check", **validation, "environment": info}, indent=2))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (args.output or (REPOSITORY / "results" / "rerun" / stamp)).resolve()
    published = (REPOSITORY / "results").resolve()
    if destination == published or destination == REPOSITORY:
        raise RuntimeError("Choose a separate output directory, not the published results or repository root.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise RuntimeError("Output already exists and is not empty; choose a new directory.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "run_manifest.json"
    manifest = {
        "mode": args.mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "completed_stages": [],
        "validation": validation,
        "environment": info,
        "runner_sha256_lf": sha256(Path(__file__), normalize_lines=True),
        "provisions_compute": False,
    }
    started = time.perf_counter()

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    try:
        import study
        if args.mode == "ablations":
            import ablations
            reference = REPOSITORY / "results" / "final_results.json"
            if not reference.is_file():
                raise RuntimeError("The published final_results.json is needed for unchanged-variant checks.")
            comparison_dir = destination / "results"
            comparison_dir.mkdir()
            shutil.copy2(reference, comparison_dir / "final_results.json")
            previous = Path.cwd()
            try:
                os.chdir(destination)
                ablations.main()
            finally:
                os.chdir(previous)
            manifest["completed_stages"].append("exploratory_ablation_rerun")
        else:
            # Data globals were loaded from the repository before changing ROOT.
            # Only result writes are redirected; frozen study.py remains intact.
            study.ROOT = destination
            for stage in ("loop1", "loop2", "loop3"):
                study.development(stage)
                manifest["completed_stages"].append(stage)
                save()
            study.ensemble_development()
            manifest["completed_stages"].append("loop4")
            save()
            if args.mode == "full":
                study.final()
                manifest["completed_stages"].append("final")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({"status": "completed", "mode": args.mode, "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
