# Training-only continuation boundary

The continuation does not use original Fold 5 for feature selection, tuning,
error analysis, threshold selection, or development metrics. Previous test
results remain historical results; this boundary cannot make already viewed
scores into fresh confirmation.

Create the isolated bundle from the frozen local source files:

```bash
python continuation/prepare_training.py
python continuation/prepare_training.py --check
```

The preparation step mechanically selects target rows from original Folds 1-4.
It does not inspect or summarize excluded target values. The resulting ignored
directory contains only `train_only.npz` and `train_only.manifest.json` under
`data/continuation/`. Original Fold 5 targets, old forecasts, WindNet prediction
arrays, and held-out plots are absent from this export.

The bundle retains the original offsets 6-19 for training-side validation.
`fit_eligible` is true only for offsets 6-18, preserving the original fitting
script's 13-window convention. `rowid` identifies the original row without
carrying its excluded neighbors.

Each feature matrix has a corresponding `source_fold_*` array. The original
four observation-window columns always originate in the same partition as
their target row; their source IDs are corrected accordingly, including for
partitions with calendar gaps. All historical feature values whose source
fold is 5 are replaced by missing values before export. Source ID 0 denotes
an unassigned historical date and is retained. Do not interpret it as a target
fold or impute it using excluded observations.

A trainer should import `load_training_bundle` from `prepare_training` and
use this bundle alone. The loader checks the bundle's manifest, exact export
hash, fixed original input hashes, field schema, target-fold allowlist,
original window offsets, and source masking. It accepts a bundle directory
after copying the two bundle files to a worker; it does not accept an arbitrary
NPZ filename. These checks prevent accidental data routing, not malicious
tampering or independent confirmation of the experiment.

For every inner-validation split, the trainer must additionally replace
historical inputs originating in that inner validation fold with missing
values in its fitting rows. Fit imputation, scaling, feature selection,
calibration, and all model choices only on that inner fitting data. Report
development scores only from training-side inner validation. Do not import
the frozen `study.py` into the new trainer: it loads the original full target
array at import time.

This preparation does not train a model, provision compute, access credentials,
or evaluate any original held-out forecasts.
