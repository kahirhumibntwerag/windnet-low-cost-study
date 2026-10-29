# Completed continuation and locked final comparison

The [eight-page full paper](../output/pdf/windnet-full-study.pdf) and [E13 results](../results/locked_final) are complete. The [hosted final dashboard](https://windnet-research-lab-oct29.wellbat32.chatgpt.site/locked-final.html) remains owner-private; the [public source branch](https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-study) publishes the paper, results, exact input bundles, and saved weights. All 13 completed/cancelled [RogNet/FixRelay campaign records](https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee) are public.

Neither preregistered balanced-improvement gate passed in the fixed 15-model E13 panel. Fast-wind error improvements coexist with detection and overall-error trade-offs. P1's positive final ESP ablation is descriptive and was not promoted after evaluation. No tuning or model changes followed final scoring. The previously viewed historical holdouts are not fresh confirmation.

The training stage H7-H12 made **zero original-holdout evaluations**. The subsequent E13 stage was separately preregistered, frozen, and executed once. Its [final-only input bundle](../artifacts/locked_final) must not be routed back into selection. The archive retains **40 CPU-portable fitted artifacts**, eight per fold, from which the fixed mixtures can be reconstructed. [Final reproduction instructions](../results/locked_final/README.md) use a new copied bundle directory, a new output directory, the exact freeze hash, and both persistent one-shot markers.

The E13 reference runtime was **CPU**, Python 3.12.3, NumPy 2.3.1, SciPy 1.16.1, scikit-learn 1.9.1, and PyTorch 2.8.0+cu128 with CUDA unavailable and `--device cpu`. A GPU is not required. The exact prepared final bundle avoids downloading SDO images or changing source archives.

| Record | Scope and outcome |
|---|---|
| H7 | Cancelled before fitting after the channel-quality audit. |
| H8 | Classical solar-feature candidates; its stated training-CV criterion was supported. |
| H9 | Component and fast-case-weight ablations; fast error improved with an overall-error trade-off. |
| H10 | Cancelled before fitting to restore the missing prior-champion control; no negative scientific run. |
| H11 | Prior-champion recipe retraining plus additions/mixtures; no candidate passed the adoption rule, and P0 was retained. |
| H12 | Fixed training-prediction stability audit; inconclusive, with no H11 adoption or portfolio change. |
| E13 | Separate one-shot final panel; completed, `not_supported`; neither balanced gate passed. |

P0 is fresh faithful retraining of the earlier 50% tree / 50% three-seed residual-neural recipe. It is distinct from the **actual old frozen campaign winner**, which enters E13 through its saved historical forecasts. P1-P3 add ESP, lines, or all EVE inputs; P4-P6 mix P0 with fixed H9 fractions 0.25/0.50/0.75. E13 reports these seven forecasts, H9 specialist, starting wind, H8 SDO, matched classical wind, unweighted lines, weight-4 lines, the actual old frozen winner, and released WindNet. The starting wind and H9 specialist roles were declared before evaluation; no test-ranked winner was selected.

H9's literal weights were not normalized. They change effective regularization, and histogram-gradient boosting also uses them for internal binning. The earlier “loss-only” wording was too narrow; explicit imputation/scaling and evaluation remain unweighted.

The cumulative conservative study compute/storage estimate is **below US$0.22**, not a settled invoice. Temporary study pods and volumes were deleted and verified absent. See [final costs](../results/locked_final/costs.json), [the completed ledger](../results/continuation/experiment_ledger.json), and [all 13 records](../results/locked_final/experiment_timeline.json).

## Training-only continuation boundary

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

## Historical H8-H9 record: SDO-derived features

The following development results, runtimes, and cost estimate describe H8/H9 before H11 and E13. They are preserved as stage history; the completed full paper and cumulative cost are described above.

The continuation downloads the official EVE V008 merged NetCDF table (20.7 MB)
and extracts scalar irradiances. No image is downloaded. The feature manifest
records the source checksum and exact wavelength/channel metadata. Six EUV
lines and three ESP bands produce 62 columns: natural-log measurements at
calendar lags 5–8, four-day mean, endpoint trend, and two log-ratios per lag.
Positive-value and missing-data rules use no fitted global statistics. EVE
measurements from excluded source folds are masked, including before computing
inner-fitting summaries. The 1216 and 366 channels were excluded prospectively
because of documented filtering and stability concerns. H7 was cancelled
before model execution; its public protocol remains available.

These are retrospectively calibrated science measurements, not a historical
as-issued archive. The short-wavelength instrument failure in May 2014 leaves
missing features; there is no interpolation across that failure. The lags
refer to nominal target-minus-three-day issuance; original calendar gaps are
not repaired or reclassified as uniformly three-day leads.

H8 tests the added solar features in ridge/tree families and convex ensembles.
H9 fixes the learner pair and its 25% ridge / 75% tree mixture while comparing
wind only, ESP only, EUV lines only, and all solar features. It also tests
literal fitting-loss multipliers 2 and 4 for observed speeds ≥500 km/s and
small convex mixtures with unweighted forecasts. These weights are not
normalized, so both class emphasis and effective regularization change.

Both use four original training-side folds, original 13-window fitting and
14-window validation conventions, fitting-source masks, and whole-block
purging when target dates overlap across fitting and validation. No target
date overlap occurred in these four splits. All imputation/scaling is fitted
on the inner fitting rows. Candidate choice uses these same CV outcomes;
selected scores therefore overestimate independent performance.

| Training-CV model | Overall RMSE | Observed ≥500 RMSE | Observed ≥600 RMSE | CSI ≥500 |
|---|---:|---:|---:|---:|
| Starting wind-only ensemble | 80.72 | 133.35 | 189.14 | 0.136 |
| H8: solar features | 81.30 | 130.29 | 184.24 | 0.170 |
| H9: weighted EUV-line ensemble | 81.92 | 123.91 | 177.14 | 0.220 |

Speeds/errors are km/s. There are 1,512 validation windows, including 291
observed ≥500 and 82 observed ≥600 cases. Equal-weight duplicate aggregation
leaves 1,452 nominal issue–target pairs and is reported as a sensitivity view.
Each step requires ≥2% fast500 RMSE improvement and ≤1% overall degradation
from its immediate incumbent. This permits cumulative drift: H9 is 1.49%
worse overall than the starting model, despite 7.08% lower fast500 RMSE.
Future loops should fix the overall-error anchor rather than compound it.

The final H9 selection uses only the EUV-line block: 25% of the 4×-weighted
pair plus 75% of its unweighted pair. Against 27-day persistence, paired MSE
skill is 0.195 on 218 observed ≥500 cases and 0.026 on 60 observed ≥600 cases.
The reference is available on 1,128/1,512 windows; missing references are not
filled using excluded measurements. H9 still has zero hits and 82 misses at
≥600, with one false alarm. Conditional error skill and event detection are
different outcomes. All MAE, bias, contingency, POD/FAR/FPR/CSI/TSS/HSS scores,
reference coverage, per-fold reports, and rejected candidates are retained.

The H8 forecast was reproduced inside H9 to maximum absolute error
1.14×10⁻¹³ km/s before selection. Runpod used Python 3.12.9, NumPy 2.3.1,
SciPy 1.16.1 and scikit-learn 1.9.1. Training took 15.8 s for H8 and 82.2 s
for H9. Feature-preparation versions are recorded separately. Both temporary
pods and their volume were deleted and their absence verified. The cumulative
conservative compute/storage estimate is below US$0.14, including the earlier
study; this is not a settled invoice.

## Reproduce historical H8-H9 training stages

The public `artifacts/continuation/` directory contains the exact isolated
training and scalar-feature bundles (about 0.57 MB), including manifests. It
contains no original Fold 5 targets. You can reproduce model fitting directly
from those files without re-downloading images or the changing LASP archive:

```bash
python -m pip install -r continuation/requirements.txt
python -m continuation.train --bundle artifacts/continuation --output results/rerun/h8 --stage h8
```

Alternatively, after obtaining the frozen source inputs using the root instructions,
regenerate the bundles yourself:

```bash
python -m pip install -r continuation/requirements.txt
python continuation/prepare_training.py
python continuation/fetch_eve.py
python continuation/check_metrics.py
python continuation/check_selection.py
python -m continuation.train --bundle data/continuation --output results/rerun/h8 --stage h8
```

For H9, put copies of the H8 results and predictions in the isolated bundle
directory, then run:

```bash
python -m continuation.fast_loss --bundle data/continuation --output results/rerun/h9 --stage h9
```

H9 requires matching recorded numerical-package/Python versions and rejects
existing output filenames. It also requires the selected H8 member spec and
forecast reproduction tolerance. Source code at commit `ec86d2e` is the exact
H9 execution version. A later defensive selector check only handles undefined
overall MSE after numerical overflow; it does not alter any recorded finite
candidate choice. Retrieve the original code commit for byte-identical source
hashes. `latest` LASP files change; checksum equality is required for exact
input reproduction, even though the same named V8 channels remain accessible.

Public protocols/outcomes: [campaign](https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee),
[H8](https://fixrelay.vercel.app/experiments/a772eb55-1daf-4787-84df-4fdd8f662421),
[H9](https://fixrelay.vercel.app/experiments/f3956e16-c32c-42cb-9d67-6b8ac6aa1970).
The completed full paper and separately frozen E13 comparison are linked above.
H7-H12 did not reopen original holdouts for tuning or error analysis; E13's
one-shot historical comparison is recorded separately and was followed by no tuning.
