# Frozen four-metric WindNet continuation (F2)

Frozen primary improved all four original metrics numerically.

The primary was selected exclusively using four-fold training-bank validation, a strict four-metric gate and a preregistered stability rule. Its code, recipe, epochs, seeds and data identities were frozen and pushed before final access. This is a locked historical comparison, not fresh blind confirmation.

| Metric | Released WindNet | Frozen primary | Direction |
|---|---:|---:|---|
| Mean-fold RMSE (km/s) | 80.276038 | 76.254191 | lower |
| Fisher-z Pearson r | 0.550034 | 0.578886 | higher |
| Mean reduced chi-square | 26.828811 | 24.539980 | lower |
| Mean original HSE threat score | 0.199591 | 0.275187 | higher |

All models use exactly **2,198 original evaluation windows across 157 released blocks**, the same four 211-Angstrom images per forecast, and the original H4/D3 window definitions. Every held-fold fit uses the other four original folds. Preprocessing, target anchors and weighting statistics use fitting rows only. The released reference forecasts were hash-checked and their observation rows aligned exactly. Test labels enter reference alignment and scoring, never estimator fit or predict calls. Scores were exported only after every fold/model fit completed.

## Frozen recipe

Ratio-normalized 4x4 pretrained activation energies plus log global energies, flattened four-image history; fit-only StandardScaler with scale floor0.01; weighted ridge alpha10000, precision lambda4; 0.625 location model +0.375 prior M6; gain1.2 about the fitting target mean, bias-5 km/s. Fixed three-seed port inside M6: seeds19/43/101, epochs38/27/36. All inputs derive from the same original images; no speed-history predictor or extra solar observations.

## Fixed comparisons and ablations

| Model | Mean RMSE | Fisher r | Mean chi-square | Mean HSE TS |
|---|---:|---:|---:|---:|
| PRIMARY_F2 | 76.2542 | 0.5789 | 24.5400 | 0.2752 |
| windnet_port | 80.9640 | 0.5224 | 26.3016 | 0.1788 |
| prior_M6 | 76.7688 | 0.5659 | 26.4313 | 0.1788 |
| location_raw | 75.4676 | 0.5524 | 29.2231 | 0.2109 |
| without_precision_weight | 75.8598 | 0.5766 | 26.4379 | 0.2507 |
| without_prior_mixture | 77.1408 | 0.5524 | 26.9529 | 0.2941 |
| without_amplitude_calibration | 74.2538 | 0.5789 | 26.7126 | 0.1948 |
| released_windnet | 80.2760 | 0.5500 | 26.8288 | 0.1996 |

The primary remains the preselected M10 model. Ablations were not substituted after scoring.

## Conditional uncertainty

| Primary minus released reference | Paired whole-block 95% interval |
|---|---:|
| rmse | [-6.937455, -1.059557] |
| r | [-0.007967, 0.066828] |
| chi2 | [-5.180616, 0.512145] |
| hse_ts | [0.010298, 0.147709] |

These 2,000-draw intervals are conditional and not adjusted for selection; blocks can overlap. RMSE and HSE intervals exclude zero in the favorable direction. Correlation and chi-square intervals include zero. Therefore the claim is numerical improvement on this historical benchmark, not established superiority on all metrics.

## Integrity and reproducibility

- Freeze SHA256: `57db805d21920576fd18f5ce46069aee2b81dc5a5771f7f28b8ba1e0dc61131f`.
- Final evaluation passes:1; post-test model changes:0.
- Eight synthetic checks passed before final access. All aggregate metrics were independently recomputed from exported fold metrics.
- All40 fitted checkpoint files were checked against the manifest, including file sizes and SHA256 values. The131MB archive contains fitted models/normalizers and the freeze manifest; no raw holdout observations.
- Checkpoint archive SHA256: `f6507a979324156f999c2a09cee9340081a5fb779844ef661aa2d136a8a5e660`.
- Exact source bytes are in `artifacts/event_loop/frozen_sources.zip`; every entry matches the freeze manifest. Git normalizes line endings for two non-model files, so use that archive for exact manifest verification.
- Code/results branch: `codex/windnet-matched-loop`; freeze was pushed in commit `07d5193` before F2 execution.
- Public campaign: https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa
- F2 record: https://fixrelay.vercel.app/experiments/be200d9f-d6c3-43b1-8a6f-15895ca0f79a

## Limits

Earlier historical holdout scores had already been viewed before this phase, and Folds1-4 are development data. The agent did not inspect raw holdout labels/images or use final feedback for a new hypothesis or model change. This does not erase prior exposure or broad adaptive training-validation search. The five-fold aggregate is exploratory. The matched PyTorch port has recorded implementation differences from the original TensorFlow model. Nominal D3 and the original HSE conventions are preserved; extra metrics are not substituted for the paper metrics.

Operational failures, temporary-disk loss and recovery are retained in `m10_transport_retry.json`. Recovery reproduced controls, fold metrics and row audits exactly; original candidate predictions were lost and bitwise replication of that lost bank is not claimed. Only resources created for this phase were deleted after evidence download and verification.
