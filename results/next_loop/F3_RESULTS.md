# F3: fixed regional-tail model on the original WindNet folds

The frozen primary **passed** the strict all-four numerical gate against released original WindNet. Individual metric directions: mean_rmse pass, fisher_r pass, mean_chi2 pass, mean_hse_ts pass.

The frozen primary **failed** the strict all-four numerical gate against same-runtime retrained M10. Individual metric directions: mean_rmse pass, fisher_r pass, mean_chi2 pass, mean_hse_ts fail.

All 157 original blocks and 2,198 evaluation windows are included. Every trainable model fitted the same other four original folds and received the same four 211 Å images. No candidate was substituted after testing. Hypotheses and the primary recipe were chosen from training-side validation.

## Complete fixed panel

| Model or ablation | RMSE (km/s) ↓ | Fisher r ↑ | Reduced χ² ↓ | Original HSE TS ↑ |
| --- | ---: | ---: | ---: | ---: |
| Frozen M13 primary | 75.236852 | 0.599623 | 22.707380 | 0.247444 |
| Matched WindNet port | 80.809715 | 0.524433 | 26.687221 | 0.173059 |
| Prior M6 | 76.558214 | 0.567158 | 26.547986 | 0.176764 |
| Prior M10 | 76.112349 | 0.579540 | 24.555901 | 0.270642 |
| Regional head only | 86.746904 | 0.547856 | 29.824922 | 0.349510 |
| Unit target weights | 75.500204 | 0.600460 | 22.024466 | 0.257966 |
| No regional precision weight | 74.093325 | 0.596206 | 25.059230 | 0.235040 |
| Global 1×1 regional head | 75.320928 | 0.595404 | 22.449435 | 0.224646 |
| No outer amplitude calibration | 72.985351 | 0.599623 | 24.785581 | 0.178509 |
| Released original WindNet | 80.276038 | 0.550034 | 26.828811 | 0.199591 |

The regional-only, unit-target-weight, zero-regional-precision and global 1×1 entries change only the stated component of the primary. The no-outer-calibration ablation removes gains and biases from the regional head and M10 outer mixture while retaining M6’s inner gain. Full recipes are fixed in [the freeze](f3_freeze.json).

## Conditional uncertainty

Primary minus comparator, paired whole-block bootstrap with 2,000 draws within the original folds, seed 20261029. Negative differences favor the primary for RMSE and χ²; positive differences favor it for correlation and event score.

| Comparator | RMSE difference, 95% interval | Fisher r difference | χ² difference | HSE TS difference |
| --- | ---: | ---: | ---: | ---: |
| Released original WindNet | [-7.969894, -2.104114] | [+0.014312, +0.086155] | [-6.806051, -1.543408] | [-0.015826, +0.117098] |
| Prior M10 | [-1.932044, +0.112592] | [+0.007824, +0.033451] | [-3.088020, -0.803983] | [-0.060606, +0.015832] |

These are conditional intervals, not adjusted for adaptive selection or repeated historical evaluation. A numerical improvement does not establish that every metric’s interval excludes zero. Original blocks may overlap.

## Provenance and boundaries

- The complete source/data/panel freeze was created and pushed before acquiring Fold5. Its SHA256 is `7f133633005d179d1b49b07ef3441ee9ff560d01c65ce16032f89b5248e779c9`.
- All five folds and all trainable panel entries were fitted before exporting any metric. The predictor interface received image-derived features only.
- The final runtime was Python 3.11.11, NumPy 2.1.2, SciPy 1.16.1, scikit-learn 1.7.2 and PyTorch 2.8.0.dev20250319+cu128 on an NVIDIA A40. All trainable comparators shared it.
- The released TensorFlow forecasts are a separate historical comparator; the matched neural port has documented implementation differences. Exact cross-hardware numerical reproduction of archived M10 is not claimed.
- The original nominal delay and HSE implementation conventions are preserved. Image-plane intensity proxies are not physical source-region attribution.
- Folds 1–4 were adaptively reused for development, and earlier historical aggregate results were already known. This locked exploratory benchmark is not fresh blind confirmation or a prospective chronological test.
- Historical outcomes cannot choose a replacement or the next hypothesis. Any further model idea must come from training-side evidence and a separately registered protocol.

The independent export verifier checked aggregate arithmetic, both gates, original fold counts, 26 source identities and 55 checkpoint-container hashes. It did not independently refit models or rerun metrics from raw labels. Raw holdout observations and individual historical predictions are excluded from exported checkpoints.

[Full results and fold metrics](f3_results.json) · [Row and label-boundary audit](f3_audit.json) · [Verification](f3_verification.json) · [Artifact digest](f3_artifacts.json) · [Reproduction guide](../../next_loop/README.md)
