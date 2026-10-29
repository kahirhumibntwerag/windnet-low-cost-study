# Completed matched image-only WindNet campaign

Six sequential development hypotheses were preregistered in public RogNet, followed by one frozen final evaluation of 16 new models/ablations and released WindNet. Every new model used the same four original 211-Angstrom images, the same original training partitions, and the same 2,198 final test windows. No in-situ, recurrence, EVE or extra solar observations were predictors.

**The predefined RMSE/correlation gate passed numerically; the strict all-four-metric gate failed.** The frozen primary improves RMSE by 4.37% and improves correlation and reduced chi-square versus released WindNet, while reducing high-speed-enhancement event threat score. The primary was selected on development validation and was not replaced after final scoring.

| Model | Mean RMSE (km/s) | Fisher r | Mean reduced chi-square | Mean HSE threat score | Mean MAE (km/s) |
|---|---:|---:|---:|---:|---:|
| Frozen primary | 76.7688 | 0.565856 | 26.4313 | 0.178815 | 59.2651 |
| Released WindNet | 80.2760 | 0.550034 | 26.8288 | 0.199591 | 60.5737 |
| New matched WindNet port | 80.9640 | 0.522388 | 26.3016 | 0.178846 | 61.5413 |

The primary is a 50:50 average of a three-seed WindNet architecture port and ridge regression on concatenated original image energies plus fixed spatial image summaries, followed by gain 1.2 around the fitting-label mean. All recipes, epochs and coefficients were frozen before test scoring. See `freeze.json`, `freeze_identity.json`, and the seven code/input identity checks.

## Evidence and limits

The paired conditional 95% interval for new minus released WindNet mean-fold RMSE is [-6.6149, -0.3369] km/s; for Fisher correlation it is [-0.022843, 0.053623]. The RMSE interval favors the primary, while the correlation interval includes no improvement. These whole-block bootstrap intervals are not selection adjusted, and overlapping blocks may be dependent.

Historical holdout scores were already public and seen before this campaign; Folds 1–4 supplied development data. Fold 5 image/label members were not parsed by this campaign until the final freeze. Therefore this is a locked historical benchmark, not an independent blind confirmation or proof of operational superiority. No further hypotheses or model changes followed final scores.

`final_audit.json` records identical fitting/test row identities for all models, the original 13/14 training/evaluation window asymmetry, reference hash/label alignment, and that test labels were never supplied to estimator fit/predict calls. All fitted artifacts were saved before scores. The code uses the original Kaffe weights, but the documented PyTorch port differs from TensorFlow and is a separate comparator.

## Files

- `m1_results.json` through `m6_results.json`: every development candidate, controlled loss/feature/weight ablations and calibration results, including negative outcomes.
- `final_results.json` and `final_comparison.csv`: fixed final panel, fold summaries, metric gates and uncertainty.
- `checkpoint_manifest.json` and `artifact_verification.json`: portable model identities and synthetic-only loading checks. Large weights are retained locally in `artifacts/matched/checkpoints.tar.gz`; they are not embedded in the dashboard or this GitHub branch.
- `costs.json`: elapsed-time estimate, price, allowance and worker teardown evidence; not a settled invoice.
- `../../matched/README.md`: protocol and fresh-directory reproduction command.

[Public RogNet campaign](https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa) · [Final experiment](https://fixrelay.vercel.app/experiments/585a7b68-6a43-48b0-a0af-ca538bd8b10b) · [Owner-private dashboard](https://windnet-matched-research-loop.wellbat32.chatgpt.site)
