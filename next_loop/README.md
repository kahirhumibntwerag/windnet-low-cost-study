# WindNet literature-guided continuation

This campaign tests image-only improvements beyond M10. Hypotheses, hyperparameters and adoption decisions use only validation within original Folds 1–4. Original Fold5 is hashed and discarded without parsing during development. The final historical benchmark is a separate, fully frozen stage. Earlier historical aggregate exposure and adaptive reuse of development data prevent a fresh blind-confirmation claim.

- [Public RogNet campaign](https://fixrelay.vercel.app/campaigns/1b7e08e2-70f1-4945-b613-5e785d3f76bb)
- [Dashboard](https://windnet-matched-research-loop.wellbat32.chatgpt.site/continuation.html)
- [M12/M13 complete training results and ablations](../results/next_loop/REGIONAL_RESULTS.md)
- [M13 training-selected recipe](m13_training_selection.json)
- [F3 registered protocol](f3_protocol.json) and [complete freeze](../results/next_loop/f3_freeze.json)
- [M14 topology experiment: complete negative result and matched ablations](../results/next_loop/M14_RESULTS.md)

## What is matched

Every new predictor receives the same original four 211 Å images, using the original H4/D3 windows. No extra wavelength, in-situ history, sunspot number or external solar predictor is added. Each original 20-day block contributes 13 fitting windows when in a fitting fold and 14 evaluation windows when held out. F3 evaluates all 157 original blocks and 2,198 windows. Every trainable comparator fits the same other four original folds on the same final worker.

The released original TensorFlow forecasts are a historical reference. The separately retrained PyTorch WindNet port is a matched-data comparator, with implementation differences disclosed. Reusing the same recipe on different hardware does not guarantee exact numerical reproduction of archived controls.

## Training evidence

M11 failed an archived-control reproduction prerequisite and is retained as a technical failure. Separately preregistered M11R tested paired calibration changes against controls fitted on its own L4 runtime; none qualified. M12 tested 288 regional feature/mixture configurations and did not pass the complete adoption rule. M13 tested 960 target-weight/mixture configurations. Its selected model passed the registered numerical margins, fold stability screen and positive conditional HSE-difference lower bounds against all three controls. All 192 overlapping unit-weight conditions reproduced M12 predictions exactly.

The selected regional head uses a 6×6 image-plane grid, trend encoding, linear features, precision lambda4, ridge alpha100 and 2× fitting weights for targets at least500km/s. Its forecast contributes25% of the primary, with75% from M10. The M13 training score is not an estimate of historical or prospective test performance. Conditional bootstrap intervals do not correct adaptive selection or multiple candidate checks.

M14 tested 288 sampled digital topology/morphology configurations against four frozen controls, including M13. No condition passed all four numerical margins. M13 remains the training incumbent. Nine feature/weight checks and separate export/provenance audits passed; the fitted heads and training-side predictions are preserved in the public `windnet-m14-2026-10-30` release. No original historical evaluation ran in M14. Reproduce its registered workload from a fresh checkout with `python -m next_loop.m14_entry`; it rejects an existing M14 start marker and verifies its frozen sources and controls before fitting.

## F3 fixed historical panel

The complete evaluator, source identities, original member hashes and all ten entries were locked before Fold5 acquisition. [Freeze verification](../results/next_loop/f3_freeze_verification.json) records26sourcefiles and ten checks. [Exact final runtime](../results/next_loop/f3_runtime.json): Python3.11.11, NumPy2.1.2, SciPy1.16.1, scikit-learn1.7.2, PyTorch2.8.0.dev20250319+cu128, NVIDIAA40. The image is pinned by digest in the protocol.

| Entry | Difference from frozen primary |
| --- | --- |
| Primary | Training-selected M13 recipe, fixed before testing |
| WindNet port | Retrained original-input neural port |
| M6 | Equal port/combined-ridge mixture, inner gain1.2 |
| M10 | Ratio/flatten location ridge, lambda4, alpha10000; location/M6 share.625/.375; outer gain1.2, bias−5 |
| Regional only | Regional share1 rather than.25 |
| Unit target weights | Only regional fast-target emphasis removed |
| Zero precision lambda | Only regional precision weighting removed |
| Global1×1 | Only regional spatial grid removed |
| No outer calibration | Regional and M10 outer gains1/biases0; M6 inner calibration retained |
| Released WindNet | Original saved forecasts on aligned identical evaluation rows |

Port seeds19/43/101 and epochs38/27/36 are fixed. Fitting-only centering/scaling, precision statistics and target weights are recomputed within each outer fitting split. Test labels and uncertainty values never enter prediction calls. All five folds and all trainable panel entries fit before any metric is exported.

Two strict numerical gates are reported separately: lower mean-fold RMSE and reduced chi-square, higher Fisher-z mean correlation and original HSE threat score, versus released WindNet; and the same four directions versus retrained M10. Undefined metrics fail. No post-test replacement is allowed. Every ablation is reported regardless of outcome.

## Reproduce safely in a fresh checkout

Use the pinned CUDA/PyTorch image from [f3_protocol.json](f3_protocol.json), an NVIDIA GPU, and a fresh checkout with no existing `data/` directory. No Runpod account is needed by the scientific evaluator itself. The runner installs the pinned NumPy/SciPy/scikit-learn/requests versions. Original author archives are downloaded and verified against the source manifest.

For strict byte-level source checks on Linux or another Git newline configuration, first extract the small `f3_frozen_sources.tar.gz` asset from the [F3 release](https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-f3-2026-10-30) over the fresh source tree. Git normalizes one frozen JSON file's line endings; this snapshot preserves the original bytes and includes a checked source manifest. The analogous `m14_frozen_sources.tar.gz` asset in the [M14 release](https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-m14-2026-10-30) preserves M14's source/protocol bytes. The snapshots contain no raw observations or historical predictions, and change no recipes or numerical results. Source snapshot digests are recorded in `results/next_loop/f3_sources_artifacts.json` and `m14_sources_artifacts.json`.

```bash
python -m unittest next_loop.test_f3 next_loop.test_regions next_loop.test_target_weights
python -c "import json; json.dump({'stage':'prepare'},open('f3-prepare-request.json','w'))"
python -m next_loop.f3_entry --request f3-prepare-request.json
```

The prepare stage acquires only development data and creates `exports/f3_freeze.json` and `exports/f3_freeze_identity.json`. Inspect and retain that freeze before evaluating. Copy its exact SHA256 into a separate final request:

```bash
python -c "import json; f=json.load(open('exports/f3_freeze_identity.json')); json.dump({'stage':'final','freeze_sha256':f['sha256']},open('f3-final-request.json','w'))"
python -m next_loop.f3_entry --request f3-final-request.json
```

Final acquisition is allowed only after the freeze. Source changes, mismatched hashes, and an existing final-start marker are rejected. Technical failures retain the marker and frozen choices; do not delete a marker to obtain another scientific attempt. Independent reproduction uses a new isolated workspace and reports its own runtime and freeze identity.

Aggregate results, source/data audits and checkpoint manifests go to `exports/`. Fitted models and normalization statistics are archived; raw holdout observations and individual historical predictions are excluded from the exported checkpoint archive. `verify_f3.py` checks downloaded aggregate arithmetic, gates, original fold counts, source identities and checkpoint hashes; it does not independently refit models or rerun raw-label metrics. Its expected download layout is `results/next_loop/` and `artifacts/next_loop/` as used by `f3_remote.py`.

## Inference and interpretation limits

Image-plane intensity features are proxies, not confirmed physical coronal-hole regions or WCS-based source attribution. The original nominal delay and event implementation are preserved. The historical five-fold benchmark does not establish chronological operational forecasting performance. Whole-block bootstrap intervals are conditional and not selection-adjusted, and original blocks can overlap. The published registry is a self-reported research trail, not independent reproduction or peer review.

Elicit is not used. Compute resources created for this campaign are temporary, separately tracked and removed after verified downloads. The cumulative budget isUS$5; estimates include setup/idle time and a storage buffer and are not settled provider invoices.
