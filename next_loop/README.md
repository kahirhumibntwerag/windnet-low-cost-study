# Literature-guided WindNet continuation

This user-requested new campaign continues the matched image-only question beyond M10:
https://fixrelay.vercel.app/observatory?campaign=1b7e08e2-70f1-4945-b613-5e785d3f76bb

The original public campaign and frozen studies remain unchanged. New hypotheses use training-side results only, never historical final scores or individual final forecasts. No Elicit tool is used.

## M11 preregistration

Hypothesis: a cross-fitted monotone distribution correction may improve all four original metrics beyond the fixed M10 recipe. The registered experiment has six conditions: five correction strengths (including strength0, the unchanged M10 control) and one in-sample calibration ablation that is ineligible for adoption. See `protocol.json` and the RogNet design in `registry.json`.

Scientific sources reviewed before the hypothesis:

- Collin et al. (2025), *Forecasting High-Speed Solar Wind Streams From Solar Images*. https://doi.org/10.1029/2024SW004125 . Its distribution correction motivates the experiment; its extra inputs and forecast setup are not imported.
- Collin et al. (2026), *PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images*. https://arxiv.org/abs/2609.26683 . Its attention to both timeline and peak skill motivates the balanced objective, not a new metric or an unmatched numerical baseline.

Only original Folds1-4 are retained in the worker. Original Fold5 archive members are hash-checked and discarded without parsing. All models use the same original fitting/evaluation rows, original four 211 Angstrom images, and original event implementation. Inner ridge regressions refit scalers/precision statistics without inner validation labels; fixed encoder image centering uses outer fitting images and is reused inside cross-fitting, so preprocessing is not fully nested.

## Execution and reproduction

Runpod worker `3rd3y8tivb54e8` uses one L4 at US$0.59/hour and temporary standard network volume `y4hllccc62` (10GB). The prior matched campaign estimated cost was US$1.23384; the cumulative envelope stays US$5. A one-hour worker lifetime guard is configured. Credentials stay in ignored `.secrets/`; no other account resource is changed.

Local tests: `python -m unittest next_loop.test_calibration`.
Remote registered workload: `python -m next_loop.worker` from a clean root with copies of `matched/`, `event_loop/` and `next_loop/`. Reproduction requires the original source archive and fixed encoder download referenced in `matched/source_manifest.json`. The pinned training dependencies remain in `matched/requirements.txt`.

The worker checks original fitting/validation row hashes and historical training-control metric agreement before selecting a candidate. The initial L4 controls differ from archived controls beyond the preregistered tolerance despite matching recorded Python3.12.3, NumPy2.1.2 and Torch2.8.0+cu128. The cause has not been isolated. This is a technical reproduction issue, not evidence against the calibration hypothesis. PyTorch's primary reproducibility documentation cautions that seeds alone do not guarantee matching results across platforms: https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md . The registered failure rule is retained; no tolerance is silently relaxed.

Aggregate exports are stored separately in `results/next_loop/`. The new dashboard page is built by `python -m next_loop.build_dashboard`; earlier F1/F2 dashboard records remain archived. No new original historical evaluation has been run.
