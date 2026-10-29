# Continuing matched WindNet research

The public campaign is https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa . This phase continues the earlier frozen study at the user's request. Earlier scientific sources and outcomes are retained.

## Development boundary

Only the 108 original blocks in Folds 1–4 are retained in the development worker. Each of four training-bank splits fits three folds and validates the remaining fold. All models receive the same four original 211 Å images per forecast, with original 13 fitting and 14 evaluation windows per block. Image centering, feature normalization, scalers, weighting statistics and target anchors use fitting rows only. Fold 5 archive members are hash checked and discarded without parsing during development. Earlier final forecasts and scores are not supplied to the worker. They were previously known to this conversation; no fresh blindness claim is possible.

The controls are a matched three-seed WindNet port and the previously frozen M6 image-only mixture, both retrained per split. The adoption gate requires every metric to improve against the stronger control: at least 1% lower mean-fold RMSE, +0.005 Fisher-z correlation, 0.5% lower mean reduced chi-square, and +0.02 mean original HSE threat score. Undefined scores fail. No single-metric promotion.

## Completed observations

M7: aligning auxiliary supervision to future target sequences did not pass the balanced gate. M8: none of 1,176 fixed mixtures/calibrations passed; correlation was limiting. M9: preserving spatial locations in fixed pretrained activation maps produced one all-gate pass among 2,940 candidates. Its event-score interval was positive against both controls, while other differences versus M6 remained uncertain. That is validation evidence, not a final benchmark result.

M10 preregisters precision-weight sensitivity, matched representation ablations and a stability gate: directional four-metric improvement in at least three of four leave-one-validation-fold-out aggregates, no individual fold RMSE more than 5% above its stronger control, and positive conditional 95% event-score difference bounds against both controls. Full negative and positive configurations remain in the result exports.

## Final evaluation boundary

The new final evaluator refuses to freeze without strict adoption and the M10 stability checks. It hashes scientific sources, data identities, recipes and validation results before acquiring Fold 5. One fixed historical five-fold pass fits all models before exporting scores. Test labels enter only reference alignment and metric scoring, never estimator fit or predict calls. The primary and fixed ablations cannot be replaced after scoring. Hypotheses must never follow test feedback.

Synthetic checks cover the all-four gate, paper aggregation, training-side stability, weighted ridge equivalence, and prediction without validation labels. A Runpod worker performs the actual scientific runs. The same Git branch and existing dashboard preserve the campaign history. Broad validation search, overlapping blocks and prior historical exposure limit inference even when numerical gates pass.
