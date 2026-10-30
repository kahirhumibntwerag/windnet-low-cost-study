# M14: sampled digital topology did not pass the balanced gate

All **288 registered training-side configurations** completed. None passed all four numerical margins against the four frozen controls. No candidate reached interval checking or adoption; **M13 remains the training incumbent**. No original historical holdout was evaluated.

The experiment used the original four 211 Å images and the same 1,512 validation windows from Folds 1–4. It tested sampled connected components and enclosed digital holes across eight intensity levels, then added largest-component and boundary measurements. These are digital topology descriptors, not full persistent homology or physical coronal-hole segmentation.

## Matched feature ablation

The display anchor was fixed before reading results: trend history, precision lambda 4, fast2 fitting weights, ridge alpha 100, and new-head share 0.25 with the fixed M13 prior. Only the feature family changes.

| Training-side model | RMSE ↓ | Fisher r ↑ | Reduced χ² ↓ | HSE threat score ↑ |
| --- | ---: | ---: | ---: | ---: |
| Fixed M13 incumbent | 77.097948 | 0.589186 | 24.858471 | 0.295216 |
| Betti counts | 76.409310 | 0.602897 | 22.727618 | 0.199031 |
| Morphology | 76.271393 | 0.609438 | 22.436107 | 0.232758 |
| Regional + morphology | 77.609904 | 0.590533 | 24.318187 | 0.333809 |

Morphology clears the RMSE, correlation and χ² margins but loses event score. Adding regional features clears the event and χ² margins but misses the RMSE and correlation margins. Neither row is adopted. This training-side complementarity is an observation, not proof of a causal division into physical solar-wind regimes.

The lowest-RMSE condition scores **73.469187 / 0.609219 / 29.140099 / 0.171515** (RMSE / Fisher r / χ² / HSE). Its χ² and event score fail. The highest-event-score condition scores **98.166134 / 0.482008 / 44.245268 / 0.510522** and fails the other three metrics. Reporting either extreme as an overall improvement would misrepresent the experiment.

## Protocol and verification

Three feature families, two temporal views, two precision weights, two target-weight methods, three ridge penalties and four mixture shares produce 72 raw recipes and 288 comparison conditions. Each recipe fits four training-bank splits, for 288 fitted heads. All four control forecasts reproduced their recorded metrics within 1e-9 on exact original row hashes before fitting.

Original Fold5 members were hash-checked and discarded without parsing or retention. No F3 result, checkpoint, historical prediction or residual was supplied to this worker. The source/data/control freeze preceded model fitting; its SHA256 is `d4c1c1f410821611fbe58fb2d5174c7d4cb138a622d3d09f2297dfed4276d329`.

Nine feature/weight tests passed. Separate export verification recomputed aggregate arithmetic, all four numerical gates and fold stability, checked the selection order, and verified the archive digest and saved prediction shapes/finiteness. Twenty frozen source identities and the protocol/freeze hashes matched. This does not constitute an independent model refit or raw-label metric rerun.

The training artifact archive contains 288 fitted heads and every candidate’s training-side validation forecasts. It is 4,026,636 bytes, SHA256 `2e9f0b4230c10c661d36633b44f80f4afaf742a458dd9a533c6eacb5b412ee74`. The temporary CPU pod and volume were deleted after verified download. Costs are estimates, not a settled provider invoice.

The same release also supplies a 44,700-byte exact-source snapshot for portable hash checks. Git normalizes line endings in a few source/reference files; the snapshot preserves the frozen bytes, with no numerical changes. Its SHA256 is `53738234e7d9ab173d2c4fb3a253cecf0cb8822a121b25af0bf2af95fccd9562`. Every archived source byte was checked against the original freeze and checked again after an archive round trip.

## Post-experiment literature and next question

[Jacobs et al. (1991), *Adaptive Mixtures of Local Experts*](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), motivates routing between predictors that specialize in different cases. [Breiman (1996), *Stacked Regressions*](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf), studies combinations using cross-validation forecasts and discusses the risks of fitting combinations to in-sample predictions.

Together with the **training-only** matched ablation, these sources motivate testing a bounded, deterministic gate between the two fixed morphology/regional-morphology anchors. Such a gate would be an explicitly defined adaptation, not a reproduction of a learned neural mixture-of-experts model. Gate choices must be registered before evaluation; original historical feedback cannot choose them. Reusing the validation bank still creates selection bias, even with out-of-fold base forecasts.

[All 288 conditions, fold metrics and weights](m14_results.json) · [Verification](m14_verification.json) · [Provenance](m14_provenance_verification.json) · [Protocol](../../next_loop/m14_protocol.json) · [Costs](m14_costs.json)
