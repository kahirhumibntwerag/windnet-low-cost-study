# Regional image forecasting: M12 and M13

These are training-side results on 1,512 original validation windows in Folds 1–4, with identical original fitting rows and four 211 Å images per forecast. **No new original historical test evaluation has run.** Fold5 was hash-checked and discarded without parsing or retention.

## M12: regional features improved point estimates, but did not qualify

M12 compared 288 conditions from 72 regional ridge recipes, including global-location removal, grid resolution, temporal view, linear/squared features, precision weighting and mixtures with M10. Thirteen conditions improved all four metrics by the registered margins; nine passed the fold screen. All nine conditional HSE difference intervals against M10 crossed zero, so no model was adopted.

The strongest fold-qualified numerical recipe had RMSE75.6669 km/s, Fisher r0.5950, reduced chi-square25.9770 and original HSE threat score0.2838. Its conditional HSE difference interval against M10 was[-0.0173,0.1122]. The lowest-RMSE numerical pass failed the fold screen. Neither was promoted. The global1x1 ablation had no numerical passes; this is descriptive evidence for this finite family, not proof that spatial information is universally necessary.

The initial attempt failed before fitting because of Windows path separators in the source manifest. Only path resolution was corrected; the registered hashes and scientific recipes were retained. The full failed attempt is preserved in `m12_attempt1.json`.

## M13: a candidate qualified under every registered training-side rule

After M12, a new primary-literature search motivated fitting-side emphasis on fast or underrepresented target speeds:

- Yang et al. (2021), *Delving into Deep Imbalanced Regression*: https://proceedings.mlr.press/v139/yang21m.html . Label-distribution smoothing motivates the explicitly defined density-weighting adaptation.
- Ren et al. (2022), *Balanced MSE for Imbalanced Visual Regression*: https://arxiv.org/abs/2203.16427 . Context for imbalanced objectives; the Balanced MSE algorithm was not implemented or reproduced here.
- Collin et al. (2026), *PROSWIN*: https://arxiv.org/abs/2609.26683 . Context for preserving both timeline and peak skill; its different inputs, metrics and published scores are not matched comparators.

M13 compared 960 conditions from 192 recipes, or768fold fits. Fifty passed all four numerical margins, and40passed the fold screen. In the registered ascending-RMSE interval-checking order, the first23candidates failed; the24thqualified. Later candidates were not declared failures. All192overlapping unweighted conditions reproduced M12's individual predictions exactly, with maximum absolute difference0.

| Training-side metric | Fixed M10 control | Qualified M13 |
| --- | ---: | ---: |
| Mean-fold RMSE (km/s) | 77.977148 | 77.097948 |
| Fisher-averaged correlation | 0.573047 | 0.589186 |
| Mean reduced chi-square | 26.880775 | 24.858471 |
| Mean original HSE threat score | 0.238904 | 0.295216 |

Selected recipe:6x6relative-intensity grid; four-image trend view; linear features; fit-only precision lambda4; ridge alpha100; fitting targets>=500km/s weighted2x; calibrated regional head mixed25%with75%fixed M10. The fixed outer gain is1.2 about the fitting-target mean, with bias-5km/s.

The candidate passes all four leave-one-validation-fold-out directional checks. Its worst fold RMSE ratio against the stronger control is1.00113, below the1.05limit. Conditional HSE difference95%intervals are positive against the port[0.0923,0.2550], M6[0.0596,0.1996] and M10[0.00374,0.11529]. These intervals are conditional on a broad, reused development search and are not multiplicity or selection adjusted. They do not establish independent generalization or significance for every metric.

## Matched target-weight ablation at the selected settings

Only target weights change in this table; grid, view, encoding, precision weighting, penalty, calibration and mixture share are fixed.

| Fitting target weights | RMSE | Fisher r | Reduced chi-square | HSE TS | Numerical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Unit | 77.346379 | 0.590881 | 23.998214 | 0.236756 | Fail |
| Fast1.5 | 77.185537 | 0.590077 | 24.462190 | 0.275315 | Pass |
| Fast2(selected) | 77.097948 | 0.589186 | 24.858471 | 0.295216 | Pass |
| Density | 77.222511 | 0.589867 | 24.893593 | 0.300486 | Fail |

Density weighting has the highest event point estimate here but misses the RMSE margin. Unit weighting has lower chi-square and slightly higher correlation than fast2, but fails the balanced requirement. The adopted choice is a conditional compromise under the registered rules. Fast1.5at this particular setting occurs later in the RMSE order and was not declared an interval failure after checking stopped.

## Complete evidence and next evaluation

`m12_results.json` and `m13_results.json` retain every condition, fold metric and uncertainty check; the corresponding verification files independently recompute aggregate values, gates, fold screens, checking order and original row identities, and check saved prediction shapes/finiteness and artifact digests. This is not an independent rerun of every model or ground-truth metric.

The training-only archives are published in the releases `windnet-m12-2026-10-30` and `windnet-m13-2026-10-30`; exact SHA256 values and byte counts are in the artifact manifests. They contain fitted models and training-side predictions, not original holdout observations.

After M13, a further primary-literature search reviewed Cawley & Talbot(2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, https://www.jmlr.org/beta/papers/v11/cawley10a.html . This reinforces keeping the historical evaluation separate from adaptive recipe decisions. The original WindNet protocol is https://arxiv.org/abs/2006.05825 . F3 is registered but not started. Its complete evaluator, panel and source/data identities must be frozen before original Fold5 acquisition. Report released-WindNet and incremental-M10 comparisons separately, retain every fixed ablation and do not change the selected model after test scoring. Earlier historical exposure means this cannot be called a fresh blind confirmation.

Both newly created CPU resources were deleted after evidence download, with404read-back confirmation. Cumulative matched-campaign cost is conservatively estimated atUS$1.43695 ofUS$5, not a settled invoice. See `m12_m13_costs.json`. Elicit was not used; the manuscript and unrelated solar72study were not changed.
