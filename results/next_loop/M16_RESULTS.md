# M16: fitting adjacent-window differences

M16 completed without an adopted model. All 144 registered conditions and 144 fold fits ran; none passed all four numerical margins, so no condition reached the bootstrap screen. M13 remains the incumbent. No historical M16 evaluation ran.

## Question and fixed design

M15 obtained one training-side numerical win but failed the event-score uncertainty check. M16 therefore tested whether fitting transition errors alongside point errors helps balance the metrics. This is an adaptation of derivative-aware loss design, not a reproduction of terrestrial FastNet. The earlier M8 neural gradient-loss failure remains recorded.

Each head minimizes weighted point error plus lambda times weighted adjacent-window difference error plus ridge regularization. Pairs are restricted to consecutive offsets within the same original block and its 13 fitting target windows (12 pairs per fitting block). No pair crosses blocks or reaches validation rows. These differences are between original windows, not physical per-day derivatives. Inference consumes only one original four-image H4/D3 window.

The panel crosses regional, morphology and combined trend features; ridge alpha 10/100/1000; difference strength 0/0.25/1/4; and head share 0.125/0.25/0.5/1 with fixed M10. Main weights use fitting-only precision4 and fast2; pair weights average adjacent main weights then normalize. Centering uses weighted fitting means, and outer gain1.2/bias-5 uses the ordinary fitting target mean. All recipes were frozen before fitting.

Four controls, original 108 development blocks and 1,512 training validation rows were unchanged. Original Fold5 was streamed, hashed and discarded without parsing or retention. M16 uses fixed M10 as the prior; M14 used fixed M13. Across-stage morphology comparisons therefore also change the prior and cannot isolate only the difference loss.

## Complete controlled anchor ablation

The display anchor was fixed at alpha100 and head share0.25; only feature family and difference strength vary. Full scores for all 144 conditions and per-fold event counts are in m16_results.json and the dashboard CSV.

| Condition | Mean-fold RMSE (km/s) | Fisher r | Mean reduced chi-squared | Mean original HSE TS |
|---|---:|---:|---:|---:|
| bank_windnet_port | 82.93143258 | 0.52763652 | 29.20174774 | 0.12362043 |
| bank_prior_M6 | 79.50205000 | 0.55510617 | 29.43461605 | 0.17157625 |
| fixed_M10 | 77.97714841 | 0.57304693 | 26.88077460 | 0.23890443 |
| fixed_M13 | 77.09794756 | 0.58918557 | 24.85847072 | 0.29521633 |
| M16_regional_trend_w4_targetfast2_a100_d0_share0.25 | 77.09794756 | 0.58918557 | 24.85847072 | 0.29521633 |
| M16_regional_trend_w4_targetfast2_a100_d0.25_share0.25 | 77.08223345 | 0.58939016 | 24.74494803 | 0.27877635 |
| M16_regional_trend_w4_targetfast2_a100_d1_share0.25 | 77.09170502 | 0.58920971 | 24.59696381 | 0.26877635 |
| M16_regional_trend_w4_targetfast2_a100_d4_share0.25 | 77.19561539 | 0.58742818 | 24.50210149 | 0.27123556 |
| M16_morphology_trend_w4_targetfast2_a100_d0_share0.25 | 76.49436677 | 0.60250733 | 23.44908557 | 0.21744922 |
| M16_morphology_trend_w4_targetfast2_a100_d0.25_share0.25 | 76.51203469 | 0.60233975 | 23.40705919 | 0.21744922 |
| M16_morphology_trend_w4_targetfast2_a100_d1_share0.25 | 76.58290684 | 0.60131929 | 23.37335602 | 0.20783383 |
| M16_morphology_trend_w4_targetfast2_a100_d4_share0.25 | 76.72421795 | 0.59879615 | 23.36708367 | 0.19937229 |
| M16_regional_morphology_trend_w4_targetfast2_a100_d0_share0.25 | 76.58682891 | 0.59480060 | 24.38716323 | 0.26651452 |
| M16_regional_morphology_trend_w4_targetfast2_a100_d0.25_share0.25 | 76.57108866 | 0.59485664 | 24.27992965 | 0.25075026 |
| M16_regional_morphology_trend_w4_targetfast2_a100_d1_share0.25 | 76.58195650 | 0.59434910 | 24.13399424 | 0.25075026 |
| M16_regional_morphology_trend_w4_targetfast2_a100_d4_share0.25 | 76.64625478 | 0.59247068 | 23.96847841 | 0.25460297 |

At the regional anchor, positive difference strengths lowered event threat score relative to zero strength. The small point-error reductions did not satisfy the frozen joint margins. The minimum-RMSE row in the full panel had RMSE76.23904846 and HSE0.24414251; the maximum-HSE row had TS0.44510625 and RMSE96.50848926. Neither was adopted. No model was substituted using historical feedback.

## Verification and resources

Fourteen unit tests passed locally and on the worker, including pair boundaries, weighted-ridge equivalence, objective gradients and shuffled-row pairing. Zero-strength regional alpha100/share0.25 reproduced the saved M13 predictions before any positive-strength fit; maximum absolute error was 1.1368683772161603e-13 in each fold. The four controls and original row references matched to1e-9. The 27-file source freeze, protocol identity, development hashes, pair audits and exported prediction shapes passed the independent arithmetic/provenance checks. These checks are not an independent refit or raw-label metric rerun.

Freeze SHA-256: `faa5f74c3ad06da43c77bff12c4d33358340f1a00e5825832f62a587dbbe637e`. Scientific preregistration commit: dcd8201e6952635770b3d282ac3c95ef06a70494.
Runtime: Python 3.12.16, NumPy 2.1.2, SciPy 1.16.1, scikit-learn 1.7.2.

The training archive contains 144 fitted heads and all training-bank forecasts; fixed M10 parent models remain required for inference. Exact-byte sources are archived separately for strict hash reproduction across Git newline conversion. Sizes and hashes are in m16_artifacts.json and m16_sources_artifacts.json.

Only the new M16 CPU pod and volume were deleted after verified export, and both returned404. Cumulative estimate US$1.66762243 of the original US$5 cap, including idle/setup time and a storage buffer; not a settled invoice.

## Post-experiment literature and inference limits

After this result, the primary review examined [Nitti et al. (2023)](https://academic.oup.com/mnras/article/519/2/3182/6972380), with the [author preprint](https://arxiv.org/pdf/2211.16572), section 3.2 and Appendix A, which uses an RBF kernel for coronal-hole area to solar-wind peak regression. Its inputs and event-level targets differ from our matched image-window task; no score from that study is a matched baseline. The [official scikit-learn 1.7.2 implementation](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/kernel_ridge.py) supplies squared-loss kernel regression with sample weighting. The recent [PROSWIN preprint](https://arxiv.org/abs/2609.26683) provides distributional forecasting context but uses additional channels/magnetograms and different metrics, which are excluded here.

M17 is now preregistered as a deterministic linear-versus-RBF kernel comparison on the same image features, rows and four controls. It has 144 conditions, including 36 linear and 108 RBF conditions, and is not yet implemented or run. Weighted kernel centering, fitting-only distance scales and synthetic primal/dual equivalence are required before the full source freeze. This is an adaptation, not GP uncertainty estimation or a reproduction of the cited models. The full design is in next_loop/m17_registered_design.json and RogNet experiment dddea291-a4df-408a-889a-af76c52589b6. No original historical result motivated the hypothesis.

This is an adaptive reused-development-bank study. No candidate qualified for an uncertainty check; no interval or equivalence claim is invented. Negative results bound the tested panel and do not show that every derivative-aware method fails. Digital features are proxies, not physical coronal-hole attribution.
