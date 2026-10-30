# M15: conditional routing between frozen experts

M15 completed without an adopted model. All 104 registered gates were evaluated on the same 1,512 training-bank validation windows. One candidate passed the four numerical margins and fold stability screen, but its event-score interval against M13 included zero. M13 remains the incumbent. No historical M15 evaluation ran.

## Method and matched comparison

Expert A is the fixed M14 morphology anchor; B is the fixed regional-plus-morphology anchor. Both use trend features, precision weight 4, fast-target weight 2, ridge alpha 100 and head share 0.25. With predictions A and B, routing is A + g(B-A), optionally mixed with fixed M13. Eight static gates use g in {0.125,0.25,0.5,0.75} and routing share in {0.5,1}. Ninety-six conditional gates use g = m sigmoid((B-t)/u), t in {400,450,500,550} km/s, u in {25,50,100} km/s, m in {0.25,0.5,0.75,1}, and the same two routing shares. All 104 conditions were registered before scoring.

Routing consumes only base predictions from the original four 211 Angstrom images (H4/D3). It receives no observed speed, uncertainty, date or neighboring forecast. No new base model was fitted. Original Folds 1-4 supply 108 development blocks, with original fitting/evaluation windows and the same four controls. All original Fold5 members were hashed and discarded without parsing. Four controls, two experts and original rows matched their frozen references to 1e-9 before scoring.

## Results and ablations

RMSE is the mean of fold RMSEs; correlation is Fisher-z averaged; reduced chi-squared and the original batchwise HSE threat score are averaged across folds. These are training-side results.

| Condition | RMSE (km/s) | r | Reduced chi-squared | HSE TS |
|---|---:|---:|---:|---:|
| bank_windnet_port | 82.93143258 | 0.52763652 | 29.20174774 | 0.12362043 |
| bank_prior_M6 | 79.50205000 | 0.55510617 | 29.43461605 | 0.17157625 |
| fixed_M10 | 77.97714841 | 0.57304693 | 26.88077460 | 0.23890443 |
| fixed_M13 | 77.09794756 | 0.58918557 | 24.85847072 | 0.29521633 |
| fixed_anchor_A | 76.27139305 | 0.60943807 | 22.43610669 | 0.23275818 |
| fixed_anchor_B | 77.60990409 | 0.59053331 | 24.31818693 | 0.33380877 |
| M15_conditional_t400_u50_m1_s1 | 76.09102584 | 0.60377149 | 23.00822215 | 0.31553118 |
| M15_static_g0.125_s0.5 | 76.40832117 | 0.60141234 | 23.43724234 | 0.27128303 |
| M15_static_g0.125_s1 | 76.28067478 | 0.60825978 | 22.54209191 | 0.23810578 |
| M15_static_g0.25_s0.5 | 76.47907773 | 0.60023230 | 23.54599793 | 0.27959886 |
| M15_static_g0.25_s1 | 76.33551506 | 0.60667334 | 22.68501279 | 0.23810578 |
| M15_static_g0.5_s0.5 | 76.65440818 | 0.59762969 | 23.79121087 | 0.27959886 |
| M15_static_g0.5_s1 | 76.58138936 | 0.60242063 | 23.08166154 | 0.27109974 |
| M15_static_g0.75_s0.5 | 76.87455779 | 0.59473348 | 24.07335947 | 0.29226343 |
| M15_static_g0.75_s1 | 77.00721398 | 0.59695683 | 23.62605291 | 0.30120008 |

All static ablations are listed above. The complete conditional panel and per-fold event counts are in m15_results.json and the dashboard CSV. No condition or axis range was filtered to hide a tradeoff.

The numerical pass M15_conditional_t400_u50_m1_s1 passed all four leave-one-fold-out directional checks; all fold RMSE control ratios stayed within 1.05. Its paired whole-block HSE difference intervals were:

| Comparator | Conditional 95% HSE difference interval |
|---|---:|
| bank_windnet_port | [+0.12095119, +0.27160157] |
| bank_prior_M6 | [+0.07103636, +0.23159792] |
| fixed_M10 | [+0.01488705, +0.14738462] |
| fixed_M13 | [-0.02093395, +0.06724525] |

The fixed rule requires a strictly positive lower endpoint against every comparator. The M13 comparison failed; no tolerance or interval requirement was changed. The lowest-RMSE condition was not substituted after inspecting event results.

## Verification and provenance

The three gate tests passed (stable sigmoid, bounded convex routing and input/panel validation). The stage verifier checked aggregate arithmetic, numerical gates, stability rules, candidate order, archive identity and finite prediction shapes. The provenance audit checked 26 frozen scientific source files, protocol/freeze identity, 108 development input hashes, controls and experts. These checks are not an independent refit or a raw-label metric rerun.

Freeze SHA-256: `6122c9f1af629db44ec9141bb9ac065e5ad8f955f0d76b736618b65ff09dcb3e`.
Execution: Python 3.12.16, NumPy 2.1.2, SciPy 1.16.1, scikit-learn 1.7.2. Source preregistration commit 875d216; CPU execution configuration 62be05d.

Gate recipes and training predictions are archived separately from the parent fitted models in M14/M13/M11R. Exact-byte frozen sources are supplied because Git newline conversion can change source hashes. SHA-256 values and sizes are in m15_artifacts.json and m15_sources_artifacts.json.

Only the new M15 CPU pod and volume were deleted; both returned 404 after evidence export. The cumulative estimate is US$1.64358333 of the original US$5 limit, including a storage/rounding buffer; it is not a settled invoice.

## Post-experiment literature and next hypothesis

[Collin et al., 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW004125) discusses the tension between timeline error and high-speed-stream peaks. Its extra persistence and sunspot inputs, curated data and event protocol differ from this campaign, so its reported scores are not matched baselines. M11 already tested a distribution-correction adaptation and did not pass the balanced gate.

[FastNet, 2026](https://journals.ametsoc.org/view/journals/aies/5/3/AIES-D-25-0090.1.xml), section on horizontal gradients in the loss, combines errors in raw forecast fields with errors in derived gradients. It concerns terrestrial spatial weather fields, not temporal solar-wind forecasts; it does not establish that the proposed adaptation will help here. Its separate spectral and wind-representation modifications must not be attributed to gradient loss.

Together with M15 training-side event uncertainty, this motivates M16: add a fitting-only adjacent-window difference term to the convex regional ridge objective. Pairs must stay inside the same original fitting block and its 13 fitting rows; inference still uses only one four-image window. The zero-penalty condition must reproduce M13 within 1e-7 before any positive-penalty fit. Prior M8 neural future-sequence gradient-loss failure remains part of the record; this is a different, explicitly controlled ridge adaptation. No historical test result motivated this proposal.

## Limits

The expert choice and routing panel adapt to a reused development bank. Out-of-fold base predictions do not make higher-level selection fully nested. The intervals are conditional and not selection adjusted. A point-estimate win is not robust event superiority, fresh blind confirmation or causal evidence.
