# M18: training-only polynomial interaction panel

On the original WindNet Folds 1–4 development bank, one polynomial-kernel head met the preregistered four-metric adoption gate against four fixed controls. This is training-side validation on repeatedly reused data, not independent holdout performance.

| Model | Mean fold RMSE (km/s) | Fisher mean correlation | Mean reduced chi² | Original batchwise HSE threat score |
| --- | ---: | ---: | ---: | ---: |
| Fixed M13 | 77.09795 | 0.58919 | 24.85847 | 0.29522 |
| M18 regional + morphology, polynomial offset 1, alpha 1, 50% blend | 75.99875 | 0.61383 | 23.27643 | 0.36545 |

All four leave-one-fold-out comparisons and every fold RMSE guard passed. The 2,000-draw whole-block HSE difference interval against M13 was [0.01534, 0.13438]. The RMSE and chi² difference intervals both include zero; the intervals are conditional and not selection-adjusted. Of 144 registered conditions five passed point/fold checks; the selected candidate was the first by increasing RMSE to pass all required bootstrap gates.

A separate remote scoring audit recalculated 3,600 numerical checks using saved predictions and source development rows, plus the qualifying bootstrap and ordering. This did not independently refit the model. No Fold5 or final test was opened. Sources and exact panel were frozen before scoring.

The result JSON SHA-256 is 349abe46bd111bdec4dccd80daac1f743c0b27eec3a4bc50cf118ed16059d1ba; fitted models and predictions archive SHA-256 is 2062eed10b01159d6c0d5b5a2d61c45778b259ec8e069e82a14293f3a4ecaaeb. Both are on the owner's private Runpod volume; aggregate panel rows are in m18_panel_1.json and m18_panel_2.json. No raw inputs or individual predictions are published here. No new holdout or operational result is claimed.
