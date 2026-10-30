# M19: training-only cubic interaction panel

M19 evaluated 108 preregistered cubic image-feature head mixtures on the original Folds 1–4 development bank, against the four original controls and fixed M18. No candidate met all four point-metric requirements, so M18 remains the training-qualified incumbent.

| Model | Mean fold RMSE (km/s) | Fisher mean correlation | Mean reduced chi² | Original batchwise HSE threat score |
| --- | ---: | ---: | ---: | ---: |
| Fixed M18 | 75.99875 | 0.61383 | 23.27643 | 0.36545 |
| Best-RMSE M19 (morphology, offset 2, alpha 10, share 0.25) | 75.34252 | 0.62364 | 21.40760 | 0.30997 |

This M19 candidate fails both the required 1% RMSE improvement and HSE score threshold. No conditional bootstrap was run because no point-gate candidates qualified. A separate remote audit recomputed 2,700 numerical scores and gate flags from original development rows and saved predictions; it did not independently refit models. The 108 aggregate rows are recorded in the adjacent JSON files. The validation bank is repeatedly reused, and no new holdout or operational performance is claimed.

Result SHA-256: 11e771fe489676726eee7eb0e54efe148feb1653564286d3669dd8804f673e30. Private remote fitted-model and prediction archive SHA-256: ec7d8ccd6b6dc94d0e4f7d06f50133748dcf97c81b769f28c6dd8e133fdf9996. No raw inputs or individual predictions are published here.
