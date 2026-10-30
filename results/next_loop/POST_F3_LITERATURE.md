# Literature check after F3

Searched and reviewed primary sources on 30 October 2026 after F3 completed. No historical images, labels, individual predictions or residuals were inspected. The completed F3 panel remains fixed; no test outcome selects a replacement.

## A possible next hypothesis, based on training evidence only

M13’s training bank showed that regional image features and fitting-target weighting could jointly pass the registered validation screen. This training observation, rather than the historical F3 metric differences, motivates asking whether image connectivity across intensity levels adds information beyond regional darkness fractions.

[Marchetti, Guastavino and Collin (2026), *Explainable Prediction of High-Speed Solar Wind Streams driven by Coronal Holes via Topological Data Analysis*](https://arxiv.org/html/2610.35549v1), submitted 26 October, studies persistent-homology representations of 193 Å images for a binary four-day stream-occurrence task. It reports a different dataset, labels and validation scheme. Its scores are not matched WindNet baselines or evidence that the same features improve continuous 211 Å forecasting.

**Queued hypothesis, not executed:** an image-only topological descriptor head may complement the M13 regional head on training-side validation. A future experiment would retain the original four 211 Å images, original fitting/evaluation rows and the same controls, and register a small feature/weight/mixture panel before fitting. All preprocessing and model selection would remain fitting-side only. No new wavelength, event catalog or external predictor would be imported. A threshold-component approximation must be labeled as such, rather than claimed as a reproduction of full persistent homology.

## What the weighting literature does and does not establish

[Yang et al. (2021), *Delving into Deep Imbalanced Regression*](https://proceedings.mlr.press/v139/yang21m.html), motivates smoothing continuous label and feature distributions. M13’s density weighting was an explicitly defined adaptation for ridge regression, not a reproduction of the full DIR method. Its validation tradeoffs remain in the complete M13 table; the paper does not override the campaign’s measured outcome.

## Evaluation discipline

[Cawley and Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*](https://jmlr.org/papers/v11/cawley10a.html), explains how optimizing a noisy selection criterion can bias evaluation. It supports limiting subsequent searches and disclosing selection effects. It does not make the reused training bank or historical benchmark independent.

The original four-metric numerical target against released WindNet has been met by the frozen F3 primary. This literature check queues a possible training-only idea; it does not authorize a claim of fresh confirmation or justify changing a tested recipe. The separate incremental gate against retrained M10 failed and is preserved without promoting any ablation.
