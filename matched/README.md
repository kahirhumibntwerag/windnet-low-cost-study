# Matched image-only WindNet research loop

This prospective campaign uses only the authors' released **211-Angstrom images**, four images per forecast, and the original H=4/D=3 block/window definitions. It is separate from the frozen earlier scalar/in-situ study. No previous fitted model, EVE feature, in-situ predictor or final test forecast is imported during development.

Public campaign: https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa

## Separation of development and final evaluation

The worker verifies all 157 original member hashes. Folds 1–3 fit the development models, Fold 4 supplies training-side validation, and Fold 5 members remain uninterpreted bytes until recipes are frozen. During the adaptive loop only validation aggregates are downloaded. The authenticated worker refuses to serve raw data or individual test forecasts. Every candidate shares fitting and validation row hashes. Final evaluation, when reached, uses the original five outer folds with identical training/test rows for every candidate.

Historical scores were already published and seen in the conversation before this campaign. Folds 1–4 also informed earlier research. These facts cannot be undone: the final pass is a frozen historical benchmark, not an independent blind confirmation. No hypothesis or model change may follow final scores.

## Comparator and metrics

The original Kaffe GoogLeNet weights and the original 832 energy features are retained. A PyTorch WindNet architecture port provides a newly trained matched-input comparator. It is explicitly a port, with its differences recorded; it must not be called byte-identical TensorFlow reproduction. Released WindNet predictions may be scored only in the frozen final evaluator and remain a separate historical comparator.

Paper metrics: mean fold RMSE, Fisher-z average Pearson correlation, reduced chi-square using target sigma squared, and mean fold high-speed-enhancement threat score from the original `HSEcalc.py`. Arithmetic mean correlation and MAE are secondary. The original event code has unconventional FP/FN naming; preserve its threat-score denominator, disclose the convention, and do not substitute a speed-threshold score.

`hse_original.py` is copied unchanged from Upendran et al.'s upstream commit `a996baa075ecdbd2b1c15533002692e29db43c19`; the original authors' terms apply. Citation: Upendran et al. (2020), *Solar wind prediction using deep learning*, https://doi.org/10.1029/2020SW002478.

## Compute and execution

`runner.py` accepts a stage request on the disposable Runpod worker. `remote.py` transfers code and aggregate exports through the authenticated transport. Connection credentials remain under ignored `.secrets/`; never commit them. The public protocol is `protocol.json`. Each fitting stage records software/code identities. Preserve failed/negative experiments and register each scientific hypothesis before execution.

The pilot is bounded by US$5 incremental estimated compute, at most six development hypotheses, and exactly one final scoring pass. Report a failure to improve honestly rather than adapting to the final holdouts.
