# Matched image-only WindNet research loop

This prospective campaign uses only the authors' released **211-Angstrom images**, four images per forecast, and the original H=4/D=3 block/window definitions. It is separate from the frozen earlier scalar/in-situ study. No previous fitted model, EVE feature, in-situ predictor or final test forecast is imported during development.

Public campaign: https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa

## Separation of development and final evaluation

The worker verifies all 157 original member hashes. Folds 1–3 fit the development models, Fold 4 supplies training-side validation, and Fold 5 members remain uninterpreted bytes until recipes are frozen. During the adaptive loop only validation aggregates are downloaded. The authenticated worker refuses to serve raw data or individual test forecasts. Every candidate shares fitting and validation row hashes. Final evaluation, when reached, uses the original five outer folds with identical training/test rows for every candidate.

Historical scores were already published and seen in the conversation before this campaign. Folds 1–4 also informed earlier research. These facts cannot be undone: the final pass is a frozen historical benchmark, not an independent blind confirmation. No hypothesis or model change may follow final scores.

## Comparator and metrics

The original Kaffe GoogLeNet weights and the original 832 energy features are retained. A PyTorch WindNet architecture port provides a newly trained matched-input comparator. It is explicitly a port, with its differences recorded; it must not be called byte-identical TensorFlow reproduction. Released WindNet predictions may be scored only in the frozen final evaluator and remain a separate historical comparator.

Paper metrics: mean fold RMSE, Fisher-z average Pearson correlation, reduced chi-square using target sigma squared, and mean fold high-speed-enhancement threat score from the original `HSEcalc.py`. Arithmetic mean correlation and MAE are secondary. The original event code has unconventional FP/FN naming; preserve its threat-score denominator, disclose the convention, and do not substitute a speed-threshold score.

`hse_original.py` preserves the source of Upendran et al.'s upstream commit `a996baa075ecdbd2b1c15533002692e29db43c19`, with line endings normalized to LF for portable source hashes; the original authors' terms apply. Citation: Upendran et al. (2020), *Solar wind prediction using deep learning*, https://doi.org/10.1029/2020SW002478.

## Compute and execution

`runner.py` accepts a stage request on the disposable Runpod worker. `remote.py` transfers code and aggregate exports through the authenticated transport. Connection credentials remain under ignored `.secrets/`; never commit them. The public protocol is `protocol.json`. Each fitting stage records software/code identities. Preserve failed/negative experiments and register each scientific hypothesis before execution.

The pilot is bounded by US$5 incremental estimated compute, at most six development hypotheses, and exactly one final scoring pass. Report a failure to improve honestly rather than adapting to the final holdouts.

## Reproduce the fixed final comparison

Use Python 3.12.3. The reference worker used NumPy 2.1.2, SciPy 1.16.1, scikit-learn 1.7.2 and PyTorch 2.8.0+cu128. The Python dependencies are in `requirements.txt` in this directory; install the matching CUDA 12.8 PyTorch build when matching the reference GPU environment. CPU execution is supported but is a different recorded execution environment, and will take longer. CUDA/BLAS/package differences can affect floating-point values.

From the repository root, with a **new output directory that does not exist**:

```text
python -m pip install -r matched/requirements.txt
python -m matched.reproduce_final --output results/rerun/matched-reproduction-001 --freeze results/matched/freeze.json --experiment-url https://fixrelay.vercel.app/experiments/585a7b68-6a43-48b0-a0af-ca538bd8b10b
```

This downloads the original images and Kaffe weights, checks all source/input identities, and fits the frozen recipes without rerunning model selection. It refuses to overwrite an output directory, change frozen sources, or execute a second final pass on a marked bundle. Prepared archive hashes are strict; an environment that changes serialization/compression must be reported as unable to assess exact identity rather than silently bypassing checks. Keep any independent reproduction separate from the original results.

The new code does not require the old study's data bundles or old fitted models. The original five released forecast files are retrieved by their public identifiers only inside the final evaluator. No test labels are supplied to estimator fit/predict calls. Exports contain aggregate scores, audits and portable fitted artifacts; raw test images and observations are not served to the agent or dashboard.
