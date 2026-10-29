# Completed E13 locked final comparison

This directory contains the completed one-shot, fixed 15-model comparison on the original five historical WindNet holdouts. [Full paper: eight pages](../../output/pdf/windnet-full-study.pdf). The [hosted dashboard](https://windnet-research-lab-oct29.wellbat32.chatgpt.site/locked-final.html) remains owner-private; code, paper, results, data, and weights are public in the [source branch](https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-study). The [E13 protocol/outcome](https://fixrelay.vercel.app/experiments/a83dbc31-db26-4d65-840e-8ac888103019) and all 13 [campaign records](https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee) are public.

Neither balanced-improvement gate passed (`outcome_verdict.json`). The starting wind model and H9 specialist were the predeclared overall and fast-wind primaries. P1's positive ESP-feature contrast is descriptive; it was not promoted. The actual old frozen winner is a saved historical comparator, distinct from freshly retrained P0. These previously viewed holdouts are not fresh confirmation, and no tuning or model changes followed this pass.

## Artifacts

- `final_results.json`, `comparison.csv`, and `predictions.npz`: all 15 fixed models, fold summaries, pooled fast-wind metrics, paired reference coverage/skill, and duplicate sensitivity.
- `freeze_manifest.json`, `recipe_freeze.json`, `e13_protocol.json`, and `execution_started.json`: preregistered recipes, identities, and one-shot start record.
- [`artifacts/locked_final/checkpoints.tar.gz`](../../artifacts/locked_final/checkpoints.tar.gz) and `checkpoint_verification.json`: **40 saved CPU-portable base artifacts**, eight per fold; fixed mixtures reconstruct derived forecasts. The published archive is in `artifacts/locked_final`, not this results directory. Completed serialization verification used synthetic inputs without refitting or rerunning evaluation.
- `outcome_verdict.json`, `experiment_timeline.json`, `execution_runtime.json`, and `costs.json`: outcome, all 13 records, actual runtime, and deletion/cost evidence.
- [`artifacts/locked_final/`](../../artifacts/locked_final): exact final bundle, manifest, freeze/recipe files, and checkpoint archive. This final-only bundle contains evaluation inputs and frozen references; it must not enter further selection.

## Independent CPU reproduction

The reference execution used **Python 3.12.3**, NumPy **2.3.1**, SciPy **1.16.1**, scikit-learn **1.9.1**, and PyTorch **2.8.0+cu128**, with CUDA unavailable and `--device cpu`. Its image was `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`. The CUDA-enabled package ran on CPU. A GPU, Runpod credentials, SDO image downloads, h5py, and astropy are not required when executing from the prepared final bundle.

From the repository root in a fresh Python 3.12.3 environment:

```bash
python -m venv .venv-final
# Activate .venv-final for your shell.
python -m pip install numpy==2.3.1 scipy==1.16.1 scikit-learn==1.9.1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

mkdir -p results/rerun
mkdir results/rerun/e13-reproduction-001
mkdir results/rerun/e13-reproduction-001/bundle
cp artifacts/locked_final/final_bundle.npz results/rerun/e13-reproduction-001/bundle/
cp artifacts/locked_final/final_bundle.manifest.json results/rerun/e13-reproduction-001/bundle/

python -m continuation.locked_final --bundle results/rerun/e13-reproduction-001/bundle --output results/rerun/e13-reproduction-001/output --freeze-manifest artifacts/locked_final/freeze_manifest.json --freeze-sha256 990efbef907fe906e3f5c22734058d5046b7059d4ee92273127369d5cfb78e7d --protocol-url https://fixrelay.vercel.app/experiments/a83dbc31-db26-4d65-840e-8ac888103019 --device cpu
```

Choose a directory that does not already exist and stop if directory creation fails. On PowerShell, use `New-Item`/`Copy-Item` for directory/copy steps; the Python command is identical. Do not execute against the checked-in `artifacts/locked_final` directory or overwrite `results/locked_final`.

The runner validates exact input/freeze/code identities, then exclusively creates `execution_<freeze-hash>.started.json` in the **copied bundle** and `execution_started.json` in the **new output directory**. Both survive failures. Changing only the output path does not authorize a second pass on that bundle; do not remove markers to bypass the rule. Preserve independent reproduction outputs separately, without selecting new recipes from them.

The ten frozen scientific files listed in the freeze manifest must remain unchanged. Each execution records its environment and input/code hashes. Different PyTorch builds, CPUs/BLAS implementations, or package versions can change floating-point results; a CPU-only PyTorch wheel is a different recorded build and should not be called byte-identical reproduction.

## Interpretation and cost

Mean fold RMSE/MAE weight folds equally; pooled errors weight finite original windows equally. Observed ≥500/≥600 subsets, same-case reference masks, and unique issue-target aggregation have different denominators. Fold SE is descriptive, not fresh confirmation. The panel includes P0-P6, H9 specialist, starting wind, H8 SDO, matched classical wind, two line/loss controls, the actual old frozen winner, and released WindNet; no test-ranked winner is selected.

Cumulative conservative compute/storage cost is **below US$0.22**, below the US$5 budget, and **is not a settled invoice**. Temporary study pods and volumes were deleted, with evidence in `costs.json`; unrelated resources were left unchanged.
