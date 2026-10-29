# Reproducibility notes

## Scientific sequence

Six substantive records are preserved in `results/experiment_ledger.json`:

1. **H1 — four-speed baseline:** fit original Folds 1–3, validate on Fold 4; compare ridge, fixed linear/RBF SVMs, mean, and persistence.
2. **H2 — recurrence and additional measurements:** test recent speed history, bands around one and two solar rotations, and recent magnetic/plasma values with ridge and small boosted trees. The development threshold is at least 2% RMSE improvement over H1.
3. **H3 — neural residuals:** use a 32/16-unit MLP, three fixed seeds, two weight decays, and early stopping. It improves development RMSE by only 0.90%, failing the predeclared 2% threshold.
4. **H4 — error diversity:** search 35 convex combinations of recurrence ridge, multivariate ridge/tree, and the neural ensemble. The retained ensemble must improve the accepted selection rule by at least 1%. That denominator is the accepted rule, not the best individual neural model.
5. **E5 — frozen evaluation:** preserve the original five outer holdouts. For outer Fold 5, the inner validation fold is Fold 4; for the other outer folds, it is Fold 5. Fit on the other three folds, choose parameters on the inner fold, then refit on the four outer training folds. Neural complexity must earn a 2% inner improvement over the best classical candidate, and a convex ensemble must improve the current accepted rule by 1%.
6. **E6 — controlled feature ablations:** keep ridge as the model family and the same inner/outer split. These ablations were requested after E5, so they are exploratory and do not change the frozen primary model.

The primary result is original Fold 5, which was not scored during H1–H4. The other folds were involved in design, so a five-fold mean should not be read as fully independent confirmation.

## Data and hashes

The source revision is `a996baa075ecdbd2b1c15533002692e29db43c19`. `recover_folds.py` streams the authors' compressed processed release and retains only the OMNI labels and fold identifiers. A restricted numeric unpickler rejects globals outside a short NumPy/codec allowlist. `prepare_data.py` independently matches all 3,140 block labels to the bundled daily OMNI speeds and prepares 2,198 original evaluation rows.

`verify_protocol.py` checks the original window formula, exact histories and targets, fold identities, finite targets, absence of cross-fold dates, and source-fold masking. Test offsets are 6 through 19; fitting offsets are 6 through 18, reflecting the original scripts.

- Recorded compact input SHA-256: `f4723c671bb2f9a2f21dfa491c4c14e34626836f1fa82bf979c36a39d8ed273e`.
- Frozen scientific code SHA-256: `8ec6bc81c39176c7186b3c8f0de9f52e414fbe77a32830656102eb300b332188`.

The runner records actual hashes. It requires the frozen scientific code to match after normalizing line endings. A newly serialized NPZ can have a different byte hash even when its arrays are equivalent; the runner reports this and still requires the protocol checks to pass.

## Environment

The original Runpod result recorded Python **3.12.3**, NumPy **2.1.2**, and PyTorch **2.8.0+cu128**, using image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Its SciPy and scikit-learn versions were not explicitly captured, so this repository does not claim a complete original environment lock.

`requirements.txt` pins the observed local verification stack: NumPy 2.3.1, SciPy 1.16.1, scikit-learn 1.9.1, PyTorch 2.8.0, requests 2.32.4, gdown 6.5.1, BeautifulSoup 4.15.0, Matplotlib 3.10.5, and ReportLab 5.1.0. PDF inspection additionally accepts pypdf 6.x. Under the local numerical stack, the unchanged short, recurrence, and full ridge means matched the Runpod values within `1e-7`.

Use a CUDA-enabled PyTorch installation for the complete run. Installing the numerical requirements does not by itself supply an NVIDIA driver. On a compatible PyTorch GPU image, the existing CUDA build can satisfy `torch==2.8.0`. `reproduce.py --mode check` reports GPU availability without starting training. Three fixed seeds improve repeatability, but GPU kernels, library versions, and hardware may still cause small differences; bitwise neural reproduction is not guaranteed.

## Safe rerun outputs

Run all commands from the repository root. `reproduce.py` supports:

- `--mode check`: validate inputs and the frozen source; report environment and CUDA availability; no training output.
- `--mode development`: run H1–H4 only, leaving the final holdouts unscored.
- `--mode full`: run H1–H4 and E5 on a CUDA device.
- `--mode ablations`: rerun E6 on CPU, retaining the original result only as a consistency reference.

Each training mode requires a new or empty `--output` directory, defaulting to a UTC-stamped directory under `results/rerun/`. The runner records `run_manifest.json`, including completed stages and failure status. Full/development modes write stage JSONs directly in that directory. Full mode also writes `predictions.csv` and `final_results.json`. The ablation mode writes its result under `OUTPUT/results/ablations.json` and copies the original final result there as `OUTPUT/results/final_results.json` solely for the unchanged-variant checks.

The frozen `study.py` can also run directly, but its default writes are in the repository root. The wrapper avoids accidental mixing of reruns with published results. Direct `ablations.py`, `fetch_reference.py`, and `summarize.py` write to `results/`; use them in a scratch checkout or explicitly preserve the original files first if you intend to keep the published snapshot untouched.

## Released WindNet predictions

After recovering inputs, `python fetch_reference.py` downloads only the sole saved prediction array from each author-released `211/CV_i/SDOPred43/Testing` folder. It refuses files larger than 2 MB and does not fetch model weights. It sorts original partition names like the upstream loader, then verifies every saved target against the recovered labels. The observed alignment error was exactly zero for all 2,198 targets. Downloads are saved in ignored `data/reference/`, while public comparison metadata includes their original IDs and SHA-256 hashes.

`python summarize.py` independently recomputes fold RMSE from the matched CSVs, builds the figures and comparison files, and adds a paired Fold 5 uncertainty calculation against the released WindNet arrays. Run this only after choosing which result snapshot you want in `results/`.

## Runtime and cost scope

The original study used a temporary Runpod PRO 6000 Blackwell MIG 24 GB GPU and a temporary 10 GB network volume. The main GPU run finished before resource teardown. Both resources were deleted and their absence verified. Estimated GPU rental was about US$0.12; the billing API had not posted records at cleanup, so this is not a settled total. Local ablations and dashboard/paper generation are outside that GPU estimate. `reproduce.py` itself never provisions resources or reads an API key.

## Publication and data terms

Downloaded image arrays, downloaded model checkpoints, and credentials are absent from the public repository. Original labels and compact NPZs are recovered using the scripts rather than committed. Published results include the study's predictions and their source measurements/reference alignment for review. New source code uses MIT; original upstream materials retain their own terms. Cite the original WindNet paper and data sources when using this work.
