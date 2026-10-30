# Current image-only continuation: all four WindNet metrics improved numerically

The frozen M13 primary passed the **F3 all-four numerical gate against released WindNet** on the same original 157 blocks and 2,198 evaluation windows. Mean-fold RMSE fell **80.276 â†’ 75.237 km/s**, Fisher correlation rose **0.5500 â†’ 0.5996**, reduced chi-square fell **26.829 â†’ 22.707**, and original HSE threat score rose **0.1996 â†’ 0.2474**. Hypotheses and recipe selection used training-side validation; all ten panel entries were fixed before the historical evaluation. No model was replaced after testing.

The result is narrower against the previous M10 recipe retrained on this same worker: RMSE, correlation and chi-square improved, but event score fell **0.2706 â†’ 0.2474**, so the incremental all-four gate failed. The event-score interval versus released WindNet includes zero. Adaptive reuse of Folds 1â€“4, earlier historical aggregate exposure and overlapping blocks limit inference; this is a locked exploratory benchmark, not fresh blind confirmation.

[Public continuation campaign](https://fixrelay.vercel.app/campaigns/1b7e08e2-70f1-4945-b613-5e785d3f76bb) Â· [Current dashboard](https://windnet-matched-research-loop.wellbat32.chatgpt.site/continuation.html) Â· [F3 results and all ablations](results/next_loop/F3_RESULTS.md) Â· [Training campaign](results/next_loop/REGIONAL_RESULTS.md) Â· [Code and reproduction](next_loop/README.md)

# Archived matched image-only F1 campaign

The [new research loop](results/matched/README.md) completed six public hypotheses and one frozen final comparison using the original WindNet images and identical folds. The fixed primary reduced mean RMSE **80.28 â†’ 76.77 km/s (4.37%)**, increased Fisher correlation **0.550 â†’ 0.566**, and reduced chi-square **26.83 â†’ 26.43**, but HSE threat score fell **0.200 â†’ 0.179**. The strict all-four-metric gate failed. This is a locked historical benchmark with prior holdout exposure, not fresh confirmation.

[Public RogNet campaign](https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa) Â· [Dashboard](https://windnet-matched-research-loop.wellbat32.chatgpt.site) Â· [Final panel and ablations](results/matched/final_comparison.csv) Â· [Reproduce](matched/README.md)

---

# WindNet low-cost forecasting study

**The full study is complete:** [eight-page paper](output/pdf/windnet-full-study.pdf), [locked E13 results](results/locked_final), [exact final bundle and saved weights](artifacts/locked_final), and [CPU reproduction instructions](results/locked_final/README.md). The [public source branch](https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-study) contains the paper, results, data bundle, and weights. The [hosted final dashboard](https://windnet-research-lab-oct29.wellbat32.chatgpt.site/locked-final.html) remains owner-private; the [RogNet/FixRelay campaign and all 13 records](https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee) are public.

**No overall superiority was established.** Neither preregistered balanced-improvement gate passed in E13. The fixed 15-model panel reports useful fast-wind error reductions alongside event-detection and overall-error trade-offs. P1's positive ESP-feature ablation in the final panel is descriptive; it was not promoted into a newly selected model. No tuning or model changes followed final scoring.

The final comparison preserves the original five holdouts and evaluation windows, including released WindNet forecasts and the **actual old frozen campaign winner**. That saved historical reference is distinct from **P0**, which faithfully retrains the earlier neural/tree recipe under stricter isolation. The primary and fast-wind-primary roles were fixed before E13; no winner was selected from test scores. The historical holdouts had already been viewed, so this is a locked historical comparison, not fresh confirmation.

Development H7-H12 used original Folds 1â€“4 only, with zero original-holdout evaluations in that stage. The separately frozen E13 stage made one final evaluation pass. Derived scalar SDO/EVE features were used; no SDO images were downloaded for the continuation. [Protocols, all candidates, negative findings, and cancellations](results/continuation) remain available. The exact final bundle includes evaluation inputs and manifests, while the checkpoint archive retains **40 CPU-portable fitted artifacts** across five folds.

The cumulative conservative compute/storage estimate is **below US$0.22**, under the US$5 cap; this is an estimate, not a settled invoice. Temporary study pods and volumes were deleted, with evidence in [costs](results/locked_final/costs.json). E13 ran on CPU: Python 3.12.3, NumPy 2.3.1, SciPy 1.16.1, scikit-learn 1.9.1, and PyTorch 2.8.0+cu128 with `--device cpu`. A GPU is not required for the final reproduction.

## Historical initial study

The sections below preserve the initial campaign's results and reproduction procedure. Its historical scores did not enter the training-only continuation search. Use the [E13 reproduction guide](results/locked_final/README.md) for the completed final panel; the CUDA workflow below concerns the initial study.

A reproducible study of compact solar-wind forecasts on **the original released WindNet holdouts**, with experiment records, controlled ablations, and a direct comparison to the authors' saved predictions.

**The result is mixed.** The frozen selection rule reduces five-fold mean RMSE from WindNet's **80.28 to 77.87 km/s**, but correlation falls from **0.550 to 0.515**. On original **Fold 5**, which stayed sealed during model design, WindNet performs better: **76.35 versus 78.06 km/s**. The paired uncertainty interval includes zero, so this study does **not** establish an overall improvement over WindNet.

- [Historical initial-study dashboard](https://windnet-research-lab-oct29.wellbat32.chatgpt.site/initial-study.html) (owner-private hosting).
- [Short paper with charts and ablations](output/pdf/windnet-study.pdf).
- [Complete model comparison](results/comparison.csv), [matched predictions](results/matched_predictions.csv), and [feature ablations](results/ablations.json).
- [Public FixRelay campaign](https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee), [six experiment records](results/experiment_ledger.json), and [study protocol](research_plan.json).

## Historical initial study: what was preserved

The study recovers the authors' 211-Angstrom release rather than regenerating a random split. It retains all **157 original 20-day blocks** and **2,198 evaluation windows** for history 4 and delay 3. Every target was matched exactly to the released WindNet prediction arrays. The reference reproduces the paper's rounded mean RMSE (80.28 km/s) and correlation (0.550).

The original code uses 13 training windows and 14 test windows per block; that asymmetry is preserved. Duplicate dates within a fold are retained, and no date is shared across folds. Fitting masks historical measurements originating in a withheld fold, including when those measurements occur in recurrence features.

No SDO image is a predictor. The processed author archive bundles labels with image arrays, so approximately 1.04 GB is streamed to recover labels and original block identities; image arrays are discarded. Raw SDO archives and neural image checkpoints are not downloaded. The compact model input file is approximately 265 KiB.

## Historical initial study: what the ablations show

The controlled comparison fixes the model family to ridge regression and keeps the same holdout rows. Removing the 27-day recurrence band raises mean RMSE by **4.50 km/s**; removing the 54-day band raises it by **0.67 km/s**. Magnetic variables add only a small average benefit, and plasma variables add essentially none. Longer recent speed history alone does not help.

The full ridge, neural residual model, and convex ensemble have mean RMSEs of 78.38, 78.35, and 77.87 km/s, respectively. Those are model-family comparisons rather than isolated feature effects. The requested feature ablations were planned after final scoring and are explicitly exploratory.

## Reproduce the historical initial study

Use Python 3.12 or 3.13. A CUDA GPU is needed for the neural stages; CPU-only protocol checks and ridge ablations are supported. No Runpod account or credential is required by the research runner itself.

```bash
git clone https://github.com/kahirhumibntwerag/windnet-low-cost-study.git
cd windnet-low-cost-study
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt

git clone https://github.com/Vishal-Upendran/WindNet.git upstream
git -C upstream checkout a996baa075ecdbd2b1c15533002692e29db43c19
python recover_folds.py --channel 211
python prepare_data.py
python verify_protocol.py
python reproduce.py --mode check
```

On a CUDA-enabled machine, run the four development cycles and frozen evaluation into a **new output directory**:

```bash
python reproduce.py --mode full --output results/rerun/my-run
```

The runner refuses to overwrite existing files and leaves the checked-in results unchanged. It calls the frozen scientific functions directly and records the code hash, data hashes, package versions, GPU, and completed stages. To run development without opening the final holdouts, use `--mode development`. For an independent CPU rerun of the controlled feature ablations:

```bash
python reproduce.py --mode ablations --output results/rerun/cpu-ablations
```

The CPU path also checks that the unchanged short, recurrence, and full ridge variants match the original Runpod means within `1e-7`. It uses the checked-in final results only for this consistency check, not for choosing hyperparameters.

See [reproducibility details](docs/reproducibility.md) for the environment limits, optional reference download, and output layout. The original Runpod experiment used `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` on a PRO 6000 Blackwell MIG 24 GB GPU. Estimated GPU cost was approximately US$0.12, below the US$5 cap; this is not a settled invoice. The temporary pod and volume were deleted.

## Historical initial study: interpretation and limits

Fold 5 was unscored during the four development hypotheses. Five-fold means are exploratory because Folds 1â€“4 informed architecture design. The original random-block holdouts do not establish chronological or operational forecast skill. The Fold 5 paired cluster bootstrap uses 33 clusters of overlapping released blocks and 5,000 resamples; compact minus WindNet RMSE has a 95% interval of **[-6.16, 7.95] km/s**.

The inputs differ: this study uses longer in-situ history and extra measurements, while WindNet uses solar images. It therefore does not isolate an architectural improvement. WindNet is evaluated from the sole prediction array retained in each released Testing folder; this study did not choose a new checkpoint. The upstream code used its held-out fold for checkpoint selection, an additional limitation of that historical comparator.

## Sources and license

Upendran, V., Cheung, M. C. M., Hanasoge, S., & Krishnamurthi, G. (2020), *Solar wind prediction using deep learning*, Space Weather, 18, e2020SW002478. [DOI: 10.1029/2020SW002478](https://doi.org/10.1029/2020SW002478).

- [Original WindNet source](https://github.com/Vishal-Upendran/WindNet).
- [Author-released folds and predictions](https://drive.google.com/drive/folders/18Z9-IcjtUOdaYWOQvPnB8YCKZ4kTj581).
- [NASA OMNI documentation and fill codes](https://omniweb.gsfc.nasa.gov/html/ow_data.html).

The [MIT license](LICENSE) covers this repository's new code. Upstream code, author-released predictions, measurements, and publications retain their own terms. The RogNet/FixRelay campaign and all 13 completed/cancelled records are public; direct links are in the initial and continuation ledgers and the full dashboard history. Registry access tokens are excluded, and registry records are not independent peer review.

## M15 training-only conditional experts

[Experiment report](results/next_loop/M15_RESULTS.md): 104 prediction-only gates, one numerical pass, failed HSE uncertainty versus M13; no adoption or historical evaluation. [Complete scores](results/next_loop/m15_results.json), [verified provenance](results/next_loop/m15_provenance_verification.json) and [release archives](https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-m15-2026-10-30) preserve all ablations. The training-prediction archive requires the frozen M14/M13/M11R parent models for inference. For reproduction, overlay the exact-byte m15_frozen_sources.tar.gz snapshot before strict source-hash verification and run `python -m next_loop.m15_entry` in a fresh CPU workspace with the pinned dependencies. This command acquires only the original development folds and hashes/discards Fold5. The new [Phase III campaign](campaigns/windnet_phase3/README.md) preserves the cumulative budget and holdout rules.
