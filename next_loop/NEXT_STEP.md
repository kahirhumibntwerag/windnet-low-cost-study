# Continuation state

The thread goal remains active: improve all four original WindNet metrics using training-side validation for every adaptive decision, search primary literature after each experiment, and record the complete trail in RogNet. Do not use Elicit or inspect original held-out Fold5 images, labels, residuals or individual forecasts for hypotheses.

## Current authoritative state

- New public campaign: `1b7e08e2-70f1-4945-b613-5e785d3f76bb`.
- M11 exact archived-control reproduction failed. Preserve the technical failure; do not call it an efficacy result.
- M11R completed with identical within-L4 controls and six registered conditions. No nonzero correction passed all four requirements; M10 is retained. `results/next_loop/m11r_results.json` and `verification.json` are authoritative.
- After M11R, searched literature again and registered M12 hypothesis `9e3aee34-65cc-497f-ab21-303e91f9ed0b`: localized image-conditioned forecasts rather than marginal distribution stretching. No M12 experiment design or workload has run yet.
- Read `results/next_loop/README.md` for the observed tradeoff, literature and limits. Next action is to register a finite M12 design with matched controls and ablations, then implement and execute it. Keep the existing all-four margins and stability gates.
- All temporary resources created here are deleted: pod `3rd3y8tivb54e8`, volume `y4hllccc62`, both404verified. Do not reuse their worker URLs or assume a process is alive. Other account resources were untouched.
- Cumulative matched-campaign cost estimate US$1.39999384 of US$5; see `results/next_loop/costs.json`. Separate `solar72/` work is not part of this matched campaign.
- Training control checkpoints and predictions are preserved in `artifacts/next_loop/m11_training_artifacts.tar.gz` and the public release `windnet-m11r-2026-10-30`. Its provider-side digest matches the local SHA256. These are training-side arrays only; reusing them as fixed controls requires checking exact split/row identities. The original Fold5 is not in this archive.
- Dashboard live at `https://windnet-matched-research-loop.wellbat32.chatgpt.site/continuation.html`; archived F1/F2 history is retained. Code/results are on `codex/windnet-matched-loop`.
- Manuscript and `solar72/` files were not changed or published as part of this continuation. Do not include them accidentally in commits or new worker uploads.

The registered M11R worker path intentionally reuses its own live-worker caches. It cannot run unchanged on a fresh worker because those caches are not present; prepare and audit a new M12 worker explicitly. The legacy matched/event scientific source files remain frozen. Hypotheses may use the training-only result above but must not use archived historical final scores as feedback.
