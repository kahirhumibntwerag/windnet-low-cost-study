# New campaign: M11 and M11R

Public RogNet campaign: https://fixrelay.vercel.app/campaigns/1b7e08e2-70f1-4945-b613-5e785d3f76bb

All results here are **training-side validation only** on 1,512 original windows in Folds1–4. No original historical test evaluation was performed. Fold5 archive members were hash-checked, then discarded without parsing or retaining them.

## Original M11: technical failure

The registered run failed its strict archived-control reproduction prerequisite. Original training/validation row hashes, frozen scientific code hashes and recorded Python/NumPy/Torch versions match, but numerical control scores do not. The saved failed process state and `technical_observation.json` preserve this. The cause is not isolated; no scientific calibration verdict is inferred from this failure. The original source is committed at `6591fd1`.

## M11R: matched comparison, no balanced improvement

A separate recovery experiment was registered before viewing any calibrated-candidate metrics. It used the same six recipes and same success margins, with the same already-fitted live-L4 controls for every comparison. It does **not** claim numerical reproduction of archived controls. Exact recovery conditions are in `next_loop/m11r_protocol.json`.

| Condition | RMSE km/s | Fisher r | Reduced chi-square | Original HSE threat score |
| --- | ---: | ---: | ---: | ---: |
| Fixed M10, correction0 | 77.9771 | 0.57305 | 26.8808 | 0.23890 |
| Cross-fitted correction0.25 | 78.8109 | 0.57627 | 27.3853 | 0.29729 |
| Cross-fitted correction0.5 | 80.6574 | 0.57742 | 28.6158 | 0.36207 |
| Cross-fitted correction0.75 | 83.4527 | 0.57749 | 30.5724 | 0.39836 |
| Cross-fitted correction1 | 87.1107 | 0.57698 | 33.2550 | 0.44146 |
| In-sample-map ablation (not eligible) | 78.6603 | 0.57565 | 28.2875 | 0.29511 |

No nonzero cross-fitted strength passed all four numerical requirements. All improved the event score but worsened RMSE and chi-square, and missed the required correlation margin. M10 is retained; the highest event-score result is not promoted. No candidate reached bootstrap adoption testing. `m11r_results.json` includes all fold metrics, maps, controls, source identities and archived-control differences. `verification.json` independently recomputes aggregate metrics and gates and confirms the original row identities.

The training-only model/prediction archive has 21 members, 87,920,680 bytes, SHA256 `e03668f2034eee2321234b83858f11caf40c79d78acae7df0bb64c6d61ae085b`. No raw original holdout data is included. Check the GitHub release for the downloadable archive.

## Literature searched again after the outcome

- NASA CCMC, SIML-HSS1: https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/ . Gridded image information and polynomial interactions motivate a localized next model; its in-situ recurrence inputs are excluded here.
- Abraham-Alowonle et al. (2026), DOI https://doi.org/10.1029/2025EA004523 . Image morphology and spatial preprocessing motivate local image-plane proxies; different channels, splits, event definitions and published scores are not matched comparisons.
- Collin et al. (2025), DOI https://doi.org/10.1029/2024SW004125 . Regional coronal-hole information is relevant context, not evidence that a new recipe will pass this campaign's gate.

M12 is registered as a new hypothesis: condition forecasts on localized image darkness and temporal evolution instead of a global marginal correction. No M12 workload has run; exact finite recipes and ablations remain to be registered before execution. The processed image archive lacks physical WCS, so proposed image-plane features must not be described as verified coronal-hole source regions.

## Cost and cleanup

The newly created Runpod pod and 10GB network volume were deleted after download and SHA verification; both returned404 on read-back. This campaign's conservative estimate is US$0.16615 including a US$0.02 storage/rounding buffer. The cumulative matched campaign estimate is US$1.39999 within the US$5 envelope; this is not a settled invoice. See `costs.json`. Unrelated resources were untouched. No Elicit tool was used.
