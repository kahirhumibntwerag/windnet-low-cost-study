# Frozen WindNet–M13 external comparison, 2020–2025

This is a single external generalization comparison, not another adaptive search. Both methods use their existing pre-2019 trained artifacts. No test-period fitting, calibration, checkpoint selection, or candidate promotion is allowed.

The exact author-retained 211 Å H4/D3 TensorFlow checkpoints (epochs 60, 63, 108, 290, 99) are compared with the five existing F3 M13 replicas. Each replica uses the same four images at t−6 through t−3 and the same eligible target dates. Report mean replica RMSE, Fisher-z mean correlation, mean original reduced χ² and mean original HSE threat score. Replica dates are not independent observations. This new period is directly comparable between these two methods; its absolute scores are not interchangeable with the paper's historical period.

## Integrity and preparation

`model_lock.json` locks all 15 author checkpoint components and 55 M13 artifacts before any external arrays were opened. Original image means and target scaling were recovered from corresponding author pre-2019 fitting folds, including all 20 days per block, as the original restoration code requires. Embedding normalization constants are already present in the saved graphs. Five fixed historical reference checks passed, with maximum error below 0.001 km/s. M13 loads its saved weights, scalers and coefficients; inference has no fitting or optimizer calls.

The source bridge reconstructs a fixed 2018 image from definitive JSOC synoptic FITS using published SDOMLv1 SUM preprocessing, followed by the author's `ProcessImg_common`. Its RMS difference from the released processed image is 0.001782 on the 0–256 scale. The same published version-8 AIA response table, dated 2017 and with an epoch ending in 2030, is frozen for the new period. No calibration is inferred from test images. This is a retrospective evaluation using definitive products, without a real-time latency claim.

`evaluator_freeze.json` locks all inference, acquisition, eligibility, alignment, scoring and normalization source hashes. `frozen_source_bundle.json` preserves exact source bytes as base64, including line endings, for reproduction on other operating systems. It includes no external images, targets, individual forecasts or residuals. Decode its paths into a fresh directory rather than overwriting an existing checkout. Download model artifacts only from the recorded author/F3 sources and verify every hash.

## Test protocol

Target days are 2020-01-01 through 2025-12-31 UTC. Use the first available 211 Å synoptic image nearest midnight within the same filename UTC day; require QUALITY=0 and supported finite images/exposure/radius. Do not search later in the day to replace a selected image that fails quality. Calendar 14-day target blocks are anchored on 2020-01-01; the final shorter block is retained. A block with any unavailable input or invalid target is excluded for both models. Gaps are never compacted into neighboring HSE days.

OMNI2 daily bulk speed is word 25 and its daily sigma is word 32. The latter describes dispersion across hourly values, not an average of hourly instrumental sigma. Fill 9999 is invalid. χ² uses the same finite positive-sigma mask for both methods; other metrics use all retained common target days. All ten replica forecast arrays are completed and hashed before target labels are opened. Only aggregate overall and fixed-year scores are exported.

Paired uncertainty uses 2,000 resamples of fixed calendar 28-day units, stratified by 14-day block start year, seed 20261030. All replicas and both methods share each draw. Missing calendar units are not compacted. Yearly HSE uses the intersections of the original event blocks with the calendar year and retains partial boundary segments. Numerical all-four wins and interval uncertainty are reported separately. Undefined metrics fail the all-four condition.

## Execution and records

RogNet campaign: https://fixrelay.vercel.app/campaigns/fb77fb27-bc42-4d58-840b-a8e1460c0ae3

Experiment: https://fixrelay.vercel.app/experiments/a84303f5-6cff-4af4-9aab-225c96b8b499

The existing cumulative US$5 budget applies. `compute.json` records the dedicated 8-vCPU Runpod worker at US$0.24/hour. The old adaptive campaign remains paused. Preparation failures are preserved in `status.json`; no external data were opened during those repairs. Campaign creation through the available RogNet connector is private; public sharing cannot be changed by that connector.

The staged worker runs `prepare`, `author-check`, `m13-check`, `freeze`, then exactly one `external-test`. Uploads are locked when external execution starts. Sources and model hashes are rechecked after evaluation. The worker's three-hour process guard is a compute-process backstop; resource deletion is still required and recorded separately.

Primary sources: [WindNet code](https://github.com/Vishal-Upendran/WindNet), [SDOMLv1 preprocessing](https://github.com/SDOML/SDOML/blob/master/aia_fits_to_np.py), [JSOC synoptic documentation](https://jsoc1.stanford.edu/data/aia/synoptic/README.html), [AIA response table](https://sohoftp.nascom.nasa.gov/solarsoft/sdo/aia/response/aia_V8_20171210_050627_response_table.txt), [OMNI documentation](https://omniweb.gsfc.nasa.gov/html/ow_data.html).
