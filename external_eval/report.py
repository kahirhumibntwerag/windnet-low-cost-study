"""Human-readable report from the frozen aggregate export, without new inference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'external_eval'
KEYS=['mean_rmse','fisher_r','mean_chi2','mean_hse_ts']
LABELS=['RMSE (km/s), lower is better','Fisher-z mean correlation, higher is better','Original mean reduced χ², lower is better','Original HSE threat score, higher is better']

def f(x):return 'undefined' if x is None else f'{x:.6f}'

def main():
    r=json.loads((E/'external_results.json').read_text(encoding='utf8'))
    c=r['coverage'];a=r['overall']['M13'];b=r['overall']['original_windnet']
    verdict='M13 improved all four primary metrics numerically.' if r['strict_all_four_numerical_win'] else 'M13 did not improve all four primary metrics numerically.'
    body='# Frozen original WindNet versus M13: external 2020–2025 comparison\n\n'+verdict+' The comparison used '+str(c['retained_unique_target_days'])+' common target days. Both models were frozen before test access, and neither was tuned, refitted, recalibrated or selected using this period.\n\n'
    body+='| Primary metric | Original WindNet | M13 | M13 − original | Paired 95% interval |\n|---|---:|---:|---:|---:|\n'
    for key,label in zip(KEYS,LABELS):
        ci=r['paired_uncertainty']['metrics'][key]['M13_minus_original_delta_95'];delta=a[key]-b[key] if a[key] is not None and b[key] is not None else None
        body+='| '+label+' | '+f(b[key])+' | '+f(a[key])+' | '+f(delta)+' | '+('undefined' if ci is None else '['+f(ci[0])+', '+f(ci[1])+']')+' |\n'
    body+='\nEach method has five existing historical replicas. RMSE, χ² and HSE scores are averaged across replicas; correlations are Fisher-z averaged. The repeated dates are not five independent test sets. The strict numerical all-four criterion and uncertainty are separate statements. The intervals are conditional on these frozen models and this period, using 2,000 paired resamples of fixed calendar 28-day units within block-start year, seed 20261030. They do not establish performance in a different solar cycle or adjust for every earlier development decision.\n\n'
    body+='![All four metrics by calendar year](../artifacts/external_eval/metrics_by_year.svg)\n\nCalendar-year summaries were fixed before test access and were not used to change either method. Annual HSE preserves intersections of the original event blocks with each year, including partial boundary segments.\n\n![Paired metric differences and conditional confidence intervals](../artifacts/external_eval/paired_differences.svg)\n\n'
    body+='## Matched inputs and exclusions\n\n'
    body+=f'Target dates were 2020-01-01 through 2025-12-31 UTC: {c["requested_target_days"]} requested days. The evaluator retained {c["retained_common_blocks"]} of {c["requested_blocks"]} fixed calendar 14-day event blocks and excluded {c["incomplete_common_blocks_excluded"]} incomplete blocks for both models. The shared χ² mask contains {a["positive_sigma_days"]} positive finite daily-sigma values. Input-image exclusions by frozen reason: {r["image_sources"]["excluded_by_reason"]}.\n\n'
    body+='Both methods receive the same 211 Å image dates t−6, t−5, t−4 and t−3. Images come from the definitive JSOC AIA synoptic archive, processed with the published SDOMLv1 SUM convention, the frozen 2017 version-8 AIA response table, and the author’s intensity transformation. The first available same-day image nearest midnight is selected; a failed quality flag does not trigger a search for a more favorable replacement. Calendar gaps are never compacted into neighboring HSE days. OMNI2 daily bulk speed and its daily sigma enter only the final scoring boundary. The sigma describes dispersion across hourly speeds.\n\n'
    body+='## Freeze and verification\n\n'
    body+='Original WindNet uses the actual author-retained TensorFlow checkpoints at epochs 60, 63, 108, 290 and 99. M13 loads the existing F3 trained artifacts and normalization. No model was retrained. Original image means and target scaling were reconstructed only from corresponding pre-2019 fitting folds. All five original-model reference checks passed with maximum forecast difference below 0.001 km/s. All 55 M13 artifact hashes were verified, and independent saved-recipe composition matched on pre-2019 examples. The reconstructed 2018 input image matched the released processed image to RMS 0.001782 on the 0–256 scale. This bridge is an implementation check, not external performance evidence. Individual F3 forecasts were not retained, so exact vector-by-vector cross-runtime reproduction of M13 cannot be claimed.\n\n'
    body+='All ten external replica forecast arrays were completed and hashed before the target labels were opened. The evaluator rechecked scientific-source and model hashes after scoring. Individual external images, target labels, forecasts and residuals were not exported or inspected for hypothesis generation. This report uses only the aggregate export. The earlier adaptive improvement loop remains paused; no model is promoted from this external test.\n\n'
    body+='This is a matched head-to-head comparison on a new period. Its absolute scores are not directly interchangeable with WindNet’s published 2011–2018 evaluation because the target days differ. The evaluation is retrospective, using definitive images; it does not establish real-time data availability.\n\n'
    body+='## Reproducibility\n\n'
    body+='- [Complete aggregate scores, calendar-year results and coverage](external_results.json)\n- [Aggregate CSV](aggregate_comparison.csv)\n- [Prediction lock](prediction_lock.json)\n- [Model lock](model_lock.json)\n- [Complete evaluator freeze](evaluator_freeze.json)\n- [Exact source bytes](frozen_source_bundle.json)\n- [Preprocessing and metric protocol](README.md)\n- [RogNet experiment](https://fixrelay.vercel.app/experiments/a84303f5-6cff-4af4-9aab-225c96b8b499)\n- [Dashboard](https://windnet-matched-research-loop.wellbat32.chatgpt.site/external.html)\n\n'
    body+='Frozen source commit: `d12d1fca5b667a55d0c5da994a0d0ec3d92e71c6`. Evaluator freeze SHA-256: `'+r['freeze_sha256']+'`. Prediction archive SHA-256: `'+r['predictions']['sha256']+'`. The source snapshot was published before external execution. RogNet records this separate campaign privately because its connected tools cannot change public sharing; GitHub code and aggregate artifacts are public.\n'
    (E/'RESULTS.md').write_bytes(body.encode('utf8'))
    print(json.dumps({'report':'external_eval/RESULTS.md','source':'aggregate results only','all_four_numerical_win':r['strict_all_four_numerical_win']}))

if __name__=='__main__':main()
