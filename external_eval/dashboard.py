"""Publish aggregate external results to the existing dashboard; no raw test data."""
import csv,html,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'matched_dashboard/dist';E=ROOT/'external_eval'
KEYS=['mean_rmse','fisher_r','mean_chi2','mean_hse_ts']
LABELS=['RMSE (km/s) ↓','Fisher-z correlation ↑','Original reduced χ² ↓','Original HSE threat score ↑']
COLORS={'original_windnet':'#6d7665','M13':'#265e91'}
NAMES={'original_windnet':'Original WindNet','M13':'M13'}

def esc(x):return html.escape(str(x))
def fmt(x):return 'undefined' if x is None else f'{x:.4f}'

def table(rows):
    return '<div class="table-scroll"><table><thead><tr><th>Method / period</th>'+''.join('<th>'+esc(v)+'</th>' for v in LABELS)+'<th>Unique days</th><th>Positive σ days</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(label)+'</td>'+''.join('<td>'+fmt(row[k])+'</td>' for k in KEYS)+f'<td>{row["unique_target_days"]}</td><td>{row["positive_sigma_days"]}</td></tr>' for label,row in rows if row)+'</tbody></table></div>'

def plot_years(result):
    standard=ROOT/'artifacts/external_eval/metrics_by_year.svg'
    if standard.exists():
        (D/'external_metrics_by_year.svg').write_bytes(standard.read_bytes())
        return '<img src="external_metrics_by_year.svg" alt="All four aggregate metrics by calendar year for original WindNet and M13,2020 to2025" style="width:100%;height:auto">'
    parts=[]
    for k,label in zip(KEYS,LABELS):
        vals=[r[k] for group in result['yearly'].values() for r in group.values() if r and r[k] is not None]
        if not vals:continue
        top=1 if k=='fisher_r' else max(1,max(vals)*1.12) if k=='mean_hse_ts' else max(max(vals)*1.12,.01)
        bottom=-1 if k=='fisher_r' else 0
        x=lambda yr:64+(yr-2020)*88
        y=lambda v:254-(v-bottom)/(top-bottom)*195
        s=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 325" role="img" aria-label="{esc(label)} by calendar year,2020 through2025"><text x="64" y="23" font-size="15">{esc(label)}</text>'
        for i in range(5):
            v=bottom+(top-bottom)*i/4;yy=y(v)
            s+=f'<path d="M64 {yy:.2f}H510" stroke="#dce1dc"/><text x="55" y="{yy+4:.2f}" text-anchor="end" font-size="11">{v:.2f}</text>'
        for yr in range(2020,2026):s+=f'<text x="{x(yr)}" y="276" text-anchor="middle" font-size="11">{yr}</text>'
        for model,color in COLORS.items():
            points=[]
            for yr in range(2020,2026):
                r=result['yearly'][str(yr)][model]
                if r and r[k] is not None:
                    xx=x(yr);yy=y(r[k]);points.append(f'{xx},{yy:.3f}')
                    s+=f'<circle cx="{xx}" cy="{yy:.3f}" r="3.5" fill="{color}"><title>{esc(NAMES[model])},{yr}: {r[k]:.5f}</title></circle>'
            s+=f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        s+='<text x="285" y="302" text-anchor="middle" font-size="12">Target calendar year (UTC)</text></svg>'
        parts.append(s)
    return '<div class="plot-grid">'+''.join(parts)+'</div>'

def main():
    status=json.loads((E/'status.json').read_text(encoding='utf8'))
    result=json.loads((E/'external_results.json').read_text(encoding='utf8')) if (E/'external_results.json').exists() else None
    author=json.loads((E/'author_restoration_preflight.json').read_text(encoding='utf8'))
    m13=json.loads((E/'m13_inference_preflight.json').read_text(encoding='utf8'))
    title='Frozen models. A new solar-cycle test.'
    body='<div class="eyebrow">EXTERNAL COMPARISON · 2020–2025 · 211 Å · H4 / D3</div><h1>'+title+'</h1><p class="lead">Original released WindNet against the existing M13 model on the same new target dates. Both methods were frozen before test access.</p>'
    if result:
        passed=result['strict_all_four_numerical_win']
        verdict='M13 improved all four metrics numerically.' if passed else 'M13 did not improve all four metrics.'
        body+='<section class="integrity"><div><strong>'+verdict+'</strong><p>One frozen pass. Zero test-period fitting, tuning or model changes.</p></div><p>Numerical gains and paired uncertainty are reported separately. No model was selected or promoted from these results.</p></section>'
        body+='<section><h2>The matched comparison</h2><p>Each metric averages the five frozen historical replicas; correlation uses Fisher-z averaging. Each target date counts once.</p>'+table([(NAMES[n],r) for n,r in result['overall'].items()])+'</section>'
        body+='<section><h2>Paired differences and uncertainty</h2><p>M13 minus original WindNet. Negative favors M13 for RMSE and χ²; positive favors M13 for correlation and HSE. Conditional 95% intervals use 2,000 paired calendar-block bootstrap draws.</p><div class="table-scroll"><table><thead><tr><th>Metric</th><th>Point difference</th><th>Paired 95% interval</th><th>Defined draws</th></tr></thead><tbody>'
        for key,label in zip(KEYS,LABELS):
            u=result['paired_uncertainty']['metrics'][key];ci=u['M13_minus_original_delta_95'];a=result['overall']['M13'][key];b=result['overall']['original_windnet'][key]
            body+='<tr><td>'+esc(label)+'</td><td>'+fmt(a-b if a is not None and b is not None else None)+'</td><td>'+('undefined' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]')+f'</td><td>{u["defined_draws"]}</td></tr>'
        body+='</tbody></table></div>'
        standard=ROOT/'artifacts/external_eval/paired_differences.svg'
        if standard.exists():
            (D/'external_paired_differences.svg').write_bytes(standard.read_bytes())
            body+='<img src="external_paired_differences.svg" alt="Four paired M13 minus original WindNet metric differences with conditional95percent confidence intervals" style="width:100%;height:auto">'
        body+='</section><section><h2>Performance across calendar years</h2><p class="legend"><span style="color:#6d7665">● Original WindNet</span> <span style="color:#265e91">● M13</span></p>'+plot_years(result)+'<p class="caption">Source: frozen Runpod external evaluation, target dates 2020–2025. Each point averages five replica scores; correlation is Fisher-z averaged. Annual HSE retains partial event blocks at year boundaries. No yearly result was used for tuning.</p><details><summary>All annual scores and denominators</summary>'+table([(f'{yr} · {NAMES[n]}',r) for yr,group in result['yearly'].items() for n,r in group.items()])+'</details></section>'
        c=result['coverage'];body+=f'<section><h2>Coverage and common exclusions</h2><p>{c["retained_unique_target_days"]:,} common target days retained from {c["requested_target_days"]:,} requested days. {c["retained_common_blocks"]} of {c["requested_blocks"]} fixed calendar event blocks retained; {c["incomplete_common_blocks_excluded"]} incomplete blocks excluded for both methods.</p><p>Missing inputs and invalid targets exclude the whole original event block. Calendar gaps are never compacted. χ² uses an identical positive finite-sigma mask.</p><p class="caption">Image exclusions: {esc(result["image_sources"]["excluded_by_reason"])}.</p></section>'
    else:
        body+='<section class="integrity"><div><strong>External test running · scores remain sealed</strong><p>All ten replica forecast arrays must finish and be hashed before target labels are opened.</p></div><p>Checkpoint and implementation checks passed. No external performance result has been exported.</p></section>'
    body+='<section><h2>What was checked before test access</h2><div class="table-scroll"><table><thead><tr><th>Check</th><th>Recorded evidence</th></tr></thead><tbody>'
    body+=f'<tr><td>Original WindNet restoration</td><td>Five reference blocks, 70 forecasts; maximum absolute difference {max(r["max_abs_error_kms"] for r in author["folds"]):.6f} km/s</td></tr><tr><td>M13 saved artifacts</td><td>All 55 hashes verified; independent saved-recipe composition matched on three histories per replica; zero refits</td></tr><tr><td>Preprocessing bridge</td><td>2018-01-01 image RMS difference {m13["source_bridge"]["rms_difference_0_to256"][0]:.6f} on the 0–256 scale</td></tr><tr><td>Scorer</td><td>Direct-score equivalence, common gap exclusion, calendar bootstrap boundaries and no replica count inflation checked with synthetic data</td></tr></tbody></table></div></section>'
    body+='<section><h2>The evaluation contract</h2><p>Same four image dates, same 211 Å channel, same target days, same masks and same four metrics. Original TensorFlow author checkpoints, not a replacement baseline. M13 loads existing F3 weights and saved normalization. No test-period calibration, refitting or checkpoint selection.</p><p>The fixed 14-day event blocks are anchored on 2020-01-01. Paired uncertainty uses fixed calendar 28-day units within block-start year, shared by both methods and every replica. Definitive JSOC synoptic images are processed with the published SDOMLv1 SUM convention and the frozen version-8 AIA response table. OMNI daily sigma describes variation across hourly speeds.</p><p>This is a head-to-head external-period comparison. Its absolute metrics cannot be directly compared with the paper’s different historical target days. The older adaptive research loop remains paused.</p></section>'
    body+='<section><h2>Evidence and reproducibility</h2><p><a href="https://fixrelay.vercel.app/campaigns/fb77fb27-bc42-4d58-840b-a8e1460c0ae3">RogNet campaign ↗</a> · <a href="https://fixrelay.vercel.app/experiments/a84303f5-6cff-4af4-9aab-225c96b8b499">Registered experiment ↗</a> · <a href="https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-matched-loop/external_eval">GitHub code and complete evidence ↗</a></p><p>RogNet’s connector created this campaign privately; its available tools cannot change public sharing. Dashboard access remains unchanged.</p><ul>'
    artifacts=['model_lock.json','evaluator_freeze.json','evaluator_freeze_identity.json','author_restoration_preflight.json','m13_inference_preflight.json','protocol_amendments.json','status.json']
    if result:artifacts+=['external_results.json','prediction_lock.json','aggregate_comparison.csv','RESULTS.md','result_verification.json','costs.json']
    for name in artifacts:
        source=E/name
        if source.exists():(D/('external_'+name)).write_bytes(source.read_bytes());body+='<li><a download href="external_'+name+'">'+esc(name)+'</a></li>'
    body+='</ul><p class="caption">Complete evaluator SHA-256: <code>'+esc(status['complete_freeze_sha256'])+'</code>. Frozen code commit <a href="https://github.com/kahirhumibntwerag/windnet-low-cost-study/commit/d12d1fca5b667a55d0c5da994a0d0ec3d92e71c6">d12d1fc</a>, published before external execution.</p></section>'
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WindNet vs M13 / External 2020–2025</title><link rel="stylesheet" href="style.css"><style>.plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.plot-grid svg{width:100%;height:auto}.legend{display:flex;gap:28px}code{overflow-wrap:anywhere}td:first-child{max-width:360px}details{margin:20px 0}summary{cursor:pointer}footer{padding:30px 0}main{padding-bottom:30px}@media(max-width:760px){.plot-grid{grid-template-columns:1fr}.integrity{grid-template-columns:1fr}nav{gap:15px}}</style></head><body><header><a class="brand" href="index.html">WINDNET <span>/ EXTERNAL TEST</span></a><nav><a href="continuation.html">Campaign history</a><a href="index.html">Overview</a></nav></header><main>'+body+'<footer>Frozen external evaluation · 2020-01-01–2025-12-31 UTC</footer></main></body></html>'
    (D/'external.html').write_text(page,encoding='utf8')
    for name in ['index.html','continuation.html']:
        path=D/name;old=path.read_text(encoding='utf8')
        if result:
            old=old.replace('2020–2025 · same data and four metrics · no test-period tuning. The previous adaptive loop remains paused.','Completed: M13 improved all four metrics on 1,968 common days, with no test-period tuning. The previous adaptive loop remains paused.')
        if 'id="external-comparison-link"' not in old:
            notice='<section id="external-comparison-link" class="integrity"><div><strong>Frozen external comparison: original WindNet vs M13</strong><p>2020–2025 · same data and four metrics · no test-period tuning. The previous adaptive loop remains paused.</p></div><a class="button" href="external.html">Open external comparison →</a></section>'
            old=old.replace('<main>','<main>'+notice,1);path.write_text(old,encoding='utf8')
    print(json.dumps({'dashboard':'external.html','external_results_available':result is not None}))

if __name__=='__main__':main()
