"""All registered M14 training conditions and matched feature ablations."""
import csv,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop';D=ROOT/'matched_dashboard/dist'
METRICS=['rmse','r','chi2','hse_ts']
def esc(s):return html.escape(str(s))
def fmt(v):return 'undefined' if v is None else f'{v:.4f}'

def scatter(d):
    rows=d['rows']+d['controls'];xs=[r['metrics']['rmse'] for r in rows if r['metrics']['rmse'] is not None];ys=[r['metrics']['hse_ts'] for r in rows if r['metrics']['hse_ts'] is not None]
    lo=min(xs);hi=max(xs);span=max(hi-lo,1);lo-=.03*span;hi+=.03*span;top=max(max(ys)*1.1,.1)
    x=lambda v:75+(v-lo)/(hi-lo)*800;y=lambda v:340-v/top*285
    s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 405" role="img" aria-label="Every M14 training-side condition across RMSE and original event threat score"><rect width="940" height="405" fill="#f6f7f3"/>'
    for j in range(5):
        xv=lo+(hi-lo)*j/4;yv=top*j/4
        s+=f'<path d="M{x(xv):.2f} 55V340M75 {y(yv):.2f}H875" stroke="#d8ded9"/><text x="{x(xv):.2f}" y="360" text-anchor="middle" font-size="12">{xv:.1f}</text><text x="63" y="{y(yv)+4:.2f}" text-anchor="end" font-size="12">{yv:.2f}</text>'
    colors={'betti':'#246c9d','morphology':'#c17631','regional_morphology':'#2a7b64'}
    for row in d['rows']:
        m=row['metrics'];passed=row['gate']['all_four']
        if m['rmse'] is None or m['hse_ts'] is None:continue
        color=colors[row['recipe']['family']]
        s+=f'<circle cx="{x(m["rmse"]):.2f}" cy="{y(m["hse_ts"]):.2f}" r="{4 if passed else 2.8}" fill="{color}" fill-opacity="{.95 if passed else .4}"><title>{esc(row["model"])}: RMSE{m["rmse"]:.3f}, HSE{m["hse_ts"]:.3f}; joint numerical pass {passed}</title></circle>'
    for row in d['controls']:
        m=row['metrics'];cx=x(m['rmse']);cy=y(m['hse_ts'])
        if row['model']=='fixed_M13':s+=f'<rect x="{cx-5:.2f}" y="{cy-5:.2f}" width="10" height="10" fill="#172d39"><title>Fixed M13 incumbent</title></rect>'
        else:s+=f'<path d="M{cx-4:.2f} {cy:.2f}h8M{cx:.2f} {cy-4:.2f}v8" stroke="#172d39" stroke-width="2"><title>{esc(row["model"])}</title></path>'
    return s+'<text x="75" y="25" font-size="13">All 288 conditions · Betti blue · morphology orange · regional + morphology green · M13 square</text><text x="475" y="391" text-anchor="middle" font-size="14">Mean-fold RMSE (km/s) ↓</text><text x="19" y="200" transform="rotate(-90 19 200)" text-anchor="middle" font-size="14">Mean original HSE threat score ↑</text></svg>'

def metric_table(rows,label):
    return '<div class="table-scroll"><table><thead><tr><th>'+label+'</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(r['model'])+'</td>'+''.join('<td>'+fmt(r['metrics'][k])+'</td>' for k in METRICS)+'</tr>' for r in rows)+'</tbody></table></div>'

def section():
    path=R/'m14_results.json'
    if not path.exists():return ''
    d=json.loads(path.read_text());rows=d['rows'];(D/'m14_scatter.svg').write_text(scatter(d),encoding='utf8')
    for name in ['m14_results.json','m14_preflight.json','m14_freeze.json','m14_verification.json','m14_provenance_verification.json','m14_artifacts.json','m14_costs.json']:
        if (R/name).exists():(D/name).write_bytes((R/name).read_bytes())
    (D/'m14_protocol.json').write_bytes((ROOT/'next_loop/m14_protocol.json').read_bytes())
    with (D/'m14_comparison.csv').open('w',newline='',encoding='utf8') as handle:
        w=csv.writer(handle);w.writerow(['model',*METRICS,'all_four','fold_screen','recipe_json','fold_metrics_json'])
        for row in rows:w.writerow([row['model'],*[row['metrics'][k] for k in METRICS],row['gate']['all_four'],row['stability']['pre_bootstrap_passed'],json.dumps(row['recipe']),json.dumps(row['fold_metrics'])])
    verdict='No candidate passed every adoption requirement; the M13 training incumbent is retained.' if not d['selected'] else 'A candidate qualified on training-side validation; no historical evaluation of M14 has run.'
    body='<section id="m14"><div class="eyebrow">M14 · TRAINING-SIDE ONLY · ORIGINAL FOLDS 1–4 · 1,512 WINDOWS</div><h2>Does image connectivity add information?</h2><p><strong>'+verdict+'</strong></p>'
    body+=f'<p>All {len(rows)} registered conditions are retained: {sum(r["gate"]["all_four"] for r in rows)} joint numerical passes, {sum(r["gate"]["all_four"] and r["stability"]["pre_bootstrap_passed"] for r in rows)} fold-screen passes, and {len(d["bootstrap_checks"])} interval checks.</p>'
    body+='<p>Sample connected components and enclosed digital holes across eight relative-intensity levels, then add largest-region and boundary descriptors. Compare topology alone, morphology, and morphology combined with the original regional features. These are sampled digital descriptors, not full persistent homology or physical coronal-hole attribution.</p><p>Every control and candidate uses the same original rows and four-image information. Fitting statistics use fitting labels only. The held-out original Fold5 was discarded without parsing. The F3 historical results below belong to the earlier fixed M13 primary, not M14.</p><img src="m14_scatter.svg" alt="All 288 training-only topology conditions and the four controls" style="width:100%;height:auto"><p class="caption">Axes span the full recorded range. Larger points pass all four numerical gates, which also include correlation and chi-square. A numerical pass alone does not qualify a candidate.</p>'
    if d['selected']:body+='<h3>Training-qualified candidate</h3>'+metric_table([d['selected']],'Candidate')
    body+='<h3>Four frozen controls</h3>'+metric_table(d['controls'],'Control')
    anchor=[r for r in rows if r['recipe']['view']=='trend' and r['recipe']['precision_lambda']==4 and r['recipe']['target_weight']=='fast2' and r['recipe']['ridge_alpha']==100 and r['recipe']['new_head_share']==.25]
    assert len(anchor)==3
    body+='<h3>Matched feature ablation</h3><p>Display anchor fixed before results: trend, precision lambda 4, fast2 weights, ridge alpha 100, head share 0.25. Only the feature family changes. This table does not select a replacement.</p>'+metric_table(anchor,'Feature-family condition')
    body+='<p><a href="m14_results.json" download>All fold metrics and interval checks ↓</a> · <a href="m14_comparison.csv" download>Complete comparison CSV ↓</a> · <a href="m14_protocol.json" download>Frozen protocol ↓</a> · <a href="m14_provenance_verification.json" download>Provenance checks ↓</a> · <a href="https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-m14-2026-10-30">Fitted heads and training predictions ↗</a></p><details><summary>Inspect all 288 conditions</summary><label>Filter model name <input type="search" class="regional-filter" data-target="table-M14" placeholder="family, target weight, penalty or mixture"></label><div class="table-scroll"><table id="table-M14"><thead><tr><th>Condition</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th><th>All four</th><th>Fold screen</th></tr></thead><tbody>'
    for row in rows:body+='<tr><td>'+esc(row['model'])+'</td>'+''.join('<td>'+fmt(row['metrics'][k])+'</td>' for k in METRICS)+'<td>'+('pass' if row['gate']['all_four'] else 'fail')+'</td><td>'+('pass' if row['stability']['pre_bootstrap_passed'] else 'fail')+'</td></tr>'
    return body+'</tbody></table></div></details><p>Conditional intervals are not adjusted for the adaptive search. All M14 hypotheses came from training-side M13 evidence and primary literature; historical F3 metric feedback was excluded.</p></section>'
