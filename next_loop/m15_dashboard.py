"""Full M15 gate panel with explicit point-score and uncertainty distinction."""
import csv,html,json
from pathlib import Path
from next_loop.m14_dashboard import metric_table
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop';D=ROOT/'matched_dashboard/dist'
KEYS=['rmse','r','chi2','hse_ts']
def esc(s):return html.escape(str(s))
def fmt(v):return 'undefined' if v is None else f'{v:.4f}'

def scatter(d):
    points=d['rows']+d['controls']+d['component_ablations'];lo=min(p['metrics']['rmse'] for p in points);hi=max(p['metrics']['rmse'] for p in points);span=max(hi-lo,1);lo-=.03*span;hi+=.03*span;top=max(p['metrics']['hse_ts'] for p in points)*1.1
    x=lambda v:75+(v-lo)/(hi-lo)*800;y=lambda v:340-v/top*285
    s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 405" role="img" aria-label="Every M15 training gate with four controls and two fixed expert ablations"><rect width="940" height="405" fill="#f6f7f3"/>'
    for j in range(5):
        xv=lo+(hi-lo)*j/4;yv=top*j/4
        s+=f'<path d="M{x(xv):.2f} 55V340M75 {y(yv):.2f}H875" stroke="#d8ded9"/><text x="{x(xv):.2f}" y="360" text-anchor="middle" font-size="12">{xv:.1f}</text><text x="63" y="{y(yv)+4:.2f}" text-anchor="end" font-size="12">{yv:.2f}</text>'
    for row in d['rows']:
        m=row['metrics'];passed=row['gate']['all_four'];color='#2a7b64' if row['recipe']['kind']=='conditional' else '#8a969e'
        s+=f'<circle cx="{x(m["rmse"]):.2f}" cy="{y(m["hse_ts"]):.2f}" r="{5 if passed else 3}" fill="{color}" stroke="{"#c17631" if passed else "none"}" stroke-width="2"><title>{esc(row["model"])}: RMSE{m["rmse"]:.4f}, HSE{m["hse_ts"]:.4f}; numerical pass {passed}</title></circle>'
    for row in d['controls']:
        m=row['metrics'];cx=x(m['rmse']);cy=y(m['hse_ts'])
        if row['model']=='fixed_M13':s+=f'<rect x="{cx-5:.2f}" y="{cy-5:.2f}" width="10" height="10" fill="#172d39"><title>Fixed M13 incumbent</title></rect>'
        else:s+=f'<path d="M{cx-4:.2f} {cy:.2f}h8M{cx:.2f} {cy-4:.2f}v8" stroke="#172d39" stroke-width="2"><title>{esc(row["model"])}</title></path>'
    for row in d['component_ablations']:
        m=row['metrics'];cx=x(m['rmse']);cy=y(m['hse_ts']);s+=f'<path d="M{cx:.2f} {cy-6:.2f}l6 11h-12z" fill="#246c9d"><title>{esc(row["model"])}</title></path>'
    return s+'<text x="75" y="25" font-size="13">All 104 gates · conditional green · static gray · outlined: numerical pass · M13 square · experts triangles</text><text x="475" y="391" text-anchor="middle" font-size="14">Mean-fold RMSE (km/s) ↓</text><text x="19" y="200" transform="rotate(-90 19 200)" text-anchor="middle" font-size="14">Mean original HSE threat score ↑</text></svg>'

def section():
    path=R/'m15_results.json'
    if not path.exists():return ''
    d=json.loads(path.read_text());rows=d['rows'];(D/'m15_scatter.svg').write_text(scatter(d),encoding='utf8')
    for name in ['m15_results.json','m15_preflight.json','m15_freeze.json','m15_verification.json','m15_provenance_verification.json','m15_artifacts.json','m15_sources_artifacts.json','m15_costs.json']:
        if (R/name).exists():(D/name).write_bytes((R/name).read_bytes())
    (D/'m15_protocol.json').write_bytes((ROOT/'next_loop/m15_protocol.json').read_bytes())
    with (D/'m15_comparison.csv').open('w',newline='',encoding='utf8') as handle:
        w=csv.writer(handle);w.writerow(['model',*KEYS,'all_four','fold_screen','recipe_json','fold_metrics_json'])
        for row in rows:w.writerow([row['model'],*[row['metrics'][k] for k in KEYS],row['gate']['all_four'],row['stability']['pre_bootstrap_passed'],json.dumps(row['recipe']),json.dumps(row['fold_metrics'])])
    passed=[r for r in rows if r['gate']['all_four']]
    verdict='No gate passed every adoption requirement; M13 remains the training incumbent.' if not d['selected'] else 'A gate qualified on training-side validation; no historical M15 evaluation has run.'
    body='<section id="m15"><div class="eyebrow">M15 · TRAINING-SIDE FORECAST ROUTING · 1,512 ORIGINAL VALIDATION WINDOWS</div><h2>Can conditional routing balance the two experts?</h2><p><strong>'+verdict+'</strong></p>'
    body+=f'<p>{len(rows)} registered gates · {len(passed)} joint numerical passes · {sum(r["gate"]["all_four"] and r["stability"]["pre_bootstrap_passed"] for r in rows)} fold-screen passes · {len(d["bootstrap_checks"])} interval checks. No new base model was fitted.</p>'
    body+='<p>Blend the fixed M14 morphology and regional-morphology anchor forecasts using either a constant weight or a sigmoid of the second expert’s predicted speed, then optionally mix with M13. The gate accepts predictions only. All base forecasts use identical original four-image information and rows; no observed speed, label or neighboring forecast enters the routing input.</p><img src="m15_scatter.svg" alt="All 104 static and conditional gate scores, four incumbent controls and two fixed experts" style="width:100%;height:auto"><p class="caption">Axes show the complete recorded range. A numerical pass does not imply adoption: the registered uncertainty and stability requirements also apply.</p>'
    if passed:body+='<h3>Numerical pass, kept separate from adoption</h3>'+metric_table(passed,'Numerical-pass condition')
    body+='<h3>Conditional event-score uncertainty</h3>'
    for check in d['bootstrap_checks']:
        body+='<p>'+esc(check['model'])+' · complete adoption '+('pass' if check['passed'] else 'fail')+'</p><div class="table-scroll"><table><thead><tr><th>Comparator</th><th>HSE difference, conditional 95% interval</th></tr></thead><tbody>'
        for name,value in check['bootstrap'].items():
            ci=value['hse_ts']['delta_95'];body+='<tr><td>'+esc(name)+'</td><td>'+('undefined' if ci is None else f'[{ci[0]:+.5f}, {ci[1]:+.5f}]')+'</td></tr>'
        body+='</tbody></table></div>'
    body+='<h3>Four incumbent controls</h3>'+metric_table(d['controls'],'Control')+'<h3>Fixed component ablations</h3>'+metric_table(d['component_ablations'],'Expert')
    body+='<p><a href="m15_results.json" download>Complete results and all intervals ↓</a> · <a href="m15_comparison.csv" download>All gate comparisons CSV ↓</a> · <a href="m15_protocol.json" download>Frozen protocol ↓</a> · <a href="https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-m15-2026-10-30">Gate recipes, training predictions and exact sources ↗</a></p><details><summary>Inspect all 104 conditions</summary><label>Filter gate name <input type="search" class="regional-filter" data-target="table-M15" placeholder="static, threshold, temperature or mixture"></label><div class="table-scroll"><table id="table-M15"><thead><tr><th>Condition</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th><th>All four</th><th>Fold screen</th></tr></thead><tbody>'
    for row in rows:body+='<tr><td>'+esc(row['model'])+'</td>'+''.join('<td>'+fmt(row['metrics'][k])+'</td>' for k in KEYS)+'<td>'+('pass' if row['gate']['all_four'] else 'fail')+'</td><td>'+('pass' if row['stability']['pre_bootstrap_passed'] else 'fail')+'</td></tr>'
    return body+'</tbody></table></div></details><p>Expert selection and gate tuning reuse the development bank. Out-of-fold base forecasts do not make this a fully nested evaluation. Intervals are conditional, not adjusted for adaptive selection. No original historical M15 scoring or F3 feedback was used.</p></section>'
