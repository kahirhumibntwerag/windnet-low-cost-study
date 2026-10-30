"""Current regional experiments with complete tables and original-scale scatter plots."""
import csv,html,json,math,re
from pathlib import Path
from next_loop.build_dashboard import main as build_prior
from next_loop.f3_dashboard import section as f3_section
from next_loop.m14_dashboard import section as m14_section
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop';D=ROOT/'matched_dashboard/dist'
def esc(s):return html.escape(str(s))
def fmt(v):return 'undefined' if v is None else f'{v:.4f}'
def scatter(d):
    pts=[r for r in d['rows']+d['controls'] if r['metrics']['rmse'] is not None and r['metrics']['hse_ts'] is not None]
    xs=[r['metrics']['rmse'] for r in pts];ys=[r['metrics']['hse_ts'] for r in pts];xmin=min(xs);xmax=max(xs);ymin=0;ymax=max(ys)*1.08
    xmin-=.03*(xmax-xmin);xmax+=.03*(xmax-xmin)
    x=lambda v:75+(v-xmin)/(xmax-xmin)*800;y=lambda v:340-(v-ymin)/(ymax-ymin)*285
    colors={'unit':'#8a969e','fast1.5':'#246c9d','fast2':'#c17631','density':'#2a7b64'}
    s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 405" role="img" aria-label="All registered training-side configurations, RMSE versus original HSE threat score"><rect width="940" height="405" fill="#f6f7f3"/>'
    for i in range(5):
        xv=xmin+(xmax-xmin)*i/4;yv=ymax*i/4
        s+=f'<path d="M{x(xv):.2f} 55V340M75 {y(yv):.2f}H875" stroke="#d8ded9"/><text x="{x(xv):.2f}" y="360" text-anchor="middle" font-size="12">{xv:.1f}</text><text x="63" y="{y(yv)+4:.2f}" text-anchor="end" font-size="12">{yv:.2f}</text>'
    for row in d['rows']:
        m=row['metrics']
        if m['rmse'] is None or m['hse_ts'] is None:continue
        weight=row['recipe'].get('target_weight','unit');color=colors[weight] if d['stage']=='M13' else {'[1, 1]':'#8a969e','[4, 3]':'#246c9d','[6, 6]':'#2a7b64'}[str(row['recipe']['grid'])];passed=row['gate']['all_four']
        s+=f'<circle cx="{x(m["rmse"]):.2f}" cy="{y(m["hse_ts"]):.2f}" r="{4 if passed else 2.8}" fill="{color}" fill-opacity="{.95 if passed else .35}" stroke="{color if passed else "none"}"><title>{esc(row["model"])}: RMSE{m["rmse"]:.3f}, HSE{m["hse_ts"]:.3f}; all-four numerical gate {passed}</title></circle>'
    fixed=next(r for r in d['controls'] if r['model']=='fixed_M10')['metrics']
    s+=f'<rect x="{x(fixed["rmse"])-5:.2f}" y="{y(fixed["hse_ts"])-5:.2f}" width="10" height="10" fill="#172d39"><title>Fixed M10 control</title></rect>'
    s+='<text x="475" y="391" text-anchor="middle" font-size="14">Mean-fold RMSE (km/s) ↓</text><text x="19" y="200" transform="rotate(-90 19 200)" text-anchor="middle" font-size="14">Mean original HSE threat score ↑</text>'
    s+='<text x="75" y="25" font-size="13">All configurations shown · larger outlined points pass all four numerical gates · square: M10</text>'
    return s+'</svg>'

def stage_section(stage):
    file=R/(stage.lower()+'_results.json')
    if not file.exists():return f'<section id="{stage.lower()}"><div class="eyebrow">REGISTERED · TRAINING-SIDE ONLY</div><h2>{stage} running</h2><p>No scores are displayed until the registered workload finishes.</p></section>'
    d=json.loads(file.read_text());rows=d['rows'];n=sum(r['gate']['all_four'] for r in rows);stable=sum(r['gate']['all_four'] and r['stability']['pre_bootstrap_passed'] for r in rows)
    (D/file.name).write_bytes(file.read_bytes());(D/(stage.lower()+'_protocol.json')).write_bytes((ROOT/'next_loop'/(stage.lower()+'_protocol.json')).read_bytes())
    (D/(stage.lower()+'_scatter.svg')).write_text(scatter(d),encoding='utf8')
    with (D/(stage.lower()+'_comparison.csv')).open('w',newline='',encoding='utf8') as handle:
        w=csv.writer(handle);w.writerow(['model','rmse','fisher_r','chi2','hse_ts','all_four','fold_screen','recipe_json','fold_metrics_json'])
        for r in rows:w.writerow([r['model'],*[r['metrics'][k] for k in ['rmse','r','chi2','hse_ts']],r['gate']['all_four'],r['stability']['pre_bootstrap_passed'],json.dumps(r['recipe']),json.dumps(r['fold_metrics'])])
    verdict=f'Training-qualified candidate: {esc(d["selected"]["model"])}. No original historical scoring yet.' if d['selected'] else 'No candidate passed every adoption requirement; M10 retained.'
    body=f'<section id="{stage.lower()}"><div class="eyebrow">{stage} · ORIGINAL FOLDS 1–4 · 1,512 MATCHED VALIDATION WINDOWS</div><h2>{"Regional image features" if stage=="M12" else "Fit-side tail and density weighting"}</h2><p><strong>{verdict}</strong></p><p>{len(rows)} conditions · {n} all-four numerical passes · {stable} fold-screen passes · {len(d["bootstrap_checks"])} interval checks.</p>'
    if stage=='M12':body+='<p>Compare global 1×1, regional 4×3 and 6×6 image-plane grids; flatten/trend histories; linear/squared features; precision weighting; and mixtures with M10. These intensity features are proxies, not confirmed physical coronal-hole regions.</p>'
    else:body+='<p>Compare unit weights, 1.5× or 2× emphasis on fitting targets ≥500 km/s, and an explicitly defined smoothed-label-density weighting. All weighting statistics use fitting labels only. This is an LDS-inspired ridge adaptation, not a reproduction of Balanced MSE.</p>'
    body+=f'<img src="{stage.lower()}_scatter.svg" alt="All {len(rows)} training-side configurations: RMSE against original HSE threat score" style="width:100%;height:auto"><p class="caption">Axes show the full recorded range. Numerical passes also require correlation and chi-square gains. A numerical pass alone does not qualify a model for adoption. Conditional event intervals and reused-fold limitations remain in the full export.</p>'
    if stage=='M13':body+='<p>Target-weight colors: unit gray · fast1.5 blue · fast2 orange · density green. All 192 overlapping unit-weight conditions reproduced M12 predictions exactly (maximum difference 0).</p>'
    else:body+='<p>Grid colors: global 1×1 gray · regional 4×3 blue · regional 6×6 green.</p>'
    if d['selected']:
        selected=d['selected'];body+='<h3>Training-qualified primary</h3><p>'+esc(selected['model'])+'</p><div class="table-scroll"><table><thead><tr><th>Training-side candidate</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th></tr></thead><tbody><tr><td>Qualified primary</td>'+''.join('<td>'+fmt(selected['metrics'][k])+'</td>' for k in ['rmse','r','chi2','hse_ts'])+'</tr></tbody></table></div>'
        intervals=d['bootstrap_checks'][-1]['bootstrap'];body+='<p>Conditional HSE difference 95% intervals: '+esc('; '.join(name+': '+', '.join(f'{v:+.4f}' for v in value['hse_ts']['delta_95']) for name,value in intervals.items()))+'. These intervals are not adjusted for the adaptive search or multiple candidate checks.</p>'
        keys=['grid','view','encoding','precision_lambda','ridge_alpha','new_head_share'];matched=[r for r in rows if all(r['recipe'][k]==selected['recipe'][k] for k in keys)]
        if stage=='M13':
            body+='<h3>Matched target-weight ablation</h3><p>Same selected grid, view, encoding, penalty, precision weighting and mixture share; only the fitting target weights change. Later candidates not reached by the registered checking order are not declared uncertainty failures.</p><div class="table-scroll"><table><thead><tr><th>Fitting target weights</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th><th>Numerical gate</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(r['recipe']['target_weight'])+'</td>'+''.join('<td>'+fmt(r['metrics'][k])+'</td>' for k in ['rmse','r','chi2','hse_ts'])+'<td>'+('pass' if r['gate']['all_four'] else 'fail')+'</td></tr>' for r in matched)+'</tbody></table></div>'
    body+='<div class="table-scroll"><table><thead><tr><th>Fixed control</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(r['model'])+'</td>'+''.join('<td>'+fmt(r['metrics'][k])+'</td>' for k in ['rmse','r','chi2','hse_ts'])+'</tr>' for r in d['controls'])+'</tbody></table></div>'
    body+=f'<p><a href="{stage.lower()}_results.json" download>All results, fold metrics & uncertainty ↓</a> · <a href="{stage.lower()}_comparison.csv" download>Complete comparison CSV ↓</a> · <a href="{stage.lower()}_protocol.json" download>Registered protocol ↓</a></p><details><summary>Inspect all {len(rows)} conditions and ablations</summary><label>Filter model name <input type="search" class="regional-filter" data-target="table-{stage}" placeholder="grid, weight, penalty or mixture"></label><div class="table-scroll"><table id="table-{stage}"><thead><tr><th>Condition</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th><th>All four</th><th>Fold screen</th></tr></thead><tbody>'
    for r in rows:body+='<tr><td>'+esc(r['model'])+'</td>'+''.join('<td>'+fmt(r['metrics'][k])+'</td>' for k in ['rmse','r','chi2','hse_ts'])+'<td>'+('pass' if r['gate']['all_four'] else 'fail')+'</td><td>'+('pass' if r['stability']['pre_bootstrap_passed'] else 'fail')+'</td></tr>'
    return body+'</tbody></table></div></details></section>'

def main():
    build_prior();p=D/'continuation.html';s=p.read_text(encoding='utf8')
    s=re.sub(r'<section><h2>Next hypothesis · M12</h2>.*?</section>','',s,flags=re.S)
    sections=stage_section('M13')+stage_section('M12')
    s=s.replace('<section id="method">',sections+'<section id="method">')
    s=s.replace('<nav>','<nav><a href="#m13">M13</a><a href="#m12">M12</a>',1)
    if (R/'m13_results.json').exists() and json.loads((R/'m13_results.json').read_text())['selected']:
        s=s.replace('<strong>M11R completed</strong><p>No candidate met the registered all-four gate; M10 retained.</p>','<strong>M13 qualified on training-side validation</strong><p>All four margins and the registered stability checks passed. The original historical comparison has not run; no new test-performance claim.</p>')
    s=s.replace('</body>','<script>for(const input of document.querySelectorAll(".regional-filter")){input.addEventListener("input",()=>{const q=input.value.toLowerCase();for(const row of document.querySelectorAll("#"+input.dataset.target+" tbody tr"))row.hidden=!row.cells[0].textContent.toLowerCase().includes(q);});}</script></body>')
    historical=f3_section()
    if historical:
        s=s.replace('<section id="m13">',historical+'<section id="m13">',1)
        s=s.replace('<nav>','<nav><a href="#f3">F3 historical</a>',1)
        s=s.replace('<strong>M13 qualified on training-side validation</strong><p>All four margins and the registered stability checks passed. The original historical comparison has not run; no new test-performance claim.</p>','<strong>F3 historical comparison completed</strong><p>Both all-four gates and every fixed ablation are reported below. The primary was locked before evaluation.</p>')
        s=s.replace('Original Fold5 is absent from development. No new original historical test evaluation. Prior historical exposure remains a limitation.','Original Fold5 was absent from development and acquired only after the full F3 freeze. Prior historical exposure remains a limitation.')
        s=s.replace('No original historical scoring yet.','The fixed F3 historical comparison is reported above.')
    topology=m14_section()
    if topology:
        target='<section id="f3">' if historical else '<section id="m13">'
        s=s.replace(target,topology+target,1)
        s=s.replace('<nav>','<nav><a href="#m14">M14 topology</a>',1)
        selected=json.loads((R/'m14_results.json').read_text())['selected']
        title='M14 qualified on training-side validation' if selected else 'M14 completed; M13 training incumbent retained'
        s=s.replace('<strong>F3 historical comparison completed</strong><p>Both all-four gates and every fixed ablation are reported below. The primary was locked before evaluation.</p>','<strong>'+title+'</strong><p>The complete training-only topology panel is reported below. No historical M14 evaluation has run.</p>')
    p.write_text(s,encoding='utf8')
    index=D/'index.html';text=index.read_text(encoding='utf8')
    for old in ['Training-side M11 experiments compare against fixed M10.','Training-side M12/M13 experiments compare against fixed M10.']:
        text=text.replace(old,'Training-side M12/M13 experiments compare against fixed M10. The fixed F3 historical comparison is now reported in the current campaign.' if historical else 'Training-side M12/M13 experiments compare against fixed M10.')
    if topology:
        text=re.sub(r'(<section id="current-campaign"[^>]*>.*?<p>).*?(</p>)',r'\g<1>Training-side M14 topology results and the fixed F3 historical panel are reported in the current campaign. Archived F1/F2 results below remain unchanged.\2',text,count=1,flags=re.S)
    index.write_text(text,encoding='utf8')
    print(json.dumps({'M12_complete':(R/'m12_results.json').exists(),'M13_complete':(R/'m13_results.json').exists(),'M14_complete':bool(topology),'F3_complete':bool(historical),'original_final_evaluations_M14':0}))
if __name__=='__main__':main()
