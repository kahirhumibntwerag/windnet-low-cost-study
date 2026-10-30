"""Present the complete fixed historical panel without choosing a replacement."""
import csv,html,json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop';D=ROOT/'matched_dashboard/dist'
LABELS={'PRIMARY_F3':'Frozen M13 primary','windnet_port':'Matched WindNet port','prior_M6':'Prior M6','prior_M10':'Prior M10',
        'regional_only':'Regional head only','regional_target_unit':'Unit target weights','regional_precision_zero':'No regional precision weight',
        'regional_global_1x1':'Global 1×1 regional head','without_outer_calibration':'No outer amplitude calibration','released_windnet':'Released original WindNet'}
METRICS=[('mean_rmse','RMSE (km/s) ↓'),('fisher_r','Fisher r ↑'),('mean_chi2','Reduced χ² ↓'),('mean_hse_ts','Original HSE TS ↑')]
def esc(s):return html.escape(str(s))
def fmt(v):return 'undefined' if v is None else f'{v:.4f}'

def dotplot(result):
    rows=result['rows'];s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 525" role="img" aria-label="All ten fixed F3 models and ablations across four original metrics"><rect width="1450" height="525" fill="#f6f7f3"/>'
    for j,row in enumerate(rows):
        yy=95+j*35;color='#c17631' if row['model']=='PRIMARY_F3' else '#246c9d' if row['model']=='prior_M10' else '#172d39' if row['model']=='released_windnet' else '#6e7b80'
        s+=f'<text x="12" y="{yy+4}" font-size="14" fill="{color}">{esc(LABELS[row["model"]])}</text>'
    for i,(key,title) in enumerate(METRICS):
        vals=[r['summary'][key] for r in rows if r['summary'][key] is not None];lo=min(vals);hi=max(vals);pad=max((hi-lo)*.12,1e-5);lo-=pad;hi+=pad
        left=290+290*i;scale=lambda v:left+(v-lo)/(hi-lo)*245
        s+=f'<text x="{left+122}" y="33" text-anchor="middle" font-size="17">{esc(title)}</text>'
        for tick in np.linspace(lo,hi,4):
            xx=scale(tick);s+=f'<path d="M{xx:.2f} 66V438" stroke="#d8ded9"/><text x="{xx:.2f}" y="461" text-anchor="middle" font-size="12">{tick:.2f}</text>'
        original=next(r['summary'][key] for r in rows if r['model']=='released_windnet')
        if original is not None:s+=f'<path d="M{scale(original):.2f} 66V438" stroke="#172d39" stroke-dasharray="5 5"/>'
        for j,row in enumerate(rows):
            v=row['summary'][key];yy=95+j*35;color='#c17631' if row['model']=='PRIMARY_F3' else '#246c9d' if row['model']=='prior_M10' else '#172d39' if row['model']=='released_windnet' else '#899498'
            if v is None:s+=f'<text x="{left}" y="{yy+4}" font-size="11">undefined</text>';continue
            s+=f'<circle cx="{scale(v):.2f}" cy="{yy}" r="{6 if row["model"] in ["PRIMARY_F3","prior_M10","released_windnet"] else 4}" fill="{color}"><title>{esc(LABELS[row["model"]])}: {v:.6f}</title></circle>'
    return s+'<text x="12" y="503" font-size="14">Orange: frozen primary · Blue: matched M10 · Dashed: released WindNet · Each axis shows its full recorded range.</text></svg>'

def section():
    if not (R/'f3_results.json').exists():return ''
    d=json.loads((R/'f3_results.json').read_text());(D/'f3_dotplot.svg').write_text(dotplot(d),encoding='utf8')
    for name in ['f3_results.json','f3_audit.json','f3_freeze.json','f3_freeze_identity.json','f3_freeze_verification.json','f3_verification.json','f3_costs.json']:
        if (R/name).exists():(D/name).write_bytes((R/name).read_bytes())
    (D/'f3_protocol.json').write_bytes((ROOT/'next_loop/f3_protocol.json').read_bytes())
    with (D/'f3_comparison.csv').open('w',newline='',encoding='utf8') as out:
        w=csv.writer(out);w.writerow(['model',*[k for k,t in METRICS],'fold_metrics_json'])
        for row in d['rows']:w.writerow([row['model'],*[row['summary'][k] for k,t in METRICS],json.dumps(row['fold_metrics'])])
    gates=d['gates'];vsoriginal=gates['released_windnet']['all_four'];vsm10=gates['prior_M10']['all_four']
    text='<section id="f3"><div class="eyebrow">F3 · FROZEN HISTORICAL EVALUATION · 157 ORIGINAL BLOCKS · 2,198 WINDOWS</div><h2>The fixed historical comparison</h2>'
    text+='<p><strong>All-four numerical gate versus released WindNet: '+('PASS' if vsoriginal else 'FAIL')+'. Incremental all-four gate versus retrained M10: '+('PASS' if vsm10 else 'FAIL')+'.</strong></p>'
    text+='<p>The training-selected primary and every registered ablation were fitted before any score was exported. All trainable entries use the same original folds, fitting rows, four-image inputs and final runtime. No model was replaced after testing. Released WindNet forecasts are a historical reference; the matched port is a separate retrained comparator.</p>'
    text+='<img src="f3_dotplot.svg" alt="Ten frozen models and ablations across all four original WindNet metrics" style="width:100%;height:auto"><div class="table-scroll"><table><thead><tr><th>Fixed model / ablation</th>'+''.join('<th>'+esc(t)+'</th>' for k,t in METRICS)+'</tr></thead><tbody>'
    for row in d['rows']:text+='<tr><td>'+esc(LABELS[row['model']])+'</td>'+''.join('<td>'+fmt(row['summary'][k])+'</td>' for k,t in METRICS)+'</tr>'
    text+='</tbody></table></div><h3>Conditional uncertainty</h3><p>Primary minus comparator, 95% whole-block bootstrap intervals. Negative favors the primary for RMSE and χ²; positive favors it for correlation and event score. These intervals do not adjust for the adaptive search.</p><div class="table-scroll"><table><thead><tr><th>Comparator</th><th>RMSE Δ</th><th>Fisher r Δ</th><th>χ² Δ</th><th>HSE TS Δ</th></tr></thead><tbody>'
    for ref,interval in d['primary_uncertainty'].items():
        text+='<tr><td>'+esc(LABELS[ref])+'</td>'
        for k in ['rmse','r','chi2','hse_ts']:
            ci=interval['metrics'][k]['delta_95'];text+='<td>'+('undefined' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]')+'</td>'
        text+='</tr>'
    text+='</tbody></table></div><p><strong>Interpretation limits.</strong> Folds 1–4 were adaptively reused for development and prior historical aggregate results were already known. This is a locked exploratory historical benchmark, not fresh blind confirmation. Blocks can overlap. All hypotheses in this continuation were chosen from training-side validation, and these test scores will not choose the next model.</p>'
    return text+'<p><a href="f3_results.json" download>All fold metrics and intervals ↓</a> · <a href="f3_comparison.csv" download>Complete fixed panel CSV ↓</a> · <a href="f3_freeze.json" download>Exact source/data/recipe freeze ↓</a> · <a href="f3_audit.json" download>Row and label-boundary audit ↓</a> · <a href="f3_verification.json" download>Aggregate and artifact verification ↓</a></p></section>'
