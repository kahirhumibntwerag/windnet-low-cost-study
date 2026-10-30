"""Complete training-side difference-loss panel and fixed ablation anchors."""
import csv, html, json
from pathlib import Path
from next_loop.m14_dashboard import metric_table
ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results/next_loop';D = ROOT/'matched_dashboard/dist'
KEYS = ['rmse', 'r', 'chi2', 'hse_ts']
def esc(s):return html.escape(str(s))
def fmt(v):return 'undefined' if v is None else f'{v:.4f}'


def scatter(d):
    points = d['rows']+d['controls'];lo = min(p['metrics']['rmse'] for p in points);hi = max(p['metrics']['rmse'] for p in points)
    span = max(hi-lo, 1);lo -= .03*span;hi += .03*span;top = max(max(p['metrics']['hse_ts'] for p in points)*1.1, .1)
    x = lambda v: 75+(v-lo)/(hi-lo)*800;y = lambda v: 340-v/top*285
    s = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 405" role="img" aria-label="All 144 M16 training-side conditions and four frozen controls"><rect width="940" height="405" fill="#f6f7f3"/>'
    for j in range(5):
        xv = lo+(hi-lo)*j/4;yv = top*j/4
        s += f'<path d="M{x(xv):.2f} 55V340M75 {y(yv):.2f}H875" stroke="#d8ded9"/><text x="{x(xv):.2f}" y="360" text-anchor="middle" font-size="12">{xv:.1f}</text><text x="63" y="{y(yv)+4:.2f}" text-anchor="end" font-size="12">{yv:.2f}</text>'
    for row in d['rows']:
        m = row['metrics'];color = {'regional': '#246c9d', 'morphology': '#c17631', 'regional_morphology': '#2a7b64'}[row['recipe']['family']]
        s += f'<circle cx="{x(m["rmse"]):.2f}" cy="{y(m["hse_ts"]):.2f}" r="{4 if row["gate"]["all_four"] else 2.8}" fill="{color}" fill-opacity=".55"><title>{esc(row["model"])}: RMSE {m["rmse"]:.4f}, HSE {m["hse_ts"]:.4f}; numerical pass {row["gate"]["all_four"]}</title></circle>'
    for row in d['controls']:
        m = row['metrics'];cx = x(m['rmse']);cy = y(m['hse_ts'])
        s += f'<path d="M{cx-4:.2f} {cy:.2f}h8M{cx:.2f} {cy-4:.2f}v8" stroke="#172d39" stroke-width="2"><title>{esc(row["model"])}</title></path>'
    return s+'<text x="75" y="25" font-size="13">All 144 conditions · regional blue · morphology orange · combined green · controls crosses</text><text x="475" y="391" text-anchor="middle" font-size="14">Mean-fold RMSE (km/s) ↓</text><text x="19" y="200" transform="rotate(-90 19 200)" text-anchor="middle" font-size="14">Mean original HSE threat score ↑</text></svg>'


def section():
    path = R/'m16_results.json'
    if not path.exists():return ''
    d = json.loads(path.read_text(encoding='utf8'));rows = d['rows']
    (D/'m16_scatter.svg').write_text(scatter(d), encoding='utf8')
    for name in ['results', 'preflight', 'freeze', 'zero_control', 'verification', 'provenance_verification', 'artifacts', 'sources_artifacts', 'costs']:
        source = R/f'm16_{name}.json'
        if source.exists():(D/source.name).write_bytes(source.read_bytes())
    (D/'m16_protocol.json').write_bytes((ROOT/'next_loop/m16_protocol.json').read_bytes())
    with (D/'m16_comparison.csv').open('w', newline='', encoding='utf8') as handle:
        w = csv.writer(handle);w.writerow(['model', *KEYS, 'all_four', 'fold_screen', 'recipe_json', 'fold_metrics_json'])
        for row in rows:w.writerow([row['model'], *[row['metrics'][k] for k in KEYS], row['gate']['all_four'], row['stability']['pre_bootstrap_passed'], json.dumps(row['recipe']), json.dumps(row['fold_metrics'])])
    verdict = 'A candidate qualified on training-side validation; no historical M16 evaluation has run.' if d['selected'] else 'No candidate passed every adoption requirement; M13 remains the training incumbent.'
    body = '<section id="m16"><div class="eyebrow">M16 · TRAINING-SIDE ONLY · 1,512 ORIGINAL VALIDATION WINDOWS</div><h2>Does fitting transition shape improve the joint forecast?</h2><p><strong>'+verdict+'</strong></p>'
    body += f'<p>{len(rows)} conditions · 36 raw recipes · {d["fold_fits"]} fold fits · {sum(r["gate"]["all_four"] for r in rows)} joint numerical passes · {len(d["bootstrap_checks"])} interval checks.</p>'
    body += '<p>Add an adjacent-window difference error to weighted ridge regression. Compare regional, morphology and combined features; ridge penalties 10, 100 and 1000; difference strengths 0, 0.25, 1 and 4; and four head shares with fixed M10. Every pair stays inside the original 13 fitting windows of one block. Forecasts still consume only the original four images.</p>'
    body += '<p>The zero-strength regional control reproduced saved M13 predictions within 1e−7 before any positive-strength fit. Four frozen controls and original rows matched to 1e−9. Original Fold5 was hashed and discarded without parsing.</p><img src="m16_scatter.svg" alt="Every M16 condition across RMSE and original HSE event threat score, with four controls" style="width:100%;height:auto"><p class="caption">Full recorded ranges; all conditions retained. Point-score gains also require the registered fold and uncertainty screens.</p>'
    body += '<h3>Four frozen controls</h3>'+metric_table(d['controls'], 'Control')
    anchor = [r for r in rows if r['recipe']['ridge_alpha'] == 100 and r['recipe']['new_head_share'] == .25]
    assert len(anchor) == 12
    body += '<h3>Matched difference-loss and feature ablations</h3><p>Fixed display anchor: ridge alpha 100 and head share 0.25. All three feature families and all four difference strengths are shown. The regional zero-strength row is the M13 reproduction control.</p>'+metric_table(anchor, 'Fixed anchor condition')
    for check in d['bootstrap_checks']:
        body += '<h3>Conditional event-score intervals: '+esc(check['model'])+'</h3><ul>'
        for name, value in check['bootstrap'].items():body += '<li>'+esc(name)+': '+esc(value['hse_ts']['delta_95'])+'</li>'
        body += '</ul>'
    body += '<p><a href="m16_results.json" download>Complete fold scores and intervals ↓</a> · <a href="m16_comparison.csv" download>All 144 conditions CSV ↓</a> · <a href="m16_protocol.json" download>Frozen protocol ↓</a> · <a href="m16_provenance_verification.json" download>Control and pair checks ↓</a> · <a href="https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/tag/windnet-m16-2026-10-30">Fitted heads, training predictions and exact sources ↗</a></p><details><summary>Inspect all 144 conditions</summary><label>Filter model name <input type="search" class="regional-filter" data-target="table-M16" placeholder="family, difference strength, ridge or share"></label><div class="table-scroll"><table id="table-M16"><thead><tr><th>Condition</th><th>RMSE ↓</th><th>Fisher r ↑</th><th>Reduced χ² ↓</th><th>HSE TS ↑</th><th>All four</th><th>Fold screen</th></tr></thead><tbody>'
    for row in rows:body += '<tr><td>'+esc(row['model'])+'</td>'+''.join('<td>'+fmt(row['metrics'][k])+'</td>' for k in KEYS)+'<td>'+str(row['gate']['all_four'])+'</td><td>'+str(row['stability']['pre_bootstrap_passed'])+'</td></tr>'
    return body+'</tbody></table></div></details><p>Development data are reused adaptively; intervals are conditional and not adjusted for selection. Adjacent original-window differences are not physical per-day derivatives. No historical feedback was supplied to M16.</p></section>'
