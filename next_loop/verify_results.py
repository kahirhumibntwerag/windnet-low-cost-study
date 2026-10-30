"""Recompute aggregate metrics and adoption gates from exported fold summaries."""
import hashlib,json,math,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop'
d=json.loads((R/'m11r_results.json').read_text())
assert d['stage']=='M11R' and not d['fold5_available'] and d['original_final_evaluations']==0
assert len(d['rows'])==6 and len(d['controls'])==3 and d['paired_restart']
control_metrics=[x['metrics'] for x in d['controls']]
for row in d['rows']+d['controls']:
    fm=row['fold_metrics'];assert len(fm)==4 and sum(x['n'] for x in fm)==1512
    for k in ['rmse','chi2','hse_ts']:
        assert abs(sum(x[k] for x in fm)/4-row['metrics'][k])<1e-10
    r=math.tanh(sum(math.atanh(max(-.999999,min(.999999,x['r']))) for x in fm)/4)
    assert abs(r-row['metrics']['r'])<1e-10
    if 'gate' in row:
        m=row['metrics'];g={'rmse':m['rmse']<=.99*min(x['rmse'] for x in control_metrics),'r':m['r']>=max(x['r'] for x in control_metrics)+.005,'chi2':m['chi2']<=.995*min(x['chi2'] for x in control_metrics),'hse_ts':m['hse_ts']>=max(x['hse_ts'] for x in control_metrics)+.02}
        assert all(row['gate'][k]==v for k,v in g.items()) and row['gate']['all_four']==all(g.values())
old=json.loads((ROOT/'next_loop/training_reference.json').read_text())
assert [(x['fit_row_sha256'],x['validation_row_sha256']) for x in d['row_audits']]==[(x['fit_row_sha256'],x['validation_row_sha256']) for x in old['row_audits']]
metadata=json.loads((R/'m11_artifacts.json').read_text());bundle=ROOT/'artifacts/next_loop'/metadata['filename']
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==metadata['sha256']
with tarfile.open(bundle) as archive:
    names=archive.getnames();assert not any('sealed' in n or 'test' in n or 'f5_' in n for n in names)
out={'stage':'M11R','aggregate_values_recomputed':True,'all_six_gates_recomputed':True,'identical_original_training_rows':True,'fold5_available':False,'original_final_evaluations':0,'artifact_sha256_verified':metadata['sha256'],'artifact_members':len(names),'eligible_four_metric_passes':sum(r['eligible_family'] and r['gate']['all_four'] for r in d['rows']),'adopted':d['gate_passed']}
(R/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
