"""Independent arithmetic checks of aggregate exports and saved prediction banks."""
import argparse,hashlib,io,json,math,tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def aggregate(folds):
    out={k:(sum(x[k] for x in folds)/len(folds) if all(x[k] is not None for x in folds) else None) for k in ['rmse','chi2','hse_ts']}
    out['r']=math.tanh(sum(math.atanh(max(-.999999,min(.999999,x['r']))) for x in folds)/len(folds)) if all(x['r'] is not None for x in folds) else None
    return out
def verify(stage):
    out=ROOT/'results/next_loop';d=json.loads((out/(stage.lower()+'_results.json')).read_text());meta=json.loads((out/(stage.lower()+'_artifacts.json')).read_text())
    assert d['stage']==stage and not d['fold5_available'] and d['original_final_evaluations']==0
    reference=json.loads((ROOT/'next_loop/m12_reference.json').read_text())
    assert [(x['fit_row_sha256'],x['validation_row_sha256']) for x in d['row_audits']]==[(x['fit_row_sha256'],x['validation_row_sha256']) for x in reference['row_audits']]
    cs=[aggregate(x['fold_metrics']) for x in d['controls']]
    for row in d['rows']+d['controls']:
        actual=aggregate(row['fold_metrics']);assert sum(x['n'] for x in row['fold_metrics'])==1512
        for k in actual:assert actual[k] is None and row['metrics'][k] is None or abs(actual[k]-row['metrics'][k])<1e-10
        if 'gate' not in row:continue
        if any(v is None for v in actual.values()):assert not row['gate']['all_four'];continue
        flags=[actual['rmse']<=.99*min(x['rmse'] for x in cs),actual['r']>=max(x['r'] for x in cs)+.005,actual['chi2']<=.995*min(x['chi2'] for x in cs),actual['hse_ts']>=max(x['hse_ts'] for x in cs)+.02]
        assert flags==[row['gate'][k] for k in ['rmse','r','chi2','hse_ts']] and all(flags)==row['gate']['all_four']
        passes=0
        for drop in range(4):
            m=aggregate([v for i,v in enumerate(row['fold_metrics']) if i!=drop]);c=[aggregate([v for i,v in enumerate(x['fold_metrics']) if i!=drop]) for x in d['controls']]
            passes+=m['rmse']<min(x['rmse'] for x in c) and m['r']>max(x['r'] for x in c) and m['chi2']<min(x['chi2'] for x in c) and m['hse_ts']>max(x['hse_ts'] for x in c)
        ratios=[m['rmse']/min(x['fold_metrics'][i]['rmse'] for x in d['controls']) for i,m in enumerate(row['fold_metrics'])]
        assert row['stability']['directional_passes']==passes
        assert row['stability']['pre_bootstrap_passed']==(passes>=3 and max(ratios)<=1.05)
    expected=sorted([r for r in d['rows'] if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']],key=lambda r:r['metrics']['rmse'])
    for i,check in enumerate(d['bootstrap_checks']):
        assert check['model']==expected[i]['model']
        assert check['passed']==all(x['hse_ts']['delta_95'][0]>0 for x in check['bootstrap'].values())
    assert bool(d['selected'])==any(x['passed'] for x in d['bootstrap_checks'])
    bundle=ROOT/'artifacts/next_loop'/meta['filename'];assert hashlib.sha256(bundle.read_bytes()).hexdigest()==meta['sha256']
    with tarfile.open(bundle) as archive:
        names=archive.getnames();assert not any('sealed' in n or 'test' in n or 'f5_' in n for n in names)
        data=archive.extractfile(stage.lower()+'_predictions.npz').read()
        with np.load(io.BytesIO(data),allow_pickle=False) as saved:
            assert len(saved.files)==len(d['rows'])*4
            for row in d['rows']:
                for f,fold in enumerate(row['fold_metrics'],1):
                    p=saved[row['model']+f'_cv{f}'];assert p.shape==(fold['n'],) and np.isfinite(p).all()
    report={'stage':stage,'conditions':len(d['rows']),'aggregate_metrics_recomputed':True,'all_four_gates_recomputed':True,'fold_stability_recomputed':True,'bootstrap_adoption_order_checked':True,'original_row_hashes_match':True,'prediction_bank_shapes_and_finiteness_checked':True,'artifact_sha256':meta['sha256'],'strict_gate_count':sum(r['gate']['all_four'] for r in d['rows']),'fold_stability_count':len(expected),'bootstrap_checks':len(d['bootstrap_checks']),'selected':d['selected']['model'] if d['selected'] else None,'original_final_evaluations':0,'scope_limit':'Arithmetic and identity audit from fold summaries and saved predictions; not an independent rerun of every trained model or ground-truth metric.'}
    (out/(stage.lower()+'_verification.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');verify(p.parse_args().stage)
