"""Independent aggregate, provenance and checkpoint-container audit; no raw holdouts."""
import collections,hashlib,json,tarfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/next_loop'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    result=json.loads((OUT/'f3_results.json').read_text());audit=json.loads((OUT/'f3_audit.json').read_text())
    freeze=json.loads((OUT/'f3_freeze.json').read_text());identity=json.loads((OUT/'f3_freeze_identity.json').read_text())
    assert digest(OUT/'f3_freeze.json')==identity['sha256']==result['freeze_sha256']
    for name,h in freeze['source_hashes'].items():assert digest(ROOT/name)==h,name
    assert identity['fold5_available'] is False and identity['final_evaluations']==0
    assert len(result['rows'])==10 and [r['model'] for r in result['rows']]==[p['model'] for p in freeze['panel']]
    counts=collections.Counter(m['fold'] for m in json.loads((ROOT/'matched/source_manifest.json').read_text())['members'])
    assert sum(counts.values())==157
    for fold in audit['folds']:
        held=fold['fold'];assert fold['fit_folds']==[f for f in range(1,6) if f!=held]
        assert fold['test_rows']==14*counts[held] and fold['fit_rows']==13*(157-counts[held])
        assert fold['reference']['rows']==fold['test_rows'] and fold['reference']['labels_aligned']
        assert not fold['test_labels_passed_to_fit_or_predict']
        assert not set(fold['predict_input_keys'])&{'y','sigma','history_y','future_y'}
    assert audit['total_test_windows']==2198 and audit['final_evaluation_passes']==1
    assert result['final_evaluation_passes']==1 and result['post_test_model_changes']==0
    assert audit['all_fits_completed_before_score_export'] and audit['fold5_parsed_only_after_full_freeze']
    summary={}
    def avg(v):return None if any(x is None for x in v) else float(sum(v)/len(v))
    for row in result['rows']:
        fm=row['fold_metrics'];assert len(fm)==5 and sum(m['n'] for m in fm)==2198
        for f,m in enumerate(fm,1):
            assert m['n']==14*counts[f]
            c=m['hse_counts_original'];den=c['tp']+c['fp']+c['fn'];expected=c['tp']/den if den else None
            assert m['hse_ts']==expected
        rs=[m['r'] for m in fm]
        expected={'mean_rmse':avg([m['rmse'] for m in fm]),'mean_chi2':avg([m['chi2'] for m in fm]),'mean_hse_ts':avg([m['hse_ts'] for m in fm]),
                  'fisher_r':None if any(x is None for x in rs) else float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999999999,.999999999)))))}
        for key,value in expected.items():
            actual=row['summary'][key]
            assert (value is None and actual is None) or (value is not None and actual is not None and abs(value-actual)<1e-10)
        summary[row['model']]=expected
    primary=summary['PRIMARY_F3']
    for ref,gate in result['gates'].items():
        expected={k:bool(primary[k] is not None and summary[ref][k] is not None and np.isfinite(primary[k]) and np.isfinite(summary[ref][k])
                         and sign*(primary[k]-summary[ref][k])>0) for k,sign in {'mean_rmse':-1,'fisher_r':1,'mean_chi2':-1,'mean_hse_ts':1}.items()}
        assert gate['checks']==expected and gate['all_four']==all(expected.values())
        interval=result['primary_uncertainty'][ref];assert interval['reference']==ref and interval['draws']==2000
        for item in interval['metrics'].values():assert 0<=item['defined_draws']<=2000 and (item['delta_95'] is None or item['delta_95'][0]<=item['delta_95'][1])
    meta=json.loads((OUT/'f3_artifacts.json').read_text());path=ROOT/'artifacts/next_loop'/meta['file']
    assert digest(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
    checkpoint=json.loads((OUT/'f3_checkpoint_manifest.json').read_text());manifest={m['file']:m for m in checkpoint['artifacts']}
    verified=0
    with tarfile.open(path) as archive:
        for member in archive:
            assert member.name.startswith(('models','freeze.json')) and '..' not in Path(member.name).parts
            if not member.isfile():continue
            if member.name=='freeze.json':assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==identity['sha256'];continue
            name=member.name.removeprefix('models/');m=manifest[name]
            assert member.size==m['bytes'] and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==m['sha256'];verified+=1
    assert verified==len(manifest)==checkpoint['count']==audit['checkpoint_count']
    proof={'passed':True,'freeze_sha256':identity['sha256'],'models':10,'test_windows':2198,'source_files_verified':len(freeze['source_hashes']),
           'checkpoints_verified':verified,'gates_independently_recomputed':result['gates'],
           'scope':'Aggregate arithmetic, gates, original fold counts, source identities and checkpoint-container hashes. No independent model refit or raw-label metric rerun; no individual historical data read.'}
    (OUT/'f3_verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

if __name__=='__main__':main()
