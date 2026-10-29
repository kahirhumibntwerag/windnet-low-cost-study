"""Verify final aggregates and saved models without reading any test observation."""
import argparse,hashlib,io,json,pickle,platform,tarfile,warnings
from pathlib import Path
import numpy as np
import torch
import scipy,sklearn
from matched.science import WindNetPort

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/matched'));p.add_argument('--archive',type=Path,required=True);a=p.parse_args();r=a.results
    f=json.loads((r/'freeze.json').read_text());j=json.loads((r/'final_results.json').read_text());audit=json.loads((r/'final_audit.json').read_text());manifest=json.loads((r/'checkpoint_manifest.json').read_text());checks=0
    for name,digest in f['code_hashes'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest;checks+=1
    assert len(j['rows'])==len(f['panel'])+1
    for row in j['rows']:
        ms=row['fold_metrics'];s=row['summary'];assert [m['n'] for m in ms]==[518,350,364,280,686];assert s['n']==2198
        expected={'mean_rmse':np.mean([x['rmse'] for x in ms]),'fisher_r':np.tanh(np.mean(np.arctanh([x['r'] for x in ms]))),'mean_chi2':np.mean([x['chi2'] for x in ms]),'mean_mae':np.mean([x['mae'] for x in ms]),'mean_hse_ts':np.mean([x['hse_ts'] for x in ms])}
        for k,v in expected.items():assert np.isclose(s[k],v,rtol=0,atol=1e-10);checks+=1
    assert audit['total_test_windows']==2198 and audit['final_evaluation_passes']==1
    assert audit['test_labels_never_passed_to_fit_or_predict'];assert audit['all_model_row_identity_checks_passed'];checks+=4
    count=0;neural=0;tabular=0;serialization_warnings=[]
    expected_members={'models/'+item['file']:item for item in manifest['artifacts']}
    with tarfile.open(a.archive,'r|gz') as tar:
        for member in tar:
            if member.name not in expected_members:continue
            item=expected_members[member.name]
            raw=tar.extractfile(member).read();assert len(raw)==item['bytes'];assert hashlib.sha256(raw).hexdigest()==item['sha256'];count+=1
            if item['file'].endswith('.pt'):
                state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);model=WindNetPort();model.load_state_dict(state['state_dict']);model.eval()
                with torch.inference_mode():y=model(torch.zeros(2,4,832)).numpy()*state['yscale']+state['ymin']
                assert np.isfinite(y).all();neural+=1
            elif item['file'].endswith('.pkl'):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');b=pickle.loads(raw)
                serialization_warnings.extend(str(w.message) for w in caught)
                x=np.zeros((2,b['scaler'].n_features_in_));v=b['scaler'].transform(x)
                if 'model' in b:y=b['model'].predict(v)
                else:
                    from scipy.spatial.distance import cdist
                    y=np.exp(-b['gamma']*cdist(v,b['training_embeddings'],'sqeuclidean'))@b['dual']+b['ymean']
                assert np.isfinite(y).all();tabular+=1
    assert count==manifest['count']
    result={'aggregate_and_source_checks_passed':checks,'archive_members_hash_verified':count,'neural_checkpoints_loaded_and_synthetic_predictions_finite':neural,'tabular_checkpoints_loaded_and_synthetic_predictions_finite':tabular,'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'archive_bytes':a.archive.stat().st_size,'raw_test_observations_read':False,'test_predictions_recomputed':False,'independent_reproduction':False}
    result['verification_environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__}
    result['serialization_warnings']=sorted(set(serialization_warnings))
    result['verification_scope']='Hash checks and synthetic-only portability checks; version differences do not establish exact numerical reproduction.'
    (r/'artifact_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
