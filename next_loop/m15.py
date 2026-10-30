"""Training-only frozen forecast routing; no new base model is fitted."""
import json,platform,tarfile,time
import numpy as np,scipy,sklearn
from matched.runner import DATA,EXPORT,ROOT,save_json,sha
from next_loop.m12 import load_blocks,samples
from next_loop.m14 import controls
from next_loop.regional_stats import metrics,aggregate,gate,stability,bootstrap
from next_loop.expert_gate import panel,predict

def load_experts(datasets):
    reference=json.loads((ROOT/'next_loop/m15_reference.json').read_text())
    source=DATA/'m14_training_artifacts.tar.gz';assert sha(source.read_bytes())=='2e9f0b4230c10c661d36633b44f80f4afaf742a458dd9a533c6eacb5b412ee74'
    target=DATA/'m14_anchor_predictions.npz'
    with tarfile.open(source) as archive:target.write_bytes(archive.extractfile('m14_predictions.npz').read())
    experts={};rows=[]
    with np.load(target,allow_pickle=False) as bank:
        for role,expected in reference['experts'].items():
            pp={f:bank[expected['model']+f'_cv{f}'].copy() for f in range(1,5)}
            fm=[]
            for f in range(1,5):
                assert pp[f].shape==(len(datasets[f][1]['y']),) and np.isfinite(pp[f]).all()
                m=metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']);fm.append(m)
                for k in ['rmse','r','chi2','hse_ts']:assert abs(m[k]-expected['fold_metrics'][f-1][k])<1e-9
            row={'model':'fixed_anchor_'+role,'parent_model':expected['model'],'fold_metrics':fm,'metrics':aggregate(fm)}
            for k in ['rmse','r','chi2','hse_ts']:assert abs(row['metrics'][k]-expected['metrics'][k])<1e-9
            rows.append(row);experts[role]=pp
    return experts,rows,reference

def freeze(protocol,audits,control_rows,expert_rows):
    assert not (DATA/'sealed').exists() and not (DATA/'m15.started.json').exists()
    assert all(sha((ROOT/name).read_bytes())==digest for name,digest in protocol['frozen_sources'].items())
    members=json.loads((DATA/'members.json').read_text())
    expected={m['member']:m for m in json.loads((ROOT/'matched/source_manifest.json').read_text())['members']}
    assert len(members)==157
    for m in members:
        assert m['sha256']==expected[m['member']]['release_member_sha256']
        if m['fold']==5:assert not m['retained'] and not m['parsed_for_development']
    record={'stage':'M15','scope':'Training-only Folds1-4 forecast routing','protocol_sha256':sha((ROOT/'next_loop/m15_protocol.json').read_bytes()),
            'source_hashes':protocol['frozen_sources'],'members_sha256':sha((DATA/'members.json').read_bytes()),
            'development_input_hashes':{str(p.relative_to(DATA)):sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))},
            'control_artifact_hashes':{name:sha((DATA/name).read_bytes()) for name in ['m11_training_artifacts.tar.gz','m13_training_artifacts.tar.gz','m14_training_artifacts.tar.gz']},
            'row_audits':audits,'controls':control_rows,'component_ablations':expert_rows,'panel':panel(),
            'original_final_evaluations':0,'new_base_fits':0,'fold5_available':False,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    assert len(record['development_input_hashes'])==108
    with (DATA/'m15.started.json').open('x') as handle:json.dump(record,handle,indent=2,allow_nan=False)
    (EXPORT/'m15_freeze.json').write_bytes((DATA/'m15.started.json').read_bytes())
    save_json(EXPORT/'m15_preflight.json',{'passed':True,'freeze_sha256':sha((DATA/'m15.started.json').read_bytes()),'four_controls_and_two_experts_match_1e_9':True,'source_hashes_match':True,'row_audits':audits,'controls':control_rows,'component_ablations':expert_rows,'fold5_available':False,'original_final_evaluations':0})
    return record

def run():
    started=time.time();blocks=load_blocks();datasets={f:samples(blocks,f) for f in range(1,5)}
    control_rows,cp,audits=controls(datasets);experts,ablation_rows,reference=load_experts(datasets)
    assert audits==reference['row_audits']
    protocol=json.loads((ROOT/'next_loop/m15_protocol.json').read_text());frozen=freeze(protocol,audits,control_rows,ablation_rows)
    print('M15 preflight passed: four controls and both fixed experts match original training rows; sources locked before routing',flush=True)
    all_predictions={};rows=[]
    for entry in frozen['panel']:
        pp={f:predict({'A':experts['A'][f],'B':experts['B'][f],'incumbent':cp['fixed_M13'][f]},entry['recipe']) for f in range(1,5)}
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={**entry,'fold_metrics':fm,'metrics':aggregate(fm)}
        row['gate']=gate(row['metrics'],[c['metrics'] for c in control_rows]);row['stability']=stability(row,control_rows)
        rows.append(row);all_predictions[row['model']]=pp
    assert len(rows)==104
    selected=None;checks=[]
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']),key=lambda r:r['metrics']['rmse']):
        intervals={name:bootstrap(all_predictions[row['model']],pp,datasets=datasets) for name,pp in cp.items()}
        passed=all(v['hse_ts']['delta_95'] is not None and v['hse_ts']['delta_95'][0]>0 for v in intervals.values());checks.append({'model':row['model'],'bootstrap':intervals,'passed':passed})
        if passed:selected=row;break
    assert not (DATA/'sealed').exists() and all(sha((ROOT/name).read_bytes())==digest for name,digest in frozen['source_hashes'].items())
    np.savez_compressed(DATA/'m15_predictions.npz',**{name+f'_cv{f}':p for name,pp in all_predictions.items() for f,p in pp.items()})
    recipes={'stage':'M15','panel':frozen['panel'],'experts':reference['experts'],'parent_archive_hashes':frozen['control_artifact_hashes'],
             'inference':'Evaluate original parent models on the same four images, form fixed M14 anchor A/B and M13 forecasts, then apply next_loop.expert_gate.predict. This archive contains gate recipes and training predictions; parent fitted models are in the referenced M14/M13/M11R archives.'}
    save_json(DATA/'m15_gate_recipes.json',recipes)
    result={'stage':'M15','scope':'Training-only original Folds1-4 forecast routing','rows':rows,'controls':control_rows,'component_ablations':ablation_rows,
            'selected':selected,'gate_passed':selected is not None,'bootstrap_checks':checks,'row_audits':audits,
            'new_base_fits':0,'original_final_evaluations':0,'fold5_available':False,'freeze_sha256':sha((DATA/'m15.started.json').read_bytes()),
            'source_hashes':frozen['source_hashes'],'versions':frozen['runtime'],'seconds':time.time()-started,
            'limitations':'Adaptive reused training bank and fixed expert choice from M14 validation; out-of-fold base forecasts do not remove higher-level selection bias. Conditional intervals not selection adjusted. No original historical scoring or F3 feedback.'}
    save_json(EXPORT/'m15_results.json',result)
    path=EXPORT/'m15_training_artifacts.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        for name in ['m15_predictions.npz','m15_gate_recipes.json']:archive.add(DATA/name,arcname=name)
    save_json(EXPORT/'m15_artifacts.json',{'filename':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'conditions':104,'new_base_models':0,'requires_parent_models':True,'scope':'Gate recipes and training validation predictions only; parent models remain in referenced M14/M13/M11R archives; no original holdout data'})
    print(json.dumps({'M15_complete':True,'conditions':len(rows),'numerical_all_four':sum(r['gate']['all_four'] for r in rows),'qualified':selected is not None,'selected':selected,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':run()
