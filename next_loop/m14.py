"""Registered training-only digital topology heads against four frozen controls."""
import json,platform,tarfile,time
import joblib,numpy as np,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from matched.runner import DATA,EXPORT,ROOT,save_json,sha
from next_loop.m12 import load_blocks,samples,load_controls
from next_loop.digital_topology import feature_families
from next_loop.regional_features import temporal
from next_loop.target_weights import target_weights
from next_loop.regional_stats import metrics,aggregate,gate,stability,bootstrap

def controls(datasets):
    rows,predictions,audits=load_controls(datasets)
    reference=json.loads((ROOT/'next_loop/m14_reference.json').read_text())
    archive_path=DATA/'m13_training_artifacts.tar.gz'
    assert sha(archive_path.read_bytes())=='64a73923dbc8da0f7b05cc844227edcaa52c102bd3285df2afdde7c6999e90eb'
    with tarfile.open(archive_path) as archive:
        target=DATA/'m13_control_predictions.npz'
        target.write_bytes(archive.extractfile('m13_predictions.npz').read())
    with np.load(target,allow_pickle=False) as data:
        pp={f:data[reference['selected']['model']+f'_cv{f}'].copy() for f in range(1,5)}
    fm=[]
    for f in range(1,5):
        assert pp[f].shape==(len(datasets[f][1]['y']),) and np.isfinite(pp[f]).all()
        m=metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']);fm.append(m)
        for key in ['rmse','r','chi2','hse_ts']:assert abs(m[key]-reference['selected']['fold_metrics'][f-1][key])<1e-9
    row={'model':'fixed_M13','metrics':aggregate(fm),'fold_metrics':fm}
    for key in ['rmse','r','chi2','hse_ts']:assert abs(row['metrics'][key]-reference['selected']['metrics'][key])<1e-9
    assert audits==reference['row_audits']
    rows.append(row);predictions['fixed_M13']=pp
    assert len(rows)==4
    return rows,predictions,audits

def freeze(protocol,audits,control_rows):
    assert not (DATA/'sealed').exists() and not (DATA/'m14.started.json').exists()
    assert all(sha((ROOT/p).read_bytes())==h for p,h in protocol['frozen_sources'].items())
    members=json.loads((DATA/'members.json').read_text())
    source={m['member']:m for m in json.loads((ROOT/'matched/source_manifest.json').read_text())['members']}
    assert len(members)==157
    for m in members:
        assert m['sha256']==source[m['member']]['release_member_sha256']
        if m['fold']==5:assert not m['retained'] and not m['parsed_for_development']
    record={'stage':'M14','scope':'Training-side Folds1-4 only','protocol_sha256':sha((ROOT/'next_loop/m14_protocol.json').read_bytes()),
            'source_hashes':protocol['frozen_sources'],'members_sha256':sha((DATA/'members.json').read_bytes()),
            'development_input_hashes':{str(p.relative_to(DATA)):sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))},
            'control_artifact_hashes':{name:sha((DATA/name).read_bytes()) for name in ['m11_training_artifacts.tar.gz','m13_training_artifacts.tar.gz']},
            'row_audits':audits,'controls':control_rows,'original_final_evaluations':0,'fold5_available':False,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    assert len(record['development_input_hashes'])==108
    with (DATA/'m14.started.json').open('x') as handle:json.dump(record,handle,indent=2,allow_nan=False)
    (EXPORT/'m14_freeze.json').write_bytes((DATA/'m14.started.json').read_bytes())
    save_json(EXPORT/'m14_preflight.json',{'passed':True,'freeze_sha256':sha((DATA/'m14.started.json').read_bytes()),'controls_match_1e_9':True,'source_hashes_match':True,'row_audits':audits,'controls':control_rows,'fold5_available':False,'original_final_evaluations':0})
    return record

def run():
    started=time.time();bs=load_blocks();datasets={f:samples(bs,f) for f in range(1,5)}
    control_rows,cp,audits=controls(datasets);protocol=json.loads((ROOT/'next_loop/m14_protocol.json').read_text())
    frozen=freeze(protocol,audits,control_rows)
    print('M14 preflight passed: four frozen controls and all original training rows match; Fold5 absent',flush=True)
    features={name:[] for name in ['betti','morphology','regional_morphology']}
    for i,b in enumerate(bs):
        for name,values in feature_families(b['images']).items():features[name].append(values)
        if (i+1)%20==0:print(json.dumps({'M14_topology_blocks':i+1,'seconds':time.time()-started}),flush=True)
    features={name:np.stack(values) for name,values in features.items()}
    all_predictions={};recipes={};models={};weight_audits=[]
    for family,frames in features.items():
        for view in ['flatten','trend']:
            for f,(tr,va) in datasets.items():
                four=[np.stack([frames[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])]) for a in [tr,va]]
                x,v=[temporal(z,view,'linear') for z in four]
                scaler=StandardScaler().fit(x);scaler.scale_=np.maximum(scaler.scale_,.01);x=scaler.transform(x);v=scaler.transform(v)
                positive=tr['sigma'][tr['sigma']>0];floor=float(np.quantile(positive,.05));med=float(np.median(positive));mu=float(tr['y'].mean())
                for precision in [0,4]:
                    base=1+precision*(med/np.maximum(tr['sigma'],floor))**2;base/=base.mean()
                    for method in ['unit','fast2']:
                        weights=base.copy()
                        if method!='unit':weights*=target_weights(tr['y'],method);weights/=weights.mean()
                        weight_audits.append({'fold':f,'family':family,'view':view,'precision_lambda':precision,'method':method,'min':float(weights.min()),'max':float(weights.max()),'mean':float(weights.mean())})
                        for alpha in [10,100,1000]:
                            ridge=Ridge(alpha=alpha,solver='cholesky').fit(x,tr['y'],sample_weight=weights)
                            name=f'M14_{family}_{view}_w{precision}_target{method}_a{alpha}'
                            raw=mu+1.2*(ridge.predict(v)-mu)-5
                            models[name+f'_cv{f}']={'ridge':ridge,'scaler':scaler,'target_mean':mu,'family':family,'view':view,'precision_lambda':precision,'target_weight':method,'ridge_alpha':alpha,'sigma_floor':floor,'sigma_median':med}
                            for share in [.125,.25,.5,1]:
                                key=name+f'_share{share}';all_predictions.setdefault(key,{})[f]=share*raw+(1-share)*cp['fixed_M13'][f]
                                recipes[key]={'family':family,'view':view,'precision_lambda':precision,'target_weight':method,'ridge_alpha':alpha,'new_head_share':share,'gain':1.2,'bias':-5,'prior':'fixed M13'}
            print(json.dumps({'M14_group_complete':{'family':family,'view':view},'seconds':time.time()-started}),flush=True)
    rows=[]
    for name,pp in all_predictions.items():
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={'model':name,'recipe':recipes[name],'fold_metrics':fm,'metrics':aggregate(fm)}
        row['gate']=gate(row['metrics'],[c['metrics'] for c in control_rows]);row['stability']=stability(row,control_rows);rows.append(row)
    assert len(rows)==288 and len(models)==288
    selected=None;checks=[]
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']),key=lambda r:r['metrics']['rmse']):
        intervals={name:bootstrap(all_predictions[row['model']],pp,datasets=datasets) for name,pp in cp.items()}
        passed=all(v['hse_ts']['delta_95'] is not None and v['hse_ts']['delta_95'][0]>0 for v in intervals.values());checks.append({'model':row['model'],'bootstrap':intervals,'passed':passed})
        if passed:selected=row;break
    assert not (DATA/'sealed').exists()
    assert all(sha((ROOT/p).read_bytes())==h for p,h in frozen['source_hashes'].items())
    np.savez_compressed(DATA/'m14_predictions.npz',**{name+f'_cv{f}':p for name,pp in all_predictions.items() for f,p in pp.items()})
    joblib.dump(models,DATA/'m14_models.joblib',compress=3)
    result={'stage':'M14','scope':'Original Folds1-4 training-side validation only','rows':rows,'controls':control_rows,'selected':selected,'gate_passed':selected is not None,
            'bootstrap_checks':checks,'row_audits':audits,'weight_audits':weight_audits,'original_final_evaluations':0,'fold5_available':False,
            'freeze_sha256':sha((DATA/'m14.started.json').read_bytes()),'source_hashes':frozen['source_hashes'],'versions':frozen['runtime'],'seconds':time.time()-started,
            'limitations':'Adaptive reused training bank and fixed prior control forecasts; conditional intervals not selection adjusted. Sampled digital topology, not full persistent homology or physical source attribution. No original historical scoring.'}
    save_json(EXPORT/'m14_results.json',result)
    archive_path=EXPORT/'m14_training_artifacts.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for name in ['m14_predictions.npz','m14_models.joblib']:archive.add(DATA/name,arcname=name)
    save_json(EXPORT/'m14_artifacts.json',{'filename':archive_path.name,'sha256':sha(archive_path.read_bytes()),'bytes':archive_path.stat().st_size,'models':288,'candidate_recipes':72,'scope':'Training-side fitted heads and predictions only; no original holdout data'})
    print(json.dumps({'M14_complete':True,'conditions':len(rows),'numerical_all_four':sum(r['gate']['all_four'] for r in rows),'qualified':selected is not None,'selected':selected,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':run()
