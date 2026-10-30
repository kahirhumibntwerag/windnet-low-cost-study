"""Registered target-weighting refinement; reuse only verified training caches."""
import json,platform,tarfile,time
import joblib,numpy as np,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from matched.runner import DATA,EXPORT,ROOT,save_json,sha
from next_loop.m12 import load_blocks,samples,load_controls
from next_loop.regional_features import regions,temporal
from next_loop.regional_stats import metrics,aggregate,gate,stability,bootstrap
from next_loop.target_weights import target_weights

def run():
    start=time.time();bs=load_blocks();datasets={f:samples(bs,f) for f in range(1,5)}
    controls,cp,audits=load_controls(datasets)
    protocol=json.loads((ROOT/'next_loop/m13_protocol.json').read_text())
    assert all(sha((ROOT/p.replace('\\','/')).read_bytes())==h for p,h in protocol['frozen_sources'].items())
    old=json.loads((EXPORT/'m12_results.json').read_text());oldrows={r['model']:r for r in old['rows']}
    oldpred=np.load(DATA/'m12_predictions.npz',allow_pickle=False)
    all_predictions={};recipes={};models={};weight_audits=[]
    for grid in [(4,3),(6,6)]:
        features=np.stack([regions(b['images'],grid) for b in bs])
        for view in ['flatten','trend']:
            frames={f:[np.stack([features[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])]) for a in datasets[f]] for f in range(1,5)}
            for encoding in ['linear','squares']:
                for f,(tr,va) in datasets.items():
                    x,v=[temporal(z,view,encoding) for z in frames[f]]
                    scaler=StandardScaler().fit(x);scaler.scale_=np.maximum(scaler.scale_,.01);x=scaler.transform(x);v=scaler.transform(v)
                    sig=tr['sigma'][tr['sigma']>0];floor=float(np.quantile(sig,.05));med=float(np.median(sig));mu=float(tr['y'].mean())
                    for precision in [0,4]:
                        base=1+precision*(med/np.maximum(tr['sigma'],floor))**2;base/=base.mean()
                        for method in ['unit','fast1.5','fast2','density']:
                            weights=base if method=='unit' else base*target_weights(tr['y'],method)
                            if method!='unit':weights/=weights.mean()
                            weight_audits.append({'fold':f,'grid':grid,'view':view,'encoding':encoding,'precision_lambda':precision,'method':method,'min':float(weights.min()),'max':float(weights.max()),'mean':float(weights.mean()),'fit_fast_fraction':float((tr['y']>=500).mean())})
                            for alpha in [10,100,1000]:
                                model=Ridge(alpha=alpha,solver='cholesky').fit(x,tr['y'],sample_weight=weights)
                                name=f'M13_g{grid[0]}x{grid[1]}_{view}_{encoding}_w{precision}_a{alpha}_target{method}'
                                raw=mu+1.2*(model.predict(v)-mu)-5
                                models[name+f'_cv{f}']={'ridge':model,'scaler':scaler,'target_mean':mu,'grid':grid,'view':view,'encoding':encoding,'precision_lambda':precision,'target_method':method,'sigma_floor':floor,'sigma_median':med}
                                for share in [.125,.25,.5,.75,1]:
                                    key=name+f'_share{share}';p=share*raw+(1-share)*cp['fixed_M10'][f]
                                    all_predictions.setdefault(key,{})[f]=p
                                    recipes[key]={'grid':list(grid),'view':view,'encoding':encoding,'precision_lambda':precision,'ridge_alpha':alpha,'target_weight':method,'new_head_share':share,'gain':1.2,'bias':-5,'prior':'fixed M10'}
                print(json.dumps({'M13_group_complete':{'grid':grid,'view':view,'encoding':encoding},'seconds':time.time()-start}),flush=True)
    rows=[];overlap=0;max_prediction_difference=0.
    for name,pp in all_predictions.items():
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={'model':name,'recipe':recipes[name],'fold_metrics':fm,'metrics':aggregate(fm)}
        if recipes[name]['target_weight']=='unit' and recipes[name]['new_head_share']!=.75:
            oldname=name.replace('M13_','M12_').replace('_targetunit','');expected=oldrows[oldname]
            for f in range(1,5):
                diff=float(np.max(np.abs(pp[f]-oldpred[oldname+f'_cv{f}'])));max_prediction_difference=max(max_prediction_difference,diff);assert diff<=1e-7,(oldname,diff)
                for k in ['rmse','r','chi2','hse_ts']:assert abs(fm[f-1][k]-expected['fold_metrics'][f-1][k])<=1e-7
            overlap+=1
        row['gate']=gate(row['metrics'],[c['metrics'] for c in controls]);row['stability']=stability(row,controls);rows.append(row)
    assert len(rows)==960 and len(models)==768 and overlap==192
    np.savez_compressed(DATA/'m13_predictions.npz',**{name+f'_cv{f}':p for name,pp in all_predictions.items() for f,p in pp.items()});joblib.dump(models,DATA/'m13_models.joblib',compress=3)
    selected=None;checks=[]
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']),key=lambda r:r['metrics']['rmse']):
        intervals={name:bootstrap(all_predictions[row['model']],p,datasets=datasets) for name,p in cp.items()}
        passed=all(x['hse_ts']['delta_95'][0]>0 for x in intervals.values());checks.append({'model':row['model'],'bootstrap':intervals,'passed':passed})
        if passed:selected=row;break
    result={'stage':'M13','scope':'Original Folds1-4 training-bank validation only','rows':rows,'controls':controls,'selected':selected,'gate_passed':selected is not None,'bootstrap_checks':checks,'row_audits':audits,'weight_audits':weight_audits,'M12_unit_overlap_count':overlap,'M12_unit_max_prediction_difference':max_prediction_difference,'original_final_evaluations':0,'fold5_available':False,'seconds':time.time()-start,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'source_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for pkg in ['matched','event_loop','next_loop'] for p in (ROOT/pkg).glob('*') if p.suffix in ['.py','.json']},'limitations':'Adaptive reused training bank, conditional bootstrap not selection adjusted. Density weighting is an LDS-inspired adaptation, not BalancedMSE. No original historical scoring or physical source-region attribution.'}
    save_json(EXPORT/'m13_results.json',result)
    target=EXPORT/'m13_training_artifacts.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for name in ['m13_models.joblib','m13_predictions.npz']:archive.add(DATA/name,arcname=name)
    save_json(EXPORT/'m13_artifacts.json',{'filename':target.name,'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'models':768,'candidate_recipes':192,'scope':'Training-side models and predictions only; no original holdout data'})
    print(json.dumps({'M13_complete':True,'conditions':960,'all_four_passes':sum(r['gate']['all_four'] for r in rows),'selected':selected,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':run()
