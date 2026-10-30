"""Registered regional ridge ablations with immutable matched training controls."""
import hashlib,json,platform,tarfile,time
from pathlib import Path
import joblib,numpy as np,scipy,sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from matched.runner import DATA,EXPORT,ROOT,save_json,sha
from next_loop.regional_features import regions,temporal
from next_loop.regional_stats import metrics,rowhash,aggregate,gate,stability,bootstrap

def acquire():
    # The historical acquisition routine verifies every release member and discards
    # Fold5 bytes without parsing or retaining them. No encoder is run here.
    from event_loop.runner import acquire as original_acquire
    original_acquire()

def load_blocks():
    bs=[]
    for m in json.loads((DATA/'members.json').read_text()):
        if m['fold']==5:
            assert not m['retained'] and not m['parsed_for_development'];continue
        d=np.load(DATA/'development'/m['file'].replace('.npy','.npz'))
        bs.append({'fold':m['fold'],'name':m['file'],'images':d['images'],'output':d['output']})
    assert len(bs)==108 and not (DATA/'sealed').exists()
    return bs

def samples(bs,held):
    def pack(training):
        a={k:[] for k in ['y','sigma','row','fold','block','offset']}
        for i,b in enumerate(bs):
            if (b['fold']!=held)!=training:continue
            for off in range(6,19 if training else 20):
                for key,value in [('y',b['output'][off,11]),('sigma',b['output'][off,14]),('row',b['name']+':'+str(off)),('fold',b['fold']),('block',i),('offset',off)]:a[key].append(value)
        return {k:np.array(v) for k,v in a.items()}
    return pack(True),pack(False)

def load_controls(datasets):
    path=DATA/'m11_training_artifacts.tar.gz'
    assert sha(path.read_bytes())=='e03668f2034eee2321234b83858f11caf40c79d78acae7df0bb64c6d61ae085b'
    names={'bank_windnet_port':'bank_windnet_port.npz','bank_prior_M6':'bank_prior_M6.npz','fixed_M10':'M11_crossfit_strength_0.npz'}
    predictions={}
    with tarfile.open(path) as archive:
        for name,file in names.items():
            target=DATA/file;target.write_bytes(archive.extractfile(file).read())
            with np.load(target) as d:predictions[name]={f:d[f'cv{f}'].copy() for f in range(1,5)}
    reference=json.loads((ROOT/'next_loop/m12_reference.json').read_text())
    audits=[]
    for f,(tr,va) in datasets.items():
        old=reference['row_audits'][f-1]
        assert rowhash(tr)==old['fit_row_sha256'] and rowhash(va)==old['validation_row_sha256']
        audits.append({'held_training_fold':f,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fit_rows':len(tr['y']),'validation_rows':len(va['y'])})
    controls=[]
    for name,pp in predictions.items():
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={'model':name,'metrics':aggregate(fm),'fold_metrics':fm}
        expected=next(c for c in reference['controls'] if c['model']==name)
        for key in ['rmse','r','chi2','hse_ts']:assert abs(row['metrics'][key]-expected['metrics'][key])<1e-9,(name,key)
        controls.append(row)
    return controls,predictions,audits

def run():
    started=time.time();bs=load_blocks();datasets={f:samples(bs,f) for f in range(1,5)}
    controls,cp,audits=load_controls(datasets)
    protocol=json.loads((ROOT/'next_loop/m12_protocol.json').read_text())
    assert all(sha((ROOT/p).read_bytes())==h for p,h in protocol['frozen_sources'].items())
    save_json(EXPORT/'m12_preflight.json',{'row_audits':audits,'controls':controls,'control_metrics_match_1e_9':True,'original_final_evaluations':0,'fold5_available':False,'frozen_sources_match':True})
    print('M12 preflight passed: all original rows, controls and frozen sources match',flush=True)
    all_predictions={};recipes={};models={}
    for grid in [(1,1),(4,3),(6,6)]:
        features=np.stack([regions(b['images'],grid) for b in bs])
        for view in ['flatten','trend']:
            frames={f:[np.stack([features[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])]) for a in datasets[f]] for f in range(1,5)}
            for encoding in ['linear','squares']:
                for f,(tr,va) in datasets.items():
                    x,v=[temporal(z,view,encoding) for z in frames[f]]
                    scaler=StandardScaler().fit(x);scaler.scale_=np.maximum(scaler.scale_,.01)
                    x=scaler.transform(x);v=scaler.transform(v)
                    positive=tr['sigma'][tr['sigma']>0];floor=float(np.quantile(positive,.05));med=float(np.median(positive));mu=float(tr['y'].mean())
                    for weight in [0,4]:
                        weights=1+weight*(med/np.maximum(tr['sigma'],floor))**2;weights/=weights.mean()
                        for alpha in [10,100,1000]:
                            model=Ridge(alpha=alpha,solver='cholesky').fit(x,tr['y'],sample_weight=weights)
                            name=f'M12_g{grid[0]}x{grid[1]}_{view}_{encoding}_w{weight}_a{alpha}'
                            raw=mu+1.2*(model.predict(v)-mu)-5
                            models[name+f'_cv{f}']={'ridge':model,'scaler':scaler,'target_mean':mu,'feature_grid':grid,'view':view,'encoding':encoding,'lambda':weight,'sigma_floor':floor,'sigma_median':med}
                            for share in [1,.125,.25,.5]:
                                key=name+f'_share{share}';p=share*raw+(1-share)*cp['fixed_M10'][f]
                                all_predictions.setdefault(key,{})[f]=p
                                recipes[key]={'grid':list(grid),'view':view,'encoding':encoding,'precision_lambda':weight,'ridge_alpha':alpha,'new_head_share':share,'gain':1.2,'bias':-5,'prior':'fixed M10'}
                print(json.dumps({'M12_group_complete':{'grid':grid,'view':view,'encoding':encoding},'seconds':time.time()-started}),flush=True)
    rows=[]
    for name,pp in all_predictions.items():
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={'model':name,'recipe':recipes[name],'fold_metrics':fm,'metrics':aggregate(fm)}
        row['gate']=gate(row['metrics'],[c['metrics'] for c in controls]);row['stability']=stability(row,controls);rows.append(row)
    assert len(rows)==288 and len(models)==288
    np.savez_compressed(DATA/'m12_predictions.npz',**{name+f'_cv{f}':p for name,pp in all_predictions.items() for f,p in pp.items()})
    joblib.dump(models,DATA/'m12_models.joblib',compress=3)
    selected=None;checks=[]
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']),key=lambda r:r['metrics']['rmse']):
        intervals={name:bootstrap(all_predictions[row['model']],p,datasets=datasets) for name,p in cp.items()}
        passed=all(x['hse_ts']['delta_95'][0]>0 for x in intervals.values());checks.append({'model':row['model'],'bootstrap':intervals,'passed':passed})
        if passed:selected=row;break
    result={'stage':'M12','scope':'Original Folds1-4 training-bank validation only','rows':rows,'controls':controls,'selected':selected,'gate_passed':selected is not None,'bootstrap_checks':checks,'row_audits':audits,'original_final_evaluations':0,'fold5_available':False,'seconds':time.time()-started,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'source_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for pkg in ['matched','event_loop','next_loop'] for p in (ROOT/pkg).glob('*') if p.suffix in ['.py','.json']},'limitations':'Adaptive reused training bank; fixed matched controls from prior training run; no fresh blind inference; no physical WCS/coronal-hole segmentation; conditional bootstrap not selection adjusted.'}
    save_json(EXPORT/'m12_results.json',result)
    target=EXPORT/'m12_training_artifacts.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for name in ['m12_models.joblib','m12_predictions.npz']:archive.add(DATA/name,arcname=name)
    save_json(EXPORT/'m12_artifacts.json',{'filename':target.name,'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'models':288,'candidate_recipes':72,'scope':'Training-side models and predictions only; no original holdout data'})
    print(json.dumps({'M12_complete':True,'conditions':len(rows),'all_four_passes':sum(r['gate']['all_four'] for r in rows),'selected':selected,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':
    acquire();run()
