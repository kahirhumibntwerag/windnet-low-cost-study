"""M10: fit-only variance weights and preregistered training-side stability."""
import json,time
import numpy as np
from matched.runner import DATA,EXPORT,save_json
from matched.science import metrics,rowhash
from event_loop.science import split,aggregate,gate,identity,save_predictions
from event_loop.location import representation,temporal,ridge_path
from event_loop.mixtures import bootstrap

def stability(row,controls):
    leave=[]
    for drop in range(4):
        m=aggregate([v for i,v in enumerate(row['fold_metrics']) if i!=drop]);cs=[aggregate([v for i,v in enumerate(c['fold_metrics']) if i!=drop]) for c in controls]
        passed=m['rmse']<min(c['rmse'] for c in cs) and m['r']>max(c['r'] for c in cs) and m['chi2']<min(c['chi2'] for c in cs) and m['hse_ts'] is not None and m['hse_ts']>max(c['hse_ts'] for c in cs)
        leave.append({'dropped_training_fold':drop+1,'all_four_directional':bool(passed)})
    ratios=[m['rmse']/min(c['fold_metrics'][i]['rmse'] for c in controls) for i,m in enumerate(row['fold_metrics'])]
    return {'leave_one_fold_out':leave,'directional_passes':sum(x['all_four_directional'] for x in leave),'per_fold_rmse_control_ratios':ratios,'pre_bootstrap_passed':sum(x['all_four_directional'] for x in leave)>=3 and max(ratios)<=1.05}

def run():
    t=time.time();controls=json.loads((EXPORT/'event_controls.json').read_text())['rows'];cs=[c['metrics'] for c in controls];prior=np.load(DATA/'bank_prior_M6.npz');datasets={f:split(f)[:2] for f in range(1,5)};raw={};audits=[]
    for held,(tr,va) in datasets.items():
        d=np.load(DATA/f'location_cv{held}.npz')
        for kind,view in [('ratio','flatten'),('log','flatten'),('ratio','mean')]:
            frames=representation(d['local'],d['glob'],kind);xx=[np.stack([frames[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])]) for a in [tr,va]];x,v=[temporal(z,view) for z in xx]
            for weight in [.25,.5,1,2,4]:
                preds=ridge_path(x,v,tr['y'],tr['sigma'],weight,[3000,10000,30000])
                for alpha,p in preds.items():
                    key=f'{kind}_{view}_w{weight}_a{alpha}';raw.setdefault(key,{'recipe':{'family':'location_ridge','representation':kind,'view':view,'weight_lambda':weight,'alpha':alpha},'predictions':{}})['predictions'][held]=p
            print(json.dumps({'M10_fits_completed':{'held':held,'representation':kind,'view':view},'seconds':time.time()-t}),flush=True)
        audits.append({'held_training_fold':held,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va)})
    rows=[];all_predictions={};candidates={}
    def score(name,recipe,pp):
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)];m=aggregate(fm);r={'model':name,'recipe':recipe,'metrics':m,'fold_metrics':fm,'gate':gate(m,cs)};r['stability']=stability(r,controls);rows.append(r)
        if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']:candidates[name]=pp
        for f,p in pp.items():all_predictions[name+'_cv'+str(f)]=p
    for i,(name,r) in enumerate(raw.items()):
        score('M10_raw_'+name,r['recipe'],r['predictions'])
        for w in [.5,.625,.75,.875,1.]:
            for gain in [1.,1.1,1.2,1.3]:
                for bias in [-15,-10,-5,0]:
                    pp={f:float(tr['y'].mean())+gain*(w*r['predictions'][f]+(1-w)*prior['cv'+str(f)]-float(tr['y'].mean()))+bias for f,(tr,va) in datasets.items()}
                    score(f'M10_{name}_mix{w}_g{gain}_b{bias}',{'family':'affine_location_mix','location':r['recipe'],'location_weight':w,'prior':'M6','gain':gain,'bias':bias},pp)
        if i%5==0:print(json.dumps({'M10_grid_bases_complete':i+1,'strict_gate_passes':sum(x['gate']['all_four'] for x in rows),'stability_pre_passes':len(candidates),'seconds':time.time()-t}),flush=True)
    assert len(rows)==3645
    np.savez_compressed(DATA/'m10_all_predictions.npz',**all_predictions)
    m9=json.loads((EXPORT/'m9_results.json').read_text())['selected'];m9_stability=stability(m9,controls)
    control_predictions={c['model']:{f:np.load(DATA/(c['model']+'.npz'))['cv'+str(f)] for f in range(1,5)} for c in controls}
    selected=None;checks=[]
    for row in sorted((x for x in rows if x['model'] in candidates),key=lambda x:x['metrics']['rmse']):
        pp=candidates[row['model']];ci={name:bootstrap(pp,p) for name,p in control_predictions.items()};passed=all(c['hse_ts']['delta_95'][0]>0 for c in ci.values());checks.append({'model':row['model'],'bootstrap':ci,'passed':passed})
        if passed:selected=row;save_predictions('m10_selected',pp);break
    result={'stage':'M10','scope':'Four-fold training-bank CV only; Folds1-4','rows':rows,'controls':controls,'selected':selected,'gate_passed':selected is not None,'strict_four_gate_count':sum(x['gate']['all_four'] for x in rows),'pre_bootstrap_stability_count':len(candidates),'bootstrap_checks':checks,'M9_selected_stability':m9_stability,'stability_rule':'At least3/4 leave-one-validation-fold-out aggregates directionally improve all4 vsbothcontrols; each fold RMSE<=1.05*stronger control; conditional HSE bootstrap lower95 bound>0 vsboth. Check candidates in ascendingRMSE until firstpass; unchecked latercandidates are not declared failures.','bootstrap_limitations':'Conditional, not selection adjusted; blocks can overlap.','row_audits':audits,'identity':identity(),'seconds':time.time()-t,'original_final_evaluations':0,'fold5_available':False}
    save_json(EXPORT/'m10_results.json',result);print(json.dumps({'M10_complete':True,'strict_gate_count':result['strict_four_gate_count'],'pre_stability_count':len(candidates),'selected':selected,'M9_stability':m9_stability,'seconds':time.time()-t}),flush=True)
