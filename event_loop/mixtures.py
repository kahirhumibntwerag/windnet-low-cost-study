"""Preregistered mixture/calibration grid on training-bank validation only."""
import json,time
import numpy as np
from matched.runner import DATA,EXPORT,save_json
from matched.science import fit_tabular,metrics,rowhash,block_moments,moment_score
from event_loop.science import split,aggregate,gate,identity,save_predictions

def bootstrap(predictions,controls,datasets=None):
    all_pairs=[]
    for held in range(1,5):
        va=split(held)[1] if datasets is None else datasets[held][1]
        all_pairs.append((block_moments(va,predictions[held]),block_moments(va,controls[held])))
    rng=np.random.default_rng(20261029);draws={k:[] for k in ['rmse','r','chi2','hse_ts']}
    for _ in range(2000):
        left=[];right=[]
        for a,b in all_pairs:
            ix=rng.integers(0,len(a),len(a));left.append(moment_score(a[ix]));right.append(moment_score(b[ix]))
        for key in draws:
            if key=='r':
                x=np.tanh(np.mean(np.arctanh(np.clip([m[key] for m in left],-.999999,.999999))));y=np.tanh(np.mean(np.arctanh(np.clip([m[key] for m in right],-.999999,.999999))))
            else:x=np.mean([m[key] for m in left]);y=np.mean([m[key] for m in right])
            if np.isfinite(x-y):draws[key].append(float(x-y))
    return {k:{'delta_95':np.percentile(v,[2.5,97.5]).tolist(),'defined_draws':len(v)} for k,v in draws.items()}
def run():
    t=time.time();controls=json.loads((EXPORT/'event_controls.json').read_text());m7=json.loads((EXPORT/'m7_results.json').read_text());cs=[r['metrics'] for r in controls['rows']]
    port=np.load(DATA/'bank_windnet_port.npz');prior=np.load(DATA/'bank_prior_M6.npz');bas=[]
    bas.append(('port',{'family':'port'}, {f:port['cv'+str(f)] for f in range(1,5)}));bas.append(('prior',{'family':'prior'}, {f:prior['cv'+str(f)] for f in range(1,5)}))
    datasets={};anchors={};audits=[]
    for held in range(1,5):
        tr,va,_=split(held);datasets[held]=(tr,va);anchors[held]=float(tr['y'].mean());audits.append({'held_training_fold':held,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va)})
    for source in ['energy','spatial','combined']:
        recipe={'family':'ridge','view':'flatten','source':source,'alpha':10000};pp={}
        for held,(tr,va) in datasets.items():pp[held],_=fit_tabular(tr,{k:va[k] for k in ['energy','spatial']},recipe)
        bas.append(('ridge_'+source,recipe,pp))
    for row in m7['rows']:
        d=np.load(DATA/(row['model']+'.npz'));bas.append((row['model'],row['recipe'],{f:d['cv'+str(f)] for f in range(1,5)}))
    combined=next(x for x in bas if x[0]=='ridge_combined');portbase=bas[0]
    for left in [portbase,*[x for x in bas if x[0].startswith('aligned_')]]:
        for w in [.25,.5,.75]:
            pp={f:(1-w)*left[2][f]+w*combined[2][f] for f in range(1,5)}
            bas.append((left[0]+'_ridge_mix'+str(w),{'family':'blend','left':left[1],'right':combined[1],'right_weight':w},pp))
    assert len(bas)==24
    rows=[];selected=None;selected_predictions=None;all_predictions={}
    save_json(EXPORT/'m8_execution.json',{'identity':identity(),'row_audits':audits,'base_count':len(bas),'configurations':1176,'original_final_evaluations':0,'fold5_available':False})
    for bi,(name,recipe,raw) in enumerate(bas):
        for gain in [.75,.9,1,1.1,1.2,1.3,1.4]:
            for bias in [-30,-20,-10,0,10,20,30]:
                pp={f:anchors[f]+gain*(raw[f]-anchors[f])+bias for f in range(1,5)};fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)];m=aggregate(fm);name2=f'M8_{name}_g{gain}_b{bias}';r={'model':name2,'recipe':{'family':'affine','base':recipe,'gain':gain,'bias':bias},'metrics':m,'fold_metrics':fm,'gate':gate(m,cs)};rows.append(r)
                if r['gate']['all_four'] and (selected is None or m['rmse']<selected['metrics']['rmse']):selected=r;selected_predictions=pp
                for f,p in pp.items():all_predictions[name2+'_cv'+str(f)]=p
        print(json.dumps({'mixture_bases_complete':bi+1,'eligible_so_far':sum(r['gate']['all_four'] for r in rows),'seconds':time.time()-t}),flush=True)
    np.savez_compressed(DATA/'m8_all_predictions.npz',**all_predictions)
    uncertainty=None
    if selected:
        save_predictions('m8_selected',selected_predictions);uncertainty={c['model']:bootstrap(selected_predictions,{f:np.load(DATA/(c['model']+'.npz'))['cv'+str(f)] for f in range(1,5)}) for c in controls['rows']}
    result={'stage':'M8','scope':'Four-fold training-bank CV only; Folds1-4','rows':rows,'controls':controls['rows'],'selected':selected,'gate_passed':selected is not None,'bootstrap':uncertainty,'bootstrap_limitations':'Conditional whole-block bootstrap; not selection adjusted; blocks can overlap.','row_audits':audits,'identity':identity(),'seconds':time.time()-t,'original_final_evaluations':0,'fold5_available':False}
    save_json(EXPORT/'m8_results.json',result);print(json.dumps({'M8_complete':True,'selected':selected,'eligible':sum(r['gate']['all_four'] for r in rows),'seconds':time.time()-t}),flush=True)
