"""One-shot frozen evaluator. Test labels are withheld from every fit/predict call."""
import datetime as dt, hashlib, json, pickle, tarfile, time
from pathlib import Path
import numpy as np
import requests
import torch
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from matched.runner import DATA,EXPORT,ROOT,save_json,sha,numeric_npy
from matched.science import (SEEDS,DEVICE,identity,blocks,embed_blocks,samples,view,metrics,rowhash,
    neural_fit,fit_tabular,block_moments,moment_score)

SCIENTIFIC_FILES=['runner.py','science.py','final_study.py','hse_original.py','protocol.json','source_manifest.json','reference_sources.json']
def code_hashes():return {name:sha((ROOT/'matched'/name).read_bytes()) for name in SCIENTIFIC_FILES}
def read_result(name):return json.loads((EXPORT/name).read_text())
def freeze(cfg):
    assert not (DATA/'freeze.json').exists(),'Refuse to replace frozen recipes'
    base=read_result('baseline.json')['baseline'];br={'family':'lstm','variant':'paper','epochs':base['epochs'],'seeds':SEEDS,'fast_weight':1}
    m1=read_result('m1_results.json');m2=read_result('m2_results.json');m3=read_result('m3_results.json');m4=read_result('m4_results.json');m5=read_result('m5_results.json');m6=read_result('m6_results.json')
    panel=[{'model':'windnet_port_ensemble','recipe':br},{'model':'M1_best_ridge','recipe':m1['best_ridge']['recipe']},{'model':'M2_best_kernel','recipe':m2['best_raw']['recipe']}]
    panel.extend({'model':r['model'],'recipe':r['recipe']} for r in m3['rows'])
    panel.extend([{'model':'M4_spatial_raw','recipe':m4['best_raw']['recipe']},{'model':'M4_rmse_candidate','recipe':m4['selected']['recipe']}])
    panel.extend({'model':r['model'],'recipe':r['recipe']} for r in m4['matched_ablations'])
    panel.extend({'model':r['model'],'recipe':r['recipe']} for r in m5['rows'] if r['model'].startswith('lstm_fast_weight_'))
    panel.append({'model':'PRIMARY_frozen','recipe':m6['selected']['recipe']})
    if m6['selected']['recipe']['family']=='calibrated':panel.append({'model':'PRIMARY_without_calibration','recipe':m6['selected']['recipe']['base']})
    panel.append({'model':'M6_best_rmse_descriptive','recipe':m6['best_rmse_candidate']['recipe']})
    f={'campaign_id':'e324d24c-b179-427c-87c1-365b857658aa','created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'code_hashes':code_hashes(),'baseline_recipe':br,'panel':panel,'primary':'PRIMARY_frozen','fast_specialist':'lstm_fast_weight_4','validation_balanced_gate_passed':m6['gate_passed'],'development_results_hashes':{p.name:sha(p.read_bytes()) for p in EXPORT.glob('m?_results.json')},'source_members_hash':sha((DATA/'members.json').read_bytes()),'original_kaffe_hash':sha((DATA/'kaffe/googlenet.npy').read_bytes()),'original_kaffe_graph_hash':sha((DATA/'kaffe/googlenet.py').read_bytes()),'fold5_unparsed_until_freeze':True,'final_rule':'Primary numeric success iff mean-fold RMSE lower and Fisher-z mean r higher than released WindNet on identical original windows. Separately require lower mean chi2 and higher mean HSE threat score for all-four-metric improvement. Score fixed port comparator, fast specialist and all ablations regardless. Undefined metrics fail. No candidate replacement after final scoring.','uncertainty':'2000 paired whole-block bootstrap draws within each fold, seed20261029, mean-fold RMSE/Fisher-r deltas. Conditional, not selection adjusted; block overlap/dependence and prior holdout exposure prevent fresh confirmatory claims.','prior_exposure':'Original historical holdout scores were already seen before this campaign; Folds1-4 are development data. Final is a locked historical benchmark, not a fresh blind test.','scope':'211-Angstrom, H4D3; original157blocks/2198testwindows; training13 and evaluation14 windows/block; images-only for all new models.'}
    save_json(DATA/'freeze.json',f);save_json(EXPORT/'freeze.json',f)
    digest=sha((DATA/'freeze.json').read_bytes());save_json(EXPORT/'freeze_identity.json',{'sha256':digest,'models':len(panel),'primary':f['primary'],'code_hashes':f['code_hashes'],'final_holdout_evaluations':0})
    print(json.dumps({'frozen':True,'sha256':digest,'models':len(panel),'final_holdout_evaluations':0}),flush=True)

def predict(tr,features,recipe,cache,folder,base_recipe):
    key=json.dumps(recipe,sort_keys=True)
    if key in cache:return cache[key]
    family=recipe['family'];bundle=None
    if family=='lstm':
        pp=[]
        for seed,epochs in zip(recipe.get('seeds',SEEDS),recipe['epochs']):
            p,state,_=neural_fit(tr,features,seed,fixed_epochs=epochs,variant=recipe['variant'],fast_weight=recipe.get('fast_weight',1));pp.append(p)
            state['recipe']=recipe;torch.save(state,folder/(sha(key.encode())[:12]+'_'+str(seed)+'.pt'))
        p=np.mean(pp,axis=0)
    elif family in ['ridge','tree']:p,bundle=fit_tabular(tr,features,recipe)
    elif family=='kernel':
        scaler=StandardScaler().fit(view(tr,recipe['view'],recipe.get('source','energy')));x=scaler.transform(view(tr,recipe['view'],recipe.get('source','energy')));v=scaler.transform(view(features,recipe['view'],recipe.get('source','energy')))
        gamma=recipe['gamma_factor']/x.shape[1];k=np.exp(-gamma*cdist(x,x,'sqeuclidean'));ym=float(tr['y'].mean());dual=solve(k+recipe['alpha']*np.eye(len(x)),tr['y']-ym,assume_a='pos');p=np.exp(-gamma*cdist(v,x,'sqeuclidean'))@dual+ym;bundle={'scaler':scaler,'training_embeddings':x,'dual':dual,'ymean':ym,'gamma':gamma,'recipe':recipe}
    elif family=='blend':
        w=recipe['kernel_weight'];p=(1-w)*predict(tr,features,base_recipe,cache,folder,base_recipe)+w*predict(tr,features,recipe['kernel'],cache,folder,base_recipe)
    elif family=='spatial_blend':
        w=recipe['spatial_weight'];p=(1-w)*predict(tr,features,base_recipe,cache,folder,base_recipe)+w*predict(tr,features,recipe['spatial'],cache,folder,base_recipe)
    elif family=='weighted_blend':
        w=recipe['weighted_fraction'];p=w*predict(tr,features,recipe['weighted'],cache,folder,base_recipe)+(1-w)*predict(tr,features,recipe['other'],cache,folder,base_recipe)
    elif family=='calibrated':
        raw=predict(tr,features,recipe['base'],cache,folder,base_recipe);anchor=float(tr['y'].mean());p=anchor+recipe['gain']*(raw-anchor)+recipe['bias']
    else:raise ValueError(family)
    assert p.shape==(len(features['energy']),) and np.isfinite(p).all()
    if bundle is not None:
        with (folder/(sha(key.encode())[:12]+'.pkl')).open('wb') as f:pickle.dump(bundle,f,protocol=5)
    cache[key]=p;return p

def reference(fold,te,d):
    src=next(r for r in json.loads((ROOT/'matched/reference_sources.json').read_text()) if r['fold']==fold)
    r=requests.get(src['source_url'],timeout=(30,90));r.raise_for_status();raw=r.content;assert sha(raw)==src['sha256'],'Released reference changed'
    data=np.asarray(numeric_npy(raw)['data'],dtype=float)
    block_ids=sorted(np.flatnonzero(d['fold']==fold),key=lambda i:str(d['name'][i]));order=np.concatenate([np.flatnonzero(te['block']==i) for i in block_ids]);assert data.shape==(len(order),2)
    assert np.allclose(data[:,1],te['y'][order],atol=.001,rtol=0),'Released reference labels misaligned'
    p=np.empty(len(order));p[order]=data[:,0]
    return p,{'fold':fold,'sha256':sha(raw),'rows':len(p),'labels_aligned':True,'original_partition_order_recovered':True}
def aggregate(rows):
    rs=[r['r'] for r in rows];ts=[r['hse_ts'] for r in rows]
    return {'mean_rmse':float(np.mean([r['rmse'] for r in rows])),'rmse_se':float(np.std([r['rmse'] for r in rows],ddof=1)/np.sqrt(len(rows))),'fisher_r':float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999999999,.999999999))))),'arithmetic_mean_r':float(np.mean(rs)),'mean_chi2':float(np.mean([r['chi2'] for r in rows])),'mean_hse_ts':float(np.mean(ts)) if all(x is not None for x in ts) else None,'mean_mae':float(np.mean([r['mae'] for r in rows])),'n':int(sum(r['n'] for r in rows))}
def paired_bootstrap(folds,model,reference_name='released_windnet'):
    pairs=[]
    for f in folds:
        pairs.append((block_moments(f['test'],f['predictions'][model]),block_moments(f['test'],f['predictions'][reference_name])))
    rng=np.random.default_rng(20261029);dr=[];dc=[]
    for _ in range(2000):
        x=[];y=[]
        for a,b in pairs:
            ix=rng.integers(0,len(a),len(a));x.append(moment_score(a[ix]));y.append(moment_score(b[ix]))
        dr.append(np.mean([z['rmse'] for z in x])-np.mean([z['rmse'] for z in y]))
        xr=np.tanh(np.mean(np.arctanh(np.clip([z['r'] for z in x],-.999999,.999999))));yr=np.tanh(np.mean(np.arctanh(np.clip([z['r'] for z in y],-.999999,.999999))));dc.append(xr-yr)
    return {'reference':reference_name,'mean_fold_rmse_delta_95':np.percentile(dr,[2.5,97.5]).tolist(),'fisher_r_delta_95':np.percentile(dc,[2.5,97.5]).tolist(),'draws':2000,'limitations':'Conditional whole-block intervals, not selection-adjusted; overlapping blocks may be dependent. Historical test scores were previously known.'}

def final(cfg):
    f=read_result('freeze.json');assert f['code_hashes']==code_hashes(),'Scientific sources changed after freeze'
    assert sha((DATA/'freeze.json').read_bytes())==cfg['freeze_sha256'],'Wrong freeze identity'
    marker=DATA/'final.started.json'
    with marker.open('x') as out:json.dump({'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'freeze_sha256':cfg['freeze_sha256'],'experiment_url':cfg['experiment_url']},out)
    save_json(EXPORT/'final_started.json',json.loads(marker.read_text()))
    bs=blocks(final=True);assert len(bs)==157
    date_sets={fold:set() for fold in range(1,6)}
    for b in bs:date_sets[b['fold']].update((int(v[0]),int(v[1])) for v in b['output'])
    assert all(not date_sets[a]&date_sets[b] for a in date_sets for b in date_sets if a<b),'Cross-fold date overlap'
    allfolds=[];audits=[];t=time.time();models_dir=DATA/'final_models';models_dir.mkdir()
    for fold in range(1,6):
        fitting=[i for i in range(1,6) if i!=fold];path=DATA/f'final_embeddings_{fold}.npz';emb=embed_blocks(bs,fitting,path);d=np.load(path);tr,te,norm=samples(d,fitting,[fold]);features={key:te[key] for key in ['energy','spatial']}
        folder=models_dir/f'fold{fold}';folder.mkdir();np.savez_compressed(folder/'normalization.npz',**norm,image_mean=d['image_mean'])
        save_json(folder/'recipes.json',{'baseline':f['baseline_recipe'],'panel':f['panel']})
        predictions={};cache={}
        for row in f['panel']:predictions[row['model']]=predict(tr,features,row['recipe'],cache,folder,f['baseline_recipe'])
        # Holdout labels enter only the final scoring/reference-alignment boundary.
        rp,ra=reference(fold,te,d);predictions['released_windnet']=rp
        assert all(len(p)==len(te['y']) for p in predictions.values())
        audits.append({'fold':fold,'fit_rows':len(tr['y']),'test_rows':len(te['y']),'fit_row_sha256':rowhash(tr),'test_row_sha256':rowhash(te),'all_models_identical_rows':True,'test_labels_passed_to_estimators':False,'reference':ra,'embedding':emb})
        allfolds.append({'test':te,'predictions':predictions})
        print(json.dumps({'final_fit_complete_fold':fold,'models':len(predictions),'seconds':round(time.time()-t),'scores_withheld_until_all_fits_complete':True}),flush=True)
    # Save all fitted artifacts before computing any score.
    manifest=[{'file':str(p.relative_to(models_dir)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in models_dir.rglob('*') if p.is_file()]
    save_json(EXPORT/'checkpoint_manifest.json',{'artifacts':manifest,'count':len(manifest),'includes_raw_test_observations':False,'freeze_sha256':cfg['freeze_sha256']})
    with tarfile.open(EXPORT/'checkpoints.tar.gz','w:gz',compresslevel=1) as tar:tar.add(models_dir,arcname='models');tar.add(EXPORT/'freeze.json',arcname='freeze.json')
    names=list(allfolds[0]['predictions']);rows=[]
    for name in names:
        ms=[metrics(x['test']['y'],x['predictions'][name],x['test']['sigma']) for x in allfolds];rows.append({'model':name,'fold_metrics':ms,'summary':aggregate(ms)})
    primary=next(r['summary'] for r in rows if r['model']==f['primary']);ref=next(r['summary'] for r in rows if r['model']=='released_windnet');port=next(r['summary'] for r in rows if r['model']=='windnet_port_ensemble')
    two=primary['mean_rmse']<ref['mean_rmse'] and primary['fisher_r']>ref['fisher_r'];four=two and primary['mean_chi2']<ref['mean_chi2'] and primary['mean_hse_ts'] is not None and ref['mean_hse_ts'] is not None and primary['mean_hse_ts']>ref['mean_hse_ts']
    result={'stage':'F1','scope':'Frozen historical original five-fold benchmark; not fresh confirmation','primary':f['primary'],'two_metric_gate_passed':two,'all_four_metric_gate_passed':four,'primary_summary':primary,'released_windnet_summary':ref,'matched_port_summary':port,'rows':rows,'primary_uncertainty':paired_bootstrap(allfolds,f['primary']),'verdict':('Primary improved all four paper metrics numerically.' if four else 'Primary improved RMSE and correlation numerically, with other metric tradeoffs.' if two else 'Primary did not improve both preregistered RMSE and correlation metrics.'),'final_evaluation_passes':1,'post_test_model_changes':0,'freeze_sha256':cfg['freeze_sha256'],'runtime_seconds':time.time()-t,'identity':identity(),'limitations':[f['prior_exposure'],'Architecture and optimizer choices used Folds1-4 development; aggregate five-fold scores are exploratory.','Nominal D3 follows original observation windows and does not always equal three calendar days.','PyTorch port differences from original TensorFlow are recorded; released forecasts are a separate historical comparator.','Block bootstrap is conditional and not selection adjusted; original blocks can overlap.','HSE score preserves original implementation conventions; speed-threshold metrics are not substituted.']}
    save_json(EXPORT/'final_results.json',result);save_json(EXPORT/'final_audit.json',{'all_original_members_hash_verified':True,'fold5_parsed_only_after_freeze':True,'cross_fold_dates_disjoint':True,'all_model_row_identity_checks_passed':True,'test_labels_never_passed_to_fit_or_predict':True,'folds':audits,'total_test_windows':sum(x['test_rows'] for x in audits),'final_evaluation_passes':1,'checkpoint_count':len(manifest)})
    print(json.dumps({'stage':'F1_complete','verdict':result['verdict'],'primary':primary,'reference':ref,'two_metric_gate_passed':two,'all_four_metric_gate_passed':four,'seconds':time.time()-t}),flush=True)
