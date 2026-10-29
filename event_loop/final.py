"""Prospective freeze and isolated, aggregate-only historical evaluation for F2."""
import copy,datetime as dt,json,pickle,tarfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from matched.runner import ROOT,DATA,EXPORT,save_json,sha,archive
from matched.science import blocks,embed_blocks,samples,metrics,rowhash,neural_fit,fit_tabular,DEVICE,SEEDS,block_moments,moment_score
from matched.final_study import reference,aggregate,paired_bootstrap,SCIENTIFIC_FILES
from event_loop.science import EPOCHS,identity
from event_loop.location import LocationFeatures,representation,temporal

def hashes():
    paths=[ROOT/'matched'/n for n in SCIENTIFIC_FILES]+[p for p in (ROOT/'event_loop').glob('*') if p.suffix in ['.py','.json']]
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(paths)}

def freeze(cfg):
    assert not (DATA/'event_freeze.json').exists(),'Refuse to replace frozen choices'
    assert not (DATA/'sealed').exists(),'Final data available before freeze'
    r=json.loads((EXPORT/'m10_results.json').read_text());assert r['gate_passed'] and r['selected']
    selected=r['selected'];assert selected['gate']['all_four'] and selected['stability']['pre_bootstrap_passed']
    check=next(x for x in r['bootstrap_checks'] if x['model']==selected['model']);assert check['passed']
    primary=selected['recipe']
    if primary['family']=='location_ridge':primary={'family':'affine_location_mix','location':primary,'location_weight':1.,'prior':'M6','gain':1.,'bias':0}
    assert primary['family']=='affine_location_mix'
    no_weight=copy.deepcopy(primary);no_weight['location']['weight_lambda']=0
    no_prior={**primary,'location_weight':1.};no_cal={**primary,'gain':1.,'bias':0}
    panel=[{'model':'PRIMARY_F2','recipe':primary},{'model':'windnet_port','recipe':{'family':'port'}},{'model':'prior_M6','recipe':{'family':'prior'}},{'model':'location_raw','recipe':primary['location']},{'model':'without_precision_weight','recipe':no_weight},{'model':'without_prior_mixture','recipe':no_prior},{'model':'without_amplitude_calibration','recipe':no_cal}]
    f={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'campaign_id':'e324d24c-b179-427c-87c1-365b857658aa','phase':'F2','selected_training_model':selected['model'],'primary':'PRIMARY_F2','panel':panel,'source_hashes':hashes(),'training_result_sha256':sha((EXPORT/'m10_results.json').read_bytes()),'source_members_sha256':sha((DATA/'members.json').read_bytes()),'original_kaffe_sha256':sha((DATA/'kaffe/googlenet.npy').read_bytes()),'graph_sha256':sha((DATA/'kaffe/googlenet.py').read_bytes()),'epochs':EPOCHS,'seeds':SEEDS,'final_rule':'One fixed historical five-fold pass. All-four success iff PRIMARY_F2 strictly lowers mean fold RMSE and chi2 and raises Fisher-z r and mean original HSE threat score versus the released original WindNet forecasts on identical2198windows. Undefined fails. No candidate replacement, recipe changes or hypothesis generation from test scores. Score fixed ablations regardless of outcomes.','exposure':'Earlier historical test scores are already known; Folds1-4 have been used for development. This is an exploratory locked historical benchmark, not fresh blind confirmation. Broad validation search and sequential use limit inference.','input_scope':'Original four211images H4D3;157originalblocks;13fit/14testwindows perblock. No extra predictor information. Both controls and all ablations use identical rows.'}
    f['development_input_hashes']={str(p.relative_to(DATA)):sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))}
    save_json(DATA/'event_freeze.json',f);save_json(EXPORT/'event_freeze.json',f);save_json(EXPORT/'event_freeze_identity.json',{'sha256':sha((DATA/'event_freeze.json').read_bytes()),'source_hashes':f['source_hashes'],'models':len(panel),'fold5_parsed':False,'final_evaluations':0})
    print(json.dumps({'frozen':True,'sha256':sha((DATA/'event_freeze.json').read_bytes()),'models':len(panel),'fold5_parsed':False}),flush=True)

def acquire_final(f):
    assert sha((DATA/'members.json').read_bytes())==f['source_members_sha256']
    expected={m['member']:m for m in json.loads((DATA/'members.json').read_text()) if m['fold']==5};(DATA/'sealed').mkdir()
    r,tar=archive('1JJjXjauT02z-RevZZ-8nQmHv9wr9wDrT');count=0
    with r,tar:
        for member in tar:
            if member.name not in expected:continue
            raw=tar.extractfile(member).read();m=expected[member.name];assert sha(raw)==m['sha256'];(DATA/'sealed'/m['file']).write_bytes(raw);count+=1
    assert count==49

def location_cache(bs,fitting,path,expected_energy):
    fit=[b for b in bs if b['fold'] in fitting];mean=float(sum(b['images'].sum(dtype=np.float64) for b in fit)/(len(fit)*20*256*256));model=LocationFeatures().eval();local=[];glob=[]
    with torch.inference_mode():
        for b in bs:
            ims=torch.from_numpy(b['images']).to(DEVICE)[:,None];ims=F.interpolate(ims,size=(224,224),mode='bilinear',align_corners=False).repeat(1,3,1,1)-mean;a,g=model(ims);local.append(a.cpu().numpy());glob.append(g.cpu().numpy())
    local=np.stack(local);glob=np.stack(glob);assert np.allclose(glob,expected_energy,rtol=1e-5,atol=.01)
    np.savez_compressed(path,local=local,glob=glob,image_mean=mean)
    return local,glob

def fit_location(tr,features,recipe,cache,folder):
    key=json.dumps(recipe,sort_keys=True)
    if key in cache:return cache[key]
    x=temporal(tr[recipe['representation']],recipe['view']);v=temporal(features[recipe['representation']],recipe['view']);s=StandardScaler().fit(x);s.scale_=np.maximum(s.scale_,.01);x=s.transform(x).astype(np.float64);v=s.transform(v).astype(np.float64)
    positive=tr['sigma'][tr['sigma']>0];floor=float(np.quantile(positive,.05));med=float(np.median(positive));w=1+recipe['weight_lambda']*(med/np.maximum(tr['sigma'],floor))**2;w/=w.mean();xm=np.average(x,axis=0,weights=w);ym=float(np.average(tr['y'],weights=w));xt=torch.as_tensor((x-xm)*np.sqrt(w[:,None]),device=DEVICE);vt=torch.as_tensor(v-xm,device=DEVICE);gram=(xt@xt.T).cpu().numpy();cross=(vt@xt.T).cpu().numpy();lam,q=eigh(gram,check_finite=False);lam=np.maximum(lam,0);dual=q@((q.T@((tr['y']-ym)*np.sqrt(w)))/(lam+recipe['alpha']));p=cross@dual+ym;coef=(xt.T@torch.as_tensor(dual,device=DEVICE)).cpu().numpy();del xt,vt
    bundle={'recipe':recipe,'scaler':s,'center':xm,'ymean':ym,'coefficient':coef,'sigma_floor_fit_only':floor,'sigma_median_fit_only':med}
    with (folder/('location_'+sha(key.encode())[:12]+'.pkl')).open('wb') as out:pickle.dump(bundle,out,protocol=5)
    cache[key]=p;return p

def predict(tr,features,recipe,cache,folder):
    key=json.dumps(recipe,sort_keys=True)
    if key in cache:return cache[key]
    family=recipe['family']
    if family=='port':
        pp=[]
        for seed,epochs in zip(SEEDS,EPOCHS):
            p,state,_=neural_fit(tr,{'energy':features['energy']},seed,fixed_epochs=epochs);torch.save(state,folder/f'port_seed{seed}.pt');pp.append(p)
        p=np.mean(pp,axis=0)
    elif family=='prior':
        port=predict(tr,features,{'family':'port'},cache,folder);ridge,bundle=fit_tabular(tr,{k:features[k] for k in ['energy','spatial']},{'family':'ridge','source':'combined','view':'flatten','alpha':10000});anchor=float(tr['y'].mean());p=anchor+1.2*(.5*port+.5*ridge-anchor)
        with (folder/'prior_ridge.pkl').open('wb') as out:pickle.dump(bundle,out,protocol=5)
    elif family=='location_ridge':p=fit_location(tr,features,recipe,cache,folder)
    elif family=='affine_location_mix':
        a=predict(tr,features,recipe['location'],cache,folder);b=predict(tr,features,{'family':'prior'},cache,folder);w=recipe['location_weight'];anchor=float(tr['y'].mean());p=anchor+recipe['gain']*(w*a+(1-w)*b-anchor)+recipe['bias']
    else:raise ValueError(family)
    assert p.shape==(len(features['energy']),) and np.isfinite(p).all();cache[key]=p;return p

def all_four_bootstrap(folds,model,draws=2000):
    pairs=[(block_moments(f['test'],f['predictions'][model]),block_moments(f['test'],f['predictions']['released_windnet'])) for f in folds];rng=np.random.default_rng(20261029);d={k:[] for k in ['rmse','r','chi2','hse_ts']}
    for _ in range(draws):
        left=[];right=[]
        for a,b in pairs:
            ix=rng.integers(0,len(a),len(a));left.append(moment_score(a[ix]));right.append(moment_score(b[ix]))
        for k in d:
            if k=='r':x=np.tanh(np.mean(np.arctanh(np.clip([v[k] for v in left],-.999999,.999999))));y=np.tanh(np.mean(np.arctanh(np.clip([v[k] for v in right],-.999999,.999999))))
            else:x=np.mean([v[k] for v in left]);y=np.mean([v[k] for v in right])
            if np.isfinite(x-y):d[k].append(float(x-y))
    return {'reference':'released_windnet','draws':draws,'metrics':{k:{'delta_95':np.percentile(v,[2.5,97.5]).tolist() if v else None,'defined_draws':len(v)} for k,v in d.items()},'limitations':'Conditional whole-block bootstrap, not adjusted for model selection; overlapping blocks and prior historical exposure limit inference.'}

def final(cfg):
    f=json.loads((DATA/'event_freeze.json').read_text());assert sha((DATA/'event_freeze.json').read_bytes())==cfg['freeze_sha256'];assert f['source_hashes']==hashes(),'Scientific sources changed after freeze'
    assert sha((DATA/'kaffe/googlenet.npy').read_bytes())==f['original_kaffe_sha256'] and sha((DATA/'kaffe/googlenet.py').read_bytes())==f['graph_sha256']
    for name,digest in f['development_input_hashes'].items():assert sha((DATA/name).read_bytes())==digest
    marker=DATA/'event_final.started.json'
    with marker.open('x') as out:json.dump({'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'freeze_sha256':cfg['freeze_sha256'],'experiment_url':cfg['experiment_url']},out)
    save_json(EXPORT/'event_final_started.json',json.loads(marker.read_text()));acquire_final(f)
    # The frozen legacy loader checks this marker; this is the new phase's freeze.
    save_json(DATA/'freeze.json',{'event_freeze_sha256':cfg['freeze_sha256']})
    bs=blocks(final=True);assert len(bs)==157
    dates={fold:set() for fold in range(1,6)}
    for b in bs:dates[b['fold']].update((int(v[0]),int(v[1])) for v in b['output'])
    assert all(not dates[a]&dates[b] for a in dates for b in dates if a<b)
    allfolds=[];audits=[];t=time.time();modeldir=DATA/'event_final_models';modeldir.mkdir()
    for held in range(1,6):
        fitting=[i for i in range(1,6) if i!=held];path=DATA/f'event_final_energy{held}.npz';emb=embed_blocks(bs,fitting,path);d=np.load(path);tr,te,norm=samples(d,fitting,[held]);features={k:te[k] for k in ['energy','spatial']};local,glob=location_cache(bs,fitting,DATA/f'event_final_location{held}.npz',d['energy'])
        for kind in ['ratio','log']:
            frames=representation(local,glob,kind)
            for a,target in [(tr,tr),(te,features)]:target[kind]=np.stack([frames[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])])
        folder=modeldir/f'fold{held}';folder.mkdir();np.savez_compressed(folder/'normalization.npz',**norm,image_mean=d['image_mean']);save_json(folder/'frozen_panel.json',f['panel']);cache={};predictions={}
        for r in f['panel']:predictions[r['model']]=predict(tr,features,r['recipe'],cache,folder)
        rp,ra=reference(held,te,d);predictions['released_windnet']=rp
        audits.append({'fold':held,'fit_rows':len(tr['y']),'test_rows':len(te['y']),'fit_row_sha256':rowhash(tr),'test_row_sha256':rowhash(te),'all_models_identical_rows':True,'test_labels_passed_to_fit_or_predict':False,'reference':ra,'embedding':emb});allfolds.append({'test':te,'predictions':predictions})
        del local,glob,frames,tr,features,cache
        print(json.dumps({'F2_fit_fold_complete':held,'models':len(predictions),'scores_withheld_until_all_fits_complete':True,'seconds':time.time()-t}),flush=True)
    manifest=[{'file':str(p.relative_to(modeldir)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in modeldir.rglob('*') if p.is_file()];save_json(EXPORT/'event_checkpoint_manifest.json',{'artifacts':manifest,'count':len(manifest),'freeze_sha256':cfg['freeze_sha256'],'includes_raw_holdout_observations':False})
    with tarfile.open(EXPORT/'event_checkpoints.tar.gz','w:gz',compresslevel=1) as tar:tar.add(modeldir,arcname='models');tar.add(EXPORT/'event_freeze.json',arcname='freeze.json')
    rows=[]
    for name in allfolds[0]['predictions']:
        ms=[metrics(x['test']['y'],x['predictions'][name],x['test']['sigma']) for x in allfolds];rows.append({'model':name,'fold_metrics':ms,'summary':aggregate(ms)})
    primary=next(r['summary'] for r in rows if r['model']==f['primary']);ref=next(r['summary'] for r in rows if r['model']=='released_windnet');checks={'rmse':primary['mean_rmse']<ref['mean_rmse'],'r':primary['fisher_r']>ref['fisher_r'],'chi2':primary['mean_chi2']<ref['mean_chi2'],'hse_ts':primary['mean_hse_ts'] is not None and ref['mean_hse_ts'] is not None and primary['mean_hse_ts']>ref['mean_hse_ts']};four=all(checks.values())
    result={'stage':'F2','scope':'Locked original five-fold historical comparison; not fresh blind confirmation','primary':f['primary'],'all_four_metric_gate_passed':four,'metric_checks':checks,'primary_summary':primary,'released_windnet_summary':ref,'rows':rows,'primary_uncertainty':paired_bootstrap(allfolds,f['primary']),'verdict':'Frozen primary improved all four original metrics numerically.' if four else 'Frozen primary failed the all-four historical metric requirement.','freeze_sha256':cfg['freeze_sha256'],'final_evaluation_passes':1,'post_test_model_changes':0,'runtime_seconds':time.time()-t,'identity':identity(),'limitations':[f['exposure'],'Matched PyTorch port differs from original TensorFlow implementation; released forecasts are a separate historical comparator.','Nominal D3 and HSE conventions preserve original implementation.','Bootstrap conditional, not selection adjusted; blocks can overlap.']}
    result['all_four_uncertainty']=all_four_bootstrap(allfolds,f['primary'])
    save_json(EXPORT/'event_final_results.json',result);save_json(EXPORT/'event_final_audit.json',{'folds':audits,'original_members':157,'total_test_windows':sum(a['test_rows'] for a in audits),'cross_fold_dates_disjoint':True,'fold5_parsed_only_after_freeze':True,'all_model_row_identity_checks_passed':True,'test_labels_never_passed_to_fit_or_predict':True,'final_evaluation_passes':1,'checkpoint_count':len(manifest)})
    print(json.dumps({'F2_complete':True,'all_four_metric_gate_passed':four,'primary':primary,'reference':ref,'seconds':time.time()-t}),flush=True)
