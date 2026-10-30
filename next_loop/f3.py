"""F3: lock every recipe before acquiring Fold5; export aggregate scores only."""
import datetime as dt
import json
import pickle
import tarfile
import time
from pathlib import Path
import numpy as np
from matched.runner import ROOT,DATA,EXPORT,save_json,sha
from matched.science import blocks,embed_blocks,samples,metrics,rowhash,block_moments,moment_score
from matched.final_study import reference
from event_loop.final import acquire_final,location_cache,predict as prior_predict
from event_loop.location import representation
from event_loop.science import identity,EPOCHS,SEEDS
from next_loop.f3_panel import fixed_panel,fit_regional,calibrate,aggregate,metric_gate,REGIONAL
from next_loop.regional_features import regions

CAMPAIGN='1b7e08e2-70f1-4945-b613-5e785d3f76bb'
EXPERIMENT='e1215e22-e3e3-40d9-b59e-b5f7754e85e9'
SOURCE_FILES = ([f'matched/{n}' for n in ['__init__.py','runner.py','science.py','hse_original.py','final_study.py','source_manifest.json','reference_sources.json','protocol.json']]
                +[f'event_loop/{n}' for n in ['__init__.py','runner.py','science.py','location.py','final.py','mixtures.py']]
                +[f'next_loop/{n}' for n in ['__init__.py','regional_features.py','target_weights.py','f3_panel.py','f3.py','f3_entry.py','f3_transport.py','f3_protocol.json','m13_training_selection.json','test_f3.py','test_regions.py','test_target_weights.py']])

def source_hashes():return {name:sha((ROOT/name).read_bytes()) for name in SOURCE_FILES}

def verify_selection():
    selected=json.loads((ROOT/'next_loop/m13_training_selection.json').read_text())
    assert selected['recipe_locked'] and selected['selected']['gate']['all_four']
    r=selected['selected']['recipe']
    assert {k:r[k] for k in REGIONAL}==REGIONAL
    assert r['new_head_share']==.25 and r['gain']==1.2 and r['bias']==-5 and r['prior']=='fixed M10'
    assert selected['selected_bootstrap']['passed']
    return selected

def freeze():
    assert not (DATA/'sealed').exists(),'Fold5 must be absent until full freeze'
    assert not (DATA/'f3_freeze.json').exists(),'Refuse to replace frozen choices'
    assert not (DATA/'f3_final.started.json').exists()
    selected=verify_selection()
    source=json.loads((ROOT/'matched/source_manifest.json').read_text())
    members=json.loads((DATA/'members.json').read_text())
    expected={x['member']:x for x in source['members']}
    assert len(members)==len(expected)==157 and sum(m['fold']==5 for m in members)==49
    for m in members:
        assert m['sha256']==expected[m['member']]['release_member_sha256']
        if m['fold']==5:assert not m['retained'] and not m['parsed_for_development']
    import platform,torch,scipy,sklearn
    f={'phase':'F3','created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
       'campaign_id':CAMPAIGN,'experiment_id':EXPERIMENT,'primary':'PRIMARY_F3',
       'panel':fixed_panel(),'source_hashes':source_hashes(),
       'source_members_sha256':sha((DATA/'members.json').read_bytes()),
       'original_kaffe_sha256':sha((DATA/'kaffe/googlenet.npy').read_bytes()),
       'graph_sha256':sha((DATA/'kaffe/googlenet.py').read_bytes()),
       'development_input_hashes':{str(p.relative_to(DATA)).replace('\\','/'):sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))},
       'epochs':EPOCHS,'seeds':SEEDS,'selection':selected['selected']['model'],
       'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'gpu':torch.cuda.get_device_name(0)},
       'selection_evidence_hashes':selected['training_evidence_hashes'],
       'historical_evaluation_passes':1,'post_test_model_changes':0,
       'scope':'Original157blocks;13fitting/14evaluationwindows perblock;H4D3;four211Aimages;2198evaluationwindows',
       'test_rule':'One fixed pass. Separate strict all-four numerical gates versus released WindNet and same-runtime M10. Undefined fails. All fits precede all score exports. No test-driven replacement or hypothesis generation.',
       'exposure':'Folds1-4 are reused development data and earlier historical aggregate tests are known. This is an exploratory locked historical benchmark, not fresh blind confirmation.',
       'uncertainty':'2000 paired whole-block draws within folds,seed20261029,conditional and not selection adjusted; overlapping blocks and sequential evaluation limit inference.'}
    assert len(f['development_input_hashes'])==108 and len(f['panel'])==10
    assert f['original_kaffe_sha256']=='d93955ecfb66c9cdbe0c038f7f47fbf971e46dee30930970e22fc4a1b8fa184f'
    assert f['graph_sha256']=='c40b34bd4910e175ce121a41083ab5336bd6689167eb7206f63831706335c834'
    save_json(DATA/'f3_freeze.json',f);save_json(EXPORT/'f3_freeze.json',f)
    digest=sha((DATA/'f3_freeze.json').read_bytes())
    save_json(EXPORT/'f3_freeze_identity.json',{'sha256':digest,'source_hashes':f['source_hashes'],'panel_models':10,'fold5_available':False,'fold5_parsed':False,'final_evaluations':0})
    print(json.dumps({'F3_frozen':True,'sha256':digest,'fold5_available':False}),flush=True)

def check_freeze(digest):
    path=DATA/'f3_freeze.json';assert sha(path.read_bytes())==digest
    f=json.loads(path.read_text());assert source_hashes()==f['source_hashes'],'Sources changed after freeze'
    assert sha((DATA/'members.json').read_bytes())==f['source_members_sha256']
    assert sha((DATA/'kaffe/googlenet.npy').read_bytes())==f['original_kaffe_sha256']
    assert sha((DATA/'kaffe/googlenet.py').read_bytes())==f['graph_sha256']
    for name,value in f['development_input_hashes'].items():assert sha((DATA/name).read_bytes())==value
    assert f['panel']==fixed_panel()
    return f

def frame_samples(frames,a):
    return np.stack([frames[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])])

def prediction(train,features,recipe,cache,folder):
    assert set(features)=={'energy','spatial','ratio','regional_6x6','regional_1x1'}
    key=json.dumps(recipe,sort_keys=True)
    if key in cache:return cache[key]
    if recipe['family']!='regional_mix':
        return prior_predict(train,features,recipe,cache,folder)
    reg=recipe['regional'];regkey='regional_'+str(reg['grid'][0])+'x'+str(reg['grid'][1])
    rawkey='regional_raw:'+json.dumps(reg,sort_keys=True)
    if rawkey not in cache:
        raw,bundle=fit_regional({'regions':train[regkey],'y':train['y'],'sigma':train['sigma']},
                                {'regions':features[regkey]},reg)
        with (folder/('regional_'+sha(rawkey.encode())[:12]+'.pkl')).open('wb') as out:pickle.dump(bundle,out,protocol=5)
        cache[rawkey]=raw
    mu=float(train['y'].mean());p=calibrate(cache[rawkey],mu,recipe['gain'],recipe['bias'])
    if recipe['new_head_share']!=1:
        prior=prior_predict(train,features,recipe['prior'],cache,folder)
        p=recipe['new_head_share']*p+(1-recipe['new_head_share'])*prior
    assert p.shape==(len(features['energy']),) and np.isfinite(p).all()
    cache[key]=p;return p

def bootstrap(folds,model,ref):
    pairs=[(block_moments(f['test'],f['predictions'][model]),block_moments(f['test'],f['predictions'][ref])) for f in folds]
    rng=np.random.default_rng(20261029);out={k:[] for k in ['rmse','r','chi2','hse_ts']}
    for _ in range(2000):
        left=[];right=[]
        for a,b in pairs:
            ix=rng.integers(0,len(a),len(a));left.append(moment_score(a[ix]));right.append(moment_score(b[ix]))
        for key in out:
            av=[m[key] for m in left];bv=[m[key] for m in right]
            if any(x is None or not np.isfinite(x) for x in av+bv):continue
            def avg(v):return np.tanh(np.mean(np.arctanh(np.clip(v,-.999999,.999999)))) if key=='r' else np.mean(v)
            delta=float(avg(av)-avg(bv))
            if np.isfinite(delta):out[key].append(delta)
    return {'reference':ref,'draws':2000,'metrics':{k:{'delta_95':np.percentile(v,[2.5,97.5]).tolist() if v else None,'defined_draws':len(v)} for k,v in out.items()},'limitations':'Conditional whole-block intervals; not selection adjusted, overlapping blocks, prior historical exposure.'}

def final(digest):
    f=check_freeze(digest)
    marker=DATA/'f3_final.started.json'
    with marker.open('x') as out:json.dump({'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'freeze_sha256':digest,'experiment_id':EXPERIMENT},out)
    save_json(EXPORT/'f3_final_started.json',json.loads(marker.read_text()))
    assert not (DATA/'sealed').exists()
    acquire_final(f)
    save_json(DATA/'freeze.json',{'F3_freeze_sha256':digest})
    bs=blocks(final=True);assert len(bs)==157
    dates={fold:set() for fold in range(1,6)}
    for b in bs:dates[b['fold']].update((int(v[0]),int(v[1])) for v in b['output'])
    assert all(not dates[a]&dates[b] for a in dates for b in dates if a<b)
    regional_frames={f'regional_{g[0]}x{g[1]}':np.stack([regions(b['images'],g) for b in bs]) for g in [(6,6),(1,1)]}
    allfolds=[];audits=[];t=time.time();modeldir=DATA/'f3_models';modeldir.mkdir()
    for held in range(1,6):
        fitting=[i for i in range(1,6) if i!=held]
        path=DATA/f'f3_energy{held}.npz';emb=embed_blocks(bs,fitting,path);d=np.load(path)
        train,test,norm=samples(d,fitting,[held]);features={k:test[k] for k in ['energy','spatial']}
        local,glob=location_cache(bs,fitting,DATA/f'f3_location{held}.npz',d['energy'])
        frames=representation(local,glob,'ratio')
        train['ratio']=frame_samples(frames,train);features['ratio']=frame_samples(frames,test)
        del local,glob,frames
        for key,frames in regional_frames.items():train[key]=frame_samples(frames,train);features[key]=frame_samples(frames,test)
        folder=modeldir/f'fold{held}';folder.mkdir()
        np.savez_compressed(folder/'normalization.npz',**norm,image_mean=d['image_mean'])
        save_json(folder/'frozen_panel.json',f['panel']);cache={};predictions={}
        for row in f['panel']:
            if row['recipe']['family']=='released_reference':continue
            predictions[row['model']]=prediction(train,features,row['recipe'],cache,folder)
        rp,ra=reference(held,test,d);predictions['released_windnet']=rp
        audits.append({'fold':held,'fit_folds':fitting,'fit_rows':len(train['y']),'test_rows':len(test['y']),
                       'fit_row_sha256':rowhash(train),'test_row_sha256':rowhash(test),
                       'all_models_identical_rows':True,'test_labels_passed_to_fit_or_predict':False,
                       'predict_input_keys':sorted(features),'reference':ra,'embedding':emb})
        allfolds.append({'test':test,'predictions':predictions})
        del train,features,cache,d
        print(json.dumps({'F3_fit_fold_complete':held,'models':len(predictions),'scores_withheld_until_all_fits_complete':True,'seconds':time.time()-t}),flush=True)
    assert sum(x['test_rows'] for x in audits)==2198
    assert check_freeze(digest)==f
    manifest=[{'file':str(p.relative_to(modeldir)).replace('\\','/'),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(modeldir.rglob('*')) if p.is_file()]
    save_json(EXPORT/'f3_checkpoint_manifest.json',{'artifacts':manifest,'count':len(manifest),'freeze_sha256':digest,'raw_holdout_observations_included':False})
    archive=EXPORT/'f3_checkpoints.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        tar.add(modeldir,arcname='models');tar.add(EXPORT/'f3_freeze.json',arcname='freeze.json')
    save_json(EXPORT/'f3_artifacts.json',{'file':archive.name,'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'count':len(manifest),'includes_raw_holdout_data_or_individual_predictions':False})
    rows=[]
    for name in [x['model'] for x in f['panel']]:
        ms=[metrics(x['test']['y'],x['predictions'][name],x['test']['sigma']) for x in allfolds]
        rows.append({'model':name,'fold_metrics':ms,'summary':aggregate(ms)})
    summary={r['model']:r['summary'] for r in rows};gates={ref:metric_gate(summary[f['primary']],summary[ref]) for ref in ['released_windnet','prior_M10']}
    result={'stage':'F3','scope':'Locked matched historical evaluation; not fresh blind confirmation','primary':f['primary'],'rows':rows,
            'gates':gates,'primary_summary':summary[f['primary']],
            'primary_uncertainty':{ref:bootstrap(allfolds,f['primary'],ref) for ref in gates},
            'freeze_sha256':digest,'final_evaluation_passes':1,'post_test_model_changes':0,
            'all_fits_completed_before_score_export':True,'runtime_seconds':time.time()-t,'identity':identity(),
            'limitations':[f['exposure'],f['uncertainty'],'PyTorch port differs from original TensorFlow; released forecasts are a separate historical comparator.','No exact cross-hardware numerical reproduction of archived M10 is claimed.','Nominal D3 and original HSE implementation conventions preserved.']}
    save_json(EXPORT/'f3_results.json',result)
    save_json(EXPORT/'f3_audit.json',{'folds':audits,'original_members':157,'total_test_windows':2198,'cross_fold_dates_disjoint':True,
              'fold5_parsed_only_after_full_freeze':True,'all_model_row_identity_checks_passed':True,
              'test_labels_never_passed_to_fit_or_predict':True,'all_fits_completed_before_score_export':True,
              'final_evaluation_passes':1,'checkpoint_count':len(manifest)})
    print(json.dumps({'F3_complete':True,'gates':gates,'seconds':time.time()-t}),flush=True)
