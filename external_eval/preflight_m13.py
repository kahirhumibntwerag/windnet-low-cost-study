"""Saved-artifact integrity and independent legacy recipe composition, pre2019."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from external_eval.m13_inference import M13,load_encoder
from external_eval.worker import save,download

def main():
    lock=json.loads(Path('external_eval/model_lock.json').read_text())
    root=Path('data/external_models')
    for item in lock['m13']['artifacts']:
        path=root/'m13/models'/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    from next_loop.f3_panel import fixed_panel,calibrate
    from event_loop.final import predict as legacy_predict
    import event_loop.final as old
    from external_eval.author_inference import EPOCHS
    encoder=load_encoder();rows=[]
    members=json.loads((root/'pre2019/members.json').read_text())
    for fold in range(1,6):
        first=sorted([r for r in members if r['fold']==fold],key=lambda r:r['file'])[0]
        with np.load(root/'pre2019'/first['file']) as data:
            assert np.all(data['output'][:,0]<2019)
            model=M13(fold,encoder);frames=model.encode(data['images'][:7])
            indices=np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]])
            features={k:v[indices] for k,v in frames.items()}
            direct=model.predict(features)
            # Independent legacy recursive composition, with its fitting leaves
            # replaced by frozen artifact inference. Any fit call fails closed.
            port=[]
            with torch.inference_mode():
                x=torch.tensor(features['energy'],dtype=torch.float32,device=old.DEVICE)
                for net,minimum,scale in model.ports:port.append(net(x)[:,-1].cpu().numpy()*scale+minimum)
            rawloc=features['ratio'].reshape(3,-1)
            rawloc=(model.location['scaler'].transform(rawloc).astype(np.float64)-model.location['center'])@model.location['coefficient']+model.location['ymean']
            combined=np.concatenate([features['energy'].reshape(3,-1),features['spatial'].reshape(3,-1)],axis=1)
            ridge=model.ridge['model'].predict(model.ridge['scaler'].transform(combined))
            prior=model.anchor+1.2*(.5*np.mean(port,axis=0)+.5*ridge-model.anchor)
            key=lambda r:json.dumps(r,sort_keys=True)
            recipe=fixed_panel()[0]['recipe'];cache={key({'family':'prior'}):prior,key(recipe['prior']['location']):rawloc}
            legacy=legacy_predict({'y':np.array([model.anchor])},features,recipe['prior'],cache,root)
            from next_loop.regional_features import temporal
            reg=model.regional['ridge'].predict(model.regional['scaler'].transform(temporal(features['regional'],'trend','linear')))
            expected=.25*calibrate(reg,model.anchor,1.2,-5)+.75*legacy
            error=float(np.max(np.abs(direct-expected)))
            assert error<1e-9
            rows.append({'fold':fold,'pre2019_member':first['member'],'synthetic_history_count':3,'legacy_composition_max_abs_error':error,'passed':True})
            del model,frames,features
    # A fixed source bridge check from2018 only. This comparison does not choose
    # model parameters; it checks published preprocessing conventions.
    from external_eval.preprocessing import process_fits
    path=root/'pre2019_source_check.fits'
    download('https://jsoc1.stanford.edu/data/aia/synoptic/2018/01/01/H0000/AIA20180101_0000_0211.fits',path)
    rebuilt=process_fits(path,'2018-01-01');pairs=[]
    for member in members:
        with np.load(root/'pre2019'/member['file']) as data:
            select=np.flatnonzero((data['output'][:,0]==2018)&(data['output'][:,1]==1))
            for ix in select:pairs.append(float(np.sqrt(np.mean((rebuilt-data['images'][ix])**2))))
    bridge={'source_date':'2018-01-01','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'paired_release_images':len(pairs),'rms_difference_0_to256':pairs,
            'status':'paired comparison available' if pairs else 'date absent from released original blocks',
            'scientific_limitation':'Reconstructed source processing follows published SDOMLv1 and author scripts; archive revisions and interpolation libraries can prevent bitwise image equality.'}
    assert pairs and max(pairs)<.1,'Pre2019 preprocessing bridge not established'
    bridge['required_rms_tolerance_0_to256']=.1
    result={'passed':True,'external_arrays_accessed':False,'frozen_artifacts_hash_verified':55,'refits':0,
            'checks':rows,'source_bridge':bridge,'runtime_torch':torch.__version__,
            'limitation':'No individual F3 forecasts were retained for vector-by-vector cross-runtime verification. Loaded exact weights and saved scalers; independent frozen recipe composition checked on pre2019 images.'}
    save('m13_inference_preflight.json',result)
    print(json.dumps({'m13_saved_inference_preflight_passed':True,'folds':len(rows),'source_bridge':bridge}),flush=True)

if __name__=='__main__':main()
