"""One external pass, all predictions locked before target scoring/export."""
import datetime as dt
import hashlib,json,time
from pathlib import Path
import numpy as np
from external_eval.worker import save
from external_eval.seal import verify

def main():
    freeze=verify();root=Path('data/external_sealed');root.mkdir(exist_ok=True)
    with (root/'external_test.started.json').open('x',encoding='utf8') as marker:
        json.dump({'started':dt.datetime.now(dt.timezone.utc).isoformat(),'freeze_sha256':hashlib.sha256(Path('external_eval/evaluator_freeze.json').read_bytes()).hexdigest()},marker)
    start=time.time()
    dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2026-01-01'))
    image_dates=np.arange(dates[0]-6,dates[-1]-2)
    indices=np.stack([np.arange(i,i+4) for i in range(len(dates))])
    assert np.all(image_dates[indices]==dates[:,None]+np.arange(-6,-2)[None,:])
    from external_eval.data_source import acquire_images,acquire_targets
    images,image_valid,source=acquire_images(image_dates)
    valid=np.all(image_valid[indices],axis=1)
    from external_eval.author_inference import AuthorWindNet
    from external_eval.m13_inference import M13,load_encoder
    pred={name:np.full((5,len(dates)),np.nan) for name in ['original_windnet','M13']}
    encoder=load_encoder()
    for fold in range(1,6):
        author=AuthorWindNet(fold)
        try:
            emb=author.encode(images)
            pred['original_windnet'][fold-1,valid]=author.predict_embeddings(emb[indices[valid]])
        finally:author.close()
        m13=M13(fold,encoder);features=m13.encode(images)
        pred['M13'][fold-1,valid]=m13.predict({k:v[indices[valid]] for k,v in features.items()})
        del m13,features,emb
        print(json.dumps({'replica_forecasts_completed':fold,'replicas':5,'aggregate_scores_exported':False}),flush=True)
    assert all(np.isfinite(v[:,valid]).all() for v in pred.values())
    path=root/'frozen_predictions.npz';np.savez_compressed(path,dates=dates,image_valid=valid,**pred)
    prediction_lock={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'completed_before_labels_opened':True,
                     'replicas_per_model':5,'model_names':list(pred),'timestamp':dt.datetime.now(dt.timezone.utc).isoformat()}
    (root/'prediction_lock.json').write_text(json.dumps(prediction_lock,indent=2),encoding='utf8')
    # Target labels and sigmas enter only this frozen final scoring boundary.
    y,sigma,targets=acquire_targets(dates)
    eligible=valid&np.isfinite(y)
    from external_eval.scoring import evaluate
    result=evaluate(dates,y,sigma,pred,eligible)
    result|={'status':'completed','period':['2020-01-01','2025-12-31'],'models_frozen':True,'future_tuning_or_refits':0,
             'external_evaluation_passes':1,'predictions':prediction_lock,'image_sources':source,'target_source':targets,
             'freeze_sha256':hashlib.sha256(Path('external_eval/evaluator_freeze.json').read_bytes()).hexdigest(),
             'model_lock_sha256':hashlib.sha256(Path('external_eval/model_lock.json').read_bytes()).hexdigest(),
             'runtime_seconds':time.time()-start,'metrics_comparable_to_original_paper_numbers':False,
             'comparison_scope':'Head-to-head on a common new external period. Paper2011-2018 scores have different targets and are not directly interchangeable.',
             'retrospective_preprocessing':'Definitive AIA synoptic archive; no real-time latency claim; V8 instrument response locked before external arrays.',
             'post_test_candidate_promotion':False}
    verify();save('external_results.json',result);save('prediction_lock.json',prediction_lock)
    print(json.dumps({'external_test_completed':True,'scores_exported':'external_results.json','seconds':round(time.time()-start)}),flush=True)

if __name__=='__main__':main()
