"""Read released WindNet prediction arrays, without downloading model weights."""
import concurrent.futures, csv, hashlib, importlib, json
from pathlib import Path
import numpy as np
import requests
from recover_folds import numeric_npy

ROOT=Path(__file__).resolve().parent
FOLDERS={1:'1ZjM6HLEp50NQ9edJbKDW29VH5ZHUsKIf',2:'13srtVetaWoNSc85p2RrSVbkWepMF2hn_',
    3:'12N49_mjTDL9-lgLqDl0xrMnHlUSh12XT',4:'1pjU8v3ZyECqrjGgdsczDa1ayXjZ4CFuu',5:'1b3ChwalYqnaNkenVSbGk_BHNqmK3Rlb3'}

def fetch(fold):
    session=requests.Session();module=importlib.import_module('gdown.download_folder')
    def ls(id):
        return module._parse_embedded_folder_view(sess=session,folder_id=id,verify=True,timeout=45)[1]
    model_id=next(i for i,name,ty in ls(FOLDERS[fold]) if name=='SDOPred43')
    testing=next(i for i,name,ty in ls(model_id) if name=='Testing')
    files=[(i,n) for i,n,t in ls(testing) if n.endswith('.npy')]
    if len(files)!=1:raise ValueError(f'Expected one released selected prediction array, got {files}')
    id,name=files[0]
    url=f'https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t'
    r=session.get(url,stream=True,timeout=45);r.raise_for_status()
    b=r.raw.read(2_000_001)
    if len(b)>2_000_000:raise ValueError('Refuse model/image-sized reference file')
    dest=ROOT/'data/reference';dest.mkdir(parents=True,exist_ok=True)
    (dest/f'fold{fold}_{name}').write_bytes(b)
    obj=numeric_npy(b);data=np.asarray(obj['data'])
    compact=np.load(ROOT/'data/windnet211_labels.npz');test=np.load(ROOT/'data/study_inputs.npz')
    order=sorted(np.flatnonzero(compact['folds']==fold),key=lambda j:str(compact['partitions'][j]))
    indexes=[int(bi*14+off) for bi in order for off in range(14)]
    truth=test['y'][indexes]
    if data.shape!=(len(truth),2):raise ValueError(f'Unexpected reference shape{data.shape} fold{fold}')
    if not np.allclose(data[:,1],truth,atol=.001,rtol=0):
        raise ValueError(f'Released WindNet target order does not match original labels, maxerror{np.max(np.abs(data[:,1]-truth))}')
    metrics={'fold':fold,'n':len(truth),'rmse':float(np.sqrt(np.mean((data[:,0]-truth)**2))),
        'mae':float(np.mean(np.abs(data[:,0]-truth))),'r':float(np.corrcoef(data[:,0],truth)[0,1]),
        'file_id':id,'file_name':name,'sha256':hashlib.sha256(b).hexdigest(),'source_url':url,
        'max_truth_alignment_error':float(np.max(np.abs(data[:,1]-truth)))}
    rows=[{'row':j,'fold':fold,'y':float(truth[k]),'released_windnet':float(data[k,0])} for k,j in enumerate(indexes)]
    return metrics,rows

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,range(1,6)))
    metrics=[x[0] for x in results];rows=[row for x in results for row in x[1]]
    summary={'folds':metrics,'mean_rmse':float(np.mean([m['rmse'] for m in metrics])),
        'rmse_se':float(np.std([m['rmse'] for m in metrics],ddof=1)/np.sqrt(5)),
        'fisher_mean_r':float(np.tanh(np.mean(np.arctanh([m['r'] for m in metrics])))),
        'n_rows_matched':len(rows),'selection':'The sole prediction array retained in each author-released Testing folder. No checkpoint selection by this study.'}
    (ROOT/'results/reference_metrics.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'results/reference_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))
