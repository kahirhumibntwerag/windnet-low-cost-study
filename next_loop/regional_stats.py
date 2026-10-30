import numpy as np
from matched.runner import sha
from matched.hse_original import BatchwiseHSE

def metrics(y,p,sigma):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);sigma=np.asarray(sigma,dtype=float)
    assert y.shape==p.shape and np.isfinite(p).all()
    ok=np.isfinite(sigma)&(sigma>0)
    tp,fp,fn=BatchwiseHSE(np.column_stack([p,y]),4,3)
    return {'n':len(y),'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y))),'r':float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 else None,'chi2':float(np.mean(((p[ok]-y[ok])/sigma[ok])**2)) if ok.any() else None,'sigma_valid':int(ok.sum()),'hse_ts':float(tp/(tp+fp+fn)) if tp+fp+fn else None,'hse_counts_original':{'tp':int(tp),'fp':int(fp),'fn':int(fn)}}

def rowhash(a):return sha('\n'.join(a['row']).encode())

def block_moments(a,p):
    result=[]
    for b in dict.fromkeys(a['block'].tolist()):
        ix=a['block']==b;y=a['y'][ix];v=p[ix];s=a['sigma'][ix];ok=s>0;tp,fp,fn=BatchwiseHSE(np.column_stack([v,y]),4,3)
        result.append([len(y),y.sum(),v.sum(),(y*y).sum(),(v*v).sum(),(y*v).sum(),((v-y)**2).sum(),(((v[ok]-y[ok])/s[ok])**2).sum(),ok.sum(),tp,fp,fn])
    return np.array(result,dtype=float)

def moment_score(m):
    n,sy,sp,syy,spp,syp,sse,chi,nchi,tp,fp,fn=m.sum(axis=0);den=(syy-sy*sy/n)*(spp-sp*sp/n)
    return {'rmse':float(np.sqrt(sse/n)),'r':float((syp-sy*sp/n)/np.sqrt(den)) if den>0 else np.nan,'chi2':float(chi/nchi) if nchi else np.nan,'hse_ts':float(tp/(tp+fp+fn)) if tp+fp+fn else np.nan}

def aggregate(ms):
    rs=[x['r'] for x in ms];ts=[x['hse_ts'] for x in ms]
    return {'n':int(sum(x['n'] for x in ms)),'rmse':float(np.mean([x['rmse'] for x in ms])),'r':float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999999,.999999))))),'chi2':float(np.mean([x['chi2'] for x in ms])),'hse_ts':float(np.mean(ts)) if all(x is not None for x in ts) else None,'mae':float(np.mean([x['mae'] for x in ms]))}

def gate(m,controls):
    anchors={k:([c[k] for c in controls]) for k in ['rmse','r','chi2','hse_ts']}
    if m['hse_ts'] is None or any(x is None for x in anchors['hse_ts']):return {'all_four':False,'undefined':True}
    tests={'rmse':m['rmse']<=.99*min(anchors['rmse']),'r':m['r']>=max(anchors['r'])+.005,'chi2':m['chi2']<=.995*min(anchors['chi2']),'hse_ts':m['hse_ts']>=max(anchors['hse_ts'])+.02}
    return {**tests,'all_four':all(tests.values()),'anchors':{k:(max(v) if k in ['r','hse_ts'] else min(v)) for k,v in anchors.items()}}

def stability(row,controls):
    leave=[]
    for drop in range(4):
        m=aggregate([v for i,v in enumerate(row['fold_metrics']) if i!=drop]);cs=[aggregate([v for i,v in enumerate(c['fold_metrics']) if i!=drop]) for c in controls]
        passed=m['rmse']<min(c['rmse'] for c in cs) and m['r']>max(c['r'] for c in cs) and m['chi2']<min(c['chi2'] for c in cs) and m['hse_ts'] is not None and m['hse_ts']>max(c['hse_ts'] for c in cs)
        leave.append({'dropped_training_fold':drop+1,'all_four_directional':bool(passed)})
    ratios=[m['rmse']/min(c['fold_metrics'][i]['rmse'] for c in controls) for i,m in enumerate(row['fold_metrics'])]
    return {'leave_one_fold_out':leave,'directional_passes':sum(x['all_four_directional'] for x in leave),'per_fold_rmse_control_ratios':ratios,'pre_bootstrap_passed':sum(x['all_four_directional'] for x in leave)>=3 and max(ratios)<=1.05}

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

# Undefined metrics fail adoption, rather than crashing an ablation export.
_original_aggregate=aggregate
_original_gate=gate
_original_stability=stability
def aggregate(ms):
    if all(all(x.get(k) is not None for k in ['rmse','r','chi2','hse_ts']) for x in ms):return _original_aggregate(ms)
    result={'n':sum(x['n'] for x in ms)}
    for key in ['rmse','r','chi2','hse_ts','mae']:
        v=[x.get(key) for x in ms]
        result[key]=None if any(x is None for x in v) else float(np.tanh(np.mean(np.arctanh(np.clip(v,-.999999,.999999))))) if key=='r' else float(np.mean(v))
    return result
def gate(m,controls):
    if any(x.get(k) is None for x in [m,*controls] for k in ['rmse','r','chi2','hse_ts']):return {'rmse':False,'r':False,'chi2':False,'hse_ts':False,'all_four':False,'undefined':True}
    return _original_gate(m,controls)
def stability(row,controls):
    if any(x.get(k) is None for r in [row,*controls] for x in r['fold_metrics'] for k in ['rmse','r','chi2','hse_ts']):return {'pre_bootstrap_passed':False,'undefined':True,'directional_passes':0}
    return _original_stability(row,controls)
