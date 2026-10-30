"""Matched aggregate-only scoring. Calendar gaps never become adjacent HSE days."""
import numpy as np
from matched.hse_original import BatchwiseHSE

METRICS=('mean_rmse','fisher_r','mean_chi2','mean_hse_ts')
DIRECTIONS=np.array([-1,1,-1,1])

def bootstrap_units(dates,kept):
    years=np.array([str(dates[ix[0]])[:4] for ix in kept])
    original_units=np.array([int(ix[0])//28 for ix in kept])
    return [[np.flatnonzero((years==year)&(original_units==unit)) for unit in np.unique(original_units[years==year])] for year in np.unique(years)]

def replica_moments(y,p,sigma):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    assert y.ndim==1 and p.shape==(5,len(y)) and np.isfinite(y).all() and np.isfinite(p).all()
    valid=np.isfinite(sigma)&(sigma>0)
    counts=np.array([BatchwiseHSE(np.column_stack([v,y]),4,3) for v in p],dtype=np.float64)
    err=p-y
    return np.stack([np.full(5,len(y)),np.sum(err**2,axis=1),np.sum(p,axis=1),np.sum(p**2,axis=1),
                     np.full(5,np.sum(y)),np.full(5,np.sum(y**2)),np.sum(p*y,axis=1),
                     np.full(5,np.sum(valid)),np.sum((err[:,valid]/sigma[valid])**2,axis=1),
                     counts[:,0],counts[:,1],counts[:,2]],axis=-1)

def summarize(moments):
    total=np.sum(moments,axis=0);n=total[:,0]
    if not np.all(n>0):return [None]*4
    rmse=np.sqrt(total[:,1]/n)
    variance=(total[:,3]-total[:,2]**2/n)*(total[:,5]-total[:,4]**2/n)
    r=np.divide(total[:,6]-total[:,2]*total[:,4]/n,np.sqrt(np.maximum(variance,0)),out=np.full(5,np.nan),where=variance>0)
    chi=np.divide(total[:,8],total[:,7],out=np.full(5,np.nan),where=total[:,7]>0)
    denominator=total[:,9:12].sum(axis=1)
    ts=np.divide(total[:,9],denominator,out=np.full(5,np.nan),where=denominator>0)
    values=[np.mean(rmse),np.tanh(np.mean(np.arctanh(np.clip(r,-.999999999,.999999999)))),np.mean(chi),np.mean(ts)]
    return [float(v) if np.isfinite(v) else None for v in values]

def report(moments):
    values=summarize(moments);total=np.sum(moments,axis=0)
    return {**dict(zip(METRICS,values)), 'unique_target_days':int(total[0,0]),'positive_sigma_days':int(total[0,7]),
            'replicas':5,'replica_scores_averaged':True,
            'hse_counts_per_replica':[{'tp_original':int(r[9]),'fp_original':int(r[10]),'fn_original':int(r[11])} for r in total]}

def evaluate(dates,y,sigma,predictions,eligible):
    assert set(predictions)=={'original_windnet','M13'}
    assert all(v.shape==(5,len(dates)) for v in predictions.values())
    block=np.arange(len(dates))//14
    groups=[np.flatnonzero(block==b) for b in np.unique(block)]
    kept=[ix for ix in groups if np.all(eligible[ix])]
    assert kept, 'No complete common blocks'
    banks={name:np.stack([replica_moments(y[ix],p[:,ix],sigma[ix]) for ix in kept]) for name,p in predictions.items()}
    scores={name:report(m) for name,m in banks.items()}
    a=np.array(summarize(banks['M13']),dtype=float);b=np.array(summarize(banks['original_windnet']),dtype=float)
    point={k:bool(np.isfinite(d) and d*s>0) for k,d,s in zip(METRICS,a-b,DIRECTIONS)}
    # Calendar14-day blocks are indivisible. Assign boundary-spanning blocks by
    # their start year, then pair neighboring blocks into28-day bootstrap units.
    units=bootstrap_units(dates,kept)
    rng=np.random.default_rng(20261030);deltas=[]
    for _ in range(2000):
        ix=np.concatenate([np.concatenate([u[j] for j in rng.integers(0,len(u),len(u))]) for u in units])
        x=np.array(summarize(banks['M13'][ix]),dtype=float);z=np.array(summarize(banks['original_windnet'][ix]),dtype=float)
        deltas.append(x-z)
    deltas=np.asarray(deltas)
    intervals={}
    for i,key in enumerate(METRICS):
        finite=deltas[:,i][np.isfinite(deltas[:,i])]
        intervals[key]={'M13_minus_original_delta_95':np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,'defined_draws':len(finite)}
    yearly={}
    for year in range(2020,2026):
        chunks=[ix[np.array([str(d)[:4]==str(year) for d in dates[ix]])] for ix in kept]
        chunks=[ix for ix in chunks if len(ix)]
        yearly[str(year)]={name:report(np.stack([replica_moments(y[ix],p[:,ix],sigma[ix]) for ix in chunks])) if chunks else None for name,p in predictions.items()}
    return {'overall':scores,'yearly':yearly,'strict_all_four_numerical_win':all(point.values()),'metric_point_checks':point,
            'paired_uncertainty':{'draws':2000,'seed':20261030,'metrics':intervals,'method':'paired calendar28-day units within start-year; original14-day HSE boundaries preserved; final partial units retained',
                                  'limitations':'Conditional on two frozen models and this solar-cycle period. No replicas or overlapping dates counted as independent samples. Partial annual HSE boundary blocks are retained.'},
            'coverage':{'requested_target_days':len(dates),'retained_common_blocks':len(kept),'requested_blocks':len(groups),
                        'retained_unique_target_days':sum(len(ix) for ix in kept),'incomplete_common_blocks_excluded':len(groups)-len(kept)}}
