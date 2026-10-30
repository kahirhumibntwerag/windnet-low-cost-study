"""Fixed F3 recipes and fitting-only regional head; no test labels accepted."""
import copy
import json
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from next_loop.regional_features import temporal
from next_loop.target_weights import target_weights

M10 = {'family':'affine_location_mix',
       'location':{'family':'location_ridge','representation':'ratio','view':'flatten','weight_lambda':4,'alpha':10000},
       'location_weight':.625,'prior':'M6','gain':1.2,'bias':-5}
REGIONAL = {'grid':[6,6],'view':'trend','encoding':'linear',
            'precision_lambda':4,'ridge_alpha':100,'target_weight':'fast2'}

def fixed_panel():
    primary={'family':'regional_mix','regional':copy.deepcopy(REGIONAL),
             'new_head_share':.25,'gain':1.2,'bias':-5,'prior':copy.deepcopy(M10)}
    panel=[{'model':'PRIMARY_F3','recipe':primary},
           {'model':'windnet_port','recipe':{'family':'port'}},
           {'model':'prior_M6','recipe':{'family':'prior'}},
           {'model':'prior_M10','recipe':copy.deepcopy(M10)}]
    for name,changes in [('regional_only',{'new_head_share':1}),
                         ('regional_target_unit',{'target_weight':'unit'}),
                         ('regional_precision_zero',{'precision_lambda':0}),
                         ('regional_global_1x1',{'grid':[1,1]}),
                         ('without_outer_calibration',{'gain':1.,'bias':0.})]:
        recipe=copy.deepcopy(primary)
        for key,value in changes.items():
            if key in recipe:recipe[key]=value
            else:recipe['regional'][key]=value
        if name=='without_outer_calibration':recipe['prior'].update(gain=1.,bias=0.)
        panel.append({'model':name,'recipe':recipe})
    panel.append({'model':'released_windnet','recipe':{'family':'released_reference'}})
    return panel

def calibrate(raw,mean,gain,bias):
    return mean+gain*(raw-mean)+bias

def fit_regional(train,features,recipe):
    assert set(features)=={'regions'},'Prediction input must contain only regional image features'
    x=temporal(train['regions'],recipe['view'],recipe['encoding'])
    v=temporal(features['regions'],recipe['view'],recipe['encoding'])
    scaler=StandardScaler().fit(x);scaler.scale_=np.maximum(scaler.scale_,.01)
    x=scaler.transform(x);v=scaler.transform(v)
    sig=train['sigma'][train['sigma']>0]
    assert len(sig)>0
    floor=float(np.quantile(sig,.05));med=float(np.median(sig))
    weights=1+recipe['precision_lambda']*(med/np.maximum(train['sigma'],floor))**2
    weights/=weights.mean()
    if recipe['target_weight']!='unit':
        weights*=target_weights(train['y'],recipe['target_weight']);weights/=weights.mean()
    model=Ridge(alpha=recipe['ridge_alpha'],solver='cholesky').fit(x,train['y'],sample_weight=weights)
    pred=model.predict(v)
    bundle={'ridge':model,'scaler':scaler,'recipe':recipe,'target_mean':float(train['y'].mean()),
            'sigma_floor_fit_only':floor,'sigma_median_fit_only':med,
            'normalized_weight_mean':float(weights.mean())}
    assert pred.shape==(len(v),) and np.isfinite(pred).all()
    return pred,bundle

def metric_gate(left,right):
    directions={'mean_rmse':-1,'fisher_r':1,'mean_chi2':-1,'mean_hse_ts':1}
    checks={k:bool(left.get(k) is not None and right.get(k) is not None
                   and np.isfinite(left[k]) and np.isfinite(right[k])
                   and sign*(left[k]-right[k])>0) for k,sign in directions.items()}
    return {'all_four':all(checks.values()),'checks':checks}

def aggregate(rows):
    def avg(key,fisher=False):
        a=[r.get(key) for r in rows]
        if any(x is None or not np.isfinite(x) for x in a):return None
        return float(np.tanh(np.mean(np.arctanh(np.clip(a,-.999999999,.999999999))))) if fisher else float(np.mean(a))
    r=[x['rmse'] for x in rows]
    return {'mean_rmse':avg('rmse'),'fisher_r':avg('r',True),'mean_chi2':avg('chi2'),
            'mean_hse_ts':avg('hse_ts'),'mean_mae':avg('mae'),'n':sum(x['n'] for x in rows),
            'rmse_se':float(np.std(r,ddof=1)/np.sqrt(len(r))) if all(x is not None for x in r) else None}
