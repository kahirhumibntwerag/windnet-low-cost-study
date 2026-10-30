"""Registered prediction-only convex routing; no fitting or label input."""
import numpy as np
from scipy.special import expit

def panel():
    rows=[]
    for g in [.125,.25,.5,.75]:
        for share in [.5,1]:
            rows.append({'model':f'M15_static_g{g}_s{share}','recipe':{'kind':'static','event_share':g,'incumbent_mix':share}})
    for threshold in [400,450,500,550]:
        for temperature in [25,50,100]:
            for maximum in [.25,.5,.75,1]:
                for share in [.5,1]:
                    rows.append({'model':f'M15_conditional_t{threshold}_u{temperature}_m{maximum}_s{share}',
                                 'recipe':{'kind':'conditional','threshold':threshold,'temperature':temperature,'max_share':maximum,'incumbent_mix':share}})
    assert len(rows)==104 and len({r['model'] for r in rows})==104
    return rows

def gate_weights(event_predictions,recipe):
    values=np.asarray(event_predictions,dtype=float)
    assert values.ndim==1 and np.isfinite(values).all()
    if recipe['kind']=='static':
        assert 0<=recipe['event_share']<=1
        return np.full(len(values),recipe['event_share'])
    assert recipe['kind']=='conditional' and recipe['temperature']>0 and 0<=recipe['max_share']<=1
    return recipe['max_share']*expit((values-recipe['threshold'])/recipe['temperature'])

def predict(experts,recipe):
    assert set(experts)=={'A','B','incumbent'},'Only forecast vectors may enter routing'
    a,b,inc=[np.asarray(experts[k],dtype=float) for k in ['A','B','incumbent']]
    assert a.ndim==1 and a.shape==b.shape==inc.shape and np.isfinite([a,b,inc]).all()
    share=recipe['incumbent_mix'];assert 0<=share<=1
    g=gate_weights(b,recipe)
    routed=a+g*(b-a);p=(1-share)*inc+share*routed
    assert np.isfinite(p).all()
    assert np.all(p>=np.minimum(np.minimum(a,b),inc)-1e-10) and np.all(p<=np.maximum(np.maximum(a,b),inc)+1e-10)
    return p
