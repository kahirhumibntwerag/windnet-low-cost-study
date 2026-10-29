"""Post-evaluation exploratory feature ablations, with no primary-model edits."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from study import D,S,Y,F,TRAINABLE,ALPHAS,score

VARIANTS={
    'short_4':list(range(4)),
    'recent_only':list(range(18)),
    'recent_plus_27':list(range(26)),
    'recent_plus_54':list(range(18))+list(range(26,34)),
    'speed_recurrence':list(range(34)),
    'full':list(range(78)),
    'no_27_band':list(range(18))+list(range(26,78)),
    'no_54_band':list(range(26))+list(range(34,78)),
    'no_magnetic':list(range(34))+list(range(58,78)),
    'no_plasma':list(range(58))
}

def matrix(columns,fit,ev,forbidden):
    a=D['multi'][fit][:,columns].copy();src=S['multi'][fit][:,columns]
    a[np.isin(src,forbidden)]=np.nan
    return a,D['multi'][ev][:,columns]

def fit_predict(a,b,y,alpha):
    model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
    model.fit(a,y);return model.predict(b)

def main():
    begin=time.time();results={}
    for name,columns in VARIANTS.items():
        folds=[]
        for outer in range(1,6):
            inner=4 if outer==5 else 5
            fit=(F!=outer)&(F!=inner)&TRAINABLE;val=F==inner
            a,b=matrix(columns,fit,val,[outer,inner])
            grid=[{'alpha':alpha,**score(Y[val],fit_predict(a,b,Y[fit],alpha))} for alpha in ALPHAS]
            alpha=min(grid,key=lambda item:item['rmse'])['alpha']
            fit=(F!=outer)&TRAINABLE;test=F==outer
            a,b=matrix(columns,fit,test,[outer]);pred=fit_predict(a,b,Y[fit],alpha)
            folds.append({'fold':outer,'alpha':alpha,**score(Y[test],pred),'inner_grid':grid})
        results[name]={'features':len(columns),'folds':folds,'mean_rmse':float(np.mean([f['rmse'] for f in folds])),
            'rmse_se':float(np.std([f['rmse'] for f in folds],ddof=1)/np.sqrt(5)),
            'fisher_mean_r':float(np.tanh(np.mean(np.arctanh([f['r'] for f in folds]))))}
    for name,result in results.items():result['delta_rmse_vs_full']=result['mean_rmse']-results['full']['mean_rmse']
    # Verify unchanged variants reproduce their original results exactly.
    original=json.loads(Path('results/final_results.json').read_text())['metrics']
    for ablation,previous in [('short_4','short_ridge'),('speed_recurrence','recurrence_ridge'),('full','multi_ridge')]:
        assert abs(results[ablation]['mean_rmse']-original[previous]['mean_rmse'])<1e-7
    output={'status':'exploratory; planned after primary holdout was opened','model_family':'ridge for every row',
        'variants':results,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.time()-begin}
    Path('results/ablations.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:{'features':v['features'],'rmse':v['mean_rmse'],'delta':v['delta_rmse_vs_full'],'fold5_rmse':v['folds'][4]['rmse']} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
