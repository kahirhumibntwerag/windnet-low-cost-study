"""Preregistered tabular WindNet study; stages never score Fold5 until final.

Training masks any lag measurement originating in a held-out fold. Predictors
at forecast time use only measurements at least three days before the target.
"""
import argparse, copy, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'study_inputs.npz'
if not DATA.exists():DATA=ROOT/'data/study_inputs.npz'
SOURCES=DATA.with_name('source_folds.npz')
D=np.load(DATA);S=np.load(SOURCES)
Y=D['y']; F=D['fold']; TRAINABLE=D['offset']<19
ALPHAS=[0.1,1,10,100,1000]

def score(y,p):
    r=float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-10 else None
    return {'rmse':float(np.sqrt(np.mean((y-p)**2))), 'mae':float(np.mean(np.abs(y-p))),'r':r,'n':len(y)}

def matrices(kind,fit,ev,forbidden):
    x=D[kind][fit].copy()
    src=S[kind][fit]
    mask=np.isin(src,forbidden)
    x[mask]=np.nan
    # Missing indicators are explicit and fitted within training data only.
    return x,D[kind][ev].copy(),int(mask.sum())

def ridge_grid(kind,fit,ev,forbidden):
    a,b,masked=matrices(kind,fit,ev,forbidden)
    rows=[];best=None
    for alpha in ALPHAS:
        model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
        model.fit(a,Y[fit]);p=model.predict(b);m=score(Y[ev],p)
        rows.append({'alpha':alpha,**m})
        if best is None or m['rmse']<best['metrics']['rmse']:
            best={'kind':kind,'alpha':alpha,'model':model,'pred':p,'metrics':m,'masked_training_values':masked}
    best['grid']=rows
    return best

def ridge_fixed(kind,alpha,fit,ev,forbidden):
    a,b,_=matrices(kind,fit,ev,forbidden)
    model=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
    model.fit(a,Y[fit]);return model.predict(b),model,a,b

def svr_fit(kernel,fit,ev):
    # Reproduce the paper's C/gamma/epsilon and training-only min-max scaling.
    lo=float(Y[fit].min());scale=float(Y[fit].max()-lo)
    x=(D['short'][fit]-lo)/scale;y=(Y[fit]-lo)/scale
    model=SVR(kernel=kernel,C=1e4,gamma=0.001,epsilon=0.1)
    model.fit(x,y)
    return model.predict((D['short'][ev]-lo)/scale)*scale+lo

def trees_grid(kind,fit,ev,forbidden):
    a,b,masked=matrices(kind,fit,ev,forbidden)
    rows=[];best=None
    for leaves in [7,15]:
        for leafsize in [30,60]:
            model=HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=leaves,
                min_samples_leaf=leafsize,l2_regularization=10,early_stopping=False,random_state=2026)
            model.fit(a,Y[fit]);p=model.predict(b);m=score(Y[ev],p)
            row={'leaves':leaves,'leafsize':leafsize,**m};rows.append(row)
            if best is None or m['rmse']<best['metrics']['rmse']:
                best={'kind':kind,'leaves':leaves,'leafsize':leafsize,'model':model,'pred':p,'metrics':m,'masked_training_values':masked}
    best['grid']=rows;return best

def tree_fixed(best,fit,ev,forbidden):
    a,b,_=matrices(best['kind'],fit,ev,forbidden)
    model=HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=best['leaves'],
        min_samples_leaf=best['leafsize'],l2_regularization=10,early_stopping=False,random_state=2026)
    model.fit(a,Y[fit]);return model.predict(b)

def neural(kind,fit,ev,forbidden,alpha,epochs=None,weight_decay=.1):
    import torch
    from torch import nn
    if not torch.cuda.is_available():raise RuntimeError('GPU experiment requires CUDA')
    torch.set_num_threads(2)
    device='cuda'
    base,basemodel,a,b=ridge_fixed(kind,alpha,fit,ev,forbidden)
    scaler=make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler())
    a=scaler.fit_transform(a);b=scaler.transform(b)
    x=torch.tensor(a,dtype=torch.float32,device=device);v=torch.tensor(b,dtype=torch.float32,device=device)
    base_train=basemodel.predict(D[kind][fit].copy())
    # Use exactly the masked fitting matrix for in-sample residuals as well.
    raw,_,_=matrices(kind,fit,ev,forbidden);base_train=basemodel.predict(raw)
    residual=torch.tensor((Y[fit]-base_train)/100,dtype=torch.float32,device=device)
    runs=[];predictions=[]
    for index,seed in enumerate([19,43,101]):
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=nn.Sequential(nn.Linear(x.shape[1],32),nn.SiLU(),nn.Dropout(.2),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1)).to(device)
        nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=weight_decay)
        bestloss=float('inf');beststate=None;bestepoch=0
        count=250 if epochs is None else int(epochs[index])
        for epoch in range(1,count+1):
            model.train();opt.zero_grad(set_to_none=True)
            loss=torch.mean((model(x).flatten()-residual)**2);loss.backward();opt.step()
            if epochs is None and (epoch%5==0 or epoch==1):
                model.eval()
                with torch.no_grad():p=base+model(v).flatten().cpu().numpy()*100
                value=score(Y[ev],p)['rmse']
                if value<bestloss:
                    bestloss=value;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch
        if epochs is None:model.load_state_dict(beststate)
        else:bestepoch=count
        model.eval()
        with torch.no_grad():p=base+model(v).flatten().cpu().numpy()*100
        predictions.append(p);runs.append({'seed':seed,'epochs':bestepoch,**score(Y[ev],p)})
    p=np.mean(predictions,axis=0)
    return {'pred':p,'metrics':score(Y[ev],p),'runs':runs,'epochs':[x['epochs'] for x in runs],
        'gpu_name':torch.cuda.get_device_name(0),'torch':torch.__version__,'weight_decay':weight_decay,'kind':kind,'alpha':alpha}

def clean(obj):
    return {k:v for k,v in obj.items() if k not in ['pred','model']}

def write(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')

def ensemble_grid(preds,y):
    from itertools import product
    names=list(preds);matrix=np.stack([preds[n] for n in names]);best=None;grid=[]
    for units in product(range(5),repeat=len(names)):
        if sum(units)!=4:continue
        weights=np.array(units)/4
        pred=weights@matrix;metrics=score(y,pred)
        row={'weights':dict(zip(names,weights.tolist())),**metrics};grid.append(row)
        if best is None or metrics['rmse']<best['rmse']:best=row
    return best,grid

def ensemble_development():
    fit=np.isin(F,[1,2,3])&TRAINABLE;ev=F==4;forbidden=[4,5]
    specs={kind+'_ridge':ridge_grid(kind,fit,ev,forbidden) for kind in ['recurrence','multi']}
    specs['multi_tree']=trees_grid('multi',fit,ev,forbidden)
    linear=specs['multi_ridge'];prior=json.loads((ROOT/'loop3.json').read_text());wd=prior['best']['weight_decay']
    net=neural('multi',fit,ev,forbidden,linear['alpha'],weight_decay=wd)
    preds={name:obj['pred'] for name,obj in specs.items()};preds['residual_nn']=net['pred']
    best,grid=ensemble_grid(preds,Y[ev]);best_single=min(score(Y[ev],p)['rmse'] for p in preds.values())
    output={'stage':'loop4','hypothesis':'Convex averaging of distinct errors improves over the best single model',
        'best':best,'best_single_rmse':best_single,'improvement_pct':100*(1-best['rmse']/best_single),
        'candidates':len(grid),'grid':grid,'fold5_scored':False}
    write('loop4.json',output);print(json.dumps(output),flush=True)

def development(stage):
    fit=np.isin(F,[1,2,3])&TRAINABLE;ev=F==4;forbidden=[4,5]
    if stage=='loop1':
        ridge=ridge_grid('short',fit,ev,forbidden)
        predictions={'training_mean':np.full(ev.sum(),Y[fit].mean()),'oldest_persistence':D['short'][ev,0],
            'latest_persistence':D['short'][ev,-1],'short_ridge':ridge['pred'],
            'linear_svr':svr_fit('linear',fit,ev),'rbf_svr':svr_fit('rbf',fit,ev)}
        results={name:score(Y[ev],p) for name,p in predictions.items()}
        output={'stage':stage,'split':'fit original Folds1-3; validation Fold4; Fold5 not scored',
            'short_ridge':clean(ridge),'models':results,'fit_n':int(fit.sum()),'validation_n':int(ev.sum())}
    elif stage=='loop2':
        results={}
        for kind in ['recurrence','multi']:
            results[kind+'_ridge']=clean(ridge_grid(kind,fit,ev,forbidden))
            results[kind+'_tree']=clean(trees_grid(kind,fit,ev,forbidden))
        prior=json.loads((ROOT/'loop1.json').read_text())
        bestname=min(results,key=lambda n:results[n]['metrics']['rmse'])
        output={'stage':stage,'split':'fit Folds1-3; validation Fold4; Fold5 not scored','models':results,
            'best':bestname,'rmse_improvement_over_short_ridge_pct':100*(1-results[bestname]['metrics']['rmse']/prior['models']['short_ridge']['rmse'])}
    else:
        prior=json.loads((ROOT/'loop2.json').read_text())
        linear=min([n for n in prior['models'] if n.endswith('_ridge')],key=lambda n:prior['models'][n]['metrics']['rmse'])
        spec=prior['models'][linear];base=spec['metrics']['rmse']
        bp,_,_,_=ridge_fixed(spec['kind'],spec['alpha'],fit,ev,forbidden)
        nets=[]
        for wd in [.1,1.0]:
            net=neural(spec['kind'],fit,ev,forbidden,spec['alpha'],weight_decay=wd)
            blends=[]
            for w in [0,.25,.5,.75,1.0]:
                blends.append({'weight':w,**score(Y[ev],bp+w*(net['pred']-bp))})
            net['blends']=blends;net['best_blend']=min(blends,key=lambda x:x['rmse']);nets.append(clean(net))
        best=min(nets,key=lambda n:n['best_blend']['rmse'])
        output={'stage':stage,'linear_reference':linear,'linear_rmse':base,'networks':nets,'best':best,
            'rmse_improvement_pct':100*(1-best['best_blend']['rmse']/base),'fold5_scored':False}
    write(stage+'.json',output);print(json.dumps(output),flush=True)

def final():
    rows=[];selections=[];stage3=json.loads((ROOT/'loop3.json').read_text());wd=stage3['best']['weight_decay']
    for outer in range(1,6):
        inner=4 if outer==5 else 5
        fit=(F!=outer)&(F!=inner)&TRAINABLE;val=F==inner;test=F==outer;allfit=(F!=outer)&TRAINABLE
        forbid=[outer,inner];specs={}
        for kind in ['short','recurrence','multi']:
            specs[kind+'_ridge']=ridge_grid(kind,fit,val,forbid)
        for kind in ['recurrence','multi']:
            specs[kind+'_tree']=trees_grid(kind,fit,val,forbid)
        bestlinear=min([n for n in specs if n.endswith('_ridge')],key=lambda n:specs[n]['metrics']['rmse'])
        bs=specs[bestlinear]
        net=neural(bs['kind'],fit,val,forbid,bs['alpha'],weight_decay=wd)
        blends=[{'weight':w,**score(Y[val],bs['pred']+w*(net['pred']-bs['pred']))} for w in [0,.25,.5,.75,1]]
        weight=min(blends,key=lambda x:x['rmse'])['weight']
        predictions={'training_mean':np.full(test.sum(),Y[allfit].mean()),
            'oldest_persistence':D['short'][test,0],'latest_persistence':D['short'][test,-1],
            'linear_svr':svr_fit('linear',allfit,test),'rbf_svr':svr_fit('rbf',allfit,test)}
        for name,spec in specs.items():
            predictions[name]=ridge_fixed(spec['kind'],spec['alpha'],allfit,test,[outer])[0] if name.endswith('_ridge') else tree_fixed(spec,allfit,test,[outer])
        fullnet=neural(bs['kind'],allfit,test,[outer],bs['alpha'],epochs=net['epochs'],weight_decay=wd)
        predictions['residual_nn']=fullnet['pred']
        predictions['ridge_nn_blend']=predictions[bestlinear]+weight*(fullnet['pred']-predictions[bestlinear])
        epreds={name:specs[name]['pred'] for name in ['recurrence_ridge','multi_ridge','multi_tree']}
        # H4 fixes the neural branch to the multivariate ridge baseline.
        if bestlinear=='multi_ridge':enet=net;efull=fullnet
        else:
            mb=specs['multi_ridge'];enet=neural('multi',fit,val,forbid,mb['alpha'],weight_decay=wd)
            efull=neural('multi',allfit,test,[outer],mb['alpha'],epochs=enet['epochs'],weight_decay=wd)
        epreds['residual_nn']=enet['pred']
        ebest,_=ensemble_grid(epreds,Y[val])
        predictions['convex_ensemble']=sum(w*(efull['pred'] if name=='residual_nn' else predictions[name]) for name,w in ebest['weights'].items())
        bestclassic=min(specs,key=lambda name:specs[name]['metrics']['rmse'])
        blendscore=min(blends,key=lambda x:x['rmse'])['rmse']
        # Neural complexity must earn a 2% validation improvement.
        selected='ridge_nn_blend' if blendscore<.98*specs[bestclassic]['metrics']['rmse'] else bestclassic
        current_inner=blendscore if selected=='ridge_nn_blend' else specs[bestclassic]['metrics']['rmse']
        # The final hypothesis requires a 1% validation gain over the current rule.
        if ebest['rmse']<.99*current_inner:selected='convex_ensemble'
        predictions['selected_model']=predictions[selected]
        selected_data={'outer_fold':outer,'inner_fold':inner,'selected':selected,'best_linear':bestlinear,'blend_weight':weight,
            'neural_epochs':net['epochs'],'weight_decay':wd,'ensemble_weights':ebest['weights'],
            'inner_metrics':{k:clean(v) for k,v in specs.items()}}
        selections.append(selected_data)
        for pos,index in enumerate(np.flatnonzero(test)):
            rows.append({'row':int(index),'fold':outer,'block':int(D['block'][index]),'offset':int(D['offset'][index]),
                'date':str(D['date'][index]),'y':float(Y[index]),**{k:float(p[pos]) for k,p in predictions.items()}})
        print(json.dumps({'outer_fold_complete':outer,'rows':int(test.sum()),'selected_on_inner':selected}),flush=True)
    # Score all outer predictions only after every method and prediction is fixed.
    import csv
    with (ROOT/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    names=[k for k in rows[0] if k not in ['row','fold','block','offset','date','y']]
    results={}
    for name in names:
        metrics=[]
        for outer in range(1,6):
            subset=[r for r in rows if r['fold']==outer]
            metrics.append({'fold':outer,**score(np.array([r['y'] for r in subset]),np.array([r[name] for r in subset]))})
        corrs=[m['r'] for m in metrics if m['r'] is not None]
        results[name]={'folds':metrics,'mean_rmse':float(np.mean([m['rmse'] for m in metrics])),
            'rmse_se':float(np.std([m['rmse'] for m in metrics],ddof=1)/np.sqrt(5)),
            'mean_mae':float(np.mean([m['mae'] for m in metrics])),
            'fisher_mean_r':float(np.tanh(np.mean(np.arctanh(corrs)))) if corrs else None}
    # The untouched original Fold5 is the confirmatory comparison.
    sub=[r for r in rows if r['fold']==5];blocks=sorted(set(r['block'] for r in sub));rng=np.random.default_rng(20261029)
    # Original blocks overlap within a fold. Keep overlapping intervals together
    # rather than treating duplicate observations as independent bootstrap units.
    intervals=[]
    for b in blocks:
        bd=[np.datetime64(r['date']) for r in sub if r['block']==b]
        intervals.append((min(bd)-np.timedelta64(6,'D'),max(bd),b))
    clusters=[];ends=[]
    for start,end,b in sorted(intervals):
        if clusters and start<=ends[-1]:
            clusters[-1].append(b);ends[-1]=max(ends[-1],end)
        else:clusters.append([b]);ends.append(end)
    cluster_index=[np.array([i for i,r in enumerate(sub) if r['block'] in cluster]) for cluster in clusters]
    picked=rng.integers(0,len(clusters),size=(5000,len(clusters)))
    sample_counts=np.array([len(ix) for ix in cluster_index])[picked].sum(axis=1)
    bootstrap={}
    for reference in ['short_ridge','linear_svr','recurrence_ridge']:
        sums_p=np.array([sum((sub[i]['y']-sub[i]['selected_model'])**2 for i in ix) for ix in cluster_index])
        sums_q=np.array([sum((sub[i]['y']-sub[i][reference])**2 for i in ix) for ix in cluster_index])
        differences=np.sqrt(sums_p[picked].sum(axis=1)/sample_counts)-np.sqrt(sums_q[picked].sum(axis=1)/sample_counts)
        bootstrap[reference]={'difference_rmse_selected_minus_reference_ci95':np.quantile(differences,[.025,.975]).tolist(),
            'resamples':5000,'unit':'clusters of overlapping released 20-day partitions','clusters':len(clusters)}
    metadata={'python':platform.python_version(),'numpy':np.__version__,'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_fold_isolation':'Fit inputs mask measurements from all held-out folds; evaluation inputs use causal measurements only.',
        'primary_holdout':5,'five_fold_status':'exploratory: architecture development used original Folds1-4','published_windnet_211_H4_D3':{'r':.55,'rmse':80.28,'reference_only':True}}
    write('final_results.json',{'metrics':results,'selections':selections,'confirmatory_bootstrap':bootstrap,'metadata':metadata})
    print('FINAL_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['loop1','loop2','loop3','final'],required=True);args=p.parse_args()
    begin=time.time()
    if (ROOT/'ensemble_mode.json').exists():
        (ROOT/'ensemble_mode.json').unlink();ensemble_development()
    elif args.stage=='final':final()
    else:development(args.stage)
    print(json.dumps({'stage':args.stage,'seconds':time.time()-begin}),flush=True)
