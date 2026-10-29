"""Training-bank-only science; never import prior final outputs or forecasts."""
import copy,json,platform,time
from pathlib import Path
import numpy as np
import torch
from matched.runner import ROOT,DATA,EXPORT,save_json,sha
from matched.science import (blocks,embed_blocks,samples,metrics,rowhash,neural_fit,WindNetPort,
    fit_tabular,DEVICE,SEEDS)

EPOCHS=[38,27,36]
def identity():
    return {'phase_code':{p.name:sha(p.read_bytes()) for p in sorted((ROOT/'event_loop').glob('*.py'))},'frozen_matched_sources':{p.name:sha(p.read_bytes()) for p in sorted((ROOT/'matched').glob('*.py'))},'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':DEVICE}
def aggregate(ms):
    rs=[x['r'] for x in ms];ts=[x['hse_ts'] for x in ms]
    return {'n':int(sum(x['n'] for x in ms)),'rmse':float(np.mean([x['rmse'] for x in ms])),'r':float(np.tanh(np.mean(np.arctanh(np.clip(rs,-.999999,.999999))))),'chi2':float(np.mean([x['chi2'] for x in ms])),'hse_ts':float(np.mean(ts)) if all(x is not None for x in ts) else None,'mae':float(np.mean([x['mae'] for x in ms]))}
def gate(m,controls):
    anchors={k:([c[k] for c in controls]) for k in ['rmse','r','chi2','hse_ts']}
    if m['hse_ts'] is None or any(x is None for x in anchors['hse_ts']):return {'all_four':False,'undefined':True}
    tests={'rmse':m['rmse']<=.99*min(anchors['rmse']),'r':m['r']>=max(anchors['r'])+.005,'chi2':m['chi2']<=.995*min(anchors['chi2']),'hse_ts':m['hse_ts']>=max(anchors['hse_ts'])+.02}
    return {**tests,'all_four':all(tests.values()),'anchors':{k:(max(v) if k in ['r','hse_ts'] else min(v)) for k,v in anchors.items()}}
def prepare():
    bs=blocks(False);assert len(bs)==108 and all(b['fold']!=5 for b in bs);assert not (DATA/'sealed').exists()
    audits=[]
    for held in range(1,5):
        fit=[f for f in range(1,5) if f!=held];path=DATA/f'bank_cv{held}.npz';audit=embed_blocks(bs,fit,path);audits.append(audit)
    save_json(EXPORT/'event_embedding_audit.json',{'splits':audits,'fold5_available':False,'prior_test_results_available':False,'identity':identity()})
    print('Four training-bank embedding splits ready; no Fold5 data',flush=True)
def split(held):
    d=np.load(DATA/f'bank_cv{held}.npz');tr,va,norm=samples(d,[f for f in range(1,5) if f!=held],[held])
    for a in [tr,va]:
        a['future_y']=np.stack([d['output'][int(b),int(o)-3:int(o)+1,11] for b,o in zip(a['block'],a['offset'])])
        a['future_sigma']=np.stack([d['output'][int(b),int(o)-3:int(o)+1,14] for b,o in zip(a['block'],a['offset'])])
        assert np.array_equal(a['future_y'][:,-1],a['y'])
    return tr,va,norm
def save_predictions(name,folds):
    out={}
    for held,p in folds.items():out['cv'+str(held)]=p
    np.savez_compressed(DATA/(name+'.npz'),**out)
def baseline():
    ms={'bank_windnet_port':[],'bank_prior_M6':[]};preds={k:{} for k in ms};audits=[];t=time.time();(DATA/'models').mkdir(exist_ok=True)
    for held in range(1,5):
        tr,va,norm=split(held);features={k:va[k] for k in ['energy','spatial']};pp=[]
        for seed,epochs in zip(SEEDS,EPOCHS):
            p,state,_=neural_fit(tr,features,seed,fixed_epochs=epochs);torch.save({**state,'normalization':norm},DATA/f'models/control_cv{held}_{seed}.pt');pp.append(p)
        port=np.mean(pp,axis=0);ridge,_=fit_tabular(tr,features,{'family':'ridge','view':'flatten','source':'combined','alpha':10000});mean=float(tr['y'].mean());old=mean+1.2*(.5*port+.5*ridge-mean)
        for name,p in [('bank_windnet_port',port),('bank_prior_M6',old)]:ms[name].append(metrics(va['y'],p,va['sigma']));preds[name][held]=p
        audits.append({'held_training_fold':held,'fit_rows':len(tr['y']),'validation_rows':len(va['y']),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_available':False})
        print(json.dumps({'control_training_split_complete':held,'seconds':time.time()-t}),flush=True)
    rows=[{'model':k,'metrics':aggregate(v),'fold_metrics':v} for k,v in ms.items()]
    for name,pp in preds.items():save_predictions(name,pp)
    save_json(EXPORT/'event_controls.json',{'stage':'M7_controls','scope':'Four-fold training-bank CV only; Folds1-4','rows':rows,'audits':audits,'identity':identity(),'seconds':time.time()-t,'original_final_evaluations':0})
    print(json.dumps({'controls_complete':rows}),flush=True)
def train_aligned(tr,features,seed,epochs,aux_mode,aux_weight,gradient_weight=0.,fast_weight=1.,hidden=None,dropout=.7):
    torch.manual_seed(seed);np.random.seed(seed);model=WindNetPort(hidden=hidden or 400,dropout=dropout).to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.0005)
    ymin=float(tr['y'].min());yscale=float(np.ptp(tr['y']));x=torch.tensor(tr['energy'],dtype=torch.float32,device=DEVICE);y=torch.tensor((tr['y']-ymin)/yscale,dtype=torch.float32,device=DEVICE);seq=torch.tensor((tr['future_y']-ymin)/yscale,dtype=torch.float32,device=DEVICE)
    floor=float(np.quantile(tr['sigma'][tr['sigma']>0],.05));sig=torch.tensor(np.maximum(tr['sigma'],floor)/yscale,dtype=torch.float32,device=DEVICE);ss=torch.tensor(np.maximum(tr['future_sigma'],floor)/yscale,dtype=torch.float32,device=DEVICE)
    ww=np.where(tr['y']>=500,fast_weight,1.);ww=ww/ww.mean();weight=torch.tensor(ww,dtype=torch.float32,device=DEVICE)
    for epoch in range(epochs):
        model.train()
        for ix in torch.randperm(len(x),device=DEVICE).split(128):
            out=model(x[ix]);err=out[:,-1]-y[ix];loss=(weight[ix]*(err**2+(err/sig[ix])**2)).mean()
            ae=out-seq[ix]
            if aux_mode=='chi':ae=ae/ss[ix]
            loss=loss+aux_weight*(weight[ix,None]*ae**2).mean()
            if gradient_weight:
                de=(out[:,1:]-out[:,:-1])-(seq[ix,1:]-seq[ix,:-1]);loss=loss+gradient_weight*(weight[ix,None]*de**2).mean()
            loss=loss+1e-6*sum((p**2).sum() for p in model.parameters());opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
    model.eval()
    with torch.inference_mode():p=model(torch.tensor(features['energy'],dtype=torch.float32,device=DEVICE))[:,-1].cpu().numpy()*yscale+ymin
    assert np.isfinite(p).all()
    state={'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'ymin':ymin,'yscale':yscale,'seed':seed,'epochs':epochs,'aux_mode':aux_mode,'aux_weight':aux_weight,'gradient_weight':gradient_weight,'fast_weight':fast_weight,'hidden':hidden or 400,'dropout':dropout,'normalized_weight_mean':float(ww.mean())}
    return p,state
def aligned():
    controls=json.loads((EXPORT/'event_controls.json').read_text());cs=[r['metrics'] for r in controls['rows']];rows=[];t=time.time();audit=[]
    for mode in ['mse','chi']:
        for weight in [.1,1.]:
            name=f'aligned_{mode}_w{weight}';fold_metrics=[];preds={}
            for held in range(1,5):
                tr,va,norm=split(held);features={k:va[k] for k in ['energy','spatial']};pp=[]
                for seed,epochs in zip(SEEDS,EPOCHS):
                    p,state=train_aligned(tr,features,seed,epochs,mode,weight);torch.save({**state,'normalization':norm},DATA/f'models/{name}_cv{held}_{seed}.pt');pp.append(p)
                p=np.mean(pp,axis=0);preds[held]=p;fold_metrics.append(metrics(va['y'],p,va['sigma']))
                audit.append({'model':name,'held_training_fold':held,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va)})
            m=aggregate(fold_metrics);recipe={'family':'aligned_lstm','aux_mode':mode,'aux_weight':weight,'epochs':EPOCHS,'seeds':SEEDS};row={'model':name,'recipe':recipe,'metrics':m,'fold_metrics':fold_metrics,'gate':gate(m,cs)};rows.append(row);save_predictions(name,preds);print(json.dumps({'aligned_condition_complete':row,'seconds':time.time()-t}),flush=True)
    eligible=[r for r in rows if r['gate']['all_four']];selected=min(eligible,key=lambda r:r['metrics']['rmse']) if eligible else None
    save_json(EXPORT/'m7_results.json',{'stage':'M7','scope':'Four-fold training-bank CV only; Folds1-4','rows':rows,'controls':controls['rows'],'selected':selected,'gate_passed':bool(eligible),'row_audits':audit,'identity':identity(),'seconds':time.time()-t,'original_final_evaluations':0,'fold5_available':False})
def run(cfg):
    stage=cfg['stage']
    if stage=='prepare':prepare()
    elif stage=='baseline':baseline()
    elif stage=='aligned':aligned()
    else:raise ValueError('Stage not implemented '+stage)
