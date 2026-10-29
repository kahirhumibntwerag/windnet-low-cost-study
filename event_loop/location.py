"""M9: fixed spatial pooling, training-bank-only weighted ridge ablations."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from matched.runner import DATA,EXPORT,save_json,sha
from matched.science import KaffeFeatures,blocks,DEVICE,metrics,rowhash
from event_loop.science import split,aggregate,gate,identity,save_predictions
from event_loop.mixtures import bootstrap

class LocationFeatures(KaffeFeatures):
    def forward(self,x):
        vals={'data':x}
        for name,op,ins,args,kw in self.graph:
            v=[vals[z] for z in ins]
            if op=='conv':
                kh,kwd,_,sh,sw=args;assert kh==kwd and sh==sw
                a=self.same(v[0],kh,sh) if kw.get('padding','SAME')=='SAME' else v[0]
                w,b=self.weights[name];z=F.conv2d(a,w,b,stride=(sh,sw));z=F.relu(z) if kw.get('relu',True) else z
            elif op=='max_pool':
                kh,kwd,sh,sw=args;assert kh==kwd and sh==sw
                a=self.same(v[0],kh,sh,float('-inf')) if kw.get('padding','SAME')=='SAME' else v[0];z=F.max_pool2d(a,kh,sh)
            elif op=='lrn':
                radius,alpha,beta=args;size=2*radius+1;z=F.local_response_norm(v[0],size,alpha=alpha*size,beta=beta,k=kw.get('bias',1.))
            elif op=='concat':z=torch.cat(v,dim=1)
            else:raise ValueError(op)
            vals[name]=z
        local=torch.cat([F.adaptive_avg_pool2d(vals[n]**2,(4,4)) for n in self.targets],dim=1)
        glob=torch.cat([(vals[n]**2).sum(dim=(2,3)) for n in self.targets],dim=1)
        return local.flatten(2),glob

def representation(local,glob,kind):
    z=np.log1p(local) if kind=='log' else local/(local.mean(axis=-1,keepdims=True)+1e-8)
    return np.concatenate([z.reshape(*z.shape[:-2],-1),np.log1p(glob)],axis=-1).astype(np.float32)

def prepare():
    bs=blocks(False);assert len(bs)==108 and all(b['fold']!=5 for b in bs);assert not (DATA/'sealed').exists()
    model=LocationFeatures().eval();audits=[]
    for held in range(1,5):
        path=DATA/f'location_cv{held}.npz'
        if path.exists():continue
        fit=[b for b in bs if b['fold']!=held];mean=float(sum(b['images'].sum(dtype=np.float64) for b in fit)/(len(fit)*20*256*256));local=[];glob=[]
        with torch.inference_mode():
            for i,b in enumerate(bs):
                ims=torch.from_numpy(b['images']).to(DEVICE)[:,None];ims=F.interpolate(ims,size=(224,224),mode='bilinear',align_corners=False).repeat(1,3,1,1)-mean
                a,g=model(ims);local.append(a.cpu().numpy());glob.append(g.cpu().numpy())
                if i%30==0:print(json.dumps({'location_embedding_fold':held,'blocks':i+1}),flush=True)
        np.savez_compressed(path,local=np.stack(local),glob=np.stack(glob),image_mean=mean)
        original=np.load(DATA/f'bank_cv{held}.npz');assert np.allclose(np.stack(glob),original['energy'],rtol=1e-5,atol=.01)
        audits.append({'held_training_fold':held,'image_mean_fit_only':mean,'cache_sha256':sha(path.read_bytes()),'global_energy_matches_original':True})
    save_json(EXPORT/'m9_embedding_audit.json',{'audits':audits,'fold5_available':False,'identity':identity()})

def temporal(x,kind):
    if kind=='mean':return x.mean(axis=1)
    if kind=='flatten':return x.reshape(len(x),-1)
    if kind=='trend':return np.concatenate([x[:,-1],x.mean(axis=1),x[:,-1]-x[:,0]],axis=1)
    raise ValueError(kind)

def ridge_path(x,v,y,sigma,weight_lambda,alphas):
    scaler=StandardScaler().fit(x);scaler.scale_=np.maximum(scaler.scale_,.01)
    x=scaler.transform(x).astype(np.float64);v=scaler.transform(v).astype(np.float64)
    positive=sigma[sigma>0];floor=float(np.quantile(positive,.05));med=float(np.median(positive));w=1+weight_lambda*(med/np.maximum(sigma,floor))**2;w=w/w.mean()
    xm=np.average(x,axis=0,weights=w);ym=float(np.average(y,weights=w));xt=torch.as_tensor((x-xm)*np.sqrt(w[:,None]),device=DEVICE);vt=torch.as_tensor(v-xm,device=DEVICE)
    gram=(xt@xt.T).cpu().numpy();cross=(vt@xt.T).cpu().numpy();del xt,vt
    lam,q=eigh(gram,check_finite=False);lam=np.maximum(lam,0);qy=q.T@((y-ym)*np.sqrt(w));vq=cross@q
    return {a:vq@(qy/(lam+a))+ym for a in alphas}

def run():
    t=time.time();prepare();controls=json.loads((EXPORT/'event_controls.json').read_text());cs=[r['metrics'] for r in controls['rows']];prior=np.load(DATA/'bank_prior_M6.npz');datasets={f:split(f)[:2] for f in range(1,5)};raw={};audits=[]
    for held,(tr,va) in datasets.items():
        d=np.load(DATA/f'location_cv{held}.npz')
        for kind in ['log','ratio']:
            frames=representation(d['local'],d['glob'],kind)
            xx=[np.stack([frames[int(b),int(o)-6:int(o)-2] for b,o in zip(a['block'],a['offset'])]) for a in [tr,va]]
            for view in ['mean','flatten','trend']:
                x,v=[temporal(z,view) for z in xx]
                for weight in [0,1]:
                    preds=ridge_path(x,v,tr['y'],tr['sigma'],weight,[100,1000,10000,100000,1000000])
                    for alpha,p in preds.items():
                        key=f'{kind}_{view}_w{weight}_a{alpha}';raw.setdefault(key,{'recipe':{'family':'location_ridge','representation':kind,'view':view,'weight_lambda':weight,'alpha':alpha},'predictions':{}})['predictions'][held]=p
                print(json.dumps({'location_fit_fold':held,'representation':kind,'view':view,'seconds':time.time()-t}),flush=True)
        audits.append({'held_training_fold':held,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va)})
    rows=[];all_predictions={};selected=None;selected_predictions=None
    def score(name,recipe,pp):
        nonlocal selected,selected_predictions
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)];m=aggregate(fm);r={'model':name,'recipe':recipe,'metrics':m,'fold_metrics':fm,'gate':gate(m,cs)};rows.append(r)
        if r['gate']['all_four'] and (selected is None or m['rmse']<selected['metrics']['rmse']):selected=r;selected_predictions=pp
        for f,p in pp.items():all_predictions[name+'_cv'+str(f)]=p
    for name,r in raw.items():
        score('M9_raw_'+name,r['recipe'],r['predictions'])
        for w in [.25,.5,.75,1.]:
            for gain in [.9,1.,1.2,1.4]:
                for bias in [-10,0,10]:
                    pp={f:float(tr['y'].mean())+gain*(w*r['predictions'][f]+(1-w)*prior['cv'+str(f)]-float(tr['y'].mean()))+bias for f,(tr,va) in datasets.items()}
                    score(f'M9_{name}_mix{w}_g{gain}_b{bias}',{'family':'affine_location_mix','location':r['recipe'],'location_weight':w,'prior':'M6','gain':gain,'bias':bias},pp)
    assert len(rows)==2940
    np.savez_compressed(DATA/'m9_all_predictions.npz',**all_predictions)
    uncertainty=None
    if selected:
        save_predictions('m9_selected',selected_predictions);uncertainty={c['model']:bootstrap(selected_predictions,{f:np.load(DATA/(c['model']+'.npz'))['cv'+str(f)] for f in range(1,5)}) for c in controls['rows']}
    save_json(EXPORT/'m9_results.json',{'stage':'M9','scope':'Four-fold training-bank CV only; Folds1-4','rows':rows,'controls':controls['rows'],'selected':selected,'gate_passed':selected is not None,'bootstrap':uncertainty,'bootstrap_limitations':'Conditional whole-block bootstrap; not selection adjusted; blocks can overlap.','row_audits':audits,'identity':identity(),'seconds':time.time()-t,'original_final_evaluations':0,'fold5_available':False})
    print(json.dumps({'M9_complete':True,'selected':selected,'eligible':sum(r['gate']['all_four'] for r in rows),'seconds':time.time()-t}),flush=True)
