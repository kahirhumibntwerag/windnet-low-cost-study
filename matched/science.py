"""Image-only science. No earlier study inputs or test predictions are imported."""
import ast, copy, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from matched.runner import DATA,EXPORT,ROOT,numeric_npy,save_json,sha
from matched.hse_original import BatchwiseHSE

torch.set_num_threads(4)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
SEEDS=[19,43,101]
def identity():
    return {'code':{p.name:sha(p.read_bytes()) for p in sorted((ROOT/'matched').glob('*.py'))},'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if DEVICE=='cuda' else None}
def metrics(y,p,sigma):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);sigma=np.asarray(sigma,dtype=float)
    assert y.shape==p.shape and np.isfinite(p).all()
    ok=np.isfinite(sigma)&(sigma>0)
    tp,fp,fn=BatchwiseHSE(np.column_stack([p,y]),4,3)
    return {'n':len(y),'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y))),'r':float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 else None,'chi2':float(np.mean(((p[ok]-y[ok])/sigma[ok])**2)) if ok.any() else None,'sigma_valid':int(ok.sum()),'hse_ts':float(tp/(tp+fp+fn)) if tp+fp+fn else None,'hse_counts_original':{'tp':int(tp),'fp':int(fp),'fn':int(fn)}}
def blocks(final=False):
    if final:
        assert (DATA/'freeze.json').exists(),'Final data access requires frozen recipes'
    result=[]
    for m in json.loads((DATA/'members.json').read_text()):
        f=m['fold']
        if f==5:
            if not final:continue
            obj=numeric_npy((DATA/'sealed'/m['file']).read_bytes());im=np.asarray(obj['input'],dtype=np.float32).reshape(20,256,256);out=np.asarray(obj['output'],dtype=float)
        else:
            d=np.load(DATA/'development'/m['file'].replace('.npy','.npz'));im=d['images'];out=d['output']
        result.append({'fold':f,'name':m['file'],'images':im,'output':out})
    return result

class KaffeFeatures(nn.Module):
    """Translate the released constant DAG; never execute downloaded Python source."""
    def __init__(self):
        super().__init__();self.graph=[];self.weights={}
        tree=ast.parse((DATA/'kaffe/googlenet.py').read_text())
        setup=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='setup')
        def parse(node):
            if not isinstance(node,ast.Call):return None
            op=node.func.attr
            args=[ast.literal_eval(a) for a in node.args];kw={a.arg:ast.literal_eval(a.value) for a in node.keywords}
            if op=='feed':return args
            ins=parse(node.func.value);name=kw['name'];self.graph.append((name,op,ins,args,kw));return [name]
        for st in setup.body:
            if isinstance(st,ast.Expr):parse(st.value)
        self.graph=self.graph[:next(i+1 for i,x in enumerate(self.graph) if x[0]=='inception_4a_output')]
        weights=numeric_npy((DATA/'kaffe/googlenet.npy').read_bytes())
        for name,op,_,_,_ in self.graph:
            if op=='conv':
                w=weights[name];kernel=torch.tensor(np.asarray(w['weights']).transpose(3,2,0,1).copy(),device=DEVICE);bias=torch.tensor(w['biases'],device=DEVICE)
                self.weights[name]=(kernel,bias)
        self.targets=['conv2_3x3_reduce','inception_4a_output','inception_3a_output']
    @staticmethod
    def same(x,k,stride,value=0):
        pads=[]
        for n in list(x.shape[-2:])[::-1]:
            p=max((math.ceil(n/stride)-1)*stride+k-n,0);pads.extend([p//2,p-p//2])
        return F.pad(x,pads,value=value)
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
            else:raise ValueError('Unsupported original op '+op)
            vals[name]=z
        return torch.cat([(vals[name]**2).sum(dim=(2,3)) for name in self.targets],dim=1)

def spatial(images):
    # Fixed disk geometry and intensity thresholds: all derived from the same images.
    yy,xx=np.mgrid[:256,:256];disk=(xx-127.5)**2+(yy-127.5)**2<125**2
    features=[]
    for y0,y1 in zip([0,64,128,192],[64,128,192,256]):
        for x0,x1 in zip(range(0,256,32),range(32,257,32)):
            mask=disk[y0:y1,x0:x1];v=images[:,y0:y1,x0:x1][:,mask]
            features.extend([v.mean(axis=1),v.std(axis=1)])
            features.extend([(v<th).mean(axis=1) for th in [20,40,60,80,100]])
    return np.stack(features,axis=1).astype(np.float32)

def embed_blocks(bs,fit_folds,path):
    fit=[b for b in bs if b['fold'] in fit_folds]
    mean=float(sum(b['images'].sum(dtype=np.float64) for b in fit)/(len(fit)*20*256*256))
    model=KaffeFeatures().eval();all_e=[];all_s=[];all_o=[];folds=[];names=[];t=time.time()
    with torch.inference_mode():
        for i,b in enumerate(bs):
            ims=torch.from_numpy(b['images']).to(DEVICE)[:,None]
            ims=F.interpolate(ims,size=(224,224),mode='bilinear',align_corners=False).repeat(1,3,1,1)-mean
            e=model(ims).cpu().numpy();assert e.shape==(20,832) and np.isfinite(e).all()
            all_e.append(e);all_s.append(spatial(b['images']));all_o.append(b['output']);folds.append(b['fold']);names.append(b['name'])
            if i%20==0:print(json.dumps({'embedding_blocks':i+1,'seconds':round(time.time()-t)}),flush=True)
    np.savez_compressed(path,energy=np.stack(all_e),spatial=np.stack(all_s),output=np.stack(all_o),fold=np.array(folds),name=np.array(names),image_mean=mean)
    return {'fit_folds':fit_folds,'blocks':len(bs),'embedding_dimensions':832,'image_mean_training_only':mean,'seconds':time.time()-t,'fold5_accessed':5 in folds,'original_weight_sha256':sha((DATA/'kaffe/googlenet.npy').read_bytes()),'cache_sha256':sha(path.read_bytes()),'identity':identity()}
def prepare_embeddings():
    bs=blocks(False);record=embed_blocks(bs,[1,2,3],DATA/'dev_embeddings.npz');save_json(EXPORT/'embedding_audit.json',record)
    print('Embedding complete; Fold5 untouched',flush=True)
def samples(d,fit_folds,eval_folds):
    energy=d['energy'];ff=np.isin(d['fold'],fit_folds)
    emin=energy[ff].min(axis=(0,1));escale=energy[ff].max(axis=(0,1))-emin+1e-8
    normalized=(energy-emin)/escale
    def pack(folds,training):
        a={'energy':[],'spatial':[],'y':[],'sigma':[],'history_y':[],'row':[],'fold':[],'block':[],'offset':[]}
        for i,f in enumerate(d['fold']):
            if f not in folds:continue
            for off in range(6,19 if training else 20):
                a['energy'].append(normalized[i,off-6:off-2]);a['spatial'].append(d['spatial'][i,off-6:off-2]);a['y'].append(d['output'][i,off,11]);a['sigma'].append(d['output'][i,off,14]);a['history_y'].append(d['output'][i,off-6:off-2,11]);a['row'].append(str(d['name'][i])+':'+str(off));a['fold'].append(f);a['block'].append(i);a['offset'].append(off)
        return {k:np.array(v) for k,v in a.items()}
    tr=pack(fit_folds,True);va=pack(eval_folds,False)
    assert not set(tr['row'])&set(va['row'])
    return tr,va,{'emin':emin,'escale':escale}
def view(a,kind='flatten',source='energy'):
    if source=='combined':return np.concatenate([view(a,kind,'energy'),view(a,kind,'spatial')],axis=1)
    x=a[source]
    if kind=='flatten':return x.reshape(len(x),-1)
    if kind=='mean':return x.mean(axis=1)
    if kind=='trend':return np.concatenate([x[:,-1],x.mean(axis=1),x[:,-1]-x[:,0]],axis=1)
    if kind=='last':return x[:,-1]
    raise ValueError(kind)
def rowhash(a):return sha('\n'.join(a['row']).encode())

class WindNetPort(nn.Module):
    def __init__(self,hidden=400,dropout=.7):
        super().__init__();self.rnn=nn.LSTM(832,hidden,batch_first=True);self.heads=nn.ModuleList([nn.Linear(hidden,1) for _ in range(4)]);self.dropout=dropout
        for name,p in self.rnn.named_parameters():
            if 'weight' in name:nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
                if name=='bias_ih_l0':p.data[hidden:2*hidden]=1
    def forward(self,x):
        out,_=self.rnn(x);out=F.dropout(out,self.dropout,self.training)
        return torch.stack([F.elu(h(out[:,i])).squeeze(-1) for i,h in enumerate(self.heads)],dim=1)
def neural_fit(tr,va,seed,max_epochs=300,patience=40,fixed_epochs=None,variant='paper',fast_weight=1):
    torch.manual_seed(seed);np.random.seed(seed)
    model=WindNetPort().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.0005)
    ymin=float(tr['y'].min());yscale=float(np.ptp(tr['y']));assert yscale>0
    x=torch.tensor(tr['energy'],dtype=torch.float32,device=DEVICE);y=torch.tensor((tr['y']-ymin)/yscale,dtype=torch.float32,device=DEVICE)
    seq=torch.tensor((tr['history_y']-ymin)/yscale,dtype=torch.float32,device=DEVICE)
    valid_sigma=tr['sigma'][tr['sigma']>0];floor=float(np.quantile(valid_sigma,.05)) if len(valid_sigma) else 1.
    sigma=torch.tensor(np.maximum(tr['sigma'],floor)/yscale,dtype=torch.float32,device=DEVICE)
    raw_weights=np.where(tr['y']>=500,float(fast_weight),1.);raw_weights=raw_weights/raw_weights.mean();weights=torch.tensor(raw_weights,dtype=torch.float32,device=DEVICE)
    xv=torch.tensor(va['energy'],dtype=torch.float32,device=DEVICE)
    best=float('inf');best_epoch=0;best_state=None;trace=[]
    for epoch in range(1,(fixed_epochs or max_epochs)+1):
        model.train()
        for ind in torch.randperm(len(x),device=DEVICE).split(128):
            out=model(x[ind]);err=out[:,-1]-y[ind]
            loss=(weights[ind]*err**2).mean()
            if variant in ['paper','no_variance']:loss=loss+.1*(weights[ind,None]*(out-seq[ind])**2).mean()
            if variant in ['paper','no_sequence']:loss=loss+(weights[ind]*(err/sigma[ind])**2).mean()
            loss=loss+1e-6*sum((p**2).sum() for p in model.parameters())
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
        if fixed_epochs:continue
        model.eval()
        with torch.inference_mode():pred=model(xv)[:,-1].cpu().numpy()*yscale+ymin
        rmse=float(np.sqrt(np.mean((pred-va['y'])**2)))
        if epoch%20==0:trace.append({'epoch':epoch,'validation_rmse':rmse})
        if rmse<best-1e-5:best=rmse;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=patience:break
    if not fixed_epochs:model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():pred=model(xv)[:,-1].cpu().numpy()*yscale+ymin
    state={'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'ymin':ymin,'yscale':yscale,'epochs':fixed_epochs or best_epoch,'seed':seed,'sigma_floor_training':floor,'variant':variant,'fast_weight':fast_weight,'normalized_weight_mean':float(raw_weights.mean()),'fast_training_count':int(np.sum(tr['y']>=500))}
    return pred,state,trace
def baseline():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,norm=samples(d,[1,2,3],[4]);rows=[];preds=[];epochs=[]
    (DATA/'models').mkdir(exist_ok=True)
    audit={'fit_rows':len(tr['y']),'validation_rows':len(va['y']),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'identity':identity(),'port_limitations':['PyTorch instead of TensorFlow1.7','float32 bilinear resize instead of cv2 float64','gradient clipping 10 for numerical stability','validation-RMSE early stopping instead of original test-correlation checkpoint selection']}
    save_json(EXPORT/'m1_execution.json',audit)
    for seed in SEEDS:
        t=time.time();p,state,trace=neural_fit(tr,va,seed);torch.save({**state,'normalization':norm},DATA/f'models/baseline_{seed}.pt');preds.append(p);epochs.append(state['epochs'])
        rows.append({'model':'windnet_port_seed'+str(seed),'metrics':metrics(va['y'],p,va['sigma']),'epochs':state['epochs'],'trace':trace});print(json.dumps({'model':rows[-1]['model'],'rmse':rows[-1]['metrics']['rmse'],'r':rows[-1]['metrics']['r'],'epochs':state['epochs'],'seconds':time.time()-t}),flush=True)
    p=np.mean(preds,axis=0);base={'model':'windnet_port_ensemble','metrics':metrics(va['y'],p,va['sigma']),'epochs':epochs};rows.append(base)
    save_json(EXPORT/'baseline.json',{'stage':'M1','scope':'training-side Fold4 validation only','rows':rows,'baseline':base,'audit':audit})
    np.savez_compressed(DATA/'baseline_dev_predictions.npz',pred=p,row=va['row'])
    ridge()
def ridge():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,norm=samples(d,[1,2,3],[4]);rows=[];preds={};best=None
    for kind in ['flatten','mean','trend']:
        scaler=StandardScaler().fit(view(tr,kind));x=scaler.transform(view(tr,kind));v=scaler.transform(view(va,kind))
        # A shared SVD avoids six redundant decompositions of the same design.
        xm=x.mean(axis=0);ym=tr['y'].mean();u,s,vt=np.linalg.svd(x-xm,full_matrices=False);uy=u.T@(tr['y']-ym);z=(v-xm)@vt.T
        for alpha in [1,10,100,1000,10000,100000]:
            p=z@((s/(s*s+alpha))*uy)+ym;recipe={'family':'ridge','view':kind,'source':'energy','alpha':alpha};m=metrics(va['y'],p,va['sigma']);name='ridge_'+kind+'_'+str(alpha)
            rows.append({'model':name,'recipe':recipe,'metrics':m});preds[name]=p
            if best is None or m['rmse']<best['metrics']['rmse']:best=rows[-1]
    base=json.loads((EXPORT/'baseline.json').read_text())['baseline']
    passed=best['metrics']['rmse']<=.99*base['metrics']['rmse'] and best['metrics']['r']>=base['metrics']['r']
    result={'stage':'M1','scope':'training-side Fold4 validation only','rows':rows,'best_ridge':best,'baseline':base,'gate_passed':passed,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'identity':identity()}
    save_json(EXPORT/'m1_results.json',result);np.savez_compressed(DATA/'ridge_dev_predictions.npz',**preds)
    print(json.dumps({'stage':'M1_complete','best':best,'gate_passed':passed}),flush=True)
def kernel():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,_=samples(d,[1,2,3],[4]);rows=[];predictions={};best_raw=None
    baseline=json.loads((EXPORT/'baseline.json').read_text())['baseline'];bp=np.load(DATA/'baseline_dev_predictions.npz')['pred']
    t=time.time();ym=float(tr['y'].mean())
    save_json(EXPORT/'m2_execution.json',{'identity':identity(),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False})
    for kind in ['flatten','mean','trend']:
        scaler=StandardScaler().fit(view(tr,kind));x=scaler.transform(view(tr,kind));v=scaler.transform(view(va,kind))
        dist=cdist(x,x,'sqeuclidean');dv=cdist(v,x,'sqeuclidean')
        for gf in [.1,1,10]:
            gamma=gf/x.shape[1];k=np.exp(-gamma*dist);kv=np.exp(-gamma*dv);lam,q=eigh(k,check_finite=False);lam=np.maximum(lam,0);qy=q.T@(tr['y']-ym);vq=kv@q
            for alpha in [.01,.1,1,10,100]:
                p=vq@(qy/(lam+alpha))+ym;name=f'kernel_{kind}_g{gf}_a{alpha}';recipe={'family':'kernel','view':kind,'gamma_factor':gf,'alpha':alpha,'source':'energy'}
                row={'model':name,'recipe':recipe,'metrics':metrics(va['y'],p,va['sigma'])};rows.append(row);predictions[name]=p
                if best_raw is None or row['metrics']['rmse']<best_raw['metrics']['rmse']:best_raw=row
    kp=predictions[best_raw['model']]
    for w in [0,.25,.5,.75,1]:
        p=(1-w)*bp+w*kp;name='kernel_baseline_mix_'+str(w);row={'model':name,'recipe':{'family':'blend','kernel_weight':w,'kernel':best_raw['recipe'],'baseline':True},'metrics':metrics(va['y'],p,va['sigma'])};rows.append(row);predictions[name]=p
    eligible=[r for r in rows if r['metrics']['r']>=baseline['metrics']['r']];selected=min(eligible,key=lambda r:r['metrics']['rmse']) if eligible else min(rows,key=lambda r:r['metrics']['rmse'])
    passed=selected['metrics']['rmse']<=.99*baseline['metrics']['rmse'] and selected['metrics']['r']>=baseline['metrics']['r']
    result={'stage':'M2','scope':'training-side Fold4 validation only','rows':rows,'selected':selected,'best_raw':best_raw,'incumbent':baseline,'gate_passed':passed,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'seconds':time.time()-t,'identity':identity()}
    save_json(EXPORT/'m2_results.json',result);np.savez_compressed(DATA/'kernel_dev_predictions.npz',**predictions)
    print(json.dumps({'stage':'M2_complete','selected':selected,'gate_passed':passed,'seconds':time.time()-t}),flush=True)
def loss_ablations():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,norm=samples(d,[1,2,3],[4]);rows=[];preds={};t=time.time();base=json.loads((EXPORT/'baseline.json').read_text())['baseline']
    save_json(EXPORT/'m3_execution.json',{'identity':identity(),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False})
    for variant in ['no_sequence','no_variance','mse_only']:
        pp=[];epochs=[];seed_rows=[]
        for seed in SEEDS:
            p,state,trace=neural_fit(tr,va,seed,variant=variant);torch.save({**state,'normalization':norm},DATA/f'models/{variant}_{seed}.pt');pp.append(p);epochs.append(state['epochs']);seed_rows.append({'seed':seed,'epochs':state['epochs'],'metrics':metrics(va['y'],p,va['sigma']),'trace':trace})
        p=np.mean(pp,axis=0);name='lstm_'+variant;row={'model':name,'recipe':{'family':'lstm','variant':variant,'epochs':epochs,'seeds':SEEDS},'metrics':metrics(va['y'],p,va['sigma']),'seed_rows':seed_rows};rows.append(row);preds[name]=p
        print(json.dumps({'condition':variant,'metrics':row['metrics'],'epochs':epochs}),flush=True)
    eligible=[r for r in rows if r['metrics']['r']>=base['metrics']['r']];selected=min(eligible or rows,key=lambda r:r['metrics']['rmse']);passed=selected['metrics']['rmse']<=.99*base['metrics']['rmse'] and selected['metrics']['r']>=base['metrics']['r']
    result={'stage':'M3','scope':'training-side Fold4 validation only','rows':rows,'selected':selected,'incumbent':base,'gate_passed':passed,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'seconds':time.time()-t,'identity':identity()}
    save_json(EXPORT/'m3_results.json',result);np.savez_compressed(DATA/'loss_dev_predictions.npz',**preds)
    print(json.dumps({'stage':'M3_complete','selected':selected['model'],'gate_passed':passed,'seconds':time.time()-t}),flush=True)
def fit_tabular(tr,va,recipe):
    source=recipe.get('source','energy');kind=recipe['view'];scaler=StandardScaler().fit(view(tr,kind,source));x=scaler.transform(view(tr,kind,source));v=scaler.transform(view(va,kind,source))
    if recipe['family']=='ridge':model=Ridge(alpha=recipe['alpha'],solver='cholesky')
    elif recipe['family']=='tree':model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=recipe['leaves'],l2_regularization=recipe['l2'],min_samples_leaf=20,early_stopping=False,random_state=2026)
    else:raise ValueError(recipe)
    model.fit(x,tr['y']);return model.predict(v),{'model':model,'scaler':scaler,'recipe':recipe}
def spatial_models():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,_=samples(d,[1,2,3],[4]);rows=[];preds={};best_raw=None;t=time.time()
    baseline=json.loads((EXPORT/'baseline.json').read_text())['baseline'];bp=np.load(DATA/'baseline_dev_predictions.npz')['pred']
    save_json(EXPORT/'m4_execution.json',{'identity':identity(),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False})
    recipes=[]
    for source in ['spatial','combined']:
        for kind in ['flatten','mean','trend']:
            for alpha in [10,100,1000,10000]:recipes.append({'family':'ridge','view':kind,'source':source,'alpha':alpha})
        for kind in ['mean','trend']:
            for leaves in [7,15]:
                for l2 in [1,10,100]:recipes.append({'family':'tree','view':kind,'source':source,'leaves':leaves,'l2':l2})
    for i,recipe in enumerate(recipes):
        p,_=fit_tabular(tr,va,recipe);name='spatial_'+str(i)+'_'+recipe['family']+'_'+recipe['source']+'_'+recipe['view'];row={'model':name,'recipe':recipe,'metrics':metrics(va['y'],p,va['sigma'])};rows.append(row);preds[name]=p
        if best_raw is None or row['metrics']['rmse']<best_raw['metrics']['rmse']:best_raw=row
        if i%12==0:print(json.dumps({'spatial_candidates_complete':i+1,'seconds':time.time()-t}),flush=True)
    sp=preds[best_raw['model']]
    for w in [0,.25,.5,.75,1]:
        p=(1-w)*bp+w*sp;name='spatial_baseline_mix_'+str(w);row={'model':name,'recipe':{'family':'spatial_blend','spatial_weight':w,'spatial':best_raw['recipe'],'baseline':True},'metrics':metrics(va['y'],p,va['sigma'])};rows.append(row);preds[name]=p
    ablations=[]
    for source in ['energy','spatial','combined']:
        recipe={**best_raw['recipe'],'source':source};p,_=fit_tabular(tr,va,recipe);row={'model':'matched_feature_ablation_'+source,'recipe':recipe,'metrics':metrics(va['y'],p,va['sigma'])};ablations.append(row);preds[row['model']]=p
    eligible=[r for r in rows if r['metrics']['r']>=baseline['metrics']['r']];selected=min(eligible or rows,key=lambda r:r['metrics']['rmse']);passed=selected['metrics']['rmse']<=.99*baseline['metrics']['rmse'] and selected['metrics']['r']>=baseline['metrics']['r']
    result={'stage':'M4','scope':'training-side Fold4 validation only','rows':rows+ablations,'selected':selected,'best_raw':best_raw,'matched_ablations':ablations,'incumbent':baseline,'gate_passed':passed,'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'seconds':time.time()-t,'identity':identity()}
    save_json(EXPORT/'m4_results.json',result);np.savez_compressed(DATA/'spatial_dev_predictions.npz',**preds)
    print(json.dumps({'stage':'M4_complete','selected':selected,'gate_passed':passed,'seconds':time.time()-t}),flush=True)
def weighted_models():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,norm=samples(d,[1,2,3],[4]);rows=[];preds={};t=time.time();base=json.loads((EXPORT/'baseline.json').read_text())['baseline']
    m4=json.loads((EXPORT/'m4_results.json').read_text())['selected'];sp=np.load(DATA/'spatial_dev_predictions.npz')[m4['model']]
    save_json(EXPORT/'m5_execution.json',{'identity':identity(),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False})
    for weight in [2,4]:
        pp=[];epochs=[];seed_rows=[]
        for seed in SEEDS:
            p,state,trace=neural_fit(tr,va,seed,fast_weight=weight);torch.save({**state,'normalization':norm},DATA/f'models/weighted{weight}_{seed}.pt');pp.append(p);epochs.append(state['epochs']);seed_rows.append({'seed':seed,'epochs':state['epochs'],'normalized_weight_mean':state['normalized_weight_mean'],'fast_training_count':state['fast_training_count'],'metrics':metrics(va['y'],p,va['sigma'])})
        p=np.mean(pp,axis=0);name='lstm_fast_weight_'+str(weight);recipe={'family':'lstm','variant':'paper','epochs':epochs,'seeds':SEEDS,'fast_weight':weight};row={'model':name,'recipe':recipe,'metrics':metrics(va['y'],p,va['sigma']),'seed_rows':seed_rows};rows.append(row);preds[name]=p
        for w in [0,.25,.5,.75,1]:
            mix=w*p+(1-w)*sp;name=f'fast{weight}_spatial_mix_{w}';row={'model':name,'recipe':{'family':'weighted_blend','weighted_fraction':w,'weighted':recipe,'other':m4['recipe']},'metrics':metrics(va['y'],mix,va['sigma'])};rows.append(row);preds[name]=mix
        print(json.dumps({'fast_weight':weight,'metrics':rows[-1]['metrics'],'epochs':epochs}),flush=True)
    eligible=[r for r in rows if r['metrics']['rmse']<=.99*base['metrics']['rmse'] and r['metrics']['r']>=base['metrics']['r'] and r['metrics']['chi2']<=base['metrics']['chi2'] and r['metrics']['hse_ts'] is not None and r['metrics']['hse_ts']>=base['metrics']['hse_ts']]
    selected=min(eligible,key=lambda r:r['metrics']['rmse']) if eligible else None
    rmse_eligible=[r for r in rows if r['metrics']['r']>=m4['metrics']['r']];best_rmse=min(rmse_eligible or rows,key=lambda r:r['metrics']['rmse'])
    result={'stage':'M5','scope':'training-side Fold4 validation only','rows':rows,'selected':selected,'best_rmse_candidate':best_rmse,'baseline':base,'rmse_incumbent':m4,'gate_passed':bool(eligible),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'seconds':time.time()-t,'identity':identity()}
    save_json(EXPORT/'m5_results.json',result);np.savez_compressed(DATA/'weighted_dev_predictions.npz',**preds)
    print(json.dumps({'stage':'M5_complete','selected':selected,'best_rmse':best_rmse,'gate_passed':bool(eligible),'seconds':time.time()-t}),flush=True)
def block_moments(a,p):
    result=[]
    for b in dict.fromkeys(a['block'].tolist()):
        ix=a['block']==b;y=a['y'][ix];v=p[ix];s=a['sigma'][ix];ok=s>0;tp,fp,fn=BatchwiseHSE(np.column_stack([v,y]),4,3)
        result.append([len(y),y.sum(),v.sum(),(y*y).sum(),(v*v).sum(),(y*v).sum(),((v-y)**2).sum(),(((v[ok]-y[ok])/s[ok])**2).sum(),ok.sum(),tp,fp,fn])
    return np.array(result,dtype=float)
def moment_score(m):
    n,sy,sp,syy,spp,syp,sse,chi,nchi,tp,fp,fn=m.sum(axis=0);den=(syy-sy*sy/n)*(spp-sp*sp/n)
    return {'rmse':float(np.sqrt(sse/n)),'r':float((syp-sy*sp/n)/np.sqrt(den)) if den>0 else np.nan,'chi2':float(chi/nchi) if nchi else np.nan,'hse_ts':float(tp/(tp+fp+fn)) if tp+fp+fn else np.nan}
def bootstrap_validation(a,p,base):
    m=block_moments(a,p);b=block_moments(a,base);rng=np.random.default_rng(20261029);diff={k:[] for k in ['rmse','r','chi2','hse_ts']}
    for _ in range(2000):
        ix=rng.integers(0,len(m),len(m));x=moment_score(m[ix]);y=moment_score(b[ix])
        for k in diff:
            v=x[k]-y[k]
            if np.isfinite(v):diff[k].append(v)
    return {k:{'delta_95':np.percentile(v,[2.5,97.5]).tolist(),'defined_draws':len(v)} for k,v in diff.items()}
def calibrate():
    d=np.load(DATA/'dev_embeddings.npz');tr,va,_=samples(d,[1,2,3],[4]);base=json.loads((EXPORT/'baseline.json').read_text())['baseline'];base_recipe={'family':'lstm','variant':'paper','epochs':base['epochs'],'seeds':SEEDS,'fast_weight':1};bp=np.load(DATA/'baseline_dev_predictions.npz')['pred']
    m3=json.loads((EXPORT/'m3_results.json').read_text());ns=next(r for r in m3['rows'] if r['model']=='lstm_no_sequence');nsp=np.load(DATA/'loss_dev_predictions.npz')[ns['model']]
    m4=json.loads((EXPORT/'m4_results.json').read_text())['selected'];sp=np.load(DATA/'spatial_dev_predictions.npz')[m4['model']]
    m5=json.loads((EXPORT/'m5_results.json').read_text());ww=np.load(DATA/'weighted_dev_predictions.npz');bases=[('baseline',base_recipe,bp),('no_sequence',ns['recipe'],nsp),('spatial',m4['recipe'],sp)]
    for weight in [2,4]:
        row=next(r for r in m5['rows'] if r['model']=='lstm_fast_weight_'+str(weight));bases.append(('weight'+str(weight),row['recipe'],ww[row['model']]))
    for frac in [.25,.5,.75]:
        row=next(r for r in m5['rows'] if r['model']=='fast4_spatial_mix_'+str(frac));bases.append(('weight4_mix'+str(frac),row['recipe'],ww[row['model']]))
    rows=[];preds={};anchor=float(tr['y'].mean());t=time.time()
    save_json(EXPORT/'m6_execution.json',{'identity':identity(),'fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False})
    for name,recipe,p in bases:
        for gain in [.9,1,1.1,1.2,1.3,1.4,1.5]:
            for bias in [-20,-10,0,10,20]:
                v=anchor+gain*(p-anchor)+bias;label=f'cal_{name}_g{gain}_b{bias}';row={'model':label,'recipe':{'family':'calibrated','base':recipe,'gain':gain,'bias':bias,'anchor':'fitting_target_mean'},'metrics':metrics(va['y'],v,va['sigma'])};rows.append(row);preds[label]=v
    eligible=[r for r in rows if r['metrics']['rmse']<=.99*base['metrics']['rmse'] and r['metrics']['r']>=base['metrics']['r']-1e-12 and r['metrics']['chi2']<=base['metrics']['chi2']+1e-10 and r['metrics']['hse_ts'] is not None and r['metrics']['hse_ts']>=base['metrics']['hse_ts']-1e-12]
    selected=min(eligible,key=lambda r:r['metrics']['rmse']) if eligible else m4
    rmse_eligible=[r for r in rows if r['metrics']['r']>=base['metrics']['r']-1e-12];best_rmse=min(rmse_eligible,key=lambda r:r['metrics']['rmse'])
    chosen=preds[selected['model']] if eligible else sp
    audit=bootstrap_validation(va,chosen,bp)
    result={'stage':'M6','scope':'training-side Fold4 validation only','rows':rows,'selected':selected,'best_rmse_candidate':best_rmse,'gate_passed':bool(eligible),'bootstrap':audit,'bootstrap_limitations':'Conditional whole-block bootstrap, not selection-adjusted; overlapping blocks may be dependent. Cannot establish fresh confirmation.','fit_row_sha256':rowhash(tr),'validation_row_sha256':rowhash(va),'fold5_accessed':False,'seconds':time.time()-t,'identity':identity()}
    save_json(EXPORT/'m6_results.json',result);np.savez_compressed(DATA/'calibration_dev_predictions.npz',**preds)
    print(json.dumps({'stage':'M6_complete','selected':selected,'best_rmse':best_rmse,'gate_passed':bool(eligible),'bootstrap':audit,'seconds':time.time()-t}),flush=True)
def run(cfg):
    stage=cfg['stage']
    if stage in ['freeze','final']:
        from matched.final_study import freeze,final
        (freeze if stage=='freeze' else final)(cfg)
    elif stage=='embed':prepare_embeddings()
    elif stage=='baseline':baseline()
    elif stage=='ridge':ridge()
    elif stage=='kernel':kernel()
    elif stage=='neural':calibrate() if cfg.get('mode')=='calibrate' else weighted_models() if cfg.get('mode')=='weighted' else loss_ablations()
    elif stage=='spatial':spatial_models()
    else:raise ValueError('Stage not implemented '+stage)
