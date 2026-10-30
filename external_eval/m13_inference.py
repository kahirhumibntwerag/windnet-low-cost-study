"""Pure inference from F3 artifacts. No fit method, optimizer or target input."""
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matched.science as legacy
from event_loop.location import LocationFeatures, representation
from next_loop.regional_features import regions, temporal
from next_loop.f3_panel import REGIONAL

class PortableUnpickler(pickle.Unpickler):
    """NumPy2 renamed core to _core; decode frozen numeric arrays on NumPy1.26."""
    def find_class(self,module,name):
        if module=='numpy._core' or module.startswith('numpy._core.'):
            module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)

class M13:
    def __init__(self,fold,encoder,root=Path('data/external_models')):
        self.encoder=encoder
        folder=root/'m13/models'/f'fold{fold}'
        with np.load(folder/'normalization.npz') as norm:
            self.minimum=norm['emin'];self.scale=norm['escale'];self.mean=float(norm['image_mean'])
        self.ports=[]
        for seed in [19,43,101]:
            state=torch.load(folder/f'port_seed{seed}.pt',map_location='cpu',weights_only=False)
            model=legacy.WindNetPort().to(legacy.DEVICE)
            model.load_state_dict(state['state_dict']);model.eval()
            self.ports.append((model,state['ymin'],state['yscale']))
        def read(name):
            with (folder/name).open('rb') as file:return PortableUnpickler(file).load()
        self.ridge=read('prior_ridge.pkl');self.location=read('location_0bccf2b75ed2.pkl')
        candidates=[read(p.name) for p in sorted(folder.glob('regional_*.pkl'))]
        candidates=[c for c in candidates if c['recipe']==REGIONAL]
        assert len(candidates)==1
        self.regional=candidates[0];self.anchor=self.regional['target_mean']

    @torch.inference_mode()
    def encode(self,images,batch_size=16):
        energy=[];ratio=[]
        for start in range(0,len(images),batch_size):
            x=torch.from_numpy(np.asarray(images[start:start+batch_size],dtype=np.float32)).to(legacy.DEVICE)[:,None]
            x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False).repeat(1,3,1,1)-self.mean
            local,glob=self.encoder(x);local=local.cpu().numpy();glob=glob.cpu().numpy()
            energy.append((glob-self.minimum)/self.scale);ratio.append(representation(local,glob,'ratio'))
        return {'energy':np.concatenate(energy),'ratio':np.concatenate(ratio),
                'spatial':legacy.spatial(np.asarray(images,dtype=np.float32)),
                'regional':regions(np.asarray(images,dtype=np.float32),[6,6])}

    @torch.inference_mode()
    def predict(self,features):
        assert set(features)=={'energy','ratio','spatial','regional'}
        x=torch.as_tensor(features['energy'],dtype=torch.float32,device=legacy.DEVICE)
        pp=[]
        for model,minimum,scale in self.ports:
            pp.append(model(x)[:,-1].cpu().numpy()*scale+minimum)
        port=np.mean(pp,axis=0)
        combined=np.concatenate([features['energy'].reshape(len(x),-1),features['spatial'].reshape(len(x),-1)],axis=1)
        ridge=self.ridge['model'].predict(self.ridge['scaler'].transform(combined))
        prior=self.anchor+1.2*(.5*port+.5*ridge-self.anchor)
        loc=features['ratio'].reshape(len(x),-1)
        loc=(self.location['scaler'].transform(loc).astype(np.float64)-self.location['center'])@self.location['coefficient']+self.location['ymean']
        m10=self.anchor+1.2*(.625*loc+.375*prior-self.anchor)-5
        reg=temporal(features['regional'],'trend','linear')
        reg=self.regional['ridge'].predict(self.regional['scaler'].transform(reg))
        reg=self.anchor+1.2*(reg-self.anchor)-5
        result=.25*reg+.75*m10
        assert result.shape==(len(x),) and np.isfinite(result).all()
        return result

def load_encoder(root=Path('data/external_models')):
    # The legacy forward path is reused verbatim. Acquisition/fitting routines
    # are never called, and its DATA root is scoped to the frozen encoder only.
    legacy.DATA=root
    model=LocationFeatures().eval()
    return model
