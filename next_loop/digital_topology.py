"""Sampled digital Betti/morphology features; no persistent-homology claim."""
import numpy as np
from scipy import ndimage
from next_loop.regional_features import regions

THRESHOLDS=(.2,.3,.4,.5,.6,.7,.8,.9)

def component_stats(foreground,domain_area):
    a=np.asarray(foreground,dtype=bool)
    assert a.ndim==2 and domain_area>0
    # Dual digital connectivity prevents diagonal foreground contacts from
    # simultaneously connecting background across those same contacts.
    labels,n0=ndimage.label(a,structure=np.ones((3,3),dtype=np.uint8))
    background,n_background=ndimage.label(~a,structure=ndimage.generate_binary_structure(2,1))
    border=np.unique(np.concatenate([background[0],background[-1],background[:,0],background[:,-1]]))
    # The registered full-disk foreground is strictly inside the frame, so the
    # complement has exactly one exterior component. Check that assumption.
    exterior=border[border>0]
    assert len(exterior)==1
    n1=max(n_background-1,0)
    largest=int(np.bincount(labels.ravel())[1:].max()) if n0 else 0
    transitions=np.count_nonzero(a[1:]!=a[:-1])+np.count_nonzero(a[:,1:]!=a[:,:-1])
    return np.array([np.log1p(n0),np.log1p(n1),largest/domain_area,transitions/(4*domain_area)],dtype=float)

def topology(images):
    images=np.asarray(images,dtype=np.float32)
    assert images.ndim==3 and images.shape[1:]==(256,256) and np.isfinite(images).all()
    yy,xx=np.mgrid[:256,:256];disk=(xx-127.5)**2+(yy-127.5)**2<112.5**2
    median=np.median(images[:,disk],axis=1);den=np.maximum(median,1e-6)
    relative=images/den[:,None,None]
    result=np.empty((len(images),len(THRESHOLDS),4),dtype=float)
    for i,image in enumerate(relative):
        for j,threshold in enumerate(THRESHOLDS):
            result[i,j]=component_stats(disk&(image<threshold),int(disk.sum()))
    assert np.isfinite(result).all()
    return result

def feature_families(images):
    base=regions(images,(6,6));topo=topology(images);n=len(images)
    return {'betti':np.concatenate([topo[:,:,:2].reshape(n,-1),base[:,-2:]],axis=1),
            'morphology':np.concatenate([topo.reshape(n,-1),base[:,-2:]],axis=1),
            'regional_morphology':np.concatenate([base,topo.reshape(n,-1)],axis=1)}
