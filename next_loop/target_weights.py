"""Declared fitting-only target weights; no validation arguments."""
import numpy as np
def target_weights(y,method):
    y=np.asarray(y,dtype=float);assert y.ndim==1 and np.isfinite(y).all()
    if method=='unit':return np.ones(len(y))
    if method=='fast1.5':return np.where(y>=500,1.5,1.)
    if method=='fast2':return np.where(y>=500,2.,1.)
    assert method=='density'
    bins=np.clip(np.floor(y/20).astype(int),0,49)
    hist=np.bincount(bins,minlength=50).astype(float)
    offsets=np.arange(-3,4);kernel=np.exp(-.5*(offsets/1.5)**2);kernel/=kernel.sum()
    density=np.convolve(hist,kernel,mode='same')[bins]
    weights=1/np.sqrt(1+density);weights/=weights.mean()
    return np.clip(weights,.5,3.)
