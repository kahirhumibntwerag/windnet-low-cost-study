"""Fixed image-plane intensity proxies, not physical coronal-hole segmentation."""
import numpy as np

def regions(images, grid):
    images=np.asarray(images,dtype=np.float32)
    assert images.ndim==3 and images.shape[1:]==(256,256) and np.isfinite(images).all()
    yy,xx=np.mgrid[:256,:256];disk=(xx-127.5)**2+(yy-127.5)**2<112.5**2
    values=images[:,disk];median=np.median(values,axis=1);den=np.maximum(median,1e-6)
    relative=images/den[:,None,None];out=[]
    rows,cols=grid;ys=np.linspace(0,256,rows+1,dtype=int);xs=np.linspace(0,256,cols+1,dtype=int)
    for y0,y1 in zip(ys[:-1],ys[1:]):
        for x0,x1 in zip(xs[:-1],xs[1:]):
            mask=disk[y0:y1,x0:x1]
            if not mask.any():out.extend([np.zeros(len(images))]*4);continue
            v=relative[:,y0:y1,x0:x1][:,mask]
            out.extend([(v<t).mean(axis=1) for t in [.3,.5,.7]])
            out.append(np.clip(v,0,3).mean(axis=1))
    q=np.quantile(values,[.25,.75],axis=1)
    out.extend([np.log1p(np.maximum(median,0)),(q[1]-q[0])/den])
    return np.stack(out,axis=1).astype(np.float64)

def temporal(frames,view,encoding):
    assert frames.ndim==3 and frames.shape[1]==4
    if view=='flatten':z=frames.reshape(len(frames),-1)
    elif view=='trend':z=np.concatenate([frames[:,-1],frames.mean(axis=1),frames[:,-1]-frames[:,0]],axis=1)
    else:raise ValueError(view)
    if encoding=='squares':z=np.concatenate([z,z*z],axis=1)
    else:assert encoding=='linear'
    return z
