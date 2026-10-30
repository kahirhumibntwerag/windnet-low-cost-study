"""Pinned SDOMLv1 SUM convention followed by author ProcessImg_common, 211 A."""
import datetime as dt
from pathlib import Path
import cv2
import numpy as np
from astropy.io import ascii, fits
from astropy.time import Time
from skimage.transform import SimilarityTransform, downscale_local_mean, warp

TABLE = Path('external_eval/calibration/aia_V8_20171210_050627_response_table.txt')

def degradation(date):
    # Same version8 instrument response as SDOMLv1 save_degrad.pro. The published
    # table's final epoch explicitly ends2030; no estimate from test images.
    t=Time(str(date)+'T00:00:00',scale='utc')
    table=ascii.read(TABLE)
    table=table[(table['WAVELNTH']==211)&(table['VER_NUM']==8)]
    starts=Time(table['T_START'].astype(str),scale='utc')
    stops=Time(table['T_STOP'].astype(str),scale='utc')
    ix=np.flatnonzero((starts<=t)&(t<stops))
    assert len(ix)==1, 'No supported calibration epoch'
    i=ix[0];elapsed=(t-starts[i]).to_value('day')
    factor=float(table['EFF_AREA'][i]/table['EFF_AREA'][0])*(1+float(table['EFFA_P1'][i])*elapsed+float(table['EFFA_P2'][i])*elapsed**2+float(table['EFFA_P3'][i])*elapsed**3)
    assert np.isfinite(factor) and factor>0
    return factor

def sdoml_v1(image, header, date, output_size=256):
    assert int(header['QUALITY'])==0
    assert int(header['WAVELNTH'])==211 and float(header['LVL_NUM'])==1.5
    x=np.asarray(image,dtype=np.float64).copy()
    assert x.shape==(1024,1024) and np.isfinite(x).all()
    mask=(x>0).astype(float);x[x<=0]=0
    factor=976.0/float(header['RSUN_OBS'])
    shift=x.shape[0]/2-factor*x.shape[0]/2
    transform=SimilarityTransform(scale=factor,translation=(shift,shift))
    x=warp(x,transform.inverse,preserve_range=True,mode='edge',output_shape=x.shape)
    mask=warp(mask,transform.inverse,preserve_range=True,mode='edge',output_shape=mask.shape)
    x=x/(mask+1e-8)
    x/=max(float(header['EXPTIME']),.01)*degradation(date)
    down=1024//output_size
    return (downscale_local_mean(x,(down,down))*down*down).astype(np.float32)

def windnet_common(image):
    x=np.asarray(image).copy()
    if x.shape[0]!=256:x=cv2.resize(x.astype(np.float64),(256,256))
    else:x/=4.0
    tmp=np.log10(x+1e-5);minv=np.min(x)
    tmp=np.clip(tmp,np.log10(25.0+minv),np.log10(2500.0+minv))
    span=np.max(tmp)-np.min(tmp)
    assert np.isfinite(tmp).all() and span>0
    return (tmp-np.min(tmp))*256.0/span

def process_fits(path, calendar_date):
    with fits.open(path,memmap=False) as f:
        return windnet_common(sdoml_v1(f[1].data,f[1].header,calendar_date))
