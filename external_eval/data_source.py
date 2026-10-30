"""Data acquisition called only by the frozen external-test entry point."""
import concurrent.futures,hashlib,io,json,re,time
from pathlib import Path
import numpy as np
import requests
from astropy.io import fits
from external_eval.preprocessing import sdoml_v1,windnet_common

BASE='https://jsoc1.stanford.edu/data/aia/synoptic/'
OMNI='https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_01_av.dat'

def get(url,allow404=False):
    for attempt in range(4):
        try:
            r=requests.get(url,timeout=(20,90))
            if r.status_code==404 and allow404:return None
            r.raise_for_status();return r
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)

def choose_url(date):
    year,month,day=str(date).split('-');code=year+month+day
    root=BASE+f'{year}/{month}/{day}/'
    direct=root+f'H0000/AIA{code}_0000_0211.fits'
    response=get(direct,allow404=True)
    if response is not None:return direct,response
    listing=get(root,allow404=True)
    if listing is None:return None,None
    hours=sorted(set(re.findall(r'href="(H\d{4}/)"',listing.text)))
    for hour in hours:
        page=get(root+hour,allow404=True)
        if page is None:continue
        names=sorted(set(re.findall(r'href="(AIA'+code+r'_\d{4}_0211\.fits)"',page.text)))
        if names:
            url=root+hour+names[0]
            return url,get(url)
    return None,None

def acquire_images(image_dates):
    root=Path('data/external_sealed');root.mkdir(parents=True,exist_ok=True)
    def one(date):
        url,response=choose_url(date)
        row={'date':str(date),'url':url}
        if response is None:return row|{'valid':False,'reason':'missing_source'},None
        raw=response.content;row|={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        # Source-only exclusions, never adjusted using forecasts or target values.
        try:
            with fits.open(io.BytesIO(raw),memmap=False) as hdus:
                h=hdus[1].header;image=hdus[1].data
                if int(h['QUALITY'])!=0:return row|{'valid':False,'reason':'quality_flag'},None
                if image.shape!=(1024,1024) or not np.isfinite(image).all():return row|{'valid':False,'reason':'nonfinite_or_shape'},None
                if int(h['WAVELNTH'])!=211 or float(h['LVL_NUM'])!=1.5:return row|{'valid':False,'reason':'wrong_channel_or_level'},None
                if not (np.isfinite(float(h['EXPTIME'])) and float(h['EXPTIME'])>0 and np.isfinite(float(h['RSUN_OBS'])) and float(h['RSUN_OBS'])>0):return row|{'valid':False,'reason':'invalid_exposure_or_radius'},None
                value=windnet_common(sdoml_v1(image,h,str(date))).astype(np.float32)
                return row|{'valid':True,'reason':None,'processed_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'observation_time':str(h.get('T_OBS'))},value
        except (OSError,IndexError,KeyError,ValueError,AssertionError):
            return row|{'valid':False,'reason':'unreadable_or_processing_invalid'},None
    images=np.zeros((len(image_dates),256,256),dtype=np.float32);valid=np.zeros(len(image_dates),dtype=bool);rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,value) in enumerate(pool.map(one,image_dates)):
            rows.append(row)
            if value is not None:images[i]=value;valid[i]=True
            if (i+1)%100==0:print(json.dumps({'external_images_acquired':i+1,'requested':len(image_dates),'individual_values_exported':False}),flush=True)
    manifest=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    (root/'image_source_manifest.json').write_bytes(manifest)
    np.savez_compressed(root/'processed_images.npz',images=images,dates=image_dates,valid=valid)
    return images,valid,{'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'requested_images':len(rows),'valid_images':int(valid.sum()),
                        'excluded_by_reason':{reason:sum(r['reason']==reason for r in rows) for reason in sorted(set(r['reason'] for r in rows if r['reason']))},
                        'source':BASE,'preprocessing':'SDOMLv1 SUM256 then author ProcessImg_common211; pinned2017 V8 calibration'}

def acquire_targets(target_dates):
    raw=get(OMNI).content
    root=Path('data/external_sealed');(root/'omni_01_av.dat').write_bytes(raw)
    desired={str(d):i for i,d in enumerate(target_dates)}
    y=np.full(len(target_dates),np.nan);sigma=np.full(len(target_dates),np.nan);seen=set()
    for line in raw.decode('ascii').splitlines():
        values=line.split()
        if len(values)<32:continue
        year=int(values[0])
        if not 2020<=year<=2025:continue
        day=int(values[1]);date=str(np.datetime64(f'{year}-01-01')+np.timedelta64(day-1,'D'))
        if date not in desired:continue
        assert date not in seen,'Duplicate daily target';seen.add(date);i=desired[date]
        speed=float(values[24]);std=float(values[31])
        if np.isfinite(speed) and 0<speed<9999:y[i]=speed
        if np.isfinite(std) and 0<=std<9999:sigma[i]=std
    np.savez_compressed(root/'targets.npz',dates=target_dates,y=y,sigma=sigma)
    return y,sigma,{'url':OMNI,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'speed_word_one_based':25,'sigma_word_one_based':32,
                    'daily_rows_found':len(seen),'requested_days':len(target_dates),'valid_speed_days':int(np.isfinite(y).sum()),'positive_sigma_days':int((np.isfinite(sigma)&(sigma>0)).sum()),
                    'definition':'daily arithmetic mean bulk speed; daily sigma across hourly speeds; no minimum-hour threshold; fill9999 invalid'}
