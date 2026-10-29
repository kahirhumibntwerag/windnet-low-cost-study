"""Build causal tabular inputs while preserving every released evaluation row."""
import datetime as dt, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    d=np.load(ROOT/'data/windnet211_labels.npz')
    out,folds,parts=d['output'],d['folds'],d['partitions']
    full=np.load(ROOT/'upstream/BenchmarkModels/Persistence_27/OMNIWEB_data_dict.npy',allow_pickle=True,encoding='latin1').item()
    keys=list(full)
    dates=np.array([np.datetime64(dt.date(int(y),1,1)+dt.timedelta(days=int(j)-1)) for y,j in zip(full['YEAR'],full['DOY'])])
    lookup={str(day):i for i,day in enumerate(dates)}
    allvals=np.stack([full[k] for k in keys],axis=1).astype(float)
    # NASA OMNI fill code for proton density and its dispersion is 999.9.
    for k in [10,13]:
        allvals[allvals[:,k]>=999.0,k]=np.nan
    def past(date,lag,col):
        i=lookup.get(str(date-np.timedelta64(int(lag),'D')))
        return np.nan if i is None else allvals[i,col]
    xshort=[]; xrec=[]; xmulti=[]; target=[]; sigma=[]; samplefold=[]; block=[]; offsets=[]; sampledates=[]
    labels_check=[]; block_dates=[]
    for bi,arr in enumerate(out):
        bd=np.array([np.datetime64(dt.date(int(y),1,1)+dt.timedelta(days=int(j)-1)) for y,j in arr[:,:2]])
        block_dates.append(bd)
        for j,row in enumerate(arr):
            idx=lookup[str(bd[j])]
            labels_check.append(np.isclose(row[11],allvals[idx,11]))
        for off in range(6,20):
            date=bd[off]
            short=arr[off-6:off-2,11].tolist() # t-6,t-5,t-4,t-3
            assert np.all(bd[off-6:off-2] <= date-np.timedelta64(3,'D'))
            recurrence=[past(date,lag,11) for lag in list(range(3,17))+list(range(24,32))+list(range(51,59))]
            extra=[]
            for col in [3,4,5,6,7,8,9,10,12,13,14]:
                recent=np.array([past(date,lag,col) for lag in range(3,7)])
                if col in [9,10,12,13]: recent=np.log1p(np.maximum(recent,0))
                extra.extend(recent.tolist())
            xshort.append(short); xrec.append(short+recurrence); xmulti.append(short+recurrence+extra)
            target.append(arr[off,11]); sigma.append(arr[off,14]); samplefold.append(folds[bi]);block.append(bi);offsets.append(off);sampledates.append(str(date))
    assert all(labels_check), 'Bundled OMNI speed differs from original fold labels'
    flatdays={f:set(str(x) for b,fd in zip(block_dates,folds) if fd==f for x in b) for f in range(1,6)}
    overlap={f'{a}-{b}':len(flatdays[a]&flatdays[b]) for a in range(1,6) for b in range(a+1,6)}
    assert not any(overlap.values()),'Original folds share dates: audit before proceeding'
    np.savez_compressed(ROOT/'data/study_inputs.npz',short=np.array(xshort),recurrence=np.array(xrec),multi=np.array(xmulti),
        y=np.array(target),sigma=np.array(sigma),fold=np.array(samplefold),block=np.array(block),offset=np.array(offsets),date=np.array(sampledates))
    audit={'source_commit':'a996baa075ecdbd2b1c15533002692e29db43c19','channel':211,'history':4,'delay':3,
        'n_partitions':len(parts),'partition_size':int(out.shape[1]),'n_test_rows':len(target),
        'evaluation_offsets':list(range(6,20)),'training_offsets':list(range(6,19)),
        'fold_partition_counts':{str(f):int(np.sum(folds==f)) for f in range(1,6)},
        'fold_test_rows':{str(f):int(np.sum(np.array(samplefold)==f)) for f in range(1,6)},
        'date_overlap_between_folds':overlap,'omni_speed_checks':len(labels_check),'omni_speed_checks_passed':bool(all(labels_check)),
        'earliest_target':min(sampledates),'latest_target':max(sampledates),
        'input_dimensions':{'short':len(xshort[0]),'recurrence':len(xrec[0]),'multi':len(xmulti[0])},
        'max_predictor_date_relative_to_target':-3,'image_predictors':False,
        'data_sha256':hashlib.sha256((ROOT/'data/study_inputs.npz').read_bytes()).hexdigest(),
        'raw_omni_sha256':hashlib.sha256((ROOT/'upstream/BenchmarkModels/Persistence_27/OMNIWEB_data_dict.npy').read_bytes()).hexdigest()}
    (ROOT/'data/split_audit.json').write_text(json.dumps(audit,indent=2))
    # Track the originating fold of every measurement. During fitting, mask
    # measurements from the held-out folds even when used as historical lags.
    datefold={str(day):int(f) for b,f in zip(block_dates,folds) for day in b}
    lags_short=[6,5,4,3]
    lags_rec=lags_short+list(range(3,17))+list(range(24,32))+list(range(51,59))
    lags_multi=lags_rec+[x for col in range(11) for x in [3,4,5,6]]
    sources={name:np.array([[datefold.get(str(np.datetime64(date)-np.timedelta64(lag,'D')),0) for lag in lags] for date in sampledates])
        for name,lags in [('short',lags_short),('recurrence',lags_rec),('multi',lags_multi)]}
    np.savez_compressed(ROOT/'data/source_folds.npz',**sources)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
