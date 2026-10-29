"""Recover only OMNI labels and fold identities from authors' processed release.

No AIA image is retained, trained on, or used as a predictor. The compressed
release is streamed because it bundles the labels with processed image arrays.
"""
import argparse, hashlib, io, json, pickle, tarfile, time
from pathlib import Path
import numpy as np
import requests

class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('numpy.core.multiarray','_reconstruct'),
                ('numpy._core.multiarray','_reconstruct'), ('numpy','ndarray'),
                ('numpy','dtype'), ('_codecs','encode')}:
            return super().find_class(module,name)
        raise ValueError(f'Disallowed pickle global: {module}.{name}')

def numeric_npy(content):
    f=io.BytesIO(content)
    ver=np.lib.format.read_magic(f)
    reader=np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0
    shape,fortran,dtype=reader(f)
    if not dtype.hasobject:
        raise ValueError('Expected original dictionary wrapper')
    return NumericUnpickler(f,encoding='latin1').load().item()

def run(channel):
    ids={211:'1JJjXjauT02z-RevZZ-8nQmHv9wr9wDrT',193:'1mu5_g1VmUt3tZ082dAVcZ5xx7G_LtdLh'}
    path=Path('data'); path.mkdir(exist_ok=True)
    url=f'https://drive.usercontent.google.com/download?id={ids[channel]}&export=download&confirm=t'
    r=requests.get(url,stream=True,timeout=(30,120)); r.raise_for_status()
    manifest=[]; labels=[]; start=time.time()
    with tarfile.open(fileobj=r.raw,mode='r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.npy'):
                continue
            if 'Fold' not in member.name:
                continue
            data=archive.extractfile(member).read()
            original=numeric_npy(data)
            output=np.asarray(original['output'],dtype=np.float64)
            import re
            fold=int(re.search(r'Fold([1-5])(?:/|\\)',member.name).group(1))
            manifest.append({'member':member.name,'fold':fold,'partition':Path(member.name).stem,
                'release_member_sha256':hashlib.sha256(data).hexdigest(),
                'output_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
                'shape':list(output.shape)})
            labels.append(output)
            del original,data
            if len(labels)%10==0:
                print(json.dumps({'recovered':len(labels),'elapsed_s':round(time.time()-start,1)}),flush=True)
    np.savez_compressed(path/f'windnet{channel}_labels.npz',output=np.stack(labels),
        folds=np.array([x['fold'] for x in manifest]),
        partitions=np.array([x['partition'] for x in manifest]))
    record={'source_url':url,'channel':channel,'release':'authors processed fold archive',
        'members':manifest,'images_retained':False,'predictors_from_images':False}
    (path/f'windnet{channel}_manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'complete':True,'partitions':len(labels),'folds':np.bincount([x['fold'] for x in manifest]).tolist(),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--channel',type=int,default=211)
    run(p.parse_args().channel)
