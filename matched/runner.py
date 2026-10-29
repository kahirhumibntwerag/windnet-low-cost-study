"""Stage entry point. Development stages cannot open sealed Fold5 members."""
import argparse, ast, hashlib, io, json, os, pickle, re, sys, tarfile, time
from pathlib import Path
import numpy as np
import requests

ROOT=Path.cwd(); DATA=ROOT/'data'; EXPORT=ROOT/'exports'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False))
class NumericUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) in {('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}:
            return super().find_class(module,name)
        raise ValueError('Disallowed pickle global '+module+'.'+name)
def numeric_npy(raw):
    f=io.BytesIO(raw);version=np.lib.format.read_magic(f)
    read=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    _,_,dtype=read(f)
    if not dtype.hasobject: raise ValueError('Expected numeric dictionary wrapper')
    return NumericUnpickler(f,encoding='latin1').load().item()
def archive(file_id):
    url=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    r=requests.get(url,stream=True,timeout=(30,180));r.raise_for_status()
    return r,tarfile.open(fileobj=r.raw,mode='r|gz')
def acquire():
    DATA.mkdir(exist_ok=True);(DATA/'development').mkdir(exist_ok=True);(DATA/'sealed').mkdir(exist_ok=True);(DATA/'kaffe').mkdir(exist_ok=True)
    source=json.loads((ROOT/'matched/source_manifest.json').read_text());expected={x['member']:x for x in source['members']}
    records=[]; t=time.time()
    r,tar=archive('1JJjXjauT02z-RevZZ-8nQmHv9wr9wDrT')
    with r,tar:
        for m in tar:
            if m.name not in expected: continue
            raw=tar.extractfile(m).read();e=expected[m.name];assert sha(raw)==e['release_member_sha256'],'Source member hash mismatch'
            f=e['fold'];name=f"f{f}_{Path(m.name).stem}.npy"
            if f==5:
                (DATA/'sealed'/name).write_bytes(raw)
                parsed=False
            else:
                obj=numeric_npy(raw);im=np.asarray(obj['input']);out=np.asarray(obj['output'],dtype=np.float64)
                assert len(im)==20 and len(out)==20
                np.savez_compressed(DATA/'development'/name.replace('.npy','.npz'),images=im.astype(np.float32).reshape(20,256,256),output=out)
                parsed=True
            records.append({'member':m.name,'fold':f,'sha256':sha(raw),'parsed_for_development':parsed,'file':name})
            if len(records)%20==0:print(json.dumps({'stage':'acquire','members':len(records),'seconds':round(time.time()-t)}),flush=True)
    assert len(records)==len(expected)==157
    save_json(DATA/'members.json',records)
    save_json(EXPORT/'acquisition.json',{'members':len(records),'source_member_hashes_verified':True,'fold5_members_unparsed':sum(x['fold']==5 for x in records),'development_folds':[1,2,3,4],'images':'original 211-Angstrom release','seconds':time.time()-t})
    print('Original images recovered; Fold5 remains uninterpreted bytes',flush=True)
    r,tar=archive('14DzwqXckiZBblxJ7MMzV8hJMcdeudCPU')
    files=[]
    with r,tar:
        for m in tar:
            if not m.isfile():continue
            name=Path(m.name).name
            if name in ['googlenet.npy','googlenet.py','network.py']:
                raw=tar.extractfile(m).read();(DATA/'kaffe'/name).write_bytes(raw)
                files.append({'name':name,'sha256':sha(raw),'bytes':len(raw)})
                if name.endswith('.py'):(EXPORT/name.replace('.py','.txt')).write_bytes(raw)
    save_json(EXPORT/'kaffe_manifest.json',{'files':files})
    print(json.dumps({'stage':'acquire_complete','kaffe_files':files}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();cfg=json.loads(Path(a.request).read_text())
    EXPORT.mkdir(exist_ok=True)
    if cfg['stage']=='acquire':acquire()
    else:
        from matched.science import run
        run(cfg)
if __name__=='__main__':main()
