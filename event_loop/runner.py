import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from matched.runner import DATA,EXPORT,ROOT,archive,numeric_npy,save_json,sha

def acquire():
    DATA.mkdir(exist_ok=True);(DATA/'development').mkdir(exist_ok=True);(DATA/'kaffe').mkdir(exist_ok=True)
    expected={x['member']:x for x in json.loads((ROOT/'matched/source_manifest.json').read_text())['members']};records=[];t=time.time()
    r,tar=archive('1JJjXjauT02z-RevZZ-8nQmHv9wr9wDrT')
    with r,tar:
        for m in tar:
            if m.name not in expected:continue
            raw=tar.extractfile(m).read();e=expected[m.name];assert sha(raw)==e['release_member_sha256'];f=e['fold'];name=f"f{f}_{Path(m.name).stem}.npy"
            if f!=5:
                obj=numeric_npy(raw);np.savez_compressed(DATA/'development'/name.replace('.npy','.npz'),images=np.asarray(obj['input'],dtype=np.float32).reshape(20,256,256),output=np.asarray(obj['output'],dtype=float))
            records.append({'member':m.name,'fold':f,'file':name,'sha256':sha(raw),'parsed_for_development':f!=5,'retained':f!=5})
            if len(records)%30==0:print(json.dumps({'members_verified':len(records),'seconds':time.time()-t}),flush=True)
    assert len(records)==157;save_json(DATA/'members.json',records)
    r,tar=archive('14DzwqXckiZBblxJ7MMzV8hJMcdeudCPU')
    with r,tar:
        for m in tar:
            if m.isfile() and Path(m.name).name in ['googlenet.npy','googlenet.py']:(DATA/'kaffe'/Path(m.name).name).write_bytes(tar.extractfile(m).read())
    assert sha((DATA/'kaffe/googlenet.npy').read_bytes())=='d93955ecfb66c9cdbe0c038f7f47fbf971e46dee30930970e22fc4a1b8fa184f'
    save_json(EXPORT/'event_acquisition.json',{'original_members_hash_verified':157,'development_blocks':108,'fold5_members_parsed':0,'fold5_members_retained':0,'prior_final_results_available':False,'seconds':time.time()-t})
    print('Development data ready; no Fold5 data retained',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();cfg=json.loads(Path(a.request).read_text());EXPORT.mkdir(exist_ok=True)
    if cfg['stage']=='acquire':acquire();return
    if importlib.util.find_spec('sklearn') is None:subprocess.check_call([sys.executable,'-m','pip','install','--quiet','scikit-learn==1.7.2','scipy==1.16.1'])
    if cfg['stage']=='mixtures':
        from event_loop.mixtures import run
        run();return
    from event_loop.science import run
    run(cfg)
if __name__=='__main__':main()
