import argparse,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('op',choices=['health','state','upload','run','download']);p.add_argument('args',nargs='*');a=p.parse_args();c=json.loads((ROOT/'.secrets/event_pod.json').read_text());h={'Authorization':'Bearer '+c['token']}
    if a.op in ['health','state']:
        r=requests.get(c['url']+'/'+a.op,headers=h,timeout=45);r.raise_for_status();print(r.text)
    elif a.op=='upload':
        state=requests.get(c['url']+'/state',headers=h,timeout=45);state.raise_for_status()
        if state.json()['running']:raise RuntimeError('Refuse package mutation while an experiment is running')
        for package in ['matched','event_loop']:
            for f in sorted((ROOT/package).glob('*')):
                if f.suffix not in ['.py','.json']:continue
                r=requests.post(c['url']+'/upload/'+package+'/'+f.name,headers=h,data=f.read_bytes(),timeout=90);r.raise_for_status()
        print('Uploaded scientific packages and public protocols; no prior results or credentials uploaded')
    elif a.op=='run':
        r=requests.post(c['url']+'/run',headers=h,json=json.loads(a.args[0]),timeout=45);r.raise_for_status();print(r.text)
    else:
        dest=ROOT/'results/event_loop';dest.mkdir(parents=True,exist_ok=True)
        for name in a.args:
            with requests.get(c['url']+'/file/'+name,headers=h,timeout=180,stream=True) as r:
                r.raise_for_status()
                with (dest/name).open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
            print(name,(dest/name).stat().st_size)
if __name__=='__main__':main()
