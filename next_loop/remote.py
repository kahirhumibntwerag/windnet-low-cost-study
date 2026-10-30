"""Authenticated transport; never upload historical final exports or raw local data."""
import argparse, json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('op',choices=['health','state','upload','run','download']);ap.add_argument('names',nargs='*');args=ap.parse_args()
    c=json.loads((ROOT/'.secrets/next_pod.json').read_text());h={'Authorization':'Bearer '+c['token']}
    if args.op in ['health','state']:
        r=requests.get(c['url']+'/'+args.op,headers=h,timeout=45);r.raise_for_status();print(r.text)
    elif args.op=='upload':
        r=requests.get(c['url']+'/state',headers=h,timeout=45);r.raise_for_status();assert not r.json()['running']
        for pkg in ['matched','event_loop','next_loop']:
            for p in sorted((ROOT/pkg).glob('*')):
                if p.suffix not in ['.py','.json']:continue
                if pkg=='next_loop' and p.name in ['registry.json','execution.json']:continue
                r=requests.post(c['url']+'/upload/'+pkg+'/'+p.name,headers=h,data=p.read_bytes(),timeout=90);r.raise_for_status()
        print('Uploaded code and training-only reference; no historical final results')
    elif args.op=='run':
        r=requests.post(c['url']+'/run',headers=h,json={'stage':'m11'},timeout=45);r.raise_for_status();print(r.text)
    else:
        out=ROOT/'results/next_loop';out.mkdir(parents=True,exist_ok=True)
        for name in args.names:
            dest=(ROOT/'artifacts/next_loop'/name) if name.endswith('.gz') else out/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(c['url']+'/file/'+name,headers=h,timeout=180,stream=True) as r:
                r.raise_for_status()
                with dest.open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
            print(name,dest.stat().st_size)
if __name__=='__main__':main()
