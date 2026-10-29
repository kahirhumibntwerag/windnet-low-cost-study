"""CLI transport; credentials stay in ignored .secrets, output is aggregate-only."""
import argparse,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('op',choices=['health','state','upload','run','download']);p.add_argument('args',nargs='*');a=p.parse_args()
    c=json.loads((ROOT/'.secrets/matched_pod.json').read_text());h={'Authorization':'Bearer '+c['token']}
    if a.op in ['health','state']:
        r=requests.get(c['url']+'/'+a.op,headers=h,timeout=45);r.raise_for_status();print(r.text)
    elif a.op=='upload':
        for f in sorted((ROOT/'matched').glob('*')):
            if f.suffix in ['.py','.json']:
                r=requests.post(c['url']+'/upload/'+f.name,headers=h,data=f.read_bytes(),timeout=90);r.raise_for_status();print(f.name,r.json().get('sha256'))
    elif a.op=='run':
        cfg=json.loads(a.args[0]);r=requests.post(c['url']+'/run',headers=h,json=cfg,timeout=45);r.raise_for_status();print(r.text)
    elif a.op=='download':
        dest=ROOT/'results/matched';dest.mkdir(parents=True,exist_ok=True)
        for name in a.args:
            r=requests.get(c['url']+'/file/'+name,headers=h,timeout=180);r.raise_for_status();(dest/name).write_bytes(r.content);print(name,len(r.content))
if __name__=='__main__':main()
