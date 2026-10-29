import argparse,json,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'.secrets/pod.json').read_text())
HEAD={'Authorization':'Bearer '+C['token']}
def get(path):
    r=requests.get(C['url']+path,headers=HEAD,timeout=(10,60));r.raise_for_status();return r
def post(path,data=b''):
    r=requests.post(C['url']+path,headers=HEAD,data=data,timeout=(10,90));r.raise_for_status();return r.json()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('op',choices=['health','upload','state','run','download']);p.add_argument('arg',nargs='?');args=p.parse_args()
    if args.op=='health':print(get('/health').text)
    if args.op=='state':print(get('/state').text)
    if args.op=='upload':
        for source in [ROOT/'study.py',ROOT/'data/study_inputs.npz',ROOT/'data/source_folds.npz']:
            print(post('/upload/'+source.name,source.read_bytes()))
    if args.op=='run':
        if args.arg=='loop4':
            print(post('/upload/ensemble_mode.json',b'{"stage":"loop4"}'))
            print(post('/run/loop3'))
        else:print(post('/run/'+args.arg))
    if args.op=='download':
        dest=ROOT/'results';dest.mkdir(exist_ok=True)
        for name in args.arg.split(','):
            b=get('/file/'+name).content;(dest/name).write_bytes(b);print(name,len(b))
