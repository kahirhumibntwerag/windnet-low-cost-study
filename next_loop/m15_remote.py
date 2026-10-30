"""Restricted M15 code upload and training-only result downloads."""
import argparse,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('op',choices=['health','state','upload','run','download']);ap.add_argument('names',nargs='*');args=ap.parse_args()
    credentials=json.loads((ROOT/'.secrets/m15_pod.json').read_text());headers={'Authorization':'Bearer '+credentials['token']};url=credentials['url']
    if args.op in ['health','state']:
        response=requests.get(url+'/'+args.op,headers=headers,timeout=45);response.raise_for_status();print(response.text)
    elif args.op=='upload':
        protocol=json.loads((ROOT/'next_loop/m15_protocol.json').read_text())
        paths=[*protocol['frozen_sources'],'next_loop/m15_protocol.json']
        for name in paths:
            path=ROOT/name
            response=requests.post(url+'/upload/'+path.parent.name+'/'+path.name,headers=headers,data=path.read_bytes(),timeout=90);response.raise_for_status()
        print('Uploaded frozen M15 source and training-control references only; no F3 or historical exports')
    elif args.op=='run':
        response=requests.post(url+'/run',headers=headers,json={'stage':'m15'},timeout=45);response.raise_for_status();print(response.text)
    else:
        for name in args.names:
            assert '/' not in name and '\\' not in name and '..' not in name and name.startswith('m15_')
            path=ROOT/('artifacts/next_loop' if name.endswith('.gz') else 'results/next_loop')/name;path.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(url+'/file/'+name,headers=headers,timeout=180,stream=True) as response:
                response.raise_for_status()
                with path.open('wb') as out:
                    for chunk in response.iter_content(1024*1024):out.write(chunk)
            print(name,path.stat().st_size)

if __name__=='__main__':main()
