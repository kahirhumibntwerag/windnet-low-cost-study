"""Code-only upload and aggregate/checkpoint download for the new F3 worker."""
import argparse,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('op',choices=['health','state','upload','prepare','final','download']);ap.add_argument('names',nargs='*');args=ap.parse_args()
    c=json.loads((ROOT/'.secrets/f3_pod.json').read_text());headers={'Authorization':'Bearer '+c['token']}
    def get(path):
        r=requests.get(c['url']+path,headers=headers,timeout=45);r.raise_for_status();return r
    if args.op in ['health','state']:print(get('/'+args.op).text)
    elif args.op=='upload':
        assert not get('/state').json()['running']
        files=[]
        for pkg in ['matched','event_loop']:files.extend((ROOT/pkg).glob('*.py'))
        files.extend(ROOT/'matched'/n for n in ['source_manifest.json','reference_sources.json','protocol.json'])
        files.extend(ROOT/'next_loop'/n for n in ['__init__.py','regional_features.py','target_weights.py','f3_panel.py','f3.py','f3_entry.py','f3_transport.py','f3_protocol.json','m13_training_selection.json','test_f3.py','test_regions.py','test_target_weights.py'])
        for p in files:
            r=requests.post(c['url']+'/upload/'+p.parent.name+'/'+p.name,headers=headers,data=p.read_bytes(),timeout=90);r.raise_for_status()
        print('Uploaded F3 scientific code and training selection only; no raw local data, historical outputs or paper')
    elif args.op in ['prepare','final']:
        cfg={'stage':args.op}
        if args.op=='final':
            assert len(args.names)==1;cfg['freeze_sha256']=args.names[0]
        r=requests.post(c['url']+'/run',headers=headers,json=cfg,timeout=45);r.raise_for_status();print(r.text)
    else:
        for name in args.names:
            assert '/' not in name and '\\' not in name and '..' not in name
            dest=ROOT/('artifacts/next_loop' if name.endswith('.gz') else 'results/next_loop')/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(c['url']+'/file/'+name,headers=headers,timeout=180,stream=True) as r:
                r.raise_for_status()
                with dest.open('wb') as out:
                    for b in r.iter_content(1024*1024):out.write(b)
            print(name,dest.stat().st_size)

if __name__=='__main__':main()
