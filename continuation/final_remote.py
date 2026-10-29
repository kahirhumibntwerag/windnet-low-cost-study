"""Transfer only the sealed E13 bundle and executable sources."""
import argparse
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT/'.secrets/final_pod.json').read_text())
HEAD = {'Authorization': 'Bearer '+C['token']}


def get(path):
    r = requests.get(C['url']+path, headers=HEAD, timeout=(15, 120))
    r.raise_for_status()
    return r


def post(path, data=b''):
    r = requests.post(C['url']+path, headers=HEAD, data=data, timeout=(15, 120))
    r.raise_for_status()
    return r.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['health', 'state', 'upload', 'run', 'download'])
    args = parser.parse_args()
    if args.operation in ('health', 'state'):
        print(get('/'+args.operation).text)
    elif args.operation == 'upload':
        manifest = json.loads((ROOT/'results/locked_final/freeze_manifest.json').read_text())
        files = [ROOT/'continuation'/name for name in manifest['code_sha256']]
        files += [ROOT/'results/locked_final/freeze_manifest.json']
        files += [ROOT/'data/locked_final'/name for name in ['final_bundle.npz', 'final_bundle.manifest.json']]
        for path in files:
            r = post('/upload/'+path.name, path.read_bytes())
            print({'saved': r['saved'], 'bytes': r['bytes']}, flush=True)
        print('All frozen files transferred.', flush=True)
    elif args.operation == 'run':
        print(post('/run'))
    else:
        for name in ['final_results.json', 'predictions.npz', 'comparison.csv', 'checkpoints.tar.gz', 'execution_started.json']:
            target = ROOT/'results/locked_final'/name
            if target.exists():
                raise FileExistsError('Refusing to overwrite '+str(target))
            b = get('/file/'+name).content
            target.write_bytes(b)
            print(name, len(b), flush=True)
