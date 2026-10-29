"""Transfer explicitly allowed training files to our disposable worker."""
import argparse
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT/'.secrets/continuation_pod.json').read_text())
HEAD = {'Authorization': 'Bearer ' + C['token']}


def get(path):
    r = requests.get(C['url'] + path, headers=HEAD, timeout=(15, 45))
    r.raise_for_status()
    return r


def post(path, data=b''):
    r = requests.post(C['url'] + path, headers=HEAD, data=data, timeout=(15, 60))
    r.raise_for_status()
    return r.json()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('op', choices=['health', 'state', 'upload', 'run', 'download'])
    p.add_argument('stage', nargs='?', default='h8', choices=['h8', 'h9'])
    a = p.parse_args()
    if a.op in ('health', 'state'):
        print(get('/' + a.op).text)
    elif a.op == 'upload':
        files = [ROOT/'continuation'/name for name in ('prepare_training.py', 'models.py', 'metrics.py', 'train.py')]
        if a.stage == 'h9':
            files += [ROOT/'continuation/fast_loss.py']
        files += [ROOT/'data/continuation'/name for name in ('train_only.npz', 'train_only.manifest.json', 'eve_features.npz', 'eve_features.manifest.json')]
        for source in files:
            print(post('/upload/' + source.name, source.read_bytes()))
    elif a.op == 'run':
        print(post('/run/' + a.stage))
    elif a.op == 'download':
        dest = ROOT/'results/continuation'
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ('_results.json', '_predictions.npz'):
            name = a.stage + suffix
            b = get('/file/' + name).content
            (dest/name).write_bytes(b)
            print(name, len(b))
