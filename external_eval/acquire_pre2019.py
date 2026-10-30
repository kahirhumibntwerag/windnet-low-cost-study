"""Recover author training inputs only. No external-period arrays are accessed."""
import json
import tarfile
import time
from pathlib import Path
import numpy as np
from matched.runner import archive, numeric_npy, sha

DEST = Path('data/external_models/pre2019')

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    members = json.loads(Path('matched/source_manifest.json').read_text(encoding='utf8'))['members']
    expected = {x['member']: x for x in members}
    records = []
    start = time.time()
    response, source = archive('1JJjXjauT02z-RevZZ-8nQmHv9wr9wDrT')
    with response, source:
        for entry in source:
            if entry.name not in expected:
                continue
            spec = expected[entry.name]
            path = DEST / f"f{spec['fold']}_{Path(entry.name).stem}.npz"
            raw = source.extractfile(entry).read()
            assert sha(raw) == spec['release_member_sha256']
            if not path.exists():
                obj = numeric_npy(raw)
                out = np.asarray(obj['output'], dtype=np.float64)
                assert np.all(out[:, 0] < 2019), 'Not pre2019'
                np.savez_compressed(path, images=np.asarray(obj['input'], dtype=np.float64).reshape(20,256,256), output=out)
            records.append({'file': path.name, 'fold': spec['fold'], 'member': entry.name, 'source_sha256': sha(raw), 'processed_sha256': sha(path.read_bytes())})
            if len(records) % 20 == 0:
                print(json.dumps({'pre2019_blocks': len(records), 'seconds': round(time.time()-start)}), flush=True)
    assert len(records) == 157
    (DEST/'members.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf8')
    target=Path('data/external_models/kaffe');target.mkdir(exist_ok=True)
    response, source = archive('14DzwqXckiZBblxJ7MMzV8hJMcdeudCPU')
    with response, source:
        for entry in source:
            if entry.isfile() and Path(entry.name).name in ['googlenet.npy','googlenet.py']:
                (target/Path(entry.name).name).write_bytes(source.extractfile(entry).read())
    assert sha((target/'googlenet.npy').read_bytes()) == 'd93955ecfb66c9cdbe0c038f7f47fbf971e46dee30930970e22fc4a1b8fa184f'
    assert sha((target/'googlenet.py').read_bytes()) == 'c40b34bd4910e175ce121a41083ab5336bd6689167eb7206f63831706335c834'
    print(json.dumps({'pre2019_acquisition_complete': True, 'blocks':len(records), 'seconds':round(time.time()-start)}), flush=True)

if __name__=='__main__': main()
