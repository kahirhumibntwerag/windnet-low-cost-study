"""Reproduce a frozen final recipe in an exclusively new directory, without tuning."""
import argparse,json,shutil,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--freeze',required=True,type=Path);p.add_argument('--experiment-url',required=True);a=p.parse_args()
    frozen=a.freeze.resolve().read_bytes();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent;shutil.copytree(source,root/'matched',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # A new process imports all module paths relative to the new run directory.
    import subprocess
    def run(stage,**kw):
        q=root/'request.json';q.write_text(json.dumps({'stage':stage,**kw}));subprocess.check_call([sys.executable,'-u','-m','matched.runner','--request','request.json'],cwd=root)
    run('acquire')
    (root/'data/freeze.json').write_bytes(frozen);(root/'exports/freeze.json').write_bytes(frozen)
    import hashlib
    run('final',freeze_sha256=hashlib.sha256(frozen).hexdigest(),experiment_url=a.experiment_url)
    print(root/'exports/final_results.json')
if __name__=='__main__':main()
