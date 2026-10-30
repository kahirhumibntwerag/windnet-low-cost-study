"""Two-stage F3 entry: acquire development and freeze, then explicitly evaluate."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.request).read_text())
    assert cfg['stage'] in ['prepare','final']
    if cfg['stage']=='prepare':
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet','numpy==2.1.2','scipy==1.16.1','scikit-learn==1.7.2','requests==2.32.3'])
        import torch
        assert torch.cuda.is_available() and torch.__version__.startswith('2.8.0')
        from matched.runner import EXPORT,save_json
        EXPORT.mkdir(exist_ok=True)
        from event_loop.runner import acquire
        acquire()
        from next_loop.f3 import freeze
        freeze()
        import numpy,scipy,sklearn,platform
        save_json(EXPORT/'f3_runtime.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
    else:
        assert set(cfg)=={'stage','freeze_sha256'}
        from next_loop.f3 import final
        final(cfg['freeze_sha256'])

if __name__=='__main__':main()
