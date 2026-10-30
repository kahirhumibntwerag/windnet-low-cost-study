"""One-purpose preregistered workload; never invoke an original final stage."""
import json, subprocess, sys, tarfile
from matched.runner import EXPORT, DATA, ROOT, save_json, sha

def main():
    EXPORT.mkdir(exist_ok=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy==2.1.2', 'scipy==1.16.1', 'scikit-learn==1.7.2', 'requests==2.32.3'])
    from event_loop.runner import acquire
    from event_loop.science import prepare, baseline
    from event_loop.location import prepare as prepare_location
    paired_restart = (ROOT/'next_loop/m11r_protocol.json').exists()
    if not paired_restart:
        acquire()
        prepare()
        baseline()
        prepare_location()
    else:
        assert (EXPORT/'event_controls.json').exists() and (DATA/'location_cv4.npz').exists()
        print('M11R uses the identical live-worker training caches and saved controls; original M11 remains failed',flush=True)
    from next_loop.m11 import run
    run(paired_restart=paired_restart)
    target = EXPORT / 'm11_training_artifacts.tar.gz'
    with tarfile.open(target, 'w:gz') as archive:
        archive.add(DATA/'models', arcname='models')
        for pattern in ['M11_*.npz', 'bank_prior_M6.npz', 'bank_windnet_port.npz']:
            for path in DATA.glob(pattern):
                archive.add(path, arcname=path.name)
    save_json(EXPORT/'m11_artifacts.json', {'filename':target.name,'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'scope':'training-side fitted controls and prediction arrays only; no Fold5'})

if __name__ == '__main__':
    main()
