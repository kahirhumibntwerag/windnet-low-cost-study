"""One-purpose M15 CPU workload with training artifacts only."""
import subprocess,sys

def main():
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','numpy==2.1.2','scipy==1.16.1','scikit-learn==1.7.2','requests==2.32.3','joblib==1.5.2'])
    from matched.runner import DATA,EXPORT,sha
    import requests
    DATA.mkdir(exist_ok=True);EXPORT.mkdir(exist_ok=True)
    assert not (DATA/'sealed').exists() and not (DATA/'m15.started.json').exists()
    sources=[('windnet-m11r-2026-10-30','m11_training_artifacts.tar.gz','e03668f2034eee2321234b83858f11caf40c79d78acae7df0bb64c6d61ae085b'),
             ('windnet-m13-2026-10-30','m13_training_artifacts.tar.gz','64a73923dbc8da0f7b05cc844227edcaa52c102bd3285df2afdde7c6999e90eb'),
             ('windnet-m14-2026-10-30','m14_training_artifacts.tar.gz','2e9f0b4230c10c661d36633b44f80f4afaf742a458dd9a533c6eacb5b412ee74')]
    for release,name,digest in sources:
        target=DATA/name
        if not target.exists():
            with requests.get('https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/download/'+release+'/'+name,stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with target.open('wb') as out:
                    for chunk in response.iter_content(1024*1024):out.write(chunk)
        assert sha(target.read_bytes())==digest
    from event_loop.runner import acquire
    acquire()
    (EXPORT/'m15_acquisition.json').write_bytes((EXPORT/'event_acquisition.json').read_bytes())
    from next_loop.m15 import run
    run()

if __name__=='__main__':main()
