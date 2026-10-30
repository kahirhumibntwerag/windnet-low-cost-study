"""One-purpose CPU M17 workload; no historical final stages are available."""
import subprocess,sys

def main():
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','numpy==2.1.2','scipy==1.16.1','scikit-learn==1.7.2','requests==2.32.3','joblib==1.5.2'])
    subprocess.check_call([sys.executable,"-m","unittest","next_loop.test_kernel_head","next_loop.test_regions","next_loop.test_topology","next_loop.test_target_weights"])
    from matched.runner import DATA,EXPORT,sha
    import requests
    DATA.mkdir(exist_ok=True);EXPORT.mkdir(exist_ok=True)
    assert not (DATA/'sealed').exists() and not (DATA/'m17.started.json').exists()
    artifacts=[('windnet-m11r-2026-10-30','m11_training_artifacts.tar.gz','e03668f2034eee2321234b83858f11caf40c79d78acae7df0bb64c6d61ae085b'),
               ('windnet-m13-2026-10-30','m13_training_artifacts.tar.gz','64a73923dbc8da0f7b05cc844227edcaa52c102bd3285df2afdde7c6999e90eb')]
    for release,name,digest in artifacts:
        path=DATA/name
        if not path.exists():
            url='https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/download/'+release+'/'+name
            with requests.get(url,stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with path.open('wb') as out:
                    for chunk in response.iter_content(1024*1024):out.write(chunk)
        assert sha(path.read_bytes())==digest
    from event_loop.runner import acquire
    acquire()
    (EXPORT/'m17_acquisition.json').write_bytes((EXPORT/'event_acquisition.json').read_bytes())
    from next_loop.m17 import run
    run()

if __name__=='__main__':main()
