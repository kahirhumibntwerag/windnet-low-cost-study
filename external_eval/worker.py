"""Staged preparation and single sealed external comparison."""
import concurrent.futures,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path

def save(name,value):
    root=Path('exports');root.mkdir(exist_ok=True)
    (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def download(url,path,digest=None):
    import requests
    if path.exists() and (digest is None or hashlib.sha256(path.read_bytes()).hexdigest()==digest):return
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            with requests.get(url,timeout=(30,180),stream=True) as r:
                r.raise_for_status()
                with path.with_suffix(path.suffix+'.part').open('wb') as out:
                    for raw in r.iter_content(1024*1024):out.write(raw)
            if digest:assert hashlib.sha256(path.with_suffix(path.suffix+'.part').read_bytes()).hexdigest()==digest,'Hash mismatch'
            path.with_suffix(path.suffix+'.part').replace(path);return
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)

def prepare():
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','tensorflow-cpu==2.15.1','numpy==1.26.4','opencv-python-headless==4.9.0.80','scipy==1.12.0','scikit-learn==1.7.2','scikit-image==0.24.0','astropy==6.1.7','requests==2.32.3'])
    root=Path('data/external_models');root.mkdir(parents=True,exist_ok=True)
    specs=json.loads(Path('external_eval/author_checkpoint_hashes.json').read_text())['files']
    def get(spec):
        download('https://drive.usercontent.google.com/download?id='+spec['file_id']+'&export=download&confirm=t',root/f"author_fold{spec['fold']}"/spec['file'],spec['sha256'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,specs))
    print(json.dumps({'original_checkpoint_files_verified':len(specs)}),flush=True)
    lock=json.loads(Path('external_eval/model_lock.json').read_text())
    # Exact existing public F3 release, never retrained in this campaign.
    archive=root/'f3_checkpoints.tar.gz'
    download('https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/download/windnet-f3-2026-10-30/f3_checkpoints.tar.gz',archive,'4afe902645f1820414a0c8621d953166f6ef99142a16851ebac69f1174a4603e')
    dest=root/'m13';dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for entry in tar:
            if not entry.isfile() or not entry.name.startswith('models/'):continue
            target=(dest/entry.name).resolve();assert target.is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(entry).read())
    refs=json.loads(Path('matched/reference_sources.json').read_text())
    for ref in refs:download(ref['source_url'],Path('data/reference')/f"fold{ref['fold']}_{ref['file_name']}",ref['sha256'])
    from external_eval.acquire_pre2019 import main
    main()
    from external_eval.author_inference import recover_normalization
    recover_normalization()
    from external_eval.preflight_author import main
    main()
    save('author_normalization.json',json.loads(Path('external_eval/author_normalization.json').read_text()))
    save('author_restoration_preflight.json',json.loads(Path('external_eval/author_restoration_preflight.json').read_text()))
    save('runtime_versions.json',{'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()})
    print(json.dumps({'preparation_complete':True,'external_arrays_accessed':False}),flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    elif stage=='author-check':
        from external_eval.preflight_author import main
        save('author_normalization.json',json.loads(Path('data/external_models/author_normalization.json').read_text()))
        save('runtime_versions.json',{'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()})
        main();save('author_restoration_preflight.json',json.loads(Path('external_eval/author_restoration_preflight.json').read_text()))
    elif stage=='m13-check':
        from external_eval.preflight_m13 import main
        main()
    elif stage=='freeze':
        from external_eval.seal import main
        main()
    elif stage=='external-test':
        from external_eval.run import main
        main()
    else:raise ValueError(stage)
