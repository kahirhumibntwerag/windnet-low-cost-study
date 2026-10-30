"""Preserve exact scientific source bytes across Git newline conversion."""
import argparse,gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['f3','m14']);args=ap.parse_args();stage=args.stage
    freeze_path=ROOT/'results/next_loop'/f'{stage}_freeze.json';freeze=json.loads(freeze_path.read_text())
    sources=dict(freeze['source_hashes']);protocol=f'next_loop/{stage}_protocol.json'
    if protocol not in sources:sources[protocol]=freeze['protocol_sha256']
    payload={};converted=[]
    for name,digest in sorted(sources.items()):
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT.resolve())
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest,name
        blob=subprocess.check_output(['git','show','HEAD:'+name])
        if hashlib.sha256(blob).hexdigest()!=digest:
            assert blob.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'),name
            converted.append(name)
        payload[name]=raw
    manifest={'stage':stage.upper(),'source_hashes':sources,'experiment_freeze_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
              'git_newline_conversion_paths':converted,'purpose':'Exact bytes for frozen source hash checks on any platform; no numerical or recipe changes','raw_observations_included':False}
    payload['FROZEN_SOURCE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    payload['FROZEN_EXPERIMENT_FREEZE.json']=freeze_path.read_bytes()
    target=ROOT/'artifacts/next_loop'/f'{stage}_frozen_sources.tar.gz';assert not target.exists(),'Refuse to replace source snapshot'
    with target.open('wb') as out,gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
        for name,raw in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0
            archive.addfile(info,io.BytesIO(raw))
    with tarfile.open(target) as archive:
        assert sorted(archive.getnames())==sorted(payload)
        for member in archive:
            assert member.isfile() and archive.extractfile(member).read()==payload[member.name]
    meta={'filename':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'scientific_source_files':len(sources),'all_source_bytes_match_original_freeze':True,
          'archive_roundtrip_verified':True,'git_newline_conversion_paths':converted,'raw_holdout_observations_or_forecasts_included':False}
    (ROOT/'results/next_loop'/f'{stage}_sources_artifacts.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta))

if __name__=='__main__':main()
