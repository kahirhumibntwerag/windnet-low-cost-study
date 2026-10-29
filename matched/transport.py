"""Authenticated, bounded transport for this campaign's disposable Runpod worker."""
import hashlib, hmac, json, os, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path('/workspace/matched-windnet'); ROOT.mkdir(parents=True,exist_ok=True)
TOKEN=os.environ['MATCHED_TOKEN']; PROCESS=None
START=time.time()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def auth(self): return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def reply(self,obj,code=200):
        raw=json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if not self.auth(): return self.reply({'error':'unauthorized'},401)
        if self.path=='/health': return self.reply({'ready':True,'elapsed_seconds':time.time()-START})
        if self.path=='/state':
            log=ROOT/'worker.log'
            return self.reply({'running':PROCESS is not None and PROCESS.poll() is None,'returncode':None if PROCESS is None else PROCESS.poll(),'log':log.read_text(errors='replace')[-3500:] if log.exists() else ''})
        if self.path.startswith('/file/'):
            name=self.path[6:]
            # Data and individual holdout predictions are never exposed by this transport.
            if '/' in name or '\\' in name or '..' in name or not name.endswith(('.json','.csv','.txt','.gz')): return self.reply({'error':'invalid'},400)
            p=ROOT/'exports'/name
            if not p.is_file(): return self.reply({'error':'missing'},404)
            raw=p.read_bytes(); self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw); return
        return self.reply({'error':'missing'},404)
    def do_POST(self):
        global PROCESS
        if not self.auth(): return self.reply({'error':'unauthorized'},401)
        n=int(self.headers.get('Content-Length','0'))
        if n>25_000_000: return self.reply({'error':'too large'},413)
        raw=self.rfile.read(n)
        if self.path.startswith('/upload/'):
            name=self.path[8:]
            if '/' in name or '\\' in name or '..' in name: return self.reply({'error':'invalid'},400)
            target=ROOT/'matched';target.mkdir(exist_ok=True);(target/name).write_bytes(raw)
            return self.reply({'saved':name,'sha256':hashlib.sha256(raw).hexdigest()})
        if self.path=='/run':
            if PROCESS is not None and PROCESS.poll() is None: return self.reply({'error':'busy'},409)
            cfg=json.loads(raw); stage=cfg['stage']
            if stage not in ['acquire','embed','baseline','ridge','spatial','kernel','neural','freeze','final','export']: return self.reply({'error':'invalid stage'},400)
            (ROOT/'request.json').write_text(json.dumps(cfg))
            with (ROOT/'worker.log').open('w') as log:
                env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
                PROCESS=subprocess.Popen([sys.executable,'-u','-m','matched.runner','--request','request.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            return self.reply({'started':stage,'pid':PROCESS.pid})
        return self.reply({'error':'missing'},404)

print('MATCHED_WORKER listening 8888',flush=True)
ThreadingHTTPServer(('0.0.0.0',8888),Handler).serve_forever()
