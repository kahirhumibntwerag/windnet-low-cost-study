"""Authenticated scoped Runpod worker. Exports metadata/aggregate JSON only."""
import hashlib,hmac,json,os,subprocess,sys,time,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path('/workspace/external-eval');ROOT.mkdir(parents=True,exist_ok=True)
TOKEN=os.environ['EXTERNAL_TOKEN'];PROCESS=None;START=time.time();SEALED=False

def guard():
    time.sleep(3*3600)
    if PROCESS is not None and PROCESS.poll() is None:PROCESS.terminate()
    os._exit(0)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def reply(self,obj,code=200):
        raw=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        if self.path=='/health':return self.reply({'ready':True,'elapsed_seconds':time.time()-START})
        if self.path=='/state':return self.reply({'running':PROCESS is not None and PROCESS.poll() is None,'returncode':None if PROCESS is None else PROCESS.poll(),'sealed':SEALED,'log':(ROOT/'worker.log').read_text(errors='replace')[-5000:] if (ROOT/'worker.log').exists() else ''})
        if self.path.startswith('/artifact/'):
            name=self.path[10:]
            if '/' in name or '\\' in name or '..' in name or not name.endswith(('.json','.txt','.csv')):return self.reply({'error':'invalid'},400)
            path=ROOT/'exports'/name
            if not path.is_file():return self.reply({'error':'missing'},404)
            raw=path.read_bytes();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        return self.reply({'error':'missing'},404)
    def do_POST(self):
        global PROCESS,SEALED
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        n=int(self.headers.get('Content-Length','0'))
        if n>30_000_000:return self.reply({'error':'too large'},413)
        raw=self.rfile.read(n)
        if self.path.startswith('/upload/'):
            if SEALED or (PROCESS is not None and PROCESS.poll() is None):return self.reply({'error':'uploads locked'},409)
            rel=Path(self.path[8:]);target=(ROOT/rel).resolve()
            if not target.is_relative_to(ROOT) or rel.parts[0] not in ['external_eval','matched','event_loop','next_loop'] or target.suffix not in ['.py','.json','.txt']:return self.reply({'error':'invalid'},400)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            return self.reply({'sha256':hashlib.sha256(raw).hexdigest()})
        if self.path=='/run':
            stage=json.loads(raw).get('stage')
            if stage not in ['prepare','author-check','m13-check','freeze','external-test']:return self.reply({'error':'invalid stage'},400)
            if PROCESS is not None and PROCESS.poll() is None:return self.reply({'error':'already running'},409)
            if stage=='external-test':
                if SEALED:return self.reply({'error':'only one external attempt'},409)
                SEALED=True
            with (ROOT/'worker.log').open('a') as log:
                env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
                PROCESS=subprocess.Popen([sys.executable,'-u','-m','external_eval.worker',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            return self.reply({'started':stage})
        return self.reply({'error':'missing'},404)

if __name__=='__main__':
    threading.Thread(target=guard,daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0',8888),Handler).serve_forever()
