"""Authenticated code-only transport restricted to the M15 CPU workload."""
import hashlib,hmac,json,os,subprocess,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path('/workspace/m15-windnet');ROOT.mkdir(parents=True,exist_ok=True)
TOKEN=os.environ['EVENT_TOKEN'];PROCESS=None;START=time.time()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def reply(self,obj,code=200):
        raw=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        if self.path=='/health':return self.reply({'ready':True,'elapsed_seconds':time.time()-START})
        if self.path=='/state':return self.reply({'running':PROCESS is not None and PROCESS.poll() is None,'returncode':None if PROCESS is None else PROCESS.poll(),'log':(ROOT/'worker.log').read_text(errors='replace')[-4500:] if (ROOT/'worker.log').exists() else ''})
        if self.path.startswith('/file/'):
            name=self.path[6:]
            if '/' in name or '\\' in name or '..' in name or not name.endswith(('.json','.gz')):return self.reply({'error':'invalid'},400)
            path=ROOT/'exports'/name
            if not path.is_file():return self.reply({'error':'missing'},404)
            self.send_response(200);self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
            with path.open('rb') as source:
                while raw:=source.read(1024*1024):self.wfile.write(raw)
            return
        return self.reply({'error':'missing'},404)
    def do_POST(self):
        global PROCESS
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        n=int(self.headers.get('Content-Length','0'))
        if n>25_000_000:return self.reply({'error':'too large'},413)
        raw=self.rfile.read(n)
        if self.path.startswith('/upload/'):
            if PROCESS is not None:return self.reply({'error':'Uploads locked after workload start'},409)
            pieces=self.path.split('/')[2:]
            if len(pieces)!=2 or pieces[0] not in ['matched','event_loop','next_loop'] or any(c in pieces[1] for c in ['..','/','\\']) or not pieces[1].endswith(('.py','.json')):return self.reply({'error':'invalid'},400)
            folder=ROOT/pieces[0];folder.mkdir(exist_ok=True);(folder/pieces[1]).write_bytes(raw)
            return self.reply({'saved':pieces[1],'sha256':hashlib.sha256(raw).hexdigest()})
        if self.path=='/run':
            if PROCESS is not None:return self.reply({'error':'Only one registered workload attempt permitted'},409)
            if json.loads(raw)!={'stage':'m15'}:return self.reply({'error':'Only training-only M15 permitted'},400)
            with (ROOT/'worker.log').open('w') as log:
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                PROCESS=subprocess.Popen([sys.executable,'-u','-m','next_loop.m15_entry'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            return self.reply({'started':'m15','pid':PROCESS.pid})
        return self.reply({'error':'missing'},404)

if __name__=='__main__':
    print('Training-only M15 worker listening on0.0.0.0:8888',flush=True)
    ThreadingHTTPServer(('0.0.0.0',8888),Handler).serve_forever()
