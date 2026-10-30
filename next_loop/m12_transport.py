"""Authenticated worker transport with explicit package/stage boundaries."""
import hashlib,hmac,json,os,subprocess,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path('/workspace/regional-windnet');ROOT.mkdir(parents=True,exist_ok=True)
TOKEN=os.environ['EVENT_TOKEN'];PROCESS=None;START=time.time()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def reply(self,obj,code=200):
        b=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        if self.path=='/health':return self.reply({'ready':True,'elapsed_seconds':time.time()-START})
        if self.path=='/state':return self.reply({'running':PROCESS is not None and PROCESS.poll() is None,'returncode':None if PROCESS is None else PROCESS.poll(),'log':(ROOT/'worker.log').read_text(errors='replace')[-4500:] if (ROOT/'worker.log').exists() else ''})
        if self.path.startswith('/file/'):
            name=self.path[6:]
            if '/' in name or '\\' in name or '..' in name or not name.endswith(('.json','.csv','.gz')):return self.reply({'error':'invalid'},400)
            p=ROOT/'exports'/name
            if not p.is_file():return self.reply({'error':'missing'},404)
            self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
            with p.open('rb') as f:
                while b:=f.read(1024*1024):self.wfile.write(b)
            return
        return self.reply({'error':'missing'},404)
    def do_POST(self):
        global PROCESS
        if not self.auth():return self.reply({'error':'unauthorized'},401)
        n=int(self.headers.get('Content-Length','0'))
        if n>25_000_000:return self.reply({'error':'too large'},413)
        b=self.rfile.read(n)
        if self.path.startswith('/upload/'):
            pieces=self.path.split('/')[2:]
            if len(pieces)!=2 or pieces[0] not in ['matched','event_loop','next_loop'] or any(x in pieces[1] for x in ['..','/','\\']):return self.reply({'error':'invalid'},400)
            p=ROOT/pieces[0];p.mkdir(exist_ok=True);(p/pieces[1]).write_bytes(b);return self.reply({'saved':pieces[1],'sha256':hashlib.sha256(b).hexdigest()})
        if self.path=='/run':
            if PROCESS is not None and PROCESS.poll() is None:return self.reply({'error':'busy'},409)
            cfg=json.loads(b)
            if cfg != {'stage':'m12'}:return self.reply({'error':'Only preregistered M12 is permitted'},400)
            with (ROOT/'worker.log').open('w') as log:
                env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
                PROCESS=subprocess.Popen([sys.executable,'-u','-m','next_loop.m12_entry'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            return self.reply({'started':cfg['stage'],'pid':PROCESS.pid})
        return self.reply({'error':'missing'},404)
import threading
threading.Timer(3600,lambda: os._exit(0)).start()
print('Training-only M12 worker listening 8888; max lifetime3600s',flush=True)
ThreadingHTTPServer(('0.0.0.0',8888),Handler).serve_forever()
