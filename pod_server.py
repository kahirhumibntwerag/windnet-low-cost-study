"""Small authenticated transport for our temporary Runpod research worker."""
import hmac,json,os,subprocess,sys,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path('/workspace/windnet-study');ROOT.mkdir(parents=True,exist_ok=True)
TOKEN=os.environ['STUDY_TOKEN']; PROCESS=None; LAUNCHED=time.time()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
    def reply(self,obj,status=200):
        b=json.dumps(obj).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if not self.authorized(): return self.reply({'error':'unauthorized'},401)
        if self.path=='/health':return self.reply({'ready':True,'pid':os.getpid(),'elapsed_s':time.time()-LAUNCHED})
        if self.path=='/state':
            return self.reply({'running':PROCESS is not None and PROCESS.poll() is None,'returncode':None if PROCESS is None else PROCESS.poll(),
                'files':[p.name for p in ROOT.iterdir() if p.is_file()],
                'log':(ROOT/'run.log').read_text(errors='replace')[-6000:] if (ROOT/'run.log').exists() else ''})
        if self.path.startswith('/file/'):
            name=self.path[len('/file/'):]
            if '/' in name or '\\' in name or '..' in name:return self.reply({'error':'bad path'},400)
            p=ROOT/name
            if not p.exists():return self.reply({'error':'missing'},404)
            b=p.read_bytes();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
        return self.reply({'error':'missing'},404)
    def do_POST(self):
        global PROCESS
        if not self.authorized(): return self.reply({'error':'unauthorized'},401)
        if self.path.startswith('/upload/'):
            name=self.path[len('/upload/'):]
            if '/' in name or '\\' in name or '..' in name:return self.reply({'error':'bad path'},400)
            b=self.rfile.read(int(self.headers.get('Content-Length','0')));(ROOT/name).write_bytes(b)
            return self.reply({'saved':name,'bytes':len(b)})
        if self.path.startswith('/run/'):
            stage=self.path.split('/')[-1]
            if stage not in ['loop1','loop2','loop3','final']:return self.reply({'error':'bad stage'},400)
            if PROCESS is not None and PROCESS.poll() is None:return self.reply({'error':'busy'},409)
            with (ROOT/'run.log').open('w') as log:
                PROCESS=subprocess.Popen([sys.executable,'-u','study.py','--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return self.reply({'started':stage,'pid':PROCESS.pid})
        return self.reply({'error':'missing'},404)

print('STUDY_TRANSPORT listening on 0.0.0.0:8888',flush=True)
ThreadingHTTPServer(('0.0.0.0',8888),Handler).serve_forever()
