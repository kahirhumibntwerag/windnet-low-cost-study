"""Temporary authenticated worker; only training-side continuation files allowed."""
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path('/workspace/windnet-continuation')
ROOT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ['STUDY_TOKEN']
PROCESS = None
STARTED = None
UPLOADS = {'prepare_training.py', 'models.py', 'metrics.py', 'train.py',
           'train_only.npz', 'train_only.manifest.json', 'eve_features.npz',
           'eve_features.manifest.json', 'protocol.json'}
OUTPUTS = {'h7_results.json', 'h7_predictions.npz', 'h8_results.json',
           'h8_predictions.npz', 'run.log'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN)

    def do_GET(self):
        if not self.authorized():
            return self.reply({'error': 'unauthorized'}, 401)
        if self.path == '/health':
            return self.reply({'ready': True, 'training_only': True})
        if self.path == '/state':
            if PROCESS is not None and PROCESS.poll() is None and time.time() - STARTED > 900:
                PROCESS.kill()
            return self.reply({'running': PROCESS is not None and PROCESS.poll() is None,
                               'returncode': None if PROCESS is None else PROCESS.poll(),
                               'log': (ROOT/'run.log').read_text(errors='replace')[-2000:] if (ROOT/'run.log').exists() else ''})
        if self.path.startswith('/file/') and self.path[6:] in OUTPUTS:
            p = ROOT / self.path[6:]
            if not p.exists():
                return self.reply({'error': 'missing'}, 404)
            body = p.read_bytes()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self.reply({'error': 'missing'}, 404)

    def do_POST(self):
        global PROCESS, STARTED
        if not self.authorized():
            return self.reply({'error': 'unauthorized'}, 401)
        if self.path.startswith('/upload/') and self.path[8:] in UPLOADS:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 10_000_000:
                return self.reply({'error': 'size'}, 400)
            (ROOT/self.path[8:]).write_bytes(self.rfile.read(size))
            return self.reply({'saved': self.path[8:], 'bytes': size})
        if self.path in ('/run/h7', '/run/h8'):
            if PROCESS is not None and PROCESS.poll() is None:
                return self.reply({'error': 'busy'}, 409)
            stage = self.path.split('/')[-1]
            with (ROOT/'run.log').open('w') as log:
                PROCESS = subprocess.Popen([sys.executable, '-u', 'train.py', '--bundle', '.', '--output', '.', '--stage', stage],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            STARTED = time.time()
            return self.reply({'started': stage, 'pid': PROCESS.pid})
        return self.reply({'error': 'missing'}, 404)


print('CONTINUATION listening on 0.0.0.0:8888', flush=True)
ThreadingHTTPServer(('0.0.0.0', 8888), Handler).serve_forever()
