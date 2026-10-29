"""Authenticated one-shot E13 transport, separate from research workers."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path('/workspace/windnet-locked-e13')
ROOT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ['FINAL_TOKEN']
FREEZE = os.environ['FINAL_FREEZE_SHA256']
PROTOCOL = os.environ['FINAL_PROTOCOL_URL']
PROCESS = None
SOURCES = {'prepare_locked_final.py', 'locked_final.py', 'final_uncertainty.py',
           'final_reporting.py', 'prior_champion.py', 'models.py', 'metrics.py',
           'train.py', 'prepare_training.py', 'fast_loss.py', 'freeze_manifest.json',
           'final_bundle.npz', 'final_bundle.manifest.json'}
OUTPUTS = {'final_results.json', 'predictions.npz', 'comparison.csv',
           'checkpoints.tar.gz', 'execution_started.json'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN)

    def do_GET(self):
        if not self.authorized():
            return self.reply({'error': 'unauthorized'}, 401)
        if self.path == '/health':
            return self.reply({'ready': True, 'stage': 'locked_final_only'})
        if self.path == '/state':
            return self.reply({'running': PROCESS is not None and PROCESS.poll() is None,
                               'returncode': None if PROCESS is None else PROCESS.poll(),
                               'log': (ROOT/'run.log').read_text(errors='replace')[-2400:] if (ROOT/'run.log').exists() else ''})
        if self.path.startswith('/file/') and self.path[6:] in OUTPUTS:
            if PROCESS is None or PROCESS.poll() != 0:
                return self.reply({'error': 'complete frozen panel not yet available'}, 409)
            path = ROOT/'results'/self.path[6:]
            if not path.is_file():
                return self.reply({'error': 'missing'}, 404)
            body = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self.reply({'error': 'missing'}, 404)

    def do_POST(self):
        global PROCESS
        if not self.authorized():
            return self.reply({'error': 'unauthorized'}, 401)
        if self.path.startswith('/upload/') and self.path[8:] in SOURCES:
            if PROCESS is not None or (ROOT/f'execution_{FREEZE}.started.json').exists():
                return self.reply({'error': 'execution already started; sources immutable'}, 409)
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 15_000_000:
                return self.reply({'error': 'invalid size'}, 400)
            body = self.rfile.read(size)
            target = ROOT/self.path[8:]
            if target.exists() and target.read_bytes() != body:
                return self.reply({'error': 'source identity differs'}, 409)
            target.write_bytes(body)
            return self.reply({'saved': target.name, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': size})
        if self.path == '/run':
            if PROCESS is not None or (ROOT/f'execution_{FREEZE}.started.json').exists():
                return self.reply({'error': 'one-shot execution already started'}, 409)
            if not all((ROOT/name).is_file() for name in SOURCES):
                return self.reply({'error': 'incomplete frozen transfer'}, 400)
            command = [sys.executable, '-u', 'locked_final.py', '--bundle', '.', '--output', 'results',
                       '--freeze-manifest', 'freeze_manifest.json', '--freeze-sha256', FREEZE,
                       '--protocol-url', PROTOCOL, '--device', 'cpu']
            with (ROOT/'run.log').open('w') as log:
                PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            return self.reply({'started': True, 'pid': PROCESS.pid, 'stage': 'E13', 'device': 'cpu'})
        return self.reply({'error': 'missing'}, 404)


print('LOCKED_E13 listening on 0.0.0.0:8888', flush=True)
ThreadingHTTPServer(('0.0.0.0', 8888), Handler).serve_forever()
