"""Install the pinned CPU dependencies, acquire training data and run M12."""
import subprocess,sys
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','numpy==2.1.2','scipy==1.16.1','scikit-learn==1.7.2','requests==2.32.3','joblib==1.5.2'])
from matched.runner import DATA,EXPORT,ROOT
DATA.mkdir(exist_ok=True);EXPORT.mkdir(exist_ok=True)
import requests
p=DATA/'m11_training_artifacts.tar.gz'
if not p.exists():
    with requests.get('https://github.com/kahirhumibntwerag/windnet-low-cost-study/releases/download/windnet-m11r-2026-10-30/m11_training_artifacts.tar.gz',stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        with p.open('wb') as handle:
            for chunk in response.iter_content(1024*1024):handle.write(chunk)
from next_loop.m12 import acquire,run
if not (DATA/'members.json').exists():acquire()
if (ROOT/'next_loop/m13_protocol.json').exists():
    from next_loop.m13 import run as run_m13
    print('Registered M13 follow-up on the same verified CPU data; M12 results retained',flush=True)
    run_m13()
else:run()
