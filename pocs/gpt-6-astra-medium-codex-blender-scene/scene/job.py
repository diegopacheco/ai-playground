import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parent.parent
action, profile = sys.argv[1:]
run = root / '.run'
state_path = run / f'{profile}.json'
log_path = run / 'logs' / f'{profile}.log'


def active(state):
    if not state:
        return False
    result = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'command='], capture_output=True, text=True)
    return result.returncode == 0 and str(root / 'scripts/run.sh') in result.stdout


state = json.loads(state_path.read_text()) if state_path.exists() else None
if action == 'start':
    if active(state):
        print(f"{profile}: RUNNING pid={state['pid']}")
    else:
        with log_path.open('w') as log:
            process = subprocess.Popen(['bash', str(root / 'scripts/run.sh')], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, 'PROFILE': profile})
        state_path.write_text(json.dumps({'pid': process.pid}))
        time.sleep(1)
        if process.poll() is not None and process.returncode != 0:
            raise SystemExit(f'Render failed; read {log_path}')
        print(f'{profile}: STARTED pid={process.pid}, log={log_path}')
elif action == 'stop':
    if active(state):
        os.killpg(state['pid'], signal.SIGTERM)
        for attempt in range(10):
            if not active(state):
                break
            time.sleep(1)
        else:
            os.killpg(state['pid'], signal.SIGKILL)
    state_path.unlink(missing_ok=True)
    print(f'{profile}: STOPPED')
elif action == 'status':
    if active(state):
        print(f"{profile}: RUNNING pid={state['pid']}, log={log_path}")
    else:
        video = root / 'output' / profile / 'frostbite-falls.mp4'
        print(f"{profile}: IDLE, video={'READY' if video.exists() else 'NOT BUILT'}, log={log_path}")
else:
    raise SystemExit(f'Unknown action: {action}')
