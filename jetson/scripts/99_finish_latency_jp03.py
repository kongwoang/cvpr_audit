#!/usr/bin/env python3
"""Wait for the authorized JP03 window, then collect its CPU analysis once."""
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT/'experiments/latency_opt_20261030/logs/JP03_20261031_014821'


def main():
    lock=(WINDOW/'collection.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (WINDOW/'COLLECTION_COMPLETE.json').exists()
    started=time.monotonic();last=None
    while time.monotonic()-started < 10800:
        state=json.loads((WINDOW/'state.json').read_text())
        current=(state['status'],len(state['runs']),state.get('window_released'))
        if current!=last:
            print(dt.datetime.now().astimezone().isoformat(),current,flush=True);last=current
        if state['status']=='failed':raise RuntimeError(state)
        if state['status']=='complete' and state.get('window_released'):
            assert not state['restore_errors']
            subprocess.run(['python3',str(ROOT/'scripts/96_collect_latency_pilot.py'),str(WINDOW)],cwd=ROOT,check=True,timeout=1200)
            h=ROOT/'docs/paper/handoff/LATENCY_OPT_20261030'
            files=['JP03_REPORT.md','JP03_BENCHMARK_PILOT.csv','JP03_TIMING_PARITY.csv','JP03_QUALITY_PILOT.csv','JP03_DEADLINES_PILOT.csv']
            result=dict(completed=dt.datetime.now().astimezone().isoformat(),hashes={f:hashlib.sha256((h/f).read_bytes()).hexdigest() for f in files})
            (WINDOW/'COLLECTION_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
            print('CPU collection and report complete',flush=True);return
        time.sleep(20)
    raise TimeoutError('JP03 collection wait exceeded three hours')


if __name__=='__main__':main()
