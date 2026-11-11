"""Finish both exclusive timing windows before CPU reports, then archive evidence."""
import fcntl
import json
import subprocess
import time
from pathlib import Path
from common import save
from coordinator import ssh,sync,REMOTE,EXP

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024'
WINDOWS=['optimization_r4_jetson_20261024','optimization_r4_fast_jetson_20261024']

def main():
    lock=(HERE/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state={n:json.loads((ROOT/'experiments'/n/'logs/state.json').read_text()) for n in WINDOWS}
        failed={n:s for n,s in state.items() if s['status']=='failed'}
        if failed:raise RuntimeError({n:s.get('error') for n,s in failed.items()})
        if all(s['status']=='complete' and s.get('window_released') and not s.get('restore_errors') for s in state.values()):break
        save(OUT/'STATUS.json',dict(phase='jetson_running',updated=time.time(),windows={n:s['status'] for n,s in state.items()},core_changed=False))
        time.sleep(35)
    for variant in ['direct','fast']:
        subprocess.run([str(ROOT/'scripts/py'),str(HERE/'report_jetson.py'),'--variant',variant],check=True)
    sync()
    # Remote backend audit has no GPU or network-data-dependent model execution.
    subprocess.run(['rsync','-a','--rsync-path=source '+REMOTE+'/env.sh && rsync',str(HERE/'backend.py'),'VUW:'+EXP+'/'],check=True)
    ssh('mkdir -p '+EXP+'/jetson_exports/jetson '+EXP+'/jetson_exports/jetson_fast')
    for folder in ['jetson','jetson_fast']:
        subprocess.run(['rsync','-a','--rsync-path=source '+REMOTE+'/env.sh && rsync',str(OUT/folder/'frames.csv'),'VUW:'+EXP+'/jetson_exports/'+folder+'/'],check=True)
    print(ssh('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python '+EXP+'/backend.py'),flush=True)
    while True:
        code="import json;from pathlib import Path;p=Path(%r)/'reports/nav_selector';print((p/'DONE.json').read_text() if (p/'DONE.json').exists() else (p/'STATUS.json').read_text())"%EXP
        import shlex
        nav=json.loads(ssh(shlex.join(['python','-c',code])))
        if nav.get('status')=='failed':raise RuntimeError(nav)
        if nav.get('status')=='complete':break
        time.sleep(30)
    sync()
    subprocess.run([str(ROOT/'scripts/py'),str(HERE/'consolidate.py')],check=True)
    save(HERE/'FINISH_STATUS.json',dict(status='complete',finished=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(HERE/'FINISH_STATUS.json',dict(status='failed',error=repr(e),updated=time.time()));raise
