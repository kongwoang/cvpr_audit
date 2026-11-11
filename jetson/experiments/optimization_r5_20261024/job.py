"""CPU job with durable state, source integrity and per-task lock."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from common import HERE, REMOTE, save, source_hashes, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('dataset');a=ap.parse_args()
    name=a.task+'_'+a.dataset;out=HERE/'jobs';out.mkdir(exist_ok=True)
    lock=(out/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=out/(name+'.json')
    if statepath.exists():
        old=json.loads(statepath.read_text())
        save(out/'attempts'/(name+'_'+str(int(old['started']))+'.json'),old)
    before=source_hashes();save(out/(name+'.sources_before.json'),before)
    state=dict(task=a.task,dataset=a.dataset,status='running',pid=os.getpid(),host=socket.gethostname(),started=time.time(),
               scripts={p.name:sha(p) for p in HERE.glob('*.py')})
    save(out/(name+'.json'),state)
    env=dict(os.environ,B=str(REMOTE),B_REAL=str(REMOTE),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
    with (out/(name+'.log')).open('a') as log:
        r=subprocess.run([sys.executable,str(HERE/(a.task+'.py')),'--dataset',a.dataset],cwd=REMOTE,env=env,stdout=log,stderr=subprocess.STDOUT)
    after=source_hashes();changed=[p for p in before if before[p]!=after.get(p)]
    save(out/(name+'.sources_after.json'),after)
    state.update(status='complete' if r.returncode==0 and not changed else 'failed',exit_code=r.returncode,finished=time.time(),changed_main_sources=changed)
    save(out/(name+'.json'),state);return int(state['status']!='complete')

if __name__=='__main__':sys.exit(main())
