"""Run one isolated worker with hashes, durable status, lock and recorded exit."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from common import HERE,REMOTE,save,source_hashes,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind');ap.add_argument('dataset');ap.add_argument('shard')
    a,extra=ap.parse_known_args();name='_'.join([a.kind,a.dataset,a.shard.replace('/','of')])
    jd=HERE/'jobs';jd.mkdir(exist_ok=True);state=jd/(name+'.json')
    lock=(jd/(name+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return 0
    if state.exists() and json.loads(state.read_text()).get('status')=='complete':return 0
    if state.exists():
        previous=json.loads(state.read_text())
        save(jd/'attempts'/(name+'_'+str(int(previous['started']))+'.json'),previous)
    before=source_hashes();save(jd/(name+'.sources_before.json'),before)
    env=dict(os.environ,B_REAL=str(REMOTE),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(HERE)+os.pathsep+env.get('PYTHONPATH','')
    script={'dev':'runner.py','smoke':'runner.py'}[a.kind]
    cmd=[sys.executable,str(HERE/script),'--dataset',a.dataset,'--shard',a.shard]+extra
    if a.kind in ('dev','smoke','learned','learnsmoke','calibrated','calsmoke','lift','liftsmoke','iddev','idsmoke','persist','persistsmoke','low','lowsmoke','compact','compactsmoke'):cmd+=['--tag',a.kind]
    if a.dataset=='av2' and a.kind in ('data','dev','smoke','learned','learnsmoke','calibrated','calsmoke','lift','liftsmoke','iddev','idsmoke','persist','persistsmoke','low','lowsmoke','compact','compactsmoke'):
        env.update(B=str(REMOTE/'k10_masked'),K10_MASK_Y0='1830')
        cmd=[sys.executable,str(REMOTE/'scripts/A_method/k10_masked_run.py')]+cmd[1:]
    else:env['B']=str(REMOTE)
    data=dict(name=name,host=socket.gethostname(),pid=os.getpid(),gpu=env.get('CUDA_VISIBLE_DEVICES'),status='running',
              started=time.time(),command=cmd,experiment_hashes={p.name:sha(p) for p in HERE.glob('*.py')})
    save(state,data)
    with (jd/(name+'.log')).open('a') as log:
        result=subprocess.run(cmd,cwd=REMOTE,env=env,stdout=log,stderr=subprocess.STDOUT)
    after=source_hashes();changed=[p for p,h in before.items() if after.get(p)!=h];save(jd/(name+'.sources_after.json'),after)
    data.update(status='complete' if result.returncode==0 and not changed else 'failed',finished=time.time(),exit_code=result.returncode,changed_main_sources=changed)
    save(state,data);print(json.dumps(data),flush=True)
    return int(data['status']!='complete')
if __name__=='__main__':sys.exit(main())
