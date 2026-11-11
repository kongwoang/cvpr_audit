"""Launched ONLY through the existing GPU wrapper; records real child exit status."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
HERE=ROOT/'experiments/optimization_r2_assoc_20261023'
def hashes():
    paths=list((ROOT/'code/CVPR/src/vtf').glob('*.py'))
    paths+=list((ROOT/'scripts/A_method').glob('*.py'))
    paths+=[ROOT/'out/k28_temporal/temporal_head.pt',ROOT/'out/resnet18_v3a/image_refiner.pt',ROOT/'models/yolov8s.pt']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['nusc','av2'],required=True)
    ap.add_argument('--stage',choices=['smoke','dev'],required=True);ap.add_argument('--shard',default='0/1')
    a=ap.parse_args();name=a.stage+'_'+a.dataset+'_'+a.shard.replace('/','of')
    jd=HERE/'jobs';jd.mkdir(parents=True,exist_ok=True)
    state=jd/(name+'.json')
    lock=(jd/(name+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Already running:',name,flush=True);return 0
    if state.exists() and json.loads(state.read_text()).get('status')=='complete':
        print('Already complete:',name,flush=True);return 0
    before=hashes();write(jd/(name+'.sources_before.json'),before)
    env=dict(os.environ, B_REAL=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,str(HERE/'runner.py'),'--dataset',a.dataset,'--tag',a.stage,'--shard',a.shard]
    if a.stage=='smoke':
        names=sorted({c['unit'] for c in json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases'] if c['dataset']==a.dataset})
        cmd+=['--units',','.join(names)]
    if a.dataset=='av2':
        env.update(B=str(ROOT/'k10_masked'),K10_MASK_Y0='1830')
        cmd=[sys.executable,str(ROOT/'scripts/A_method/k10_masked_run.py')]+cmd[1:]
    else: env['B']=str(ROOT)
    data=dict(name=name,host=socket.gethostname(),pid=os.getpid(),gpu=env.get('CUDA_VISIBLE_DEVICES'),
              start=time.time(),status='running',command=cmd,
              experiment_sources={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest()
                  for p in ('runner.py','sandbox_loop.py','helpers.py','protocol.md','inloop_protocol.md','proactive.py','estimators.py','bounded_variant.py','association.py')})
    write(state,data)
    with (jd/(name+'.log')).open('a') as log:
        result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    after=hashes();changed=[p for p in before if before[p]!=after.get(p)]
    write(jd/(name+'.sources_after.json'),after)
    data.update(finish=time.time(),exit_code=result.returncode,changed_main_sources=changed,
                status='complete' if result.returncode==0 and not changed else 'failed')
    write(state,data)
    print(json.dumps(data),flush=True)
    return 0 if data['status']=='complete' else 1
if __name__=='__main__':sys.exit(main())
