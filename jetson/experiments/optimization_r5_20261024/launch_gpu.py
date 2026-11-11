"""Local launcher: source server env before writes and use its unchanged wrapper."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
EXP=ROOT+'/experiments/optimization_r5_20261024'
def launch(host,name,jobs):
    code=f'''import os,json,subprocess,fcntl,time
from pathlib import Path
b=Path({ROOT!r});e=Path({EXP!r});d=e/'logs';d.mkdir(exist_ok=True)
lock=(d/({name!r}+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
saved=d/({name!r}+'.json')
if saved.exists():
 p=json.loads(saved.read_text());c=Path('/proc')/str(p['wrapper_pid'])/'cmdline'
 if c.exists() and b'optimization_r5_20261024' in c.read_bytes():print(json.dumps(p));raise SystemExit
f=(d/({name!r}+'_wrapper.log')).open('a')
cmd=['bash',str(b/'run_when_gpu_free.sh'),str(d/({name!r}+'_group.log')),'python',str(e/'group.py'),{json.dumps(jobs)!r}]
p=subprocess.Popen(cmd,cwd=b,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,SHARE='1',MIN_FREE_MB='10000'))
obj=dict(host={host!r},wrapper_pid=p.pid,jobs={jobs!r},started=time.time(),command=cmd)
saved.write_text(json.dumps(obj,indent=2));print(json.dumps(obj))
'''
    command='source '+ROOT+'/env.sh && '+shlex.join(['python','-c',code])
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    result=json.loads(r.stdout);p=Path(__file__).parent/'launches';p.mkdir(exist_ok=True)
    (p/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('host');ap.add_argument('name');ap.add_argument('jobs')
    a=ap.parse_args();print(json.dumps(launch(a.host,a.name,json.loads(a.jobs)),indent=2))
