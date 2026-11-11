"""Idempotent wrapper launch. GPU selection remains wholly in the frozen wrapper."""
import argparse
import json
import shlex
import subprocess

REMOTE='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
EXP=REMOTE+'/experiments/optimization_r2_20261023'

def launch(host,stage,dataset=None):
    name=stage+('_'+dataset if dataset else '')
    code=f'''import os,subprocess,json,fcntl
from pathlib import Path
b=Path({REMOTE!r});e=Path({EXP!r});(e/'logs').mkdir(exist_ok=True)
name={name!r};saved=e/'logs'/('launch_'+name+'.json')
lock=(e/'logs'/('launch_'+name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
if saved.exists():
    prior=json.loads(saved.read_text());proc=Path('/proc')/str(prior['wrapper_pid'])/'cmdline'
    if proc.exists() and b'group.py' in proc.read_bytes() and b'optimization_r2_20261023' in proc.read_bytes():
        print(json.dumps(prior));raise SystemExit(0)
states=['smoke_nusc_0of1','smoke_av2_0of1'] if {stage!r}=='smoke' else ['dev_'+{dataset!r}+'_'+str(i)+'of2' for i in range(2)]
if all((e/'jobs'/(s+'.json')).exists() and json.loads((e/'jobs'/(s+'.json')).read_text()).get('status')=='complete' for s in states):
    print(json.dumps(dict(status='already_complete',jobs=states)));raise SystemExit(0)
cmd=['bash',str(b/'run_when_gpu_free.sh'),str(e/'logs'/(name+'_group.log')),'python',str(e/'group.py'),{stage!r}]
if {dataset!r}:cmd+=['--dataset',{dataset!r}]
f=(e/'logs'/(name+'_wrapper.log')).open('a')
p=subprocess.Popen(cmd,cwd=b,env=dict(os.environ,SHARE='1',MIN_FREE_MB='10000'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(host={host!r},wrapper_pid=p.pid,name=name,command=cmd)
saved.write_text(json.dumps(data,indent=2)+'\\n');print(json.dumps(data))
'''
    command='source '+REMOTE+'/env.sh && '+shlex.join(['python','-c',code])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    return json.loads(result.stdout)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('host',choices=['VUW','VUW1']);ap.add_argument('stage',choices=['smoke','dev']);ap.add_argument('--dataset',choices=['nusc','av2'])
    a=ap.parse_args();assert a.stage=='smoke' or a.dataset
    print(json.dumps(launch(a.host,a.stage,a.dataset),indent=2))
