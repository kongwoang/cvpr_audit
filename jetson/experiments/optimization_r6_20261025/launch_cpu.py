"""Launch an idempotent isolated CPU task after sourcing server env."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
ROOT='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
EXP=ROOT+'/experiments/optimization_r6_20261025'

def launch(host,task,dataset):
    code=f'''import os,json,subprocess,fcntl
from pathlib import Path
e=Path({EXP!r});j=e/'jobs';j.mkdir(exist_ok=True);name={task+'_'+dataset!r}
lock=(j/(name+'.launch.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
state=j/(name+'.json')
if state.exists():
 d=json.loads(state.read_text());cmd=Path('/proc')/str(d.get('pid',0))/'cmdline'
 if d.get('status')=='complete' or (cmd.exists() and b'optimization_r6_20261025/job.py' in cmd.read_bytes()):print(json.dumps(d));raise SystemExit
log=(j/(name+'.wrapper.log')).open('a')
p=subprocess.Popen(['python',str(e/'job.py'),{task!r},{dataset!r}],cwd={ROOT!r},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host={host!r},pid=p.pid,task={task!r},dataset={dataset!r})))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        'source '+ROOT+'/env.sh && '+shlex.join(['python','-c',code])],capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    d=json.loads(r.stdout);out=Path(__file__).parent/'launches';out.mkdir(exist_ok=True)
    (out/(task+'_'+dataset+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(d)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('host');ap.add_argument('task');ap.add_argument('dataset');a=ap.parse_args();launch(a.host,a.task,a.dataset)
