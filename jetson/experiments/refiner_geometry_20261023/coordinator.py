"""Queue CPU attribution after the original immutable development records finish."""
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
B='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
REMOTE=B+'/experiments/refiner_geometry_20261023'
OUT=ROOT/'docs/paper/handoff/REFINER_GEOMETRY_20261023'


def ssh(code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW1',shlex.join(['python3','-c',code])],
        capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr[-800:])
    return r.stdout


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=__pycache__',str(HERE)+'/', 'VUW1:'+REMOTE+'/'],check=True)
    while True:
        code="import pathlib,json; p=pathlib.Path("+repr(B+"/experiments/w1_dev_20261023/jobs")+"); print(json.dumps([json.loads(f.read_text()).get('status') for f in p.glob('dev*.json') if '.sources_' not in f.name]))"
        try:states=json.loads(ssh(code))
        except Exception:time.sleep(30);continue
        assert not any(s=='failed' for s in states),states
        if len(states)==4 and all(s=='complete' for s in states):break
        (OUT/'STATUS.json').write_text(json.dumps(dict(status='waiting_original_records',states=states))+'\n')
        time.sleep(30)
    code=f'''import os,subprocess,json
from pathlib import Path
e=Path({REMOTE!r});(e/'logs').mkdir(exist_ok=True);f=(e/'logs/job.log').open('a')
p=subprocess.Popen(['python','-u',str(e/'runner.py')],cwd={B!r},env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
'''
    cmd='source '+B+'/env.sh && '+shlex.join(['python3','-c',code])
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW1',cmd],capture_output=True,text=True,timeout=60)
    r.check_returncode();pid=int(r.stdout.strip())
    while True:
        code="import json,pathlib; p=pathlib.Path("+repr(REMOTE+"/reports/state.json")+"); print(p.read_text() if p.exists() else '{}')"
        try:state=json.loads(ssh(code))
        except Exception:time.sleep(30);continue
        state['remote_pid']=pid;(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
        assert state.get('status')!='failed',state
        if state.get('status')=='complete':break
        time.sleep(30)
    subprocess.run(['rsync','-a','VUW1:'+REMOTE+'/reports/',str(OUT)+'/'],check=True)
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',units=452),indent=2)+'\n')


if __name__=='__main__':main()
