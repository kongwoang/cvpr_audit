"""Local dependency coordinator; independent of this chat's lifetime."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from launch import launch,ROOT as REMOTE,EXP

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R3_20261023'
def ssh(cmd,timeout=180):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW','source '+REMOTE+'/env.sh && '+cmd],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    return r.stdout
def states():
    code="import json;from pathlib import Path;p=Path(%r)/'jobs';print(json.dumps({f.stem:json.loads(f.read_text()) for f in p.glob('*.json') if '.sources_' not in f.name}))"%EXP
    return json.loads(ssh(shlex.join(['python','-c',code])))
def done(st,names):return all(st.get(s,{}).get('status')=='complete' for s in names)
def persist(phase,**kw):
    OUT.mkdir(parents=True,exist_ok=True);data=dict(phase=phase,updated=time.time(),**kw)
    tmp=OUT/'STATUS.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(OUT/'STATUS.json')
def analyze(stage):
    ssh('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python '+EXP+'/analyze.py --stage '+stage,timeout=900)
def sync():
    OUT.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--rsync-path=source '+REMOTE+'/env.sh && rsync','VUW:'+EXP+'/reports/',str(OUT)+'/'],check=True)
    subprocess.run(['rsync','-a','--rsync-path=source '+REMOTE+'/env.sh && rsync','VUW:'+EXP+'/jobs/',str(HERE/'job_status')+'/'],check=True)
    subprocess.run(['rsync','-a','--include=*/','--include=*.json','--include=*.csv','--exclude=*',
        '--rsync-path=source '+REMOTE+'/env.sh && rsync','VUW:'+EXP+'/models/',str(OUT/'training')+'/'],check=True)
def main():
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launched=set();completed=set();cases=json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases']
    while True:
        st=states();failed={k:v for k,v in st.items() if v['status']=='failed'}
        # Initial import failures were repaired and retried; latest status is authoritative.
        if failed:raise RuntimeError({k:v.get('exit_code') for k,v in failed.items()})
        for ds,host in [('nusc','VUW'),('av2','VUW1')]:
            key='fit_'+ds
            if key not in launched and done(st,[f'encode_{ds}_{i}of2' for i in range(2)]):
                launch(host,key,[['fit',ds,'0/2','--objective','geometry'],['fit',ds,'1/2','--objective','decision']]);launched.add(key)
        if 'learnsmoke' not in launched and done(st,[f'fit_{ds}_{i}of2' for ds in ['nusc','av2'] for i in range(2)]):
            jobs=[['learnsmoke',ds,'0/1','--units',','.join(sorted({c['unit'] for c in cases if c['dataset']==ds}))] for ds in ['nusc','av2']]
            launch('VUW','learned_smoke',jobs);launched.add('learnsmoke')
        if 'learned_dev' not in launched and done(st,['learnsmoke_nusc_0of1','learnsmoke_av2_0of1']):
            analyze('learnsmoke')
            launch('VUW','learned_nusc',[['learned','nusc','0/2'],['learned','nusc','1/2']])
            launch('VUW1','learned_av2_a',[['learned','av2','0/4'],['learned','av2','1/4']])
            launch('VUW','learned_av2_b',[['learned','av2','2/4'],['learned','av2','3/4']]);launched.add('learned_dev')
        for stage in ('dev','learned'):
            names=[f'{stage}_nusc_{i}of2' for i in range(2)]+[f'{stage}_av2_{i}of4' for i in range(4)]
            if stage not in completed and done(st,names):analyze(stage);completed.add(stage);sync()
        persist('running',launched=sorted(launched),completed=sorted(completed),jobs={k:v['status'] for k,v in st.items()})
        print(time.strftime('%H:%M:%S'),{k:v['status'] for k,v in st.items()},flush=True)
        if completed=={'dev','learned'}:
            sync();persist('inference_complete',completed=sorted(completed));break
        time.sleep(45)
if __name__=='__main__':
    try:main()
    except BaseException as e:persist('coordinator_failed',error=repr(e));raise
