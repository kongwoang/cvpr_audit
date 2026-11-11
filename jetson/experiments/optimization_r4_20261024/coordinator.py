"""Supervise the fixed R4 population and bring every summary back to the writer."""
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path
from launch import ROOT as REMOTE, EXP
from common import save

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024'

def ssh(command, timeout=1200):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW',
        'source '+REMOTE+'/env.sh && '+command],text=True,timeout=timeout)

def states():
    code="import json;from pathlib import Path;p=Path(%r)/'jobs';print(json.dumps({f.stem:json.loads(f.read_text()) for f in p.glob('*.json') if '.sources_' not in f.name}))"%EXP
    return json.loads(ssh(shlex.join(['python','-c',code])))

def run(script,*args):
    print(ssh('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.join(['python',EXP+'/'+script,*args])),flush=True)

def sync():
    OUT.mkdir(parents=True,exist_ok=True)
    for source,dest in [('reports/',str(OUT)+'/'),('jobs/',str(HERE/'job_status')+'/')]:
        subprocess.run(['rsync','-a','--rsync-path=source '+REMOTE+'/env.sh && rsync','VUW:'+EXP+'/'+source,dest],check=True)
    mirror=ROOT/'results/vuw_summaries/OPTIMIZATION_R4_20261024'
    subprocess.run(['rsync','-a',str(OUT)+'/',str(mirror)+'/'],check=True)

def main():
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('analyze.py','--stage','readout')
    run('audit_rescues.py','--stage','readout','--report','readout')
    run('analyze.py','--stage','smoke')
    sync()
    names=[f'dev_nusc_{i}of2' for i in range(2)]+[f'dev_av2_{i}of4' for i in range(4)]
    failures=0
    while True:
        try:st=states();failures=0
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            failures+=1
            if failures>3:raise
            time.sleep(30);continue
        failed={k:v.get('exit_code') for k,v in st.items() if v['status']=='failed'}
        if failed:raise RuntimeError(failed)
        save(OUT/'STATUS.json',dict(phase='inference_running',updated=time.time(),jobs={k:v['status'] for k,v in st.items()},core_changed=False))
        print(time.strftime('%H:%M:%S'),{k:v['status'] for k,v in st.items()},flush=True)
        if all(st.get(n,{}).get('status')=='complete' for n in names):break
        time.sleep(40)
    run('analyze.py','--stage','dev')
    run('audit_rescues.py','--stage','dev','--report','dev')
    run('cost_audit.py')
    code='''import json
from pathlib import Path
from common import source_hashes,save,HERE,REMOTE
current=source_hashes();states={f.stem:json.loads(f.read_text()) for f in (HERE/'jobs').glob('*.json') if '.sources_' not in f.name}
assert all(s['status']=='complete' and s['exit_code']==0 and not s['changed_main_sources'] for s in states.values())
for f in (HERE/'jobs').glob('*.sources_before.json'):
 before=json.loads(f.read_text());after=json.loads(f.with_name(f.name.replace('.sources_before.','.sources_after.')).read_text())
 assert before==after and all(current.get(k)==v for k,v in before.items())
rows=[]
for stage in ['readout','dev']:
 for ds in ['nusc','av2']:
  expected={p.stem for p in (REMOTE/'results/k30/dev_devref'/ds).glob('*.pkl')}
  actual={p.stem for p in (HERE/'results'/stage/ds).glob('*.pkl')};assert actual==expected
  frames=0
  for unit in actual:
   audit=json.loads((HERE/'results'/stage/ds/(unit+'.parity.json')).read_text());assert not any(audit['differences'].values());frames+=audit['frames']
  rows.append(dict(stage=stage,dataset=ds,units=len(actual),frames=frames,baseline_differences=0))
save(HERE/'reports/server_completion_audit.json',dict(status='complete',jobs=len(states),all_exit_zero=True,unit_suites=rows,main_source_weight_hashes_unchanged=True,test_data_used=False))
'''
    print(ssh('cd '+EXP+' && '+shlex.join(['python','-c',code])))
    sync();save(OUT/'STATUS.json',dict(phase='inference_and_analysis_complete',updated=time.time(),core_changed=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'STATUS.json',dict(phase='coordinator_failed',error=repr(e),updated=time.time()));raise
