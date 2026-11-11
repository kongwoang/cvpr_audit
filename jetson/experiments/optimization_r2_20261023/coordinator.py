"""Durable smoke/parity gate, guarded two-host development jobs, complete handoff."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from launch import launch,REMOTE,EXP

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R2_20261023'

def save(phase,**kw):
    data=dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw)
    for path in [HERE/'coordinator_state.json',OUT/'STATUS.json']:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(data),flush=True)

def ssh(host,command,timeout=90):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        'source '+REMOTE+'/env.sh && '+command],text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError((host,r.returncode,r.stderr[-2000:],r.stdout[-2000:]))
    return r.stdout

def states():
    code=f'''import json,pathlib
p=pathlib.Path({EXP!r})/'jobs'
print(json.dumps({{f.stem:json.loads(f.read_text()) for f in p.glob('*.json') if '.sources_' not in f.name}}))
'''
    return json.loads(ssh('VUW1',shlex.join(['python','-c',code])))

def wait(names,phase):
    errors=0
    while True:
        try:s=states();errors=0
        except Exception as e:
            errors+=1;save(phase,connection_errors=errors,error=str(e));time.sleep(30);continue
        jobs={n:s.get(n,{'status':'queued'}) for n in names};save(phase,jobs=jobs)
        if any(j['status']=='failed' for j in jobs.values()):raise RuntimeError(('failed jobs',jobs))
        if all(j['status']=='complete' for j in jobs.values()):return
        time.sleep(30)

def analyze(stage):
    command=f'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 B_REAL={REMOTE} python {EXP}/analyze.py --stage {stage}'
    print(ssh('VUW1',command,timeout=1200),flush=True)
    folder=OUT/stage;folder.mkdir(exist_ok=True)
    subprocess.run(['rsync','-az','VUW1:'+EXP+'/reports/'+stage+'/',str(folder)+'/'],check=True)
    subprocess.run(['rsync','-az','VUW1:'+EXP+'/jobs/',str(HERE/'job_status')+'/'],check=True)

def main():
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save('waiting_smoke')
    wait(['smoke_nusc_0of1','smoke_av2_0of1'],'smoke_running')
    analyze('smoke')
    launches=[launch('VUW1','dev','nusc'),launch('VUW','dev','av2')]
    (HERE/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
    save('dev_queued',launches=launches)
    wait(['dev_nusc_0of2','dev_nusc_1of2','dev_av2_0of2','dev_av2_1of2'],'dev_running')
    analyze('dev')
    hashes=json.loads((HERE/'fork_manifest.json').read_text())
    changed=[p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (OUT/'INLOOP_REPORT.md').write_text((OUT/'dev/REPORT.md').read_text()+
        '\nSource integrity: local frozen source hashes unchanged; every remote job checked main source/weights before and after.\n')
    subprocess.run([str(ROOT/'scripts/py'),str(HERE/'summarize.py')],cwd=ROOT,check=True,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    save('complete',main_source_unchanged=True,report=str(OUT/'REPORT.md'))

if __name__=='__main__':
    try:main()
    except Exception as exc:save('failed',error=repr(exc));raise
