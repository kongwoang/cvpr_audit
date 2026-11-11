"""Local durable coordinator: smoke gate -> two hosts -> analysis -> local handoff.
No source deployment, no model selection, no main-code mutation.
"""
import hashlib
import fcntl
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REMOTE='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
EXP=REMOTE+'/experiments/w1_bounded_20261023'
HANDOFF=ROOT/'docs/paper/handoff/W1_BOUNDED_20261023'
HANDOFF.mkdir(parents=True,exist_ok=True)

def status(phase,**kwargs):
    state=dict(phase=phase,updated=time.time(),pid=__import__('os').getpid(),**kwargs)
    for path in [HERE/'coordinator_state.json',HANDOFF/'STATUS.json']:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(state),flush=True)

def ssh(host,cmd,timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,cmd],text=True,capture_output=True,timeout=timeout)
    if r.returncode: raise RuntimeError((host,r.returncode,r.stderr[-1500:],r.stdout[-1500:]))
    return r.stdout

def remote_py(host,code):return ssh(host,shlex.join(['python3','-c',code]))

def states():
    raw=remote_py('VUW1',f'''import json
from pathlib import Path
p=Path({EXP!r})/'jobs'
print(json.dumps([json.loads(f.read_text()) for f in p.glob('*.json') if '.sources_' not in f.name]))
''')
    return {r['name']:r for r in json.loads(raw)}

def wait_jobs(names,phase):
    errors=0
    while True:
        try:
            all_states=states();errors=0
        except Exception as exc:
            errors+=1;status(phase,connection_errors=errors,error=str(exc))
            time.sleep(30);continue
        jobs={k:all_states.get(k,{'status':'queued'}) for k in names}
        status(phase,jobs=jobs)
        failed=[k for k,v in jobs.items() if v['status']=='failed']
        if failed:raise RuntimeError(('job failed',failed))
        if all(v['status']=='complete' for v in jobs.values()):return
        time.sleep(30)

def analyze(stage):
    cmd=f'cd {REMOTE} && source {REMOTE}/env.sh && export B_REAL={REMOTE} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && python {EXP}/analyze.py --stage {stage}'
    print(ssh('VUW1',shlex.join(['bash','-c',cmd]),timeout=300),flush=True)
    dest=HANDOFF/stage;dest.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','VUW1:'+EXP+'/reports/'+stage+'/',str(dest)+'/'],check=True)
    subprocess.run(['rsync','-a','VUW1:'+EXP+'/jobs/',str(HERE/'job_status')+'/'],check=True)

def launch(host,dataset):
    code=f'''import os,subprocess,json,fcntl
from pathlib import Path
b=Path({REMOTE!r});e=Path({EXP!r})
lock=(e/'logs/launch_{dataset}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
saved=e/'logs/launch_{dataset}.json'
if saved.exists():
    prior=json.loads(saved.read_text())
    proc=Path('/proc')/str(prior['wrapper_pid'])/'cmdline'
    if proc.exists() and b'group.py' in proc.read_bytes() and b'w1_bounded_20261023' in proc.read_bytes():
        print(prior['wrapper_pid']);raise SystemExit(0)
states=[e/'jobs'/f'dev_{dataset}_{{i}}of2.json' for i in range(2)]
if all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in states):
    print(0);raise SystemExit(0)
f=(e/'logs/dev_{dataset}_wrapper.log').open('a')
p=subprocess.Popen(['bash',str(b/'run_when_gpu_free.sh'),str(e/'logs/dev_{dataset}_group.log'),
    'python',str(e/'group.py'),'dev','--dataset',{dataset!r}],cwd=b,
    env=dict(os.environ,SHARE='1',MIN_FREE_MB='10000'),stdin=subprocess.DEVNULL,stdout=f,
    stderr=subprocess.STDOUT,start_new_session=True)
saved.write_text(json.dumps(dict(host={host!r},wrapper_pid=p.pid,dataset={dataset!r})))
print(p.pid)
'''
    pid=int(remote_py(host,code).strip())
    return dict(host=host,dataset=dataset,wrapper_pid=pid)

def main():
    lock=(HERE/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('waiting_smoke')
    wait_jobs(['smoke_nusc_0of1','smoke_av2_0of1'],'smoke_running')
    analyze('smoke')
    # Baseline parity has been asserted for every smoke unit by both runner and analyzer.
    launches=[launch('VUW1','nusc'),launch('VUW','av2')]
    (HERE/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
    status('dev_queued',launches=launches)
    wait_jobs(['dev_nusc_0of2','dev_nusc_1of2','dev_av2_0of2','dev_av2_1of2'],'dev_running')
    while True:
        original=json.loads(ssh('VUW1',shlex.join(['python3','-c',
            "import json,pathlib; p=pathlib.Path('"+REMOTE+"/experiments/w1_dev_20261023/jobs'); print(json.dumps([json.loads(f.read_text()).get('status') for f in p.glob('dev*.json') if '.sources_' not in f.name]))"])))
        if len(original)==4 and all(x=='complete' for x in original):break
        if any(x=='failed' for x in original):raise RuntimeError(('original development run failed',original))
        status('waiting_original_controls',original_status=original);time.sleep(30)
    analyze('dev')
    manifest=json.loads((HERE/'fork_manifest.json').read_text())
    changed=[p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError(('local main source changed',changed))
    report=(HANDOFF/'dev/REPORT.md').read_text()
    report+='\nSource integrity: local frozen hashes match the pre-experiment manifest; every remote job checked main source and weight hashes before/after.\n'
    report+='\nRaw records and exact commands: experiments/w1_bounded_20261023 on VUW shared workspace; mirrored job metadata in the local experiment folder.\n'
    (HANDOFF/'REPORT.md').write_text(report)
    status('complete',main_source_unchanged=True,report=str(HANDOFF/'REPORT.md'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=repr(exc));raise
