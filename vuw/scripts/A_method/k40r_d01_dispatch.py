"""D01: durable, sequential VUW wrapper queue; parity-verified unit checkpoints."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import time
from k40r_runcheck import room
from k41_job_exec import save


def main():
    b=Path(os.environ['B']);assert b==Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
    assert socket.gethostname().split('.')[0]=='cuda12','D01 belongs on VUW'
    out=b/'logs/k40r/D01';out.mkdir(parents=True,exist_ok=True)
    lock=(b/'logs/k41').joinpath('cuda12_gpu_queue.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(b,out)


def run(b,out):
    if (out/'state.json').exists():
        previous=json.loads((out/'state.json').read_text())
        for stage in previous.get('stages',[]):
            pid=stage.get('wrapper_pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                assert b'run_when_gpu_free' not in command,('prior wrapper still alive',pid)
    for p in out.glob('*.execution.json'):
        prior=json.loads(p.read_text())
        if prior['status']=='running':
            raise RuntimeError('Interrupted/active execution requires process audit before resume: '+str(p))
    manifest=json.loads((b/'scripts/A_method/k40r_mean_manifest.json').read_text())
    state=dict(task='D01',host=socket.gethostname(),controller_pid=os.getpid(),started_unix=time.time(),
               ceiling_gpu_hours=40,manifest=manifest,stages=[])
    def persist():save(out/'state.json',state)
    def hashes():
        for name,expected in manifest.items():
            actual=hashlib.md5((b/'scripts/A_method'/name).read_bytes()).hexdigest()
            assert actual==expected,('registered code changed',name)
    route_manifest=json.loads((b/'results/k40r/routes/manifest.json').read_text())
    for rel,expected in route_manifest['routing_hashes'].items():
        assert hashlib.md5((b/'results/k40r/routes'/rel).read_bytes()).hexdigest()==expected,rel
    save(out/'routing_manifest.json',route_manifest)
    refs={}
    for ds in ('nusc','av2','ahd'):
        refs[ds]={}
        for p in sorted((b/'results/k30/match'/ds).glob('*.pkl')):
            rows=pickle.load(p.open('rb'));refs[ds][p.stem]=[(int(r['frame']),bool(r['gt'])) for r in rows]
        assert (len(refs[ds]),sum(sum(g for _,g in r) for r in refs[ds].values()))==dict(nusc=(228,9181),av2=(150,11818),ahd=(24,15006))[ds]
    save(out/'reference_frames.json',refs)
    stages=[(phase,ds) for phase in ('smoke','full') for ds in ('nusc','av2','ahd')]
    try:
        for phase,ds in stages:
            hashes();room(b)
            units=list(refs[ds])[:(2 if ds=='nusc' else 1)] if phase=='smoke' else list(refs[ds])
            expected={u:refs[ds][u][:80] if phase=='smoke' else refs[ds][u] for u in units}
            label=f'{phase}_{ds}';tag='smokeD01' if phase=='smoke' else ''
            result=b/'results/k40r'/('mean_history_'+tag if tag else 'mean_history')/ds
            def validate():
                done=[]
                for unit,frames in expected.items():
                    p=result/(unit+'.pkl');meta=p.with_suffix('.done.json')
                    if not p.exists() or not meta.exists():return None
                    report=json.loads(meta.read_text());assert not report['failures']
                    assert report['script_md5']==manifest['k40r_mean_history.py']
                    rows=pickle.load(p.open('rb'))
                    assert [(int(r['frame']),bool(r['gt'])) for r in rows]==frames,(label,unit,'frame manifest')
                    assert report['comparisons']==10*len(rows)
                    done.append(report)
                return dict(units=len(done),frames=sum(x['rows'] for x in done),
                            evaluated_frames=sum(x['evaluated_frames'] for x in done),
                            frozen_comparisons=sum(x['comparisons'] for x in done))
            prior=validate()
            if prior:
                state['stages'].append(dict(id=label,status='complete_checkpoint',validation=prior));persist();continue
            statuses=list(out.glob('*.execution.json'))
            consumed=sum(float(json.loads(p.read_text()).get('elapsed_seconds',0)) for p in statuses)
            remaining=40*3600-consumed;assert remaining>0,'D01 aggregate occupancy ceiling reached'
            attempt=1
            while (out/f'{label}_{attempt}.execution.json').exists():attempt+=1
            status=out/f'{label}_{attempt}.execution.json'
            cmd=['python','scripts/A_method/k40r_mean_history.py','--dataset',ds,'--part','mean']
            if phase=='smoke':cmd+=['--units',','.join(units),'--max_frames','80','--tag',tag]
            if ds=='av2':
                cmd=['env','B_REAL='+str(b),'B='+str(b/'k10_masked'),'K10_MASK_Y0=1830',
                     'python','scripts/A_method/k10_masked_run.py']+cmd[1:]
            launch=['./run_when_gpu_free.sh',str(out/f'{label}_{attempt}.log'),'python','scripts/A_method/k41_job_exec.py',
                    '--status',str(status.relative_to(b)),'--seconds',str(remaining),'--']+cmd
            item=dict(id=label,status='queued',command=launch,execution_status=str(status),queued_unix=time.time())
            state['stages'].append(item);state['status']='queued_or_running';persist()
            env=dict(os.environ,MIN_FREE_MB='12000',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            env.pop('SHARE',None)
            proc=subprocess.Popen(launch,cwd=b,env=env);item['wrapper_pid']=proc.pid;persist();proc.wait()
            # The legacy wrapper's exit code alone does not certify the child.
            assert status.exists(),('missing child status',label)
            execution=json.loads(status.read_text());assert execution['status']=='complete',execution
            checks=validate();assert checks,('missing validated unit output',label)
            item.update(status='complete',validation=checks,execution=execution);persist()
        state['status']='complete';state['finished_unix']=time.time();persist()
    except BaseException as e:
        state.update(status='failed',error=repr(e),finished_unix=time.time());persist();raise


if __name__=='__main__':main()
