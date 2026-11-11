"""Lightweight status handoff and bounded recovery of killed immutable GPU shards."""
import json
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_20261023'
B='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
EXPS=('w1_dev_20261023','w1_bounded_20261023')


def alive(pid):
    return bool(pid and (Path('/proc')/str(pid)/'cmdline').exists())


def remote(host,code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.join(['python3','-c',code])],
                     capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr[-700:])
    return r.stdout


def snapshot():
    code='''import json,pathlib
B=pathlib.Path('''+repr(B)+''');out={}
for name in '''+repr(EXPS)+''':
 p=B/'experiments'/name;jobs=[]
 for f in (p/'jobs').glob('dev*.json'):
  if '.sources_' in f.name:continue
  j=json.loads(f.read_text());pid=j.get('pid');q=pathlib.Path('/proc')/str(pid)/'cmdline'
  j['worker_alive']=q.exists() and name.encode() in q.read_bytes()
  jobs.append(j)
 out[name]={'jobs':jobs,'completed_units':{d:len(list((p/'results/dev'/d).glob('*.pkl'))) for d in ('nusc','av2')}}
print(json.dumps(out))
'''
    # Both hosts share files, but process liveness must be checked on the owning host.
    first=json.loads(remote('VUW',code));second=json.loads(remote('VUW1',code))
    for exp in EXPS:
        a={j['name']:j for j in first[exp]['jobs']};b={j['name']:j for j in second[exp]['jobs']}
        first[exp]['jobs']=[(b[n] if 'cuda13' in j.get('host','') else j) for n,j in a.items()]
    return first


def restart_group(host,exp,dataset,pair=None):
    path=B+'/experiments/'+exp
    group_args=(['python',path+'/group4.py','--pair',str(pair)] if pair is not None else
                ['python',path+'/group.py','dev','--dataset',dataset])
    code=f"""import os,subprocess,json
from pathlib import Path
e=Path({path!r});f=(e/'logs/watchdog_{dataset}_{pair}_wrapper.log').open('a')
p=subprocess.Popen(['bash',{B+'/run_when_gpu_free.sh'!r},str(e/'logs/watchdog_{dataset}_{pair}_group.log')]+{group_args!r},
 cwd={B!r},env=dict(os.environ,SHARE='1',MIN_FREE_MB='10000'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
    cmd='source '+B+'/env.sh && '+shlex.join(['python3','-c',code])
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,cmd],capture_output=True,text=True,timeout=60)
    r.check_returncode();return int(r.stdout.strip())


def main():
    OUT.mkdir(exist_ok=True);HERE.mkdir(exist_ok=True)
    lock=(HERE/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    eventfile=OUT/'recovery_events.json'
    events=json.loads(eventfile.read_text()) if eventfile.exists() else []
    while True:
        state=dict(updated=time.time(),pid=os.getpid(),server={},local={})
        try:
            state['server']=snapshot()
            for exp,entry in state['server'].items():
                for j in entry['jobs']:
                    if j['status']=='complete':continue
                    killed=(j['status']=='failed' and j.get('exit_code') in (-9,137) and not j.get('changed_main_sources'))
                    stale=j['status']=='running' and not j['worker_alive']
                    if not (killed or stale):continue
                    dataset=j['name'].split('_')[1];host='VUW1' if 'cuda13' in j['host'] else 'VUW'
                    shard=j['name'].split('_')[-1]
                    pair=int(shard.split('of')[0])//2 if exp=='w1_bounded_20261023' and shard.endswith('of4') else None
                    key=exp+':'+dataset+(':'+str(pair) if pair is not None else '');past=[e for e in events if e['key']==key]
                    if len(past)>=2 or (past and time.time()-past[-1]['time']<900):continue
                    # Do not duplicate a detached child if only its supervisor disappeared.
                    orphan_code='''import pathlib,json
root='''+repr(B+'/experiments/'+exp+'/runner.py')+''';found=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().split(bytes([0]))
  if root.encode() in a and '''+repr(dataset.encode())+''' in a and '''+repr(j['name'].split('_')[-1].replace('of','/').encode())+''' in a:found.append(str(p.parent))
 except (OSError,PermissionError):pass
print(json.dumps(found))'''
                    if json.loads(remote(host,orphan_code)):
                        state.setdefault('attention',[]).append(dict(exp=exp,job=j['name'],reason='orphan child still running'))
                        continue
                    event=dict(key=key,time=time.time(),previous_state=j,host=host)
                    event['wrapper_pid']=restart_group(host,exp,dataset,pair);events.append(event)
                    (OUT/'recovery_events.json').write_text(json.dumps(events,indent=2)+'\n')
                    active=ROOT/'experiments'/exp/'active_coordinator.json'
                    coordinator=Path(json.loads(active.read_text())['path']) if active.exists() else ROOT/'experiments'/exp/'coordinator.py'
                    assert coordinator.parent==ROOT/'experiments'/exp
                    pidfile=coordinator.with_name('coordinator.pid')
                    oldpid=int(pidfile.read_text()) if pidfile.exists() else 0
                    if not alive(oldpid):
                        with coordinator.with_name('coordinator.log').open('a') as f:
                            p=subprocess.Popen(['/usr/bin/python3',str(coordinator)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        pidfile.write_text(str(p.pid)+'\n')
        except Exception as exc:
            state['monitor_error']=repr(exc)
        for exp,filename in [(e,'coordinator_state.json') for e in EXPS]+[
                ('runtime_20261023','logs/state.json'),('temporal_jetson_20261023','logs/state.json')]:
            file=ROOT/'experiments'/exp/filename
            if file.exists():state['local'][exp]=json.loads(file.read_text())
        done={name:(ROOT/'docs/paper/handoff'/name/'DONE.json').exists()
              for name in ('RUNTIME_20261023','NAVSIM_REUSE_20261023','TEMPORAL_JETSON_20261023','REFINER_GEOMETRY_20261023')}
        state['completed_reports']=done
        state['status']='complete' if all(done.values()) and all(state['local'].get(e,{}).get('phase')=='complete' for e in EXPS) else 'running'
        tmp=OUT/'STATUS.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'STATUS.json')
        lines=['# Optimization development phase — live handoff','',
            'Separate experiments only; frozen method/weights/K34 unchanged. All executed rows retained.',
            'Updated epoch: '+str(state['updated']),
            'Detailed review: ../W1_DEV_20261023/OPTIMIZATION_REVIEW.md','',
            '| Work | Status | Progress |','|---|---|---|']
        for exp in EXPS:
            s=state['local'].get(exp,{});progress=state['server'].get(exp,{}).get('completed_units',{})
            lines.append(f"| {exp} | {s.get('phase','unknown')} | {progress} |")
        for exp in ('runtime_20261023','temporal_jetson_20261023'):
            s=state['local'].get(exp,{})
            lines.append(f"| {exp} | {s.get('status','unknown')} | {len(s.get('runs',[]))} completed processes |")
        lines+=['','NAVSIM initial CPU pass: exact scores/actions/field/support on 1,000 tokens; 7x7 mean 116→13 ms,',
            '3x3 28→7.7 ms. Independent three-repetition/two-host replication is in ../NAVSIM_REUSE_20261023/.',
            'Runtime buffer results: ../RUNTIME_20261023/. Temporal Jetson results: ../TEMPORAL_JETSON_20261023/.',
            'Full dev factorial: ../W1_BOUNDED_20261023/. Original screen: ../W1_DEV_20261023/.',
            'Refiner geometry attribution: ../REFINER_GEOMETRY_20261023/.',
            'Budget controls include historical pixels but guards may overrun; no exact equal-cost superiority claim.',
            'No method is promoted, and existing test data are not an untouched confirmatory set.']
        if state.get('monitor_error'):lines+=['','Monitor connection error: '+state['monitor_error']]
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        if state['status']=='complete':break
        time.sleep(300)


if __name__=='__main__':main()
