"""Local persistent coordinator for CPU K34 jobs on VUW and VUW1."""
import fcntl
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

R=Path(__file__).resolve().parents[1];B='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR';L=R/'logs/k34cpu'
L.mkdir(parents=True,exist_ok=True)
STATE={'started_at':datetime.now().astimezone().isoformat(),'status':'running','phase':'initialization','jobs':{}}

def save():
    STATE['updated_at']=datetime.now().astimezone().isoformat()
    tmp=L/'state.tmp';tmp.write_text(json.dumps(STATE,indent=2)+'\n');tmp.replace(L/'state.json')

def ssh(host,command):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=30',host,command],text=True,timeout=90)

def launch(host,name,script,args=(),nav=False):
    env=B+'/scripts/common/'+('env_navsim.sh' if nav else 'env.sh')
    cmd=['python','-u',B+'/scripts/A_method/k34_job.py',name,'python','-u',B+'/scripts/'+script,*args]
    shell='source '+shlex.quote(env)+'; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
    shell+='nohup '+shlex.join(cmd)+' > '+shlex.quote(B+'/logs/k34/'+name+'.log')+' 2>&1 < /dev/null & echo $!'
    pid=ssh(host,shell).strip();STATE['jobs'][name]={'host':host,'pid':pid,'log':B+'/logs/k34/'+name+'.log'};save()
    print('LAUNCH',name,host,pid,flush=True)

def remote_states():
    code="import json,pathlib; p=pathlib.Path('"+B+"/results/k34/jobs'); print(json.dumps({f.stem:json.loads(f.read_text()) for f in p.glob('*.json')}))"
    return json.loads(ssh('VUW','source '+B+'/env.sh; python -c '+shlex.quote(code)))

def wait_jobs(names):
    failures=0
    while True:
        try:s=remote_states();failures=0
        except (subprocess.SubprocessError,OSError) as e:
            failures+=1;print('SSH RETRY',failures,str(e),flush=True);time.sleep(30);continue
        for name in names:
            if name in s:STATE['jobs'][name].update(s[name])
        save()
        bad=[name for name in names if s.get(name,{}).get('status')=='failed']
        if bad:raise RuntimeError('CPU jobs failed: '+','.join(bad))
        complete=sum(s.get(name,{}).get('status')=='complete' for name in names)
        print(STATE['phase'],complete,'/',len(names),'complete',flush=True)
        if complete==len(names):return
        time.sleep(30)

def main():
    lock=(L/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save()
    # Validated smoke units are the identical CPU replay, not a selected timing sample.
    code="""import pathlib,pickle,shutil
B=pathlib.Path('"""+B+"""')
for ds in ('nusc','av2','ahd'):
 p=B/'results/k34/replay'/ds;p.mkdir(parents=True,exist_ok=True)
 for f in (B/'results/k34/replay_smoke'/ds).glob('*.pkl'):
  q=pickle.load(f.open('rb'));assert all(x['noref_matches_apdump'] and x['ref_matches_apdump'] for x in q['checks'])
  if not (p/f.name).exists():shutil.copy2(f,p/f.name)
 assert (B/'results/k34/episodes'/ds/'DONE.json').exists()
"""
    ssh('VUW','source '+B+'/env.sh; python -c '+shlex.quote(code))
    STATE['phase']='K34-1 replay';names=[]
    for host,ds,shards in [('VUW','nusc',1),('VUW','ahd',2),('VUW1','av2',2)]:
        for i in range(shards):
            name=f'replay_{ds}_{i}of{shards}';names.append(name)
            launch(host,name,'A_method/k34_replay.py',['--dataset',ds,'--shard',f'{i}/{shards}'])
    wait_jobs(names)
    STATE['phase']='K34-1 operating';names=[]
    for host,ds in [('VUW','nusc'),('VUW1','av2'),('VUW','ahd')]:
        name='operating_'+ds;names.append(name);launch(host,name,'A_method/k34_operating.py',['--dataset',ds])
    wait_jobs(names)
    code="""import pathlib,json
B=pathlib.Path('"""+B+"""');p=B/'results/k34'
assert all((p/s/d/'DONE.json').exists() for s in ('episodes','operating') for d in ('nusc','av2','ahd'))
(p/'GATE_K34_1_COMPLETE.json').write_text(json.dumps({'complete':True,'jetson_K34_3':'complete 2026-10-22'}))
"""
    ssh('VUW','source '+B+'/env.sh; python -c '+shlex.quote(code))
    STATE['phase']='K34-2 coverage';names=[]
    for host,ds in [('VUW','nusc'),('VUW1','av2'),('VUW','ahd')]:
        name='coverage_'+ds;names.append(name);launch(host,name,'A_method/k34_coverage.py',['--dataset',ds])
    wait_jobs(names)
    STATE['phase']='K34-4 selector'
    launch('VUW1','selector','A_navsim/k34_selector_cost.py',nav=True);wait_jobs(['selector'])
    STATE['phase']='pull and handoff';save()
    dest=R/'results/vuw_summaries/k34';dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--include=*/','--include=*.csv','--include=*.json','--exclude=*',
                    'VUW:'+B+'/results/k34/',str(dest)+'/'],check=True)
    subprocess.run([str(R/'scripts/py'),str(R/'scripts/k34_cpu_handoff.py')],check=True)
    STATE['phase']='complete';STATE['status']='complete';save();print('K34 CPU COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        STATE.update(status='failed',error=repr(e));save();raise
