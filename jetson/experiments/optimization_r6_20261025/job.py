"""Isolated supervised worker; source hashes and atomic progress."""
import argparse,fcntl,json,os,socket,subprocess,sys,time
from common import HERE,REMOTE,save,source_hashes,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('dataset');p.add_argument('shard',nargs='?',default='0/1');a=p.parse_args()
    name='_'.join([a.task,a.dataset,a.shard.replace('/','of')]);jd=HERE/'jobs';jd.mkdir(exist_ok=True);state=jd/(name+'.json')
    lock=(jd/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state.exists():
        old=json.loads(state.read_text())
        if old.get('status')=='complete':return 0
        save(jd/'attempts'/(name+'_'+str(int(old['started']))+'.json'),old)
    before=source_hashes();save(jd/(name+'.sources_before.json'),before)
    gpu=a.task=='collect';n='4' if gpu else '1'
    env=dict(os.environ,B=str(REMOTE),B_REAL=str(REMOTE),OMP_NUM_THREADS=n,MKL_NUM_THREADS=n,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(HERE)+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,str(HERE/(a.task+'.py')),'--dataset',a.dataset,'--shard',a.shard]
    if gpu and a.dataset=='av2':
        env.update(B=str(REMOTE/'k10_masked'),K10_MASK_Y0='1830')
        cmd=[sys.executable,str(REMOTE/'scripts/A_method/k10_masked_run.py')]+cmd[1:]
    d=dict(name=name,host=socket.gethostname(),pid=os.getpid(),gpu=env.get('CUDA_VISIBLE_DEVICES'),status='running',started=time.time(),command=cmd,scripts={p.name:sha(p) for p in HERE.glob('*.py')},protocol_sha256=sha(HERE/'protocol.md'))
    save(state,d)
    with (jd/(name+'.log')).open('a') as log:r=subprocess.run(cmd,cwd=REMOTE,env=env,stdout=log,stderr=subprocess.STDOUT)
    after=source_hashes();changed=[p for p in before if before[p]!=after.get(p)];save(jd/(name+'.sources_after.json'),after)
    d.update(status='complete' if r.returncode==0 and not changed else 'failed',exit_code=r.returncode,finished=time.time(),changed_main_sources=changed);save(state,d);print(json.dumps(d),flush=True)
    return int(d['status']!='complete')
if __name__=='__main__':sys.exit(main())
