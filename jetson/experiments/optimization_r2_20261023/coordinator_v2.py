"""Resume unchanged AV2 experiment on four shards after a documented checkpoint."""
import fcntl
import hashlib
import json
import shlex
import time
from pathlib import Path
import coordinator as M


def launch_pair(host,pair):
    code=f'''import os,subprocess,json
from pathlib import Path
b=Path({M.REMOTE!r});e=Path({M.EXP!r});saved=e/'logs/launch_av2_pair{pair}_v2.json'
if saved.exists():
 old=json.loads(saved.read_text());p=Path('/proc')/str(old['wrapper_pid'])/'cmdline'
 if p.exists() and b'group4.py' in p.read_bytes():print(old['wrapper_pid']);raise SystemExit(0)
states=[e/'jobs'/f'dev_av2_{{i}}of4.json' for i in {list(range(2*pair,2*pair+2))!r}]
if all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in states):print(0);raise SystemExit(0)
if all(p.exists() and json.loads(p.read_text()).get('status') in ('running','complete') for p in states):
 active=[json.loads(p.read_text()) for p in states if json.loads(p.read_text()).get('status')=='running']
 if active and all((Path('/proc')/str(j['pid'])/'cmdline').exists() for j in active):print(0);raise SystemExit(0)
f=(e/'logs/av2_pair{pair}_v2_wrapper.log').open('a')
p=subprocess.Popen(['bash',str(b/'run_when_gpu_free.sh'),str(e/'logs/av2_pair{pair}_v2_group.log'),
 'python',str(e/'group4.py'),'--pair','{pair}'],cwd=b,env=dict(os.environ,SHARE='1',MIN_FREE_MB='10000'),
 stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
saved.write_text(json.dumps(dict(host={host!r},pair={pair},wrapper_pid=p.pid,started={time.time()})))
print(p.pid)
'''
    command=shlex.join(['python','-c',code])
    pid=int(M.ssh(host,command).strip())
    return dict(host=host,pair=pair,wrapper_pid=pid)


def main():
    lock=(M.HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (M.HERE/'repartition.json').exists()
    launches=[launch_pair('VUW',0),launch_pair('VUW1',1)]
    (M.HERE/'launches_v2.json').write_text(json.dumps(launches,indent=2)+'\n')
    M.save('dev_queued_four_shards',launches=launches)
    names=['dev_nusc_0of2','dev_nusc_1of2']+[f'dev_av2_{i}of4' for i in range(4)]
    M.wait(names,'dev_running_four_shards')
    M.analyze('dev')
    manifest=json.loads((M.HERE/'fork_manifest.json').read_text())
    changed=[p for p,h in manifest.items() if hashlib.sha256((M.ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    text=(M.OUT/'dev/REPORT.md').read_text()
    text+='\nMain source/weight integrity checked by every job; local frozen hashes unchanged. Scheduling only: remaining AV2 resumed across four disjoint shards.\n'
    (M.OUT/'INLOOP_REPORT.md').write_text(text)
    import os,subprocess
    subprocess.run([str(M.ROOT/'scripts/py'),str(M.HERE/'summarize.py')],cwd=M.ROOT,check=True,
                   env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    M.save('complete',main_source_unchanged=True,report=str(M.OUT/'REPORT.md'),scheduling='four AV2 shards')


if __name__=='__main__':
    try:main()
    except BaseException as exc:M.save('failed',error=repr(exc));raise
