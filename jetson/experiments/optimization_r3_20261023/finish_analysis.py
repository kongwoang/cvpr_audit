"""Finish every full-development report and audit once its supervised jobs finish."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE,REMOTE,save,source_hashes

STAGES=['dev','learned','calibrated','lift','persist','iddev','low','compact']

def run(script,*args):
    subprocess.run([sys.executable,str(HERE/script),*args],check=True,
                   env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))

def main():
    lock=(HERE/'finish_analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=set();started=time.time()
    while len(done)<len(STAGES):
        states={f.stem:json.loads(f.read_text()) for f in (HERE/'jobs').glob('*.json') if '.sources_' not in f.name}
        failed={k:v['exit_code'] for k,v in states.items() if v['status']=='failed'}
        assert not failed,failed
        for stage in STAGES:
            names=[f'{stage}_nusc_{i}of2' for i in range(2)]+([] if stage=='iddev' else [f'{stage}_av2_{i}of4' for i in range(4)])
            if stage in done or not all(states.get(n,{}).get('status')=='complete' for n in names):continue
            # Existing coordinators own other full reports; this worker owns low.
            report=HERE/'reports'/stage
            if stage=='low' and not (report/'REPORT.md').exists():run('analyze.py','--stage',stage)
            if not (report/'REPORT.md').exists():continue
            if not (report/'attribution_STATUS.json').exists():run('audit_rescues.py','--stage',stage,'--report',stage)
            if stage in ('low','compact'):run('audit_fallback.py','--stage',stage)
            done.add(stage)
        save(HERE/'reports/ANALYSIS_STATUS.json',dict(status='running',completed=sorted(done),started=started,updated=time.time()))
        if len(done)<len(STAGES):time.sleep(40)
    states={f.stem:json.loads(f.read_text()) for f in (HERE/'jobs').glob('*.json') if '.sources_' not in f.name}
    assert all(v['status']=='complete' for v in states.values()),{k:v['status'] for k,v in states.items() if v['status']!='complete'}
    frozen=source_hashes();snapshots=[]
    for f in sorted((HERE/'jobs').glob('*.sources_before.json')):
        before=json.loads(f.read_text());after=json.loads(f.with_name(f.name.replace('.sources_before.','.sources_after.')).read_text())
        assert before==after and all(frozen.get(k)==v for k,v in before.items()),f
        snapshots.append(f.name)
    units=[]
    for stage in STAGES:
        for ds in (['nusc'] if stage=='iddev' else ['nusc','av2']):
            expected={p.stem for p in (REMOTE/'results/k30/dev_devref'/ds).glob('*.pkl')}
            actual={p.stem for p in (HERE/'results'/stage/ds).glob('*.pkl')};assert actual==expected
            total=0
            for unit in actual:
                p=json.loads((HERE/'results'/stage/ds/(unit+'.parity.json')).read_text());assert not any(p['differences'].values());total+=p['frames']
            units.append(dict(stage=stage,dataset=ds,units=len(actual),frames=total,baseline_differences=0))
    attempts=[json.loads(f.read_text()) for f in (HERE/'jobs/attempts').glob('*.json')]
    save(HERE/'reports/server_completion_audit.json',dict(status='complete',jobs=len(states),all_exit_zero=all(v['exit_code']==0 for v in states.values()),
        unit_suites=units,main_source_weight_hashes_unchanged=True,hash_snapshots=len(snapshots),current_main_hashes=frozen,
        prior_attempts=[dict(name=v['name'],status=v['status'],exit_code=v.get('exit_code')) for v in attempts],
        test_data_used=False,started=started,finished=time.time()))
    run('summarize.py')
    save(HERE/'reports/ANALYSIS_STATUS.json',dict(status='complete',completed=sorted(done),started=started,finished=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as e:save(HERE/'reports/ANALYSIS_STATUS.json',dict(status='failed',error=repr(e)));raise
