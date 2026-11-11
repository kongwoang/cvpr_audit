"""Wait for all follow-up smoke checks, then group serial suites per GPU worker."""
import fcntl
import time
import coordinator as C
from launch import launch
from common import save

def main():
    lock=(C.HERE/'followups.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    smoke=[f'{stage}_{ds}_0of1' for stage in ['calsmoke','liftsmoke','persistsmoke'] for ds in ['nusc','av2']]+['idsmoke_nusc_0of1']
    launched=False;reported=set()
    while True:
        st=C.states();failed={k:v.get('exit_code') for k,v in st.items() if v['status']=='failed'}
        if failed:raise RuntimeError(failed)
        if not launched and C.done(st,smoke):
            for stage in ['calsmoke','liftsmoke','persistsmoke']:
                C.analyze(stage)
            C.ssh('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python '+C.EXP+'/analyze.py --stage idsmoke --dataset-only nusc',timeout=900)
            launch('VUW','followups_nusc',[['followups','nusc','0/2'],['followups','nusc','1/2']])
            launch('VUW1','followups_av2_a',[['followups','av2','0/4'],['followups','av2','1/4']])
            launch('VUW','followups_av2_b',[['followups','av2','2/4'],['followups','av2','3/4']]);launched=True
        for stage in ['calibrated','lift','persist','iddev']:
            names=[f'{stage}_nusc_{i}of2' for i in range(2)]+([] if stage=='iddev' else [f'{stage}_av2_{i}of4' for i in range(4)])
            if stage not in reported and C.done(st,names):
                if stage=='iddev':C.ssh('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python '+C.EXP+'/analyze.py --stage iddev --dataset-only nusc',timeout=900)
                else:C.analyze(stage)
                reported.add(stage);C.sync()
        state=dict(phase='running' if len(reported)<4 else 'complete',launched=launched,reported=sorted(reported),updated=time.time())
        save(C.OUT/'FOLLOWUPS_STATUS.json',state);print(state,flush=True)
        if len(reported)==4:break
        time.sleep(45)
if __name__=='__main__':
    try:main()
    except BaseException as e:save(C.OUT/'FOLLOWUPS_STATUS.json',dict(phase='failed',error=repr(e)));raise
