"""Finish compact readout control; fixed suites and automatic report generation."""
import fcntl
import json
import time
import coordinator as C
from launch import launch
from common import save

def main():
    lock=(C.HERE/'compact.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=json.loads((C.HERE/'case_manifest_with_severity.json').read_text())['cases'];launched=set()
    while True:
        st=C.states();failed={k:v.get('exit_code') for k,v in st.items() if v['status']=='failed'}
        if failed:raise RuntimeError(failed)
        if 'smoke' not in launched and C.done(st,[f'compactfit_{ds}_0of1' for ds in ['nusc','av2']]):
            jobs=[['compactsmoke',ds,'0/1','--units',','.join(sorted({c['unit'] for c in cases if c['dataset']==ds}))] for ds in ['nusc','av2']]
            launch('VUW1','compact_smoke',jobs);launched.add('smoke')
        if 'dev' not in launched and C.done(st,[f'compactsmoke_{ds}_0of1' for ds in ['nusc','av2']]):
            C.analyze('compactsmoke')
            launch('VUW','compact_nusc',[['compact','nusc','0/2'],['compact','nusc','1/2']])
            launch('VUW1','compact_av2_a',[['compact','av2','0/4'],['compact','av2','1/4']])
            launch('VUW','compact_av2_b',[['compact','av2','2/4'],['compact','av2','3/4']]);launched.add('dev')
        names=[f'compact_nusc_{i}of2' for i in range(2)]+[f'compact_av2_{i}of4' for i in range(4)]
        if C.done(st,names):C.analyze('compact');C.sync();save(C.OUT/'COMPACT_STATUS.json',dict(phase='complete',updated=time.time()));break
        save(C.OUT/'COMPACT_STATUS.json',dict(phase='running',launched=sorted(launched),updated=time.time()));time.sleep(45)
if __name__=='__main__':
    try:main()
    except BaseException as e:save(C.OUT/'COMPACT_STATUS.json',dict(phase='failed',error=repr(e)));raise
