"""Two small frozen-checkpoint audit workers, launched only via GPU wrapper."""
import os
from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
jobs=[]
for ds in ['nusc','av2']:
    log=(p/'logs'/('head_'+ds+'.log')).open('a')
    jobs.append((subprocess.Popen([sys.executable,str(p/'audit_head.py'),'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')),log))
codes=[j.wait() for j,_ in jobs]
for _,log in jobs:log.close()
print('HEAD AUDIT EXIT',codes,flush=True)
sys.exit(int(any(codes)))
