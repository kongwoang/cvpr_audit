"""Wait for four-shard completion, then run the immutable-record CPU cost audit."""
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
B='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'
E=B+'/experiments/w1_bounded_20261023'


def main():
    while json.loads((ROOT/'experiments/w1_bounded_20261023/coordinator_state.json').read_text()).get('phase')!='complete':
        time.sleep(30)
    subprocess.run(['scp','-q',str(ROOT/'experiments/w1_bounded_20261023/cost_audit.py'),'VUW1:'+E+'/cost_audit.py'],check=True)
    command='source '+B+'/env.sh && export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && python '+E+'/cost_audit.py'
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW1',shlex.join(['bash','-c',command])],check=True,timeout=300)
    out=ROOT/'docs/paper/handoff/W1_BOUNDED_20261023/cost_audit'
    subprocess.run(['rsync','-a','VUW1:'+E+'/reports/cost_audit/',str(out)+'/'],check=True)


if __name__=='__main__':main()
