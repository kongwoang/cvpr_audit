#!/usr/bin/env bash
# K31 variant B2K, test run (registered in handoff/K31/STATUS.md): k31_eval.py part b2v with the dev-selected candidate
# ($1, A or B) on the three test splits, 3 shards per dataset, one group (3 processes) per run_when_gpu_free.sh call
# (SHARE=1, MIN_FREE_MB 3500). Finished units are skipped.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k31; mkdir -p $L
M=${1:?B2K mode A or B}
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
cmd() {
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k31_eval.py --dataset av2 $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k31_eval.py --dataset $ds $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
for ds in nusc av2 ahd; do
  a=()
  for ((i = 0; i < 3; i++)); do a+=("$(cmd $ds --part b2v --b2k_modes $M --shard $i/3)"); done
  grp $L/b2v_${ds}.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
done
echo "$(date +%T) K31 B2V LAUNCHED (mode $M)"
