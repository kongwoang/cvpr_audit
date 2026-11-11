#!/usr/bin/env bash
# K31 items 3 / 4 (pre-registered in handoff/K31/STATUS.md): k31_eval.py parts curvdev (dev split: nuScenes new val, AV2
# CFG-DEV with the ego mask), curv and b2 (three test splits), 3 shards per dataset and part, one group (3 processes) per
# run_when_gpu_free.sh call (SHARE=1, MIN_FREE_MB 3500). Finished units are skipped.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k31; mkdir -p $L
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
three() {  # $1 part, $2 dataset, $3 extra args
  local p=$1 ds=$2 x=$3 a=() i
  for ((i = 0; i < 3; i++)); do a+=("$(cmd $ds --part $p --shard $i/3 $x)"); done
  grp $L/r34_${p}_${ds}.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
}
for ds in nusc av2; do three curvdev $ds "--split dev"; done
for ds in nusc av2 ahd; do three curv $ds ""; done
for ds in nusc av2 ahd; do three b2 $ds ""; done
echo "$(date +%T) K31 R34 CHAIN LAUNCHED"
