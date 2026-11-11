#!/usr/bin/env bash
# K39 VUW (handoff/K39/PREREG.md K39-2 + K39-4): experiments/k39_vuw_20261027/k39_eval.py --part k39 on the three test splits
# (AV2 with the K10 ego mask). usage: k39_vuw_chain.sh smoke | full
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
H=$B/experiments/k39_vuw_20261027/k39_eval.py
S=$B/k10_masked
L=$B/logs/k39; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
g() { local nm=$1; shift; local c="{ $1 & p0=\$!; $2 & p1=\$!; $3 & p2=\$!; wait \$p0; wait \$p1; wait \$p2; }"
      SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/vuw_group_$nm.log bash -c "$c" > /dev/null 2>&1 < /dev/null & }
if [[ "$1" == "smoke" ]]; then
  n="$T K39_SMOKE_CHECK=1 B=$B python $H --dataset nusc --part k39 --tag smoke --units scene-0110 > $L/smoke_nusc.log 2>&1"
  a="$T K39_SMOKE_CHECK=1 B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --part k39 --tag smoke --max_units 1 > $L/smoke_av2.log 2>&1"
  h="$T K39_SMOKE_CHECK=1 B=$B python $H --dataset ahd --part k39 --tag smoke --units seq00 --max_units 1 > $L/smoke_ahd.log 2>&1"
  g smoke "$n" "$a" "true"
  echo "$(date +%T) K39 VUW smoke launched"
  exit 0
fi
n() { echo "$T B=$B python $H --dataset $1 --part k39 --shard $2 > $L/vuw_$1_${2/\//of}.log 2>&1"; }
a() { echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --part k39 --shard $1 > $L/vuw_av2_${1/\//of}.log 2>&1"; }
g g1 "$(n nusc 0/3)" "$(n nusc 1/3)" "$(n nusc 2/3)"
g g2 "$(a 0/3)" "$(a 1/3)" "$(a 2/3)"
g g3 "$(n ahd 0/3)" "$(n ahd 1/3)" "$(n ahd 2/3)"
echo "$(date +%T) K39 VUW full launched"
