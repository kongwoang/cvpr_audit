#!/usr/bin/env bash
# K37-1: experiments/k37_thr_20261026/k30_thr.py --part thr. $1 = dev | test | hold; $2 = host tag (log names only).
# Groups of three processes per run_when_gpu_free.sh call (memory: 2-3 shards per call). AV2 with the K10 ego mask.
# hold = the K35-4 holdout through k24_holdout (K35_AHD_SPLIT=train), dataset ahd; only after PREREG (2nd holdout use).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
H=$B/experiments/k37_thr_20261026/k30_thr.py
S=$B/k10_masked
L=$B/logs/k37; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
n() { echo "$T B=$B python $H --dataset $1 --part thr --split $2 --shard $3 > $L/k1_${2}_$1_${3/\//of}.log 2>&1"; }
a() { echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --part thr --split $1 --shard $2 > $L/k1_${1}_av2_${2/\//of}.log 2>&1"; }
g() { local nm=$1; shift; local c="{ $1 & p0=\$!; $2 & p1=\$!; $3 & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
      SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/k1_group_$nm.log bash -c "$c" > /dev/null 2>&1 < /dev/null & }
case $1 in
  dev)  g dev1 "$(n nusc dev 0/2)" "$(n nusc dev 1/2)" "$(a dev 0/4)"
        g dev2 "$(a dev 1/4)" "$(a dev 2/4)" "$(a dev 3/4)" ;;
  test) g test1 "$(n nusc test 0/2)" "$(n nusc test 1/2)" "$(n ahd test 0/1)"
        g test2 "$(a test 0/3)" "$(a test 1/3)" "$(a test 2/3)" ;;
  hold) HH=$B/experiments/k37_thr_20261026/k30_thr_h.py
        h() { echo "$T B=$B K35_AHD_SPLIT=train python $HH --dataset ahd --part thr --split test --shard $1 > $L/k1_hold_${1/\//of}.log 2>&1"; }
        g hold "$(h 0/3)" "$(h 1/3)" "$(h 2/3)" ;;
  *) echo "usage: $0 dev|test|hold"; exit 1 ;;
esac
echo "$(date +%T) K37-1 $1 launched"
