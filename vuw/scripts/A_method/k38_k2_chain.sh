#!/usr/bin/env bash
# K38-2: experiments/k38_probe_20261026/k30_probe.py --part probe on the three test splits (AV2 with the K10 ego mask).
# Two run_when_gpu_free.sh groups of three processes.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
H=$B/experiments/k38_probe_20261026/k30_probe.py
S=$B/k10_masked
L=$B/logs/k38; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
n() { echo "$T B=$B python $H --dataset $1 --part probe --shard $2 > $L/k2_$1_${2/\//of}.log 2>&1"; }
a() { echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --part probe --shard $1 > $L/k2_av2_${1/\//of}.log 2>&1"; }
g() { local nm=$1; shift; local c="{ $1 & p0=\$!; $2 & p1=\$!; $3 & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
      SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/k2_group_$nm.log bash -c "$c" > /dev/null 2>&1 < /dev/null & }
g g1 "$(n nusc 0/2)" "$(n nusc 1/2)" "$(n ahd 0/1)"
g g2 "$(a 0/3)" "$(a 1/3)" "$(a 2/3)"
echo "$(date +%T) K38-2 launched"
