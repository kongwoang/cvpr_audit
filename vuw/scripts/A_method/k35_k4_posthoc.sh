#!/usr/bin/env bash
# K35-4 (v7a passed the holdout kill rule): v7a once, POST-HOC, on the three test splits through
# experiments/k35_holdout_20261025/r3/runner_t.py (build_posthoc.py). Two run_when_gpu_free.sh groups of three:
# nuScenes 0/2, 1/2 + Argoverse-HD; AV2 (K10 ego mask) 0/3, 1/3, 2/3.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
H=$B/experiments/k35_holdout_20261025/r3/runner_t.py
S=$B/k10_masked
L=$B/logs/k35/k4; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
c1="{ $T B=$B python $H --dataset nusc --shard 0/2 > $L/posthoc_nusc_0of2.log 2>&1 & p0=\$!; \
$T B=$B python $H --dataset nusc --shard 1/2 > $L/posthoc_nusc_1of2.log 2>&1 & p1=\$!; \
$T B=$B python $H --dataset ahd > $L/posthoc_ahd.log 2>&1 & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
c2="{ $T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --shard 0/3 > $L/posthoc_av2_0of3.log 2>&1 & p0=\$!; \
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --shard 1/3 > $L/posthoc_av2_1of3.log 2>&1 & p1=\$!; \
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $H --dataset av2 --shard 2/3 > $L/posthoc_av2_2of3.log 2>&1 & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/posthoc_group1.log bash -c "$c1" > /dev/null 2>&1 < /dev/null &
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/posthoc_group2.log bash -c "$c2" > /dev/null 2>&1 < /dev/null &
echo "$(date +%T) K35-4 POSTHOC launched"
