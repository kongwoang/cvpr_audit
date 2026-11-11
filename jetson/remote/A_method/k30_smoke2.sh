#!/usr/bin/env bash
# K30 smoke of k30_eval2.py: apdump and speed on the first nuScenes scene, apdump on the first AV2 log (ego mask)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
for p in apdump speed; do $T B=$B python $B/scripts/A_method/k30_eval2.py --dataset nusc --part $p --max_units 1 --tag smoke || echo "FAIL nusc $p"; done
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k30_eval2.py --dataset av2 --part apdump --max_units 1 --tag smoke || echo "FAIL av2 apdump"
echo "K30 SMOKE2 DONE"
