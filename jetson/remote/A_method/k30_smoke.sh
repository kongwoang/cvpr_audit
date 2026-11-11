#!/usr/bin/env bash
# K30 smoke: every part of k30_eval.py on the first unit of nuScenes, and part match on the first AV2 log (ego mask) and
# the first Argoverse-HD sequence (--max_units 1, tag smoke; results/k30/<part>_smoke/)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
for p in match axis idm cons; do $T B=$B python $B/scripts/A_method/k30_eval.py --dataset nusc --part $p --max_units 1 --tag smoke || echo "FAIL nusc $p"; done
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k30_eval.py --dataset av2 --part match --max_units 1 --tag smoke || echo "FAIL av2 match"
$T B=$B python $B/scripts/A_method/k30_eval.py --dataset ahd --part match --max_units 1 --tag smoke || echo "FAIL ahd match"
echo "K30 SMOKE DONE"
