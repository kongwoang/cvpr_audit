#!/usr/bin/env bash
# K31 R2 smoke: parts ttcpx and idmS of k31_eval.py on the first unit of each test split (--max_units 1, tag smoke;
# results/k31/r2/<ds>_<part>_smoke/). AV2 through k10_masked_run.py (ego mask).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
for p in ttcpx idmS; do
  $T B=$B python $B/scripts/A_method/k31_eval.py --dataset nusc --part $p --max_units 1 --tag smoke || echo "FAIL nusc $p"
  $T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k31_eval.py --dataset av2 --part $p --max_units 1 --tag smoke || echo "FAIL av2 $p"
  $T B=$B python $B/scripts/A_method/k31_eval.py --dataset ahd --part $p --max_units 1 --tag smoke || echo "FAIL ahd $p"
done
echo "K31 R2 SMOKE DONE"
