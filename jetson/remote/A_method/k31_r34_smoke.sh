#!/usr/bin/env bash
# K31 items 3 / 4 smoke: parts curvdev (dev split), curv and b2 (test) on the first unit (tag smoke).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
$T B=$B python $B/scripts/A_method/k31_eval.py --dataset nusc --part curvdev --split dev --max_units 1 --tag smoke || echo "FAIL nusc curvdev"
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k31_eval.py --dataset av2 --part curvdev --split dev --max_units 1 --tag smoke || echo "FAIL av2 curvdev"
for p in curv b2; do
  $T B=$B python $B/scripts/A_method/k31_eval.py --dataset nusc --part $p --max_units 1 --tag smoke || echo "FAIL nusc $p"
  $T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k31_eval.py --dataset av2 --part $p --max_units 1 --tag smoke || echo "FAIL av2 $p"
done
echo "K31 R34 SMOKE DONE"
