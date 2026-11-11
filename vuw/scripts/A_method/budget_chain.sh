#!/usr/bin/env bash
# Budget curves on the NEW TEST scenes (and the dev set for the strict x1 point):
#  (1) crop baselines never exceeding the per-frame budget (--strict) at 0.5 / 1 / 2 x v3's pixels;
#  (2) v3's own operating points (rounds, crop scale, probe closing speed), both refiners.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/budget
mkdir -p $L
G=$B/run_when_gpu_free.sh
S=scripts/A_method/baselines_eq.py
$G $L/strict_x1.log python $S --threads 4 --strict --arms random uncertainty temporal_crit traj_support v3 --tag _strict
$G $L/strict_x1_dev.log python $S --set old --split train+val --ref eval__old_train+val_v3_h1.6.pkl --threads 4 --strict \
  --arms temporal_crit uncertainty v3 --tag _strict
for k in 0.5 2; do
  $G $L/strict_x$k.log python $S --threads 4 --strict --budget_scale $k --arms random uncertainty temporal_crit v3 --tag _strict_b$k
done
REF="$B/out/resnet18/image_refiner.pt $B/out/resnet18_v2/image_refiner.pt"
for cfg in "10 1 0.6" "10 3 0.4" "10 3 0.8" "0 3 0.6" "0 3 0.8" "0 1 0.4"; do
  set -- $cfg
  $G $L/v3_vc$1_r$2_s$3.log python scripts/A_method/nusc_ablate.py --set new --split test --device cuda:0 --threads 4 \
    --v_close $1 --max_rounds $2 --scale $3 --tag _curve_vc$1_r$2_s$3 --refiners $REF
done
echo "$(date +%T) BUDGET CHAIN DONE" >> $L/done.log
