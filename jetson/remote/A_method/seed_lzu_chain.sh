#!/usr/bin/env bash
# Second seed of LZU-ft and its uniform control (same recipe), then the evaluation (results/lzu/summary_newtest_s1.csv).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/seeds
mkdir -p $L
for arm in lzu plainrect; do
  SEED=1 MIN_FREE_MB=14000 $B/run_when_gpu_free.sh $L/lzu_train_$arm.log python scripts/A_method/lzu_baseline.py --stage train --arm $arm
  [ -f out/lzu/${arm}_s1/weights/last.pt ] || { echo "SEED LZU ABORT $arm" >> $L/lzu_train_$arm.log; exit 1; }
done
SEED=1 $B/run_when_gpu_free.sh $L/lzu_eval.log python scripts/A_method/lzu_baseline.py --stage eval
echo "SEED LZU CHAIN DONE" >> $L/lzu_eval.log
