#!/usr/bin/env bash
# Baseline (5): LZU (official zoom/unzoom) around YOLOv8s -- saliency, frozen eval, fine-tune LZU + control, full eval.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/lzu
mkdir -p $L
S=scripts/A_method/lzu_baseline.py
G=$B/run_when_gpu_free.sh
ok() { tail -3 $1 | grep -q "LZU $2 DONE"; }
$G $L/sal.log python $S --stage sal; ok $L/sal.log SAL || { echo "LZU ABORT sal" >> $L/sal.log; exit 1; }
$G $L/eval_frozen.log python $S --stage eval --stretch_check --tag _frozen
for arm in lzu plainrect; do
  $G $L/train_$arm.log python $S --stage train --arm $arm
  ok $L/train_$arm.log TRAIN || { echo "LZU ABORT train $arm" >> $L/train_$arm.log; exit 1; }
done
$G $L/eval.log python $S --stage eval --stretch_check; ok $L/eval.log EVAL || { echo "LZU ABORT eval" >> $L/eval.log; exit 1; }
echo "LZU CHAIN DONE" >> $L/eval.log
