#!/usr/bin/env bash
# Second seed of the fine-tuned detectors (plain = u960-ft / v3 verifier, warped = FOVEA-ft), same recipe, then the
# fine-tuned comparison again (results/fovea/summary_ft_newtest_s1.csv).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/ft_seed1
mkdir -p $L
for arm in plain warped; do
  SEED=1 MIN_FREE_MB=14000 $B/run_when_gpu_free.sh $L/train_$arm.log python scripts/A_ext/d68_train.py $arm fovea_ft 15 yolov8s 960 32 ${arm}_s1
  [ -f out/fovea_ft/${arm}_s1/weights/best.pt ] || { echo "FT SEED ABORT $arm" >> $L/train_$arm.log; exit 1; }
done
FT_SUFFIX=_s1 $B/run_when_gpu_free.sh $L/eval.log python scripts/A_method/fovea_ft.py --stage eval
echo "FT SEED CHAIN DONE" >> $L/eval.log
