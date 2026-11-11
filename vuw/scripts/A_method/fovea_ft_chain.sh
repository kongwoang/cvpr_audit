#!/usr/bin/env bash
# FOVEA with fine-tuning + fine-tuned controls: build warped/plain sets, train YOLOv8s at 960 on each, evaluate.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
G=$B/run_when_gpu_free.sh
$G logs/A_method/fovea_ft_build.log python scripts/A_method/fovea_ft.py --stage build
for arm in warped plain; do
  $G logs/A_method/fovea_ft_train_$arm.log python scripts/A_ext/d68_train.py $arm fovea_ft 15 yolov8s 960 32 $arm
done
$G logs/A_method/fovea_ft_eval.log python scripts/A_method/fovea_ft.py --stage eval
echo "$(date +%T) FOVEA FT DONE" >> logs/A_method/fovea_ft_eval.log
