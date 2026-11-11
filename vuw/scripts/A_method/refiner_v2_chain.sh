#!/usr/bin/env bash
# Refiner v2 (all nuScenes train/val scenes) and its comparison with the frozen refiner on the NEW VAL scenes only.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
python refiner_data_v2.py > logs/refiner_data_v2.log 2>&1 || exit 1
$B/run_when_gpu_free.sh logs/refiner_v2_train.log python train_refiner_remote.py --arch resnet18 --datasets KITTI nuScenesAll --tag _v2
python nusc_ablate.py --set new --split val --device cpu --threads 10 --scale 0.6 --tag _refv2_val \
  --refiners $B/out/resnet18/image_refiner.pt $B/out/resnet18_v2/image_refiner.pt > logs/eval_refv2_val.log 2>&1
echo "$(date +%T) REFV2 DONE" >> logs/eval_refv2_val.log
