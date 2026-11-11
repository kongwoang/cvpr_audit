#!/usr/bin/env bash
# v3 development runs (old train+val): truncation guard + scene history; probe height 1.6 vs 4.0 m; ResNet-18 refiner.
cd /vol/dl-nguyenb5-solar/users/hoangpc/CVPR && . ./env.sh
for H in 1.6 4.0; do
  python nusc_full_eval.py --set old --split train+val --device cpu --threads 6 --scale 0.6 --obj_h $H --tag _v3_h$H \
    --refiners $B/out/resnet18/image_refiner.pt > logs/eval_dev_v3_h$H.log 2>&1 &
done
wait
