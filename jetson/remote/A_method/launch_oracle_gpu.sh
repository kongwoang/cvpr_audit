#!/usr/bin/env bash
# DEEP's nuScenes oracle braking cell on the new test scenes, on a free GPU.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B
exec $B/run_when_gpu_free.sh logs/A_method/oracle_newtest_gpu.log python scripts/A_method/nusc_ablate.py --set new --split test \
  --threads 4 --device cuda:0 --scale 0.6 --geometry oracle --refiners $B/out/resnet18/image_refiner.pt --tag _oracle
