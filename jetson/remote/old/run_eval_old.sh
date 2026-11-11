#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
OMP_NUM_THREADS=8 python nusc_full_eval.py --set old --split test --device cpu --threads 8 --refiners $B/out/resnet18/image_refiner.pt > $B/logs/eval_old_test_ref.log 2>&1
