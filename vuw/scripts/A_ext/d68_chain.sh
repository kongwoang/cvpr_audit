#!/usr/bin/env bash
# D68 chain: build the three arms, then train each when a GPU is free.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
python d68_build.py > logs/d68_build.log 2>&1 || exit 1
for arm in plain rand dsf; do
  $B/run_when_gpu_free.sh $B/logs/d68_train_$arm.log python d68_train.py $arm
done
echo "$(date +%T) D68 TRAIN DONE" >> logs/d68_build.log
