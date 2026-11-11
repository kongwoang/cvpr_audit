#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
setsid nohup python train_refiner_remote.py --arch small --device cpu --threads 8 --tag _small_cpu > $B/logs/train_refiner_small_cpu.log 2>&1 < /dev/null &
setsid nohup ./run_when_gpu_free.sh $B/logs/train_refiner_resnet.log python train_refiner_remote.py --arch resnet18 --tag _resnet18 > /dev/null 2>&1 < /dev/null &
