#!/usr/bin/env bash
# development selection of the crop scale on DEEP's old train+val scenes (the old test and all new scenes untouched)
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
for sc in 0.4 0.6; do
  OMP_NUM_THREADS=6 python nusc_full_eval.py --set old --split train+val --device cpu --threads 6 --scale $sc --tag _s$sc \
     --refiners $B/out/resnet18/image_refiner.pt > $B/logs/eval_dev_s$sc.log 2>&1 &
done
wait
