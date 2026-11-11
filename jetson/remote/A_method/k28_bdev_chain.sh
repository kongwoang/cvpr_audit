#!/usr/bin/env bash
# K28 family B on DEVELOPMENT data: after the three refiners are trained, the v5.1-S loop with YOLOv8m / YOLOv8l / YOLO11m
# on the crops (own refiner) and whole frame with the same detector (k28_dev.py --part ver:<det>).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
until grep -q "K28 B TRAIN DONE" $L/bdata_done.log 2>/dev/null; do sleep 300; done
for d in yolov8m yolov8l yolo11m; do
  [ -f $B/out/resnet18_v3a_$d/image_refiner.pt ] || { echo "missing refiner $d"; exit 1; }
done
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-7000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
for d in yolov8m yolov8l yolo11m; do
  N="$T python $B/scripts/A_method/k28_dev.py --dataset nusc --tag r1 --part ver:$d"
  A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_dev.py --dataset av2 --tag r1 --part ver:$d"
  grp $L/r1_ver_${d}_g0.log "$N --shard 0/2 & $N --shard 1/2 & $A --shard 0/6 & wait"
  grp $L/r1_ver_${d}_g1.log "$A --shard 1/6 & $A --shard 2/6 & $A --shard 3/6 & wait"
  grp $L/r1_ver_${d}_g2.log "$A --shard 4/6 & $A --shard 5/6 & wait"
done
wait
echo "$(date +%T) K28 B DEV DONE" > $L/bdev_done.log
