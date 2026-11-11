#!/usr/bin/env bash
# K28 family B: refiner data with YOLOv8m / YOLOv8l / YOLO11m (AV2 REF-TRAIN / REF-VAL with the ego mask; nuScenesAll
# TRAIN scenes), the D seeds of v3a (1, 2), then the three B refiners (v3a recipe, seed 0). Groups of processes per
# run_when_gpu_free.sh call.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_refdata.py --src av2"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-6000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
for d in yolov8m yolov8l yolo11m; do
  grp $L/bdata_av2_${d}_g0.log "$A --det $d --shard 0/4 & $A --det $d --shard 1/4 & wait"
  grp $L/bdata_av2_${d}_g1.log "$A --det $d --shard 2/4 & $A --det $d --shard 3/4 & wait"
done
grp $L/bdata_nusc.log "$T python $B/scripts/A_method/k28_refdata.py --src nusc --det yolov8m & $T python $B/scripts/A_method/k28_refdata.py --src nusc --det yolov8l & $T python $B/scripts/A_method/k28_refdata.py --src nusc --det yolo11m & wait"
MIN=12000 grp $L/train_yolov8s_seeds.log "$T python $B/scripts/A_method/k28_train.py --det yolov8s --seed 1 & $T python $B/scripts/A_method/k28_train.py --det yolov8s --seed 2 & wait"
wait
echo "$(date +%T) K28 B DATA DONE" > $L/bdata_done.log
for d in yolov8m yolov8l yolo11m; do
  MIN=9000 grp $L/train_$d.log "$T python $B/scripts/A_method/k28_train.py --det $d --seed 0"
done
wait
echo "$(date +%T) K28 B TRAIN DONE" >> $L/bdata_done.log
