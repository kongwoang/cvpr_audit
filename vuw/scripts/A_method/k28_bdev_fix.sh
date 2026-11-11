#!/usr/bin/env bash
# K28 family B: re-run of the YOLOv8l g0 group (nuScenes shards 0/2, 1/2 and AV2 shard 0/6), which failed at start-up with
# CUDA OOM (the GPU was full when the wrapper launched it). Finished units are skipped (k28_dev.py resumes per unit).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
d=yolov8l
N="$T python $B/scripts/A_method/k28_dev.py --dataset nusc --tag r1 --part ver:$d"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_dev.py --dataset av2 --tag r1 --part ver:$d"
SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/r1_ver_${d}_g0_rerun.log bash -c "$N --shard 0/2 & $N --shard 1/2 & $A --shard 0/6 & wait"
echo "$(date +%T) K28 B DEV V8L RERUN DONE" > $L/bdev_fix_done.log
