#!/usr/bin/env bash
# K28 family C on DEVELOPMENT data: after the temporal head is trained, k28_tdev.py (C1 / C2 and the u960 controls).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
until grep -q "K28 C TRAIN DONE" $L/cdata_done.log 2>/dev/null; do sleep 300; done
[ -f $B/out/k28_temporal/temporal_head.pt ] || { echo "missing head"; exit 1; }
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T python $B/scripts/A_method/k28_tdev.py --dataset nusc --tag r1"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_tdev.py --dataset av2 --tag r1"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
grp $L/r1_tdev_g0.log "$N --shard 0/2 & $N --shard 1/2 & $A --shard 0/6 & wait"
grp $L/r1_tdev_g1.log "$A --shard 1/6 & $A --shard 2/6 & $A --shard 3/6 & wait"
grp $L/r1_tdev_g2.log "$A --shard 4/6 & $A --shard 5/6 & wait"
wait
echo "$(date +%T) K28 C DEV DONE" > $L/cdev_done.log
