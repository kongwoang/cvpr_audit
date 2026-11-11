#!/usr/bin/env bash
# K28 family C: 3-crop temporal chains on AV2 REF-TRAIN / REF-VAL (ego mask) and nuScenes new train, then the head.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_tdata.py --src av2"
N="$T python $B/scripts/A_method/k28_tdata.py --src nusc"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-6000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
grp $L/cdata_av2_g0.log "$A --shard 0/6 & $A --shard 1/6 & $A --shard 2/6 & wait"
grp $L/cdata_av2_g1.log "$A --shard 3/6 & $A --shard 4/6 & $A --shard 5/6 & wait"
grp $L/cdata_nusc_g0.log "$N --shard 0/6 & $N --shard 1/6 & $N --shard 2/6 & wait"
grp $L/cdata_nusc_g1.log "$N --shard 3/6 & $N --shard 4/6 & $N --shard 5/6 & wait"
wait
echo "$(date +%T) K28 C DATA DONE" > $L/cdata_done.log
MIN=12000 grp $L/ttrain.log "$T python $B/scripts/A_method/k28_ttrain.py"
wait
echo "$(date +%T) K28 C TRAIN DONE" >> $L/cdata_done.log
