#!/usr/bin/env bash
# K28 round 1 (method_freeze_draft §2j), DEVELOPMENT data only: family A (+ v5.1-S reference) and family E grids.
# Each run_when_gpu_free.sh call starts a group of 3 shard processes on the picked GPU (fewer serialised launches).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T python $B/scripts/A_method/k28_dev.py --dataset nusc --tag r1"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_dev.py --dataset av2 --tag r1"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
for p in ours base; do
  for g in 0 1; do grp $L/r1_nusc_${p}_g$g.log "$(for i in 0 1 2; do echo -n "$N --part $p --shard $((3*g+i))/6 & "; done) wait"; done
  for g in 0 1 2 3; do grp $L/r1_av2_${p}_g$g.log "$(for i in 0 1 2; do echo -n "$A --part $p --shard $((3*g+i))/12 & "; done) wait"; done
done
wait
echo "$(date +%T) K28 R1 DEV DONE" > $L/r1_done.log
