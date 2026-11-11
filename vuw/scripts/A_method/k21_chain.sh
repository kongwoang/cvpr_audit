#!/usr/bin/env bash
# K21(2) second detector family (YOLO11s) + K21(3) ego-mask sensitivity. Starts after the K16 chain. Post-hoc.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k21; mkdir -p $L
until [ -f $B/logs/A_method/k16/chain_done.log ]; do sleep 600; done
R="python $B/scripts/A_method/k10_masked_run.py"
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
for i in 0 1; do u det2_nusc_$i $B/scripts/A_method/k21_det2.py --dataset nusc --shard $i/2 & sleep 120; done
for i in 0 1; do m det2_av2_$i $S/scripts/A_method/k21_det2.py --dataset av2 --shard $i/2 & sleep 120; done
for i in 0 1 2; do u mask_$i $B/scripts/A_method/k21_mask.py --shard $i/3 & sleep 120; done
wait
echo "$(date +%T) K21 CHAIN DONE" > $L/chain_done.log
