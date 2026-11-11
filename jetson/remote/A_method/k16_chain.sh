#!/usr/bin/env bash
# K16: attribution of ours' disagreement with pi(F) (nuScenes new test, AV2 val with the ego mask). Post-hoc.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k16; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
for i in 0 1; do u nusc_$i $B/scripts/A_method/k16_attr.py --dataset nusc --shard $i/2 & sleep 120; done
for i in 0 1; do m av2_$i $S/scripts/A_method/k16_attr.py --dataset av2 --shard $i/2 & sleep 120; done
wait
echo "$(date +%T) K16 CHAIN DONE" > $L/chain_done.log
