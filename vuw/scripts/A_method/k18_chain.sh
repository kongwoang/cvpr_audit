#!/usr/bin/env bash
# K18(A) crop heuristics at 0.5-4x ours' per-frame tensor px. Starts after the K21 chain (GPU sharing). Post-hoc.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k18; mkdir -p $L
until [ -f $B/logs/A_method/k21/chain_done.log ]; do sleep 600; done
R="python $B/scripts/A_method/k10_masked_run.py"
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
for i in 0 1 2; do u heur_nusc_$i $B/scripts/A_method/k18_heur.py --dataset nusc --shard $i/3 & sleep 120; done
for i in 0 1 2 3; do m heur_av2_$i $S/scripts/A_method/k18_heur.py --dataset av2 --shard $i/4 & sleep 120; done
wait
echo "$(date +%T) K18 CHAIN DONE" > $L/chain_done.log
