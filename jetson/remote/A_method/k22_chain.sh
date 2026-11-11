#!/usr/bin/env bash
# K22: Argoverse-HD crop heuristics at tensor px (K18 protocol) + group-d diagnosis (AV2 masked, Argoverse-HD). Post-hoc.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k22; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
for i in 0 1 2; do u heur_ahd_$i $B/scripts/A_method/k18_heur.py --dataset ahd --budgets 0.5,1,2 --shard $i/3 & sleep 30; done
for i in 0 1; do m diag_av2_$i $S/scripts/A_method/k22_diag.py --dataset av2 --shard $i/2 & sleep 30; done
for i in 0 1; do u diag_ahd_$i $B/scripts/A_method/k22_diag.py --dataset ahd --shard $i/2 & sleep 30; done
wait
echo "$(date +%T) K22 CHAIN DONE" > $L/chain_done.log
