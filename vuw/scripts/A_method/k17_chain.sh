#!/usr/bin/env bash
# K17 (field mechanism) + K21(1) (probe templates x closing speeds): k17_variants.py arms. Starts after the K14 chain
# (GPU sharing). Post-hoc; AV2 with the ego mask (K10 shadow tree).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k17; mkdir -p $L
until [ -f $B/logs/A_method/k14/chain_done.log ]; do sleep 600; done
R="python $B/scripts/A_method/k10_masked_run.py"
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
V=$B/scripts/A_method/k17_variants.py; VS=$S/scripts/A_method/k17_variants.py
pr() { skip=$1; out=""; for vc in 0 5 10 15; do for p in ped car truck union; do [ "vc${vc}_$p" = "$skip" ] || out="$out,vc${vc}_$p"; done; done; echo ${out#,}; }
PN=$(pr vc10_car); PA=$(pr vc0_car)
# phase 1: test splits, base
for i in 0 1 2 3; do u nusc_test_$i $V --dataset nusc --split test --arms full,so,so_ctx4,so_ctx6,so_ctx8,so_ctx12,so_rand,$PN --shard $i/4 & sleep 120; done
for i in 0 1 2 3 4 5; do m av2_test_$i $VS --dataset av2 --split test --arms full,so,so_ctx2.5,so_ctx3,so_ctx3.5,so_ctx4,so_rand,$PA --shard $i/6 & sleep 120; done
for i in 0 1; do u ahd_test_$i $V --dataset ahd --arms full,so,so_ctx2.5,so_ctx3,so_ctx4,so_ctx6,so_rand,vc5_car,vc10_car --shard $i/2 & sleep 120; done
wait
echo "$(date +%T) K17 PHASE 1 DONE" > $L/phase1_done.log
# phase 2: controlled deletion (recovery), test splits
for i in 0 1; do u nusc_del_$i $V --dataset nusc --split test --setting del_all --arms full,$PN --shard $i/2 & sleep 120; done
for i in 0 1 2; do m av2_del_$i $VS --dataset av2 --split test --setting del_all --arms full,so,$PA --shard $i/3 & sleep 120; done
u ahd_del $V --dataset ahd --setting del_all --arms full,so,vc5_car,vc10_car & sleep 120
wait
echo "$(date +%T) K17 PHASE 2 DONE" > $L/phase2_done.log
# phase 3: development splits (K21(1)), base
for i in 0 1; do u nusc_val_$i $V --dataset nusc --split val --arms full,$PN --shard $i/2 & sleep 120; done
for i in 0 1 2; do m av2_dev_$i $VS --dataset av2 --split dev --arms full,$PA --shard $i/3 & sleep 120; done
wait
echo "$(date +%T) K17 CHAIN DONE" > $L/chain_done.log
