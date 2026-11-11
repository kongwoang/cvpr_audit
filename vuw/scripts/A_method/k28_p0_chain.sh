#!/usr/bin/env bash
# K28 phase 0 (1): pivots / single round / without guard for v5.1-S on the K24 test units (post-hoc), then the
# phase-0 tables (2) (4) (5) (1).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/p0abl_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k28_phase0_abl.py --dataset nusc --shard $i/4 &
  sleep 20
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/p0abl_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k28_phase0_abl.py --dataset av2 --shard $i/4 &
  sleep 20
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/p0abl_ahd_$i.log env B_REAL=$B python $B/scripts/A_method/k28_phase0_abl.py --dataset ahd --shard $i/4 &
  sleep 20
done
wait
echo "$(date +%T) P0 ABL DONE" > $L/p0_done.log
cd $B
for p in sota loss cost abl; do python $B/scripts/A_method/k28_phase0.py --part $p > $L/p0_$p.log 2>&1; done
echo "$(date +%T) P0 TABLES DONE" >> $L/p0_done.log
