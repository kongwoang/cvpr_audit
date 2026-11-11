#!/usr/bin/env bash
# K24 CONFIRMATORY run (method_freeze_draft §2i): v5.1-S / v5.1-L (primary), v5 (reference) and the baselines for v5.1,
# once, on nuScenes new test, AV2 val (ego mask) and Argoverse-HD val.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k24; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3 4 5 6 7; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/main_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k24_test.py --dataset nusc --shard $i/8 &
  sleep 30
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/main_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k24_test.py --dataset av2 --shard $i/8 &
  sleep 30
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/main_ahd_$i.log env B_REAL=$B python $B/scripts/A_method/k24_test.py --dataset ahd --shard $i/8 &
  sleep 30
done
wait
echo "$(date +%T) K24 MAIN DONE" > $L/main_done.log
python $B/scripts/A_method/k24_analyze.py > $L/analyze.log 2>&1
echo "$(date +%T) K24 ANALYZE DONE" >> $L/main_done.log
