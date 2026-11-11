#!/usr/bin/env bash
# K27 (development data only, method_freeze_draft §2g): temporal-cue arms on nuScenes new val and AV2 CFG-DEV (ego mask).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k27; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3 4 5; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/main_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k27_dev.py --dataset nusc --shard $i/6 &
  sleep 30
done
for i in 0 1 2 3 4 5 6 7 8 9; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/main_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k27_dev.py --dataset av2 --shard $i/10 &
  sleep 30
done
wait
echo "$(date +%T) K27 MAIN DONE" > $L/main_done.log
python $B/scripts/A_method/k27_analyze.py > $L/analyze.log 2>&1
