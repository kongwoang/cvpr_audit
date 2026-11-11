#!/usr/bin/env bash
# K23 combinations (development data only): components that passed the rule alone (C1 refiner v3, C4 v_close 0 on
# nuScenes = the AV2 value), with the v3b / v3c lateral correction applied as trained (bug fix in k23_dev).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k23
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/combo_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k23_dev.py --dataset nusc --arms base,vc0 --tag combos --shard $i/4 &
  sleep 30
done
for i in 0 1 2 3 4 5; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/combo_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k23_dev.py --dataset av2 --arms base --tag combos --shard $i/6 &
  sleep 30
done
wait
echo "$(date +%T) K23 COMBOS DONE" > $L/combos_done.log
