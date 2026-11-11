#!/usr/bin/env bash
# K24 SUPPLEMENTARY pass (method_freeze_draft §2i, revised spec): nuScenes v3 + components, support only / field only
# + components, once, on the same test units; then the K24 tables (after the main pass has finished).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k24; mkdir -p $L
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3 4 5 6 7; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/supp_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k24_supp.py --dataset nusc --shard $i/8 &
  sleep 30
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/supp_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k24_supp.py --dataset av2 --shard $i/8 &
  sleep 30
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/supp_ahd_$i.log env B_REAL=$B python $B/scripts/A_method/k24_supp.py --dataset ahd --shard $i/8 &
  sleep 30
done
wait
echo "$(date +%T) K24 SUPP DONE" > $L/supp_done.log
until grep -q "MAIN DONE" $L/main_done.log 2>/dev/null; do sleep 300; done
python $B/scripts/A_method/k24_analyze.py > $L/analyze_v51.log 2>&1
echo "$(date +%T) K24 V51 ANALYZE DONE" >> $L/supp_done.log
