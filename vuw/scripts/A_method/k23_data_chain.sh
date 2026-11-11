#!/usr/bin/env bash
# K23 step 1: AV2 development split + refiner-v3 training data on REF-TRAIN / REF-VAL (ego mask). Development data only.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k23; mkdir -p $L
B=$B python $B/scripts/A_method/k23_refdata.py --split > $L/split.log 2>&1
R="python $B/scripts/A_method/k10_masked_run.py"
for i in 0 1 2 3; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/refdata_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k23_refdata.py --shard $i/4 &
  sleep 30
done
wait
echo "$(date +%T) K23 DATA DONE" > $L/data_done.log
