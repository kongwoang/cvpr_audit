#!/usr/bin/env bash
# K12(a): AV2 locality decomposition, 50 val logs, 2 shards.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k12; mkdir -p $L
for s in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/loc_$s.log python $B/scripts/A_method/k12_locality.py --shard $s/2 &
  sleep 150
done
wait
python $B/scripts/A_method/k12_locality.py --analyze > $L/loc_summary.log 2>&1
echo "$(date +%T) K12 LOC DONE" > $L/loc_done.log
