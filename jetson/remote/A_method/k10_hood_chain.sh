#!/usr/bin/env bash
# K10 part 1/3: driver-in-ego-region shares, AV2 val (2 shards), Argoverse-HD val (3), nuScenes new test (2).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k10
for job in "av2 0/2" "av2 1/2" "ahd 0/3" "ahd 1/3" "ahd 2/3" "nusc 0/2" "nusc 1/2"; do
  set -- $job
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/hood_$1_${2%/*}.log python $B/scripts/A_method/k10_hood.py --dataset $1 --shard $2 &
  sleep 150
done
wait
python $B/scripts/A_method/k10_hood.py --analyze > $L/hood_summary.log 2>&1
echo "$(date +%T) K10 HOOD DONE" > $L/hood_done.log
