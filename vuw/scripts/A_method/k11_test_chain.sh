#!/usr/bin/env bash
# K11 phase 2: test evaluation of the dev-selected latency configuration on nuScenes new test, 3 shards.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k11; mkdir -p $L
for s in 0 1 2; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/test_$s.log python $B/scripts/A_method/k11_test.py --shard $s/3 &
  sleep 150
done
wait
python $B/scripts/A_method/k11_test.py --analyze > $L/test_summary.log 2>&1
echo "$(date +%T) K11 TEST DONE" > $L/test_done.log
