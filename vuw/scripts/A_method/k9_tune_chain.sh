#!/usr/bin/env bash
# K9 dev tuning: nuScenes new val (all scenes) and AV2 dev (first 60 logs), 2 shards each, via run_when_gpu_free.sh
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh; export B
L=$B/logs/A_method/k9_tune
for ds in nusc av2; do
  mu=0; [ $ds = av2 ] && mu=60
  for s in 0 1; do
    SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/${ds}_dev_$s.log python $B/scripts/A_method/k9_tune.py --dataset $ds --split dev --shard $s/2 --max_units $mu &
    sleep 60
  done
done
wait
for ds in nusc av2; do python $B/scripts/A_method/k9_tune.py --analyze --dataset $ds --split dev > $L/${ds}_dev_summary.log 2>&1; done
echo K9 DEV DONE > $L/chain_done.log
