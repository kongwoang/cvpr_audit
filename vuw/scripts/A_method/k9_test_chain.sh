#!/usr/bin/env bash
# K9 follow-up, test splits once (post-hoc analysis): configurations selected on dev (k9_tune --split dev):
# nuScenes vc 0 / m 0.25 (+ K9's default vc 10 / m 0 and both controls), AV2 default vc 0 / m 0 (+ px-matched control).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh; export B
L=$B/logs/A_method/k9_tune
for s in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/nusc_test_$s.log python $B/scripts/A_method/k9_tune.py --dataset nusc --split test --configs 0:0.25 --shard $s/2 &
  sleep 60
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/av2_test_$s.log python $B/scripts/A_method/k9_tune.py --dataset av2 --split test --configs 0:0 --shard $s/2 &
  sleep 60
done
wait
for ds in nusc av2; do python $B/scripts/A_method/k9_tune.py --analyze --dataset $ds --split test > $L/${ds}_test_summary.log 2>&1; done
echo K9 TEST DONE > $L/test_done.log
