#!/usr/bin/env bash
# K9 follow-up (post-hoc analysis, test splits): G-vs-pixels curve of the uncertainty control at 0.75-2x ours' detector
# pixels per frame (ours = K9's default configuration), to compare at equal pixels.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh; export B
L=$B/logs/A_method/k9_tune
M=0.75,1,1.25,1.5,2
for s in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/nusc_curve_$s.log python $B/scripts/A_method/k9_tune.py --dataset nusc --split test --configs 10:0 --unc_mults $M --shard $s/2 &
  sleep 60
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/av2_curve_$s.log python $B/scripts/A_method/k9_tune.py --dataset av2 --split test --configs 0:0 --unc_mults $M --shard $s/2 &
  sleep 60
done
wait
for ds in nusc av2; do python $B/scripts/A_method/k9_tune.py --analyze --dataset $ds --split test --unc_mults $M > $L/${ds}_curve_summary.log 2>&1; done
echo K9 CURVE DONE > $L/curve_done.log
