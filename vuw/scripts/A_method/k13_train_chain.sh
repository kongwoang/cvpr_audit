#!/usr/bin/env bash
# K13(g): masked AV2 TRAIN dev check (av2_eval_v4: v3 + ref v1, v4 + ref v2, v4-lite + ref v2) in 6 log shards
# (replaces the single-process train_v4 stage of k13_chain.sh), then merge + summary. Post-hoc analysis, ego mask.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k13
R="python $B/scripts/A_method/k10_masked_run.py"
SC=$S/scripts/A_method
N=6
for i in $(seq 0 $((N - 1))); do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/train_v4_s$i.log env B=$S K10_MASK_Y0=1830 $R $SC/av2_eval_v4.py --split train --tag _confirm3 --shard $i/$N &
  sleep 120
done
wait
env B=$S python $SC/av2_eval_v4.py --split train --tag _confirm3 --merge $N > $L/train_v4_summary.log 2>&1
echo "$(date +%T) K13 TRAIN DONE" > $L/train_done.log
