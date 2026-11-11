#!/usr/bin/env bash
# Argoverse-HD val: wait for the 3D labels, smoke one sequence, run all 24 in 4 shards (GPU, shared), score.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/argoverse_hd; mkdir -p $L
until grep -qs "AV1 LABELS DONE" logs/data/argoverse_hd.log; do sleep 300; done
SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/smoke.log python scripts/A_method/argoverse_hd_eval.py --stage run --shard 0/24
grep -q "RUN DONE" $L/smoke.log || { echo "$(date +%T) AHD ABORT smoke" >> $L/chain.log; exit 1; }
for i in 0 1 2 3; do
  SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/run_$i.log python scripts/A_method/argoverse_hd_eval.py --stage run --shard $i/4 &
  sleep 300
done
wait
python scripts/A_method/argoverse_hd_eval.py --stage score > $L/score.log 2>&1
echo "$(date +%T) AHD CHAIN DONE" >> $L/chain.log
