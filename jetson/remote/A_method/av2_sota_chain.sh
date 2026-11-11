#!/usr/bin/env bash
# AV2 ports of AdaScale and Budget-Adaptive Routing: labels (GPU, after the 2-log smoke) -> ORIC -> train (GPU)
# -> AdaScale test pass (GPU) -> score against our AV2 configuration (v4-lite + ref).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/sota_ext
S=scripts/A_method
until grep -qs "exit" $L/av2_labels_smoke.log; do sleep 300; done
grep -q "AV2 SOTA LABELS DONE" $L/av2_labels_smoke.log || { echo "$(date +%T) AV2 SOTA ABORT smoke" >> $L/av2_sota_chain.log; exit 1; }
rm -f results/av2_sota/frames__*.pkl results/av2_sota/boxes__*.pkl
for sp in test val train; do
  SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/av2_labels_$sp.log python $S/av2_sota_labels.py --splits $sp &
  sleep 300
done
wait
for sp in test val train; do grep -q "LABELS DONE" $L/av2_labels_$sp.log || { echo "$(date +%T) AV2 SOTA ABORT labels $sp" >> $L/av2_sota_chain.log; exit 1; }; done
python $S/av2_sota.py --stage oric > $L/av2_oric.log 2>&1
grep -q "ORIC DONE" $L/av2_oric.log || { echo "$(date +%T) AV2 SOTA ABORT oric" >> $L/av2_sota_chain.log; exit 1; }
SHARE=1 MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/av2_train.log python $S/av2_sota.py --stage train
grep -q "TRAIN DONE" $L/av2_train.log || { echo "$(date +%T) AV2 SOTA ABORT train" >> $L/av2_sota_chain.log; exit 1; }
SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/av2_ada_eval.log python $S/av2_sota.py --stage ada_eval
grep -q "ADA_EVAL DONE" $L/av2_ada_eval.log || { echo "$(date +%T) AV2 SOTA ABORT ada_eval" >> $L/av2_sota_chain.log; exit 1; }
python $S/av2_sota.py --stage score > $L/av2_score.log 2>&1
echo "$(date +%T) AV2 SOTA CHAIN DONE" >> $L/av2_sota_chain.log
