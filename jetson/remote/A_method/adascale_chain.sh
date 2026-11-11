#!/usr/bin/env bash
# AdaScale baseline: labels (5 scales on new train/val) -> scale classifier -> new-test evaluation.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
mkdir -p logs/A_method
for st in data train eval; do
  $B/run_when_gpu_free.sh logs/A_method/adascale_$st.log python scripts/A_method/adascale.py --stage $st --device cuda:0
done
echo "$(date +%T) ADASCALE DONE" >> logs/A_method/adascale_eval.log
