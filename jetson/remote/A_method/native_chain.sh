#!/usr/bin/env bash
# Native-resolution uniform baselines (native_uniform.py): smoke on one unit, then nuScenes new test and AV2 val.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/native
mkdir -p $L
for ds in nusc av2; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/${ds}_smoke.log python scripts/A_method/native_uniform.py --dataset $ds --max_units 1 --tag _smoke
  grep -q "NATIVE UNIFORM DONE" $L/${ds}_smoke.log || { echo "$(date +%T) NATIVE ABORT smoke $ds" >> $L/chain.log; continue; }
  rm -f results/native/${ds}_smoke.pkl results/native/summary_${ds}_smoke.csv
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/${ds}.log python scripts/A_method/native_uniform.py --dataset $ds &
  sleep 300
done
wait
echo "$(date +%T) NATIVE CHAIN DONE" >> $L/chain.log
