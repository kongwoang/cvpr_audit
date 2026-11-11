#!/usr/bin/env bash
# AV2 val baselines (frozen COCO detector for every method): smoke on 2 logs, then all 150.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/av2_baselines.log
$B/run_when_gpu_free.sh $L python scripts/A_method/av2_baselines.py --max_logs 2 --tag _smoke
tail -3 $L | grep -q "AV2 BASELINES DONE" || { echo "AV2 BASELINES SMOKE FAILED" >> $L; exit 1; }
$B/run_when_gpu_free.sh $L python scripts/A_method/av2_baselines.py
echo "AV2 BASELINES CHAIN DONE" >> $L
