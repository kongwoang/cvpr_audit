#!/usr/bin/env bash
# CONFIRMATORY 2 (pre-registered, method_freeze_draft §2c): wait for the AV2 val download, run frozen v3 once.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
mkdir -p logs/A_method
until grep -q "AV2 VAL DONE" logs/download_av2.log; do sleep 300; done
$B/run_when_gpu_free.sh logs/A_method/av2_confirm2.log python scripts/A_method/av2_eval.py --device cuda:0
echo "$(date +%T) AV2 CONFIRM2 DONE" >> logs/A_method/av2_confirm2.log
