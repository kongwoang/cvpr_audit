#!/usr/bin/env bash
# CONFIRMATORY 3 (pre-registered, method_freeze_draft §2d): after the AV2 train download, smoke-test the script on one
# VAL log (outputs deleted), then run v3 / v4 / v4-lite once on all downloaded TRAIN logs.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
until grep -q "AV2 TRAIN DONE" logs/download_av2_train.log; do sleep 600; done
$B/run_when_gpu_free.sh logs/A_method/av2v4_smoke.log python scripts/A_method/av2_eval_v4.py --split val --max_logs 1 \
  --tag _smoke --device cuda:0
if ! grep -q "v4+ref - v3+ref" logs/A_method/av2v4_smoke.log; then echo "SMOKE FAILED" >> logs/A_method/av2_confirm3.log; exit 1; fi
rm -f results/av2/eval__val_smoke.pkl results/av2/eval_summary__val_smoke.csv
$B/run_when_gpu_free.sh logs/A_method/av2_confirm3.log python scripts/A_method/av2_eval_v4.py --split train --device cuda:0
echo "$(date +%T) AV2 CONFIRM3 DONE" >> logs/A_method/av2_confirm3.log
