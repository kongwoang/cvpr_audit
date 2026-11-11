#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
until grep -q "CHAIN DONE" logs/confirm_chain.log; do sleep 300; done
python d5_analysis.py > logs/d5_analysis.log 2>&1
echo "$(date +%T) D5 ANALYSIS DONE" >> logs/d5_analysis.log
