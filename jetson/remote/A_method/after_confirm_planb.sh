#!/usr/bin/env bash
cd /vol/dl-nguyenb5-solar/users/hoangpc/CVPR && . ./env.sh
until grep -q "confirmatory eval exit" logs/confirm_chain.log; do sleep 300; done
python nusc_eval_planb.py --set new --split test --threads 12 > logs/planb_newtest.log 2>&1
