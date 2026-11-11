#!/usr/bin/env bash
cd /vol/dl-nguyenb5-solar/users/hoangpc/CVPR && . ./env.sh
python nusc_eval_planb.py --set old --split train+val --threads 5 > logs/planb_dev.log 2>&1 &
python nusc_eval_planb.py --set old --split test --threads 3 > logs/planb_oldtest.log 2>&1 &
wait
python nusc_eval_planb.py --set new --split test --threads 8 > logs/planb_newtest.log 2>&1
