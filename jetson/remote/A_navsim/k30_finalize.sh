#!/usr/bin/env bash
# K30 unattended finalisation (VUW): after the navtest re-scoring (k30_nav_v2score2.sh) -> navtest EPDMS summary;
# after the Bench2Drive logging run -> in-loop metrics (k30_b2d_inloop.py); then "K30 ALL DONE".
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
until grep -qs "K30 V2 NAVTEST2 DONE" $L/v2score_done.log; do sleep 600; done
. $B/scripts/common/env_nuplan.sh
export NAVSIM_EXP_ROOT=$B/results/navsim_exp
python $B/scripts/A_navsim/k30_nav_v2_summary.py > $L/navtest_summary.log 2>&1
until grep -qs "K30 B2D LOG ALL CHUNKS COMPLETE" $L/b2dlog/dispatch.log; do sleep 900; done
source $B/env.sh
python $B/scripts/A_closedloop/k30_b2d_inloop.py > $L/b2d_inloop.log 2>&1
echo "$(date '+%F %T') K30 ALL DONE" >> $L/final_done.log
