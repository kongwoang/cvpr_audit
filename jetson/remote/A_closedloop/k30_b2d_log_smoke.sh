#!/usr/bin/env bash
# K30 V4 smoke: the logging agent (perception_pilot_log.py) on the first Bench2Drive route (K6's smoke route), branches as args
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=$B/results/b2d/k30log_smoke
mkdir -p $O
for br in "$@"; do
  rm -rf $O/$br.json $O/save_$br
  SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $B/logs/k30/b2dlog_smoke_$br.log bash -c "export PERC_BRANCH=$br; \
    bash $B/scripts/A_closedloop/b2d_run.sh $B/results/b2d/smoke/route1.xml $B/scripts/A_closedloop/perception_pilot_log.py none $O/$br.json $O/save_$br"
done
