#!/usr/bin/env bash
# Smoke: perception-in-the-loop PDM-Lite (perception_pilot.py) on the first Bench2Drive route, branches given as args.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=$B/results/b2d/smoke
for br in "$@"; do
  rm -rf $O/$br.json $O/save_$br
  PERC_BRANCH=$br MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $B/logs/A_closedloop/smoke_$br.log bash -c "export PERC_BRANCH=$br; \
    bash $B/scripts/A_closedloop/b2d_run.sh $O/route1.xml $B/scripts/A_closedloop/perception_pilot.py none $O/$br.json $O/save_$br"
done
