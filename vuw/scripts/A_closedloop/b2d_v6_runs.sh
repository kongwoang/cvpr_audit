#!/usr/bin/env bash
# K28 phase 3 (post-hoc): K6 Bench2Drive with v6-S as the ours branch (perception_pilot.py PERC_BRANCH=ours_v6, md5 23539df7),
# three runs on the same route files as K6 (run 1: results/b2d/full/routes chunk0-3 — NOTE: incl. route 11755, which K6
# excluded via chunk0b; superseded by b2d_v6_runs2.sh on 2026-10-14; runs 2 / 3:
# results/b2d/rep{2,3}/routes, 219). Only the new branch runs; the other branches of K6 are unchanged. b2d_dispatch_v6.sh =
# b2d_dispatch.sh counting only this branch's evaluators (other sessions run CARLA on this host too). MAXJ 2.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/A_closedloop/carla_env.sh
for r in full rep2 rep3; do
  B2D_OUT=$B/results/b2d/$r B2D_LOG=$B/logs/A_closedloop/v6_$r BRANCHES=ours_v6 MAXJ=${MAXJ:-2} bash $B/scripts/A_closedloop/b2d_dispatch_v6.sh
done
echo "$(date +%T) B2D V6 RUNS DONE" >> $B/logs/A_closedloop/v6_rep3/dispatch.log
