#!/usr/bin/env bash
# K28 phase 3 (post-hoc): continuation of b2d_v6_runs.sh with K6's route protocol. Route 11755 hangs the simulator for every
# branch except cheap (K6 run 1 stopped chunk0 after 4937 and ran the remaining 26 routes as chunk0b; runs 2 / 3 have no
# 11755). v6 run 1 reached 11755 in chunk0 (28 routes done, through 4937) and hung there, so run 1 continues with chunks
# 0b 1 2 3 (chunk0.json stays at 28 / 55, as for the K6 branches); runs 2 / 3 unchanged. Already-running chunk evaluators
# are detected by the dispatcher (checkpoint path in their cmdline) and not relaunched.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/A_closedloop/carla_env.sh
B2D_OUT=$B/results/b2d/full B2D_LOG=$B/logs/A_closedloop/v6_full BRANCHES=ours_v6 CHUNKS="0b 1 2 3" MAXJ=${MAXJ:-2} \
  bash $B/scripts/A_closedloop/b2d_dispatch_v6.sh
for r in rep2 rep3; do
  B2D_OUT=$B/results/b2d/$r B2D_LOG=$B/logs/A_closedloop/v6_$r BRANCHES=ours_v6 MAXJ=${MAXJ:-2} bash $B/scripts/A_closedloop/b2d_dispatch_v6.sh
done
echo "$(date +%T) B2D V6 RUNS DONE" >> $B/logs/A_closedloop/v6_rep3/dispatch.log
