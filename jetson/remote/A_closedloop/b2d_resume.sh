#!/usr/bin/env bash
# Bench2Drive's evaluator stops after a crashed route ("please restart"); its official multi-GPU scripts re-launch until
# the checkpoint is complete. This loop does the same for every (branch, chunk): wait while a process still writes the
# chunk's checkpoint, then re-launch with RESUME until progress == total (≤ 20 attempts per chunk).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=$B/results/b2d/full; L=$B/logs/A_closedloop/full
. $B/scripts/A_closedloop/carla_env.sh
progress() { python - "$1" <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))["_checkpoint"]["progress"]
    print("done" if p and p[0] >= p[1] else "todo")
except Exception:
    print("todo")
PY
}
busy() { for p in /proc/[0-9]*; do grep -qs "$1" $p/cmdline 2>/dev/null && return 0; done; return 1; }
run_chunk() {
  local br=$1 k=$2 J=$O/$br/chunk$k.json
  for a in $(seq 1 20); do
    while busy "$J"; do sleep 300; done
    [ "$(progress $J)" = done ] && { echo "$(date +%T) $br chunk$k complete" >> $L/resume.log; return; }
    echo "$(date +%T) $br chunk$k attempt $a" >> $L/resume.log
    SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/${br}_$k.log bash -c "export PERC_BRANCH=$br; \
      bash $B/scripts/A_closedloop/b2d_run.sh $O/routes/chunk$k.xml $B/scripts/A_closedloop/perception_pilot.py none $J $O/$br/save_chunk$k"
  done
}
for br in ${BRANCHES:-privileged cheap ours u1600 u960}; do
  for k in 0 1 2 3; do run_chunk $br $k & sleep 60; done
done
wait
echo "$(date +%T) B2D RESUME DONE" >> $L/resume.log
