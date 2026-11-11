#!/usr/bin/env bash
# Single Bench2Drive dispatcher (replaces b2d_chain.sh / b2d_resume.sh): at most MAXJ evaluators at a time (CARLA servers
# take 5-9 GB each and other tasks need GPU memory too); a (branch, chunk) is (re)launched with RESUME until its
# checkpoint is complete (Bench2Drive's evaluator stops after a crashed route), up to 20 attempts.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=${B2D_OUT:-$B/results/b2d/full}; L=${B2D_LOG:-$B/logs/A_closedloop/full}; mkdir -p $L
MAXJ=${MAXJ:-4}
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
n_eval() { local n=0; for p in /proc/[0-9]*; do grep -qs "leaderboard_evaluator" $p/cmdline 2>/dev/null && n=$((n+1)); done; echo $n; }
declare -A tries
while true; do
  left=0
  for br in ${BRANCHES:-privileged cheap ours u960 u1600}; do
    for k in ${CHUNKS:-0 1 2 3}; do
      J=$O/$br/chunk$k.json
      [ "$(progress $J)" = done ] && continue
      left=$((left+1))
      busy "$J" && continue
      [ "${tries[$br$k]:-0}" -ge 20 ] && continue
      while [ "$(n_eval)" -ge $MAXJ ]; do sleep 120; done
      tries[$br$k]=$(( ${tries[$br$k]:-0} + 1 ))
      echo "$(date +%T) launch $br chunk$k (attempt ${tries[$br$k]})" >> $L/dispatch.log
      SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/${br}_$k.log bash -c "export PERC_BRANCH=$br; \
        bash $B/scripts/A_closedloop/b2d_run.sh $O/routes/chunk$k.xml $B/scripts/A_closedloop/perception_pilot.py none $J $O/$br/save_chunk$k" &
      sleep 300
    done
  done
  [ $left -eq 0 ] && break
  sleep 300
done
wait
echo "$(date +%T) B2D ALL CHUNKS COMPLETE" >> $L/dispatch.log
