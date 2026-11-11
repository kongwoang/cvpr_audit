#!/usr/bin/env bash
# K30 V4 in-loop run: b2d_dispatch_v6.sh with the logging agent (perception_pilot_log.py; driving unchanged) on the rep2
# routes (219, no 11755), all six branches; a (branch, chunk) is relaunched with RESUME until its checkpoint is complete, up
# to 20 attempts. Output results/b2d/k30log/<branch>/, logs logs/k30/b2dlog/.
# v3 (2026-10-17): per-host dispatch log, run on VUW and VUW1 with disjoint BRANCHES; k30_b2d_log_join.sh writes the
# final "ALL CHUNKS COMPLETE" once both hosts are complete.
# v2 (2026-10-16): at most MAXJ (branch, chunk) JOBS at a time, counting the jobs still waiting for a GPU in
# run_when_gpu_free.sh (v1 counted only started evaluators, so queued launches all started together: 5 at once).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
O=$B/results/b2d/k30log; L=$B/logs/k30/b2dlog; mkdir -p $L $O/routes
D=$L/${DLOG:-dispatch_$(hostname -s).log}   # v3: one dispatch log per host (VUW = cuda12, VUW1 = cuda13; disjoint BRANCHES)
cp -n $B/results/b2d/rep2/routes/chunk*.xml $O/routes/
MAXJ=${MAXJ:-2}
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
n_eval() {  # distinct checkpoints of this run held by a waiting wrapper, a b2d_run.sh or an evaluator
  for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -o "k30log/[a-z0-9_]*/chunk[0-9]*\.json"; done | sort -u | wc -l; }
declare -A tries
while true; do
  left=0
  for br in ${BRANCHES:-ours_v6 cheap u960 u1600 ours privileged}; do
    for k in 0 1 2 3; do
      J=$O/$br/chunk$k.json
      [ "$(progress $J)" = done ] && continue
      left=$((left+1))
      busy "$J" && continue
      [ "${tries[$br$k]:-0}" -ge 20 ] && continue
      while [ "$(n_eval)" -ge $MAXJ ]; do sleep 120; done
      tries[$br$k]=$(( ${tries[$br$k]:-0} + 1 ))
      echo "$(date +%T) launch $br chunk$k (attempt ${tries[$br$k]})" >> $D
      SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/${br}_$k.log bash -c "export PERC_BRANCH=$br; \
        bash $B/scripts/A_closedloop/b2d_run.sh $O/routes/chunk$k.xml $B/scripts/A_closedloop/perception_pilot_log.py none $J $O/$br/save_chunk$k" &
      sleep 300
    done
  done
  [ $left -eq 0 ] && break
  sleep 300
done
wait
echo "$(date +%T) K30 B2D LOG HOST COMPLETE ($BRANCHES)" >> $D
