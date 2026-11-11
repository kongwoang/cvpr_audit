#!/usr/bin/env bash
# Usage: [MIN_FREE_MB=n] run_when_gpu_free.sh <log> <command...>. Waits for a GPU at <= 50% utilisation with
# >= MIN_FREE_MB free (default 4000; checked twice, 60 s apart, so a momentary dip does not count), then runs the command
# on it via CUDA_VISIBLE_DEVICES.  Waiting is lock-free (a job needing a big GPU does not block smaller ones); the final
# pick + launch is serialised across our jobs with a lock held until 120 s after the launch, so two of our jobs never take
# the same GPU before the first one has allocated its memory; a GPU that already runs one of our processes is never picked.
# SHARE=1 (small jobs only, user-approved 2026-10-02): ignore the utilisation limit and our one-job-per-GPU rule; only
# the free-memory check (MIN_FREE_MB) applies, so the job may share a busy GPU.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
log=$1; shift
MIN=${MIN_FREE_MB:-4000}
# GPUs already running one of OUR processes (cwd inside $B; the account is shared with other workloads) are excluded
ours() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=', ' read -r u p; do
           case "$(readlink /proc/$p/cwd 2>/dev/null)" in "$B"*) echo "$u";; esac; done | sort -u; }
pick() { busy=$([ "${SHARE:-0}" = 1 ] || ours | tr '\n' ' '); UT=$([ "${SHARE:-0}" = 1 ] && echo 100 || echo 50)
         nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
         awk -F', ' -v m=$MIN -v ut=$UT -v busy="$busy" 'index(busy, $2) == 0 && $5 <= ut && $3 - $4 >= m {print $3 - $4, $1}' |
         sort -rn | head -1 | awk '{print $2}'; }
# every GPU that passes the same checks (the second check only asks whether the picked GPU still qualifies: two GPUs
# with nearly equal free memory used to swap places between the two checks and send the job back to the queue)
eligible() { busy=$([ "${SHARE:-0}" = 1 ] || ours | tr '\n' ' '); UT=$([ "${SHARE:-0}" = 1 ] && echo 100 || echo 50)
         nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
         awk -F', ' -v m=$MIN -v ut=$UT -v busy="$busy" 'index(busy, $2) == 0 && $5 <= ut && $3 - $4 >= m {print $1}'; }
exec 9>"$B/tmp/gpu_pick.lock"
last=0
while true; do
  g=$(pick)
  if [ -n "$g" ]; then
    flock 9
    sleep 60
    eligible | grep -qx "$g" && break
    flock -u 9
  fi
  now=$(date +%s)
  [ $((now - last)) -ge 1800 ] && { echo "$(date +%T) waiting for a free GPU (>= $MIN MB)" >> "$log"; last=$now; }
  sleep 120
done
echo "$(date +%T) running on GPU $g: $*" >> "$log"
( CUDA_VISIBLE_DEVICES=$g "$@" >> "$log" 2>&1; echo "$(date +%T) exit $?" >> "$log" ) 9>&- &
pid=$!
sleep 120
flock -u 9; exec 9>&-
wait $pid
