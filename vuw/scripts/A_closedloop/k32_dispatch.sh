#!/usr/bin/env bash
# K32 dispatcher: usage k32_dispatch.sh <jobs file> [<other host's jobs file>]; each line "name|routes.xml|VAR=val VAR=val ...".
# A job's checkpoint is results/k32/<name>.json (save dir results/k32/<name>_save); it is (re)launched through
# run_when_gpu_free.sh until its checkpoint is complete (up to 20 attempts), at most MAXJ jobs at a time (waiting ones
# included). Log logs/k32/dispatch_<file>.log
# Work sharing between the hosts (2026-10-20): every launch first claims the job (atomic mkdir under results/k32/claims,
# owner = this host); a job claimed by the other host is skipped. Once every job of its own file is launched, done or
# claimed elsewhere, the dispatcher takes unstarted, unclaimed jobs of the other host's file (no checkpoint yet).
# v3 (2026-10-21): one launch per scan, so the files are re-read after every launch, and each scan first resumes this
# host's started-but-incomplete jobs (an evaluator that exited after a simulator crash), then starts new own jobs, then
# new jobs of the other file.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
JF=$1; OF=${2:-}; L=$B/logs/k32; mkdir -p $L
D=$L/dispatch_$(basename $JF .txt).log
MAXJ=${MAXJ:-2}
CL=$B/results/k32/claims; mkdir -p $CL
HN=$(hostname -s)
progress() { python3 - "$1" <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))["_checkpoint"]["progress"]
    print("done" if p and p[0] >= p[1] else "todo")
except Exception:
    print("todo")
PY
}
busy() { for p in /proc/[0-9]*; do grep -qs "$1" $p/cmdline 2>/dev/null && return 0; done; return 1; }
n_jobs() { for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -o "results/k32/[A-Za-z0-9_/]*\.json"; done | sort -u | wc -l; }
cdir() { echo "$CL/$(echo "$1" | tr '/' '_')"; }
owner() { cat "$(cdir "$1")/host" 2>/dev/null; }
claim() { local c; c=$(cdir "$1"); if mkdir "$c" 2>/dev/null; then echo "$HN" > "$c/host"; return 0; fi; sleep 1; [ "$(owner "$1")" = "$HN" ]; }
declare -A tries
launch() {   # name routes vars
  local name=$1 routes=$2 vars=$3 J=$B/results/k32/$1.json
  tries[$name]=$(( ${tries[$name]:-0} + 1 ))
  echo "$(date +%T) launch $name (attempt ${tries[$name]})" >> $D
  mkdir -p $(dirname $L/$name.log)
  SHARE=1 MIN_FREE_MB=${MINF:-9000} setsid nohup $B/run_when_gpu_free.sh $L/$name.log bash -c "export $vars; bash $B/scripts/A_closedloop/k32_run.sh $routes $J $B/results/k32/${name}_save" > /dev/null 2>&1 < /dev/null &
  sleep 200
}
# first launchable line of a file for a mode: resume (checkpoint exists, owner = this host or own file without owner) |
# new (no checkpoint; own file: not claimed by the other host; other file: unclaimed). Prints the line. Also counts the
# file's unfinished jobs that this host is responsible for (own file: all not claimed elsewhere; other: claimed by it).
scan() {   # file kind(own|other) mode(resume|new)
  local f=$1 kind=$2 mode=$3 name routes vars J o
  while IFS='|' read -r name routes vars; do
    [ -z "$name" ] && continue
    J=$B/results/k32/$name.json
    o=$(owner "$name")
    [ -n "$o" ] && [ "$o" != "$HN" ] && continue
    if [ "$mode" = resume ]; then
      [ -f "$J" ] || continue
      [ "$kind" = other ] && [ -z "$o" ] && continue
    else
      [ -f "$J" ] && continue
    fi
    [ "$(progress $J)" = done ] && continue
    busy "$J" && continue
    [ "${tries[$name]:-0}" -ge 20 ] && continue
    echo "$name|$routes|$vars"; return 0
  done < "$f"
  return 1
}
remaining() {   # unfinished jobs this host may still run or wait for
  local n=0 name routes vars J o
  while IFS='|' read -r name routes vars; do
    [ -z "$name" ] && continue
    J=$B/results/k32/$name.json; [ "$(progress $J)" = done ] && continue
    n=$((n+1))
  done < $JF
  if [ -n "$OF" ]; then
    while IFS='|' read -r name routes vars; do
      [ -z "$name" ] && continue
      J=$B/results/k32/$name.json; o=$(owner "$name")
      [ -n "$o" ] && [ "$o" != "$HN" ] && continue
      [ -z "$o" ] && [ -f "$J" ] && continue
      [ "$(progress $J)" = done ] && continue
      n=$((n+1))
    done < $OF
  fi
  echo $n
}
# jobs of the own file that already run here or have a checkpoint belong to this host
while IFS='|' read -r name routes vars; do
  [ -z "$name" ] && continue
  { [ -f $B/results/k32/$name.json ] || busy "$B/results/k32/$name.json"; } && [ -z "$(owner "$name")" ] && claim "$name"
done < $JF
echo "$(date +%T) dispatcher v3 on $HN, MAXJ $MAXJ, MINF ${MINF:-9000}, own $(basename $JF)${OF:+, shares $(basename $OF)}" >> $D
while true; do
  while [ "$(n_jobs)" -ge $MAXJ ]; do sleep 60; done
  pick=""
  for spec in "$JF own resume" "${OF:+$OF other resume}" "$JF own new" "${OF:+$OF other new}"; do
    [ -z "$spec" ] && continue
    set -- $spec
    pick=$(scan "$1" "$2" "$3") && break
  done
  if [ -n "$pick" ]; then
    IFS='|' read -r name routes vars <<< "$pick"
    claim "$name" || continue
    launch "$name" "$routes" "$vars"
    continue
  fi
  [ "$(remaining)" -eq 0 ] && break
  sleep 300
done
echo "$(date +%T) K32 DISPATCH COMPLETE $(basename $JF)" >> $D
