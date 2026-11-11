#!/usr/bin/env bash
# K32 unattended finisher, run on the kit from the user's crontab every 30 min (user, 2026-10-21: "sắp hết quota tuần, khả
# năng bạn thiết lập để đưa kết quả về kit sau khi chạy xong nhé, tự theo dõi k32 nốt"). Each run:
#   1. keeps one k32_dispatch.sh (v3) alive per host while it has work (VUW: jobs_test_vuw.txt, MAXJ 11; VUW1:
#      jobs_test_vuw1.txt, MAXJ 10; MINF 8000; each shares the other's file). A dispatcher whose log ends with
#      "K32 DISPATCH COMPLETE" is not restarted.
#   2. every 6 h: partial tables on VUW (k32_test.py --tag partial, k32_shadow.py, k32_autoreport.py --tag partial)
#   3. when all 376 registered jobs are complete, or no K32 job process has run on either host for 4 consecutive runs
#      (2 h; a stall, e.g. a job past 20 attempts): the final tables (--tag test) and the marker results/k32/ANALYSIS_DONE
#   4. pulls results/k32/tables into docs/paper/handoff/K32/auto/ (+ status.txt)
# After the final pull it writes auto/FINAL_PULLED and every later run is a no-op. Log: logs/k32_auto.log.
set -u
R=/home/kongwoang/research/CVPR
A=$R/docs/paper/handoff/K32/auto
ST=$R/logs/k32_auto.state
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
mkdir -p $A $R/logs
exec 9>$R/logs/k32_auto.lock
flock -n 9 || exit 0
log() { echo "$(date '+%F %T') $*" >> $R/logs/k32_auto.log; }
[ -f $A/FINAL_PULLED ] && exit 0
touch $ST
idle=$(grep -oP '^idle=\K\d+' $ST || echo 0); last_partial=$(grep -oP '^last_partial=\K\d+' $ST || echo 0)
now=$(date +%s)

# --- status of both hosts
status() {   # host -> "dispatcher_alive complete_logged n_job_procs"
  $SSH $1 bash -s -- $2 <<'EOF'
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR; JF=$1
a=$(ps -eo args= | grep -c "[k]32_dispatch.sh $B/results/k32/$JF.txt")
c=$(tail -1 $B/logs/k32/dispatch_$JF.log 2>/dev/null | grep -c "K32 DISPATCH COMPLETE")
n=$(ps -eo args= | grep -c "[k]32_run.sh\|[r]un_when_gpu_free.sh .*logs/k32/\(test\|shadow\)")
echo "$a $c $n"
EOF
}
todo=$($SSH VUW bash -s <<'EOF'
python3 - <<'PY'
import json
K = "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/k32"
n = 0
for f in ("jobs_test_vuw.txt", "jobs_test_vuw1.txt"):
    for ln in open(f"{K}/{f}"):
        if not ln.strip():
            continue
        try:
            p = json.load(open(f"{K}/{ln.split('|')[0]}.json"))["_checkpoint"]["progress"]
            n += not (p and p[0] >= p[1])
        except Exception:
            n += 1
print(n)
PY
EOF
)
[ -z "$todo" ] && { log "VUW unreachable"; exit 0; }
read -r a0 c0 n0 <<< "$(status VUW jobs_test_vuw)"
read -r a1 c1 n1 <<< "$(status VUW1 jobs_test_vuw1)"
log "todo $todo | VUW dispatcher $a0 complete $c0 procs $n0 | VUW1 dispatcher $a1 complete ${c1:-?} procs ${n1:-?}"

# --- 1. dispatchers
relaunch() {   # host own other maxj
  $SSH $1 bash -s -- $2 $3 $4 <<'EOF'
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR; JF=$1; OF=$2; M=$3
echo "$(date +%T) dispatcher v3 relaunched by the kit's k32_auto.sh (none was running)" >> $B/logs/k32/dispatch_$JF.log
MAXJ=$M MINF=8000 setsid nohup bash $B/scripts/A_closedloop/k32_dispatch.sh $B/results/k32/$JF.txt $B/results/k32/$OF.txt > /dev/null 2>&1 < /dev/null &
EOF
}
if [ "$todo" -gt 0 ]; then
  [ "${a0:-1}" -eq 0 ] && [ "${c0:-0}" -eq 0 ] && { relaunch VUW jobs_test_vuw jobs_test_vuw1 11; log "relaunched VUW dispatcher"; }
  [ "${a1:-1}" -eq 0 ] && [ "${c1:-0}" -eq 0 ] && { relaunch VUW1 jobs_test_vuw1 jobs_test_vuw 10; log "relaunched VUW1 dispatcher"; }
fi

# --- 2. / 3. analysis
analysis() {   # tag
  $SSH VUW "bash -c 'cd $B && nice -n 10 venv/bin/python scripts/A_closedloop/k32_test.py --tag $1 && nice -n 10 venv/bin/python scripts/A_closedloop/k32_shadow.py && venv/bin/python scripts/A_closedloop/k32_autoreport.py --tag $1'" >> $R/logs/k32_auto.log 2>&1
}
if [ "${n0:-1}" -eq 0 ] && [ "${n1:-1}" -eq 0 ]; then idle=$((idle + 1)); else idle=0; fi
final=0
if [ "$todo" -eq 0 ]; then final=1; elif [ $idle -ge 4 ]; then final=1; log "stall: no K32 job process on either host for $idle runs; final tables on what exists"; fi
if [ $final -eq 1 ]; then
  if analysis test; then
    $SSH VUW "bash -c 'date > $B/results/k32/ANALYSIS_DONE; echo todo=$todo >> $B/results/k32/ANALYSIS_DONE'"
    log "final analysis done (todo $todo)"
  else
    log "final analysis FAILED"; final=0
  fi
elif [ $((now - last_partial)) -ge 21600 ]; then
  if analysis partial; then log "partial analysis done"; else log "partial analysis FAILED (not retried for 6 h)"; fi
  last_partial=$now
fi

# --- 4. pull
mkdir -p $A/tables
rsync -a -e "$SSH" VUW:$B/results/k32/tables/ $A/tables/ 2>> $R/logs/k32_auto.log
cp -f $A/tables/REPORT_auto_*.md $A/ 2>/dev/null
{ echo "pulled $(date '+%F %T') (kit time); jobs not yet complete: $todo of 376"
  $SSH VUW "tail -3 $B/logs/k32/dispatch_jobs_test_vuw.log; tail -3 $B/logs/k32/dispatch_jobs_test_vuw1.log"; } > $A/status.txt 2>&1
if [ $final -eq 1 ]; then
  date > $A/FINAL_PULLED
  echo "- $(date '+%F %T') (kit time): **K32 test complete** (jobs not complete: $todo). Final tables were pulled by k32_auto.sh into auto/ (REPORT_auto_test.md). REPORT.md is still to be written by hand." >> $R/docs/paper/handoff/K32/STATUS.md
  log "final tables pulled into $A"
fi
printf "idle=%s\nlast_partial=%s\n" $idle $last_partial > $ST
