#!/usr/bin/env bash
# K30 V4 logging run watchdog (unattended): every 10 min, an evaluator of THIS run (cmdline holds /k30log/) whose
# b2dlog/<branch>_<chunk>.log has not changed for > 60 min is stopped together with the CARLA server(s) it started
# (exact PIDs from the process tree; nothing else is touched). The dispatcher then relaunches the chunk with RESUME.
# Ends when the dispatcher logs "K30 B2D LOG ALL CHUNKS COMPLETE".
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30/b2dlog
desc() { local p=$1; for c in $(ps -eo pid=,ppid= | awk -v p=$p '$2==p {print $1}'); do echo $c; desc $c; done; }
until grep -qs "K30 B2D LOG ALL CHUNKS COMPLETE" $L/dispatch.log; do
  for p in $(ps -eo pid=,args= | awk '/leaderboard_evaluator.py/ && /\/k30log\// && !/awk/ {print $1}'); do
    j=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -o "k30log/[a-z0-9_]*/chunk[0-9]*\.json" | head -1)
    [ -z "$j" ] && continue
    br=$(echo $j | cut -d/ -f2); k=$(echo $j | grep -o "chunk[0-9]*" | tr -d chunk)
    lg=$L/${br}_$k.log
    [ -f $lg ] || continue
    age=$(( ( $(date +%s) - $(stat -c %Y $lg) ) / 60 ))
    if [ $age -gt 60 ]; then
      kids=$(desc $p | tr '\n' ' ')
      echo "$(date '+%F %T') stop hung evaluator $p ($j), log idle ${age} min; descendants: $kids" >> $L/watchdog.log
      for c in $kids; do kill $c 2>/dev/null; done
      kill $p 2>/dev/null
    fi
  done
  sleep 600
done
echo "$(date '+%F %T') watchdog end" >> $L/watchdog.log
