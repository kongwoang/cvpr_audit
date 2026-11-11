#!/usr/bin/env bash
# Every 30 min: check the VUW jobs (confirmatory 3, ESOD). Exit (-> wakes the agent) on an anomaly or when all are done.
LOG=/home/kongwoang/research/CVPR/logs/monitor.log
prev_c=-1; prev_e=""; stall_c=0; stall_e=0
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 VUW 'cd /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/logs/A_method
    c=$(grep -c "^\s*\[" av2_confirm3.log 2>/dev/null)
    cd3=$(grep -c "AV2 CONFIRM3 DONE" av2_confirm3.log 2>/dev/null)
    ce=$(grep -ac "Traceback" av2_confirm3.log 2>/dev/null)
    ep=$(tail -c 4000 esod_train.log 2>/dev/null | tr "\r" "\n" | grep -E "^ +[0-9]+/29" | tail -1 | awk "{print \$1}")
    ed=$(grep -ac "ESOD DONE" esod_train.log 2>/dev/null)
    ee=$(grep -acE "Traceback|ABORT" esod_train.log 2>/dev/null)
    echo "$c $cd3 $ce ${ep:-none} $ed $ee"' 2>&1)
  ts=$(date "+%F %T")
  if [ $? -ne 0 ] || [ -z "$out" ]; then echo "$ts ssh failed: $out" >> $LOG; sleep 1800; continue; fi
  set -- $out; c=$1; cd3=$2; ce=$3; ep=$4; ed=$5; ee=$6
  echo "$ts confirm3 logs=$c done=$cd3 tracebacks=$ce | esod epoch=$ep done=$ed errors=$ee" >> $LOG
  [ "$ce" -gt 0 ] && { echo "ALERT confirm3 traceback"; tail -3 $LOG; exit 1; }
  [ "$ee" -gt 0 ] && { echo "ALERT esod error/abort"; tail -3 $LOG; exit 1; }
  if [ "$cd3" -eq 0 ]; then [ "$c" = "$prev_c" ] && stall_c=$((stall_c + 1)) || stall_c=0; fi
  if [ "$ed" -eq 0 ]; then [ "$ep" = "$prev_e" ] && stall_e=$((stall_e + 1)) || stall_e=0; fi
  [ $stall_c -ge 2 ] && { echo "ALERT confirm3 stalled for 1 h at $c logs"; tail -3 $LOG; exit 1; }
  [ $stall_e -ge 3 ] && { echo "ALERT esod stalled for 1.5 h at epoch $ep"; tail -3 $LOG; exit 1; }
  [ "$cd3" -gt 0 ] && [ "$ed" -gt 0 ] && { echo "ALL DONE"; tail -3 $LOG; exit 0; }
  prev_c=$c; prev_e=$ep
  sleep 1800
done
