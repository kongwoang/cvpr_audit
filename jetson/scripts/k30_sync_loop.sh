#!/usr/bin/env bash
# K30 unattended: every 2 h copy the K30 tables / figures from VUW into results/vuw_summaries/k30 and handoff/K30
# (k30_sync.sh, read-only on VUW); stops after the sync that follows "K30 ALL DONE" (VUW logs/k30/final_done.log).
L=$HOME/research/CVPR/logs/hourly_check/k30_sync_loop.log
while true; do
  bash $HOME/research/CVPR/scripts/k30_sync.sh >/dev/null 2>&1; echo "$(date '+%F %T') synced" >> $L
  if ssh VUW "grep -qs 'K30 ALL DONE' /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/logs/k30/final_done.log"; then
    echo "$(date '+%F %T') K30 ALL DONE seen; final sync done, loop ends" >> $L; break
  fi
  sleep 7200
done
