#!/usr/bin/env bash
# VUW1: wait for the D03 del_all dispatcher to finish (it holds the cuda13 queue lock), then run the D03 base dispatcher on
# cuda13 (K41_D03_HOST), then the D05 chain. Moved because VUW (cuda12) has had no eligible GPU for > 8 h.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B
while pgrep -f 'k41_d03_dispatch.py del_all' >/dev/null; do sleep 60; done
echo "$(date +%T) del_all dispatcher finished; D03 base on cuda13"
bash -c "cd $B && source env.sh >/dev/null 2>&1; export B=$B K41_D03_HOST=cuda13; cd \$B/scripts/A_method && exec python k41_d03_dispatch.py base" \
  > $B/logs/k41/D03/dispatch_base_vuw1.log 2>&1 < /dev/null
echo "$(date +%T) D03 base dispatcher exit $?"
exec bash $B/scripts/A_method/k41_d05_chain.sh >> $B/logs/k41/D05/chain_cuda13.log 2>&1
