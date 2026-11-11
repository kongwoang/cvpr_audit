#!/usr/bin/env bash
# K40 R1 smoke: release run_rows.py (fresh env) on AV2 test first log, all rows; GPU queue.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
R=$B/tmp/desp_k40; E=$B/tmp/desp_env/bin; L=$B/logs/k40; mkdir -p $L
c="cd $R && $E/python scripts/run_rows.py --dataset av2 --split test --units 1 --wf 448,512,640,960,1280,1536,native --heur tsp8,unc2,tcr0,cor8 --out runs/r1_smoke_av2 > $L/r1_smoke_av2.log 2>&1; echo rc=\$? >> $L/r1_smoke_av2.log"
SHARE=1 MIN_FREE_MB=3500 setsid nohup $B/run_when_gpu_free.sh $L/r1_smoke_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo queued
