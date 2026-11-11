#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
until grep -q "LIDAR ALL DONE" logs/extract_lidar.log; do sleep 300; done
python d72_lidar.py --set old --split train+val+test > logs/d72_old.log 2>&1 &
python d72_lidar.py --set new --split test > logs/d72_newtest.log 2>&1 &
wait
echo "$(date +%T) D72 DONE" >> logs/d72_old.log
