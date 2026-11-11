#!/usr/bin/env bash
# Detect every nuScenes scene as its CAM_FRONT images appear; stop after one pass once the download has finished.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
while true; do
  fin=0; grep -q "ALL DONE" $B/logs/download_nuscenes.log && fin=1
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python nusc_detect_vuw.py --weights $B/models/yolov8s.pt --device cpu --threads 8
  [ $fin = 1 ] && break
  sleep 600
done
echo "$(date +%T) DETECT LOOP DONE"
