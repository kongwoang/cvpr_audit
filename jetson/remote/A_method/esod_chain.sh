#!/usr/bin/env bash
# ESOD baseline: prepare nuScenes in ESOD's layout (SAM masks) on a free GPU. Training/eval appended once validated.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
mkdir -p $B/logs/A_method
L=$B/logs/A_method/esod_prepare.log
while true; do
  . $B/scripts/A_method/esod_env.sh
  g=$(pick_gpu); if [ -n "$g" ]; then sleep 60; g2=$(pick_gpu); [ "$g" = "$g2" ] && break; fi
  echo "$(date +%T) waiting for a free GPU" >> $L; sleep 120
done
cd $B/third_party/esod
echo "$(date +%T) running on GPU $g" >> $L
CUDA_VISIBLE_DEVICES=$g python $B/scripts/A_method/esod_prepare.py >> $L 2>&1
echo "$(date +%T) ESOD PREPARE exit $?" >> $L
