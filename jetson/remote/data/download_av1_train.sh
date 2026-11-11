#!/usr/bin/env bash
# K35-0 (b): Argoverse 1.1 3D-tracking TRAIN (train1-4, 65 logs = the Argoverse-HD train sequences), stream-extracted
# from the public tars without storing them: only what vtf.av1.AV1Log reads (calibration, poses, amodal 3D labels,
# ring_front_center images) and city_info.json. Nothing else is extracted; no pipeline is run on these logs here.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/data/av1_train.log; mkdir -p $(dirname $L) $B/datasets/argoverse1
cd $B/datasets/argoverse1
for i in 1 2 3 4; do
  [ -f .train${i}_done ] && continue
  echo "$(date +%T) train$i start" >> $L
  curl -sS -L --retry 5 https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/tracking_train${i}_v1.1.tar.gz | \
    tar -xz --wildcards --no-anchored "*/vehicle_calibration_info.json" "*/city_info.json" "*/poses/*" \
        "*/per_sweep_annotations_amodal/*" "*/ring_front_center/*" >> $L 2>&1 && touch .train${i}_done
  echo "$(date +%T) train$i rc=$? logs $(ls argoverse-tracking/train$i 2>/dev/null | wc -l)" >> $L
done
echo "$(date +%T) AV1.1 TRAIN DONE: $(ls -d argoverse-tracking/train*/* 2>/dev/null | wc -l) logs" >> $L
