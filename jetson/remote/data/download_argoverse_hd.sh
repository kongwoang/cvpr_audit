#!/usr/bin/env bash
# Argoverse-HD (streaming perception benchmark, Li et al. ECCV 2020). The full-dataset S3 buckets are gone (403), so the
# images come from Argoverse 1.1 tracking VAL (Argoverse-HD val = its ring_front_center, 1920x1200 @ 30 FPS): the 283 GB
# tar is streamed and only ring_front_center images, ego poses, calibration and 3D amodal labels are extracted. The
# Argoverse-HD COCO-format annotations come from the authors' annotation-only package (CMU project page).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env.sh
L=$B/logs/data/argoverse_hd.log
mkdir -p $B/datasets/argoverse_hd $B/logs/data $B/datasets/argoverse1
cd $B/datasets/argoverse_hd
if [ ! -f .ann_done ]; then
  rm -f Argoverse-HD-Full.zip
  curl -sS -L -o Argoverse-HD.zip "https://www.cs.cmu.edu/~mengtial/proj/streaming/dataset/Argoverse-HD.zip?v=1.1" >> $L 2>&1 \
    && unzip -q -o Argoverse-HD.zip >> $L 2>&1 && rm -f Argoverse-HD.zip && touch .ann_done
  echo "$(date +%T) Argoverse-HD annotations: $(find . -name '*.json' | head -5 | tr '\n' ' ')" >> $L
fi
cd $B/datasets/argoverse1
if [ ! -f .val_done ]; then
  curl -sS -L https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/tracking_val_v1.1.tar.gz | \
    tar -xz --wildcards --no-anchored "*/ring_front_center/*" "*/poses/*" "*vehicle_calibration_info.json" \
      "*/track_labels_amodal/*" "*city_info.json" >> $L 2>&1 && touch .val_done
  echo "$(date +%T) AV1.1 tracking val: $(find . -name vehicle_calibration_info.json | wc -l) logs, $(find . -path '*ring_front_center*' -name '*.jpg' | wc -l) front images" >> $L
fi
echo "$(date +%T) ARGOVERSE HD DOWNLOAD DONE" >> $L
