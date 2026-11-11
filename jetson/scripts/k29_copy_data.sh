#!/usr/bin/env bash
# K29: copy the dev splits (AV2 CFG-DEV 128 train logs, frames at stride 2 + metadata; nuScenes new val 152 scenes) and the
# 20 nuScenes new-test scenes (first 20 scene tokens in sorted order), CAM_FRONT keyframes, from VUW to ~/datasets/k29
# (a separate tree: DEEP's ~/datasets/nuscenes and ~/datasets/av2 are not touched). 8 parallel rsync streams per list.
set -u
V=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
D=$HOME/datasets/k29
cd $D
scp -q VUW:$V/tmp/k29_av2dev_files.txt VUW:$V/tmp/k29_nusc_files.txt VUW:$V/tmp/k29_nusc_test20.txt VUW:$V/tmp/k29_nusc_val.txt $D/
split -n l/8 -d k29_av2dev_files.txt av2part_
split -n l/8 -d k29_nusc_files.txt nuspart_
for f in av2part_*; do rsync -a --files-from=$f VUW:$V/datasets/av2/sensor/train/ $D/av2/sensor/train/ & done
for f in nuspart_*; do rsync -a --files-from=$f VUW:$V/datasets/nuscenes/ $D/nuscenes/ & done
wait
echo "$(date '+%F %T') K29 COPY DONE"
