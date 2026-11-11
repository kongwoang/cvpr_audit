#!/usr/bin/env bash
# Extract LIDAR_TOP keyframe sweeps from the 10 trainval blobs (for D72), two archives at a time.
D=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/datasets/nuscenes
cd $D
ex () { [ -f lidar_$1.done ] || { tar -xzf archives/v1.0-trainval$1_blobs.tgz -C $D --wildcards "samples/LIDAR_TOP/*" && touch lidar_$1.done && echo "$(date +%T) lidar $1"; }; }
for pair in "01 02" "03 04" "05 06" "07 08" "09 10"; do
  set -- $pair; ex $1 & ex $2 & wait
done
echo "$(date +%T) LIDAR ALL DONE"
