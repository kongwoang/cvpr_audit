#!/usr/bin/env bash
# Extract CAM_F0 from every completely downloaded nuPlan mini camera zip (per the download log's COMPLETE lines) and
# link each log directory into datasets/nuplan/cam_f0_flat/<log> so filename_jpg paths resolve from one root.
D=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/datasets/nuplan
LOG=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/logs/download_nuplan.log
mkdir -p $D/cam_f0_flat
while true; do
  for i in 0 1 2 3 4 5 6 7 8; do
    z=nuplan-v1.1_mini_camera_$i.zip
    grep -q "COMPLETE $z" $LOG || continue
    if [ ! -f $D/cam_f0_x$i.done ]; then
      unzip -qn $D/archives/$z "*/CAM_F0/*" -d $D/sensor_blobs && touch $D/cam_f0_x$i.done && echo "$(date +%T) extracted $i"
    fi
    for l in $D/sensor_blobs/nuplan-v1.1_mini_camera_$i/*; do
      [ -e $D/cam_f0_flat/$(basename $l) ] || ln -s $l $D/cam_f0_flat/$(basename $l)
    done
  done
  [ $(ls $D/cam_f0_x*.done 2>/dev/null | wc -l) -ge 9 ] && break
  sleep 300
done
echo "$(date +%T) EXTRACT ALL DONE"
