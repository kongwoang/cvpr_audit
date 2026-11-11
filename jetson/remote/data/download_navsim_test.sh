#!/usr/bin/env bash
# OpenScene v1.1 TEST split for NAVSIM navtest: metadata + camera blobs, keeping only CAM_F0 (no LiDAR).
# Layout (NAVSIM's): $B/datasets/navsim/{navsim_logs/test, sensor_blobs/test/<log>/CAM_F0/*.jpg}
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
D=$B/datasets/navsim
L=$B/logs/data/navsim_download.log
mkdir -p $D/navsim_logs $D/sensor_blobs $B/logs/data
cd $D
U=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
if [ ! -d navsim_logs/test ]; then
  curl -sL --retry 10 -o meta.tgz $U/openscene_metadata_test.tgz && tar -xzf meta.tgz && rm meta.tgz
  mv openscene-v1.1/meta_datas navsim_logs/test && rm -rf openscene-v1.1
  echo "$(date +%T) metadata $(ls navsim_logs/test | wc -l) logs" >> $L
fi
for s in $(seq 0 31); do
  [ -f sensor_blobs/.cam_$s.done ] && continue
  for t in 1 2 3 4 5; do
    curl -sL --retry 10 -o cam.tgz $U/openscene_sensor_test_camera/openscene_sensor_test_camera_$s.tgz && \
      tar -xzf cam.tgz --wildcards '*/CAM_F0/*' && break
    echo "$(date +%T) part $s attempt $t failed" >> $L; sleep 60
  done
  rm -f cam.tgz
  if [ -d openscene-v1.1/sensor_blobs/test ]; then
    rsync -a --remove-source-files openscene-v1.1/sensor_blobs/test/ sensor_blobs/test/ && rm -rf openscene-v1.1
    touch sensor_blobs/.cam_$s.done
    echo "$(date +%T) camera part $s done ($(du -sh sensor_blobs | cut -f1))" >> $L
  else
    echo "$(date +%T) camera part $s: no CAM_F0 extracted" >> $L
  fi
done
echo "NAVSIM TEST DOWNLOAD DONE" >> $L
