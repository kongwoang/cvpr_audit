#!/usr/bin/env bash
# OpenScene v1.1 split $SPLIT (test | mini) for NAVSIM navtest: metadata + camera blobs, keeping only CAM_F0 (no LiDAR).
# Layout (NAVSIM's): $B/datasets/navsim/{navsim_logs/$SPLIT, sensor_blobs/$SPLIT/<log>/CAM_F0/*.jpg}
SPLIT=${1:-test}
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
D=$B/datasets/navsim
L=$B/logs/data/navsim_download_$SPLIT.log
mkdir -p $D/navsim_logs $D/sensor_blobs $B/logs/data
cd $D
U=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
if [ ! -d navsim_logs/$SPLIT ]; then
  curl -sL --retry 10 -o meta_$SPLIT.tgz $U/openscene_metadata_$SPLIT.tgz && tar -xzf meta_$SPLIT.tgz && rm meta_$SPLIT.tgz
  mv openscene-v1.1/meta_datas navsim_logs/$SPLIT && rm -rf openscene-v1.1
  [ -d navsim_logs/$SPLIT/$SPLIT ] && mv navsim_logs/$SPLIT/$SPLIT navsim_logs/${SPLIT}_tmp && rmdir navsim_logs/$SPLIT && mv navsim_logs/${SPLIT}_tmp navsim_logs/$SPLIT
  echo "$(date +%T) metadata $(ls navsim_logs/$SPLIT | wc -l) logs" >> $L
fi
for s in $(seq 0 31); do
  [ -f sensor_blobs/.cam_${SPLIT}_$s.done ] && continue
  for t in 1 2 3 4 5; do
    curl -sL --retry 10 -o cam_$SPLIT.tgz $U/openscene_sensor_${SPLIT}_camera/openscene_sensor_${SPLIT}_camera_$s.tgz && \
      tar -xzf cam_$SPLIT.tgz --wildcards '*/CAM_F0/*' && break
    echo "$(date +%T) part $s attempt $t failed" >> $L; sleep 60
  done
  rm -f cam_$SPLIT.tgz
  if [ -d openscene-v1.1/sensor_blobs/$SPLIT ]; then
    rsync -a --remove-source-files openscene-v1.1/sensor_blobs/$SPLIT/ sensor_blobs/$SPLIT/ && rm -rf openscene-v1.1
    touch sensor_blobs/.cam_${SPLIT}_$s.done
    echo "$(date +%T) camera part $s done ($(du -sh sensor_blobs | cut -f1))" >> $L
  else
    echo "$(date +%T) camera part $s: no CAM_F0 extracted" >> $L
  fi
done
echo "NAVSIM $SPLIT DOWNLOAD DONE" >> $L
