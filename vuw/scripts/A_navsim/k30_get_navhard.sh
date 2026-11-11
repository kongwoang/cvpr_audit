#!/usr/bin/env bash
# K30 V8a: NAVSIM v2.2 navhard two-stage data (OpenScene on HuggingFace, the devkit's download_navhard_two_stage.sh),
# user-approved download (2026-10-16: "cứ tự tải và chạy đi, tôi approve hết miễn là cần thiết")
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
D=$B/datasets/navsim/navhard; cd $D
for f in navsim_v2.2_navhard_two_stage_scene_pickles navsim_v2.2_navhard_two_stage_curr_sensors navsim_v2.2_navhard_two_stage_hist_sensors; do
  [ -f $f.done ] && continue
  wget -q -c https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/$f.tar.gz && tar -xzf $f.tar.gz && rm $f.tar.gz && touch $f.done
  echo "$(date +%T) $f $(ls $f.done 2>/dev/null)"
done
echo "$(date +%T) NAVHARD DOWNLOAD DONE"
