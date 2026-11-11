#!/usr/bin/env bash
# K29: after k29_copy_data.sh, re-run every part list until rsync succeeds (VUW's sshd refused one of the 16 parallel
# connections); 4 streams at a time, 5 s apart. Existing files are skipped by rsync.
set -u
V=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
D=$HOME/datasets/k29
until grep -q "K29 COPY DONE" $HOME/research/CVPR/logs/k29/copy_data.log; do sleep 60; done
cd $D
one() { local src=$1 lst=$2 dst=$3 n=0; until rsync -a --files-from=$lst VUW:$src $dst; do n=$((n+1)); [ $n -ge 20 ] && { echo "FAILED $lst"; return 1; }; sleep 30; done; echo "$(date +%T) ok $lst"; }
i=0
for f in av2part_* nuspart_*; do
  case $f in av2*) one $V/datasets/av2/sensor/train/ $f $D/av2/sensor/train/ & ;; *) one $V/datasets/nuscenes/ $f $D/nuscenes/ & ;; esac
  i=$((i+1)); sleep 5
  [ $((i % 4)) -eq 0 ] && wait
done
wait
echo "$(date '+%F %T') K29 COPY VERIFIED"
