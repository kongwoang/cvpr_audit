#!/usr/bin/env bash
# K30-DEV (dev splits only, pre-registered): k30_dev.py part devref on nuScenes val (3 shards) and AV2 CFG-DEV with the ego
# mask (3 shards), two passes (the second re-runs only units a silently killed shard left out), then k30_dev_an.py
# (selection by the pre-registered rule). Unattended.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
S=$B/k10_masked
L=$B/logs/k30
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
for pass in 1 2; do
  for i in 0 1 2; do
    SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/dev_nusc_p${pass}_$i.log bash -c \
      "$T B=$B python $B/scripts/A_method/k30_dev.py --dataset nusc --part devref --shard $i/3" &
    sleep 150
    SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/dev_av2_p${pass}_$i.log bash -c \
      "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k30_dev.py --dataset av2 --part devref --shard $i/3" &
    sleep 150
  done
  wait
done
source $B/env.sh
python $B/scripts/A_method/k30_dev_an.py > $L/dev_analyze.log 2>&1
echo "$(date '+%F %T') K30 DEV DONE" >> $L/dev_analyze.log
