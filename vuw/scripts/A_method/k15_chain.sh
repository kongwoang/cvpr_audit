#!/usr/bin/env bash
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked; L=$B/logs/A_method/k15
SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/render_nusc.log python $B/scripts/A_method/k15_render.py $B/results/k15/frames_nusc.csv $B/results/k15/panels_nusc &
sleep 120
SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/render_av2.log env B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k15_render.py $B/results/k15/frames_av2.csv $B/results/k15/panels_av2 &
wait
echo "$(date +%T) K15 DONE" > $L/chain_done.log
