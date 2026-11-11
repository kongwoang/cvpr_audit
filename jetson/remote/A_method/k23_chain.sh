#!/usr/bin/env bash
# K23 (development data only): refiner v3 training (3 variants) after the data chain, then every component ALONE on
# nuScenes new val and AV2 CFG-DEV (ego mask). Combinations are a second run chosen from these results.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k23; mkdir -p $L
until [ -f $L/data_done.log ]; do sleep 300; done
for v in a b c; do
  SHARE=1 MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/train_v3$v.log python $B/scripts/A_method/k23_train_refiner.py --variant $v &
  sleep 30
done
wait
echo "$(date +%T) K23 TRAIN DONE" > $L/train_done.log
R="python $B/scripts/A_method/k10_masked_run.py"
AN=base,s0.8,s1.0,ctx2.5,ctx3,fm0.15,vc0,vc5,ts,b2,b3,base+c0.35,base+c0.45,base+c0.55,base+p
AA=base,s1280,s1536,ctx2.5,ctx3,fm0.15,vc5,vc10,ts,b2,b3,base+c0.35,base+c0.45,base+c0.55,base+p
for i in 0 1 2 3; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/alone_nusc_$i.log env B_REAL=$B python $B/scripts/A_method/k23_dev.py --dataset nusc --arms $AN --shard $i/4 &
  sleep 30
done
for i in 0 1 2 3 4 5 6 7; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/alone_av2_$i.log env B=$S B_REAL=$B K10_MASK_Y0=1830 $R $S/scripts/A_method/k23_dev.py --dataset av2 --arms $AA --shard $i/8 &
  sleep 30
done
wait
echo "$(date +%T) K23 ALONE DONE" > $L/alone_done.log
