#!/usr/bin/env bash
# nuScenes dev ablations of v3 (old train+val): one round; static probe; no guard + scene history.
cd /vol/dl-nguyenb5-solar/users/hoangpc/CVPR && . ./env.sh
R="--set old --split train+val --device cpu --threads 5 --scale 0.6 --refiners $B/out/resnet18/image_refiner.pt"
python nusc_ablate.py $R --max_rounds 1 --tag _abl_r1 > logs/abl_r1.log 2>&1 &
python nusc_ablate.py $R --v_close 0 --tag _abl_vc0 > logs/abl_vc0.log 2>&1 &
python nusc_ablate.py $R --guard none --tag _abl_noguard_scene > logs/abl_noguard_scene.log 2>&1 &
wait
