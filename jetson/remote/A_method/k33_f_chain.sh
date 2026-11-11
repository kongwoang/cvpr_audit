#!/usr/bin/env bash
# K33 F (table1_design.md §2 F): AV2 fine-tuned rows that do not exist yet. $1 = build | train_fovea | train_ufp | train_lzu
#   build        FOVEA warped set, UFP mosaic set, LZU saliency (k33_av2ft.py; one GPU group, small)
#   train_<arm>  YOLOv8s 15 epochs, imgsz 960, batch 32 (fovea / ufp: d68_train.py; lzu: k33_av2ft.py --stage train), workers 4
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k33; mkdir -p $L
E="source $B/env.sh; cd $B; export B=$B OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WORKERS=4"
case $1 in
  build)
    SHARE=1 MIN_FREE_MB=4000 setsid nohup $B/run_when_gpu_free.sh $L/f_build.log bash -c \
      "$E; python scripts/A_method/k33_av2ft.py --stage build --arm fovea && python scripts/A_method/k33_av2ft.py --stage build --arm ufp && python scripts/A_method/k33_av2ft.py --stage lzusal" > /dev/null 2>&1 < /dev/null & ;;
  train_fovea|train_ufp)
    arm=${1#train_}
    SHARE=1 MIN_FREE_MB=14000 setsid nohup $B/run_when_gpu_free.sh $L/f_train_$arm.log bash -c \
      "$E; python scripts/A_ext/d68_train.py $arm k33_av2ft 15 yolov8s 960 32 $arm" > /dev/null 2>&1 < /dev/null & ;;
  train_lzu)
    SHARE=1 MIN_FREE_MB=14000 setsid nohup $B/run_when_gpu_free.sh $L/f_train_lzu.log bash -c \
      "$E; python scripts/A_method/k33_av2ft.py --stage train --arm lzu" > /dev/null 2>&1 < /dev/null & ;;
esac
sleep 2; echo "$(date +%T) K33 F $1 launched"
