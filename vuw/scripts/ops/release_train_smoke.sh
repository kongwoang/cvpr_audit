#!/usr/bin/env bash
# Release: training-pipeline smoke in the fresh environment (1 unit per source, 1 epoch); not a reproduction.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
R=$B/tmp/desp_release
E=$B/tmp/desp_env/bin
L=$B/logs/release
c="cd $R && { $E/python training/extract_looming_data.py --src av2 --max-units 2 --out runs/train_smoke && \
$E/python training/extract_looming_data.py --src nusc --max-units 2 --out runs/train_smoke && \
$E/python training/train_looming.py --data runs/train_smoke --epochs 1 --out runs/train_smoke/looming.pt && \
$E/python training/extract_refiner_data.py --src av2 --max-units 1 --out runs/train_smoke && \
$E/python training/extract_refiner_data.py --src nuscenes --max-units 2 --out runs/train_smoke && \
$E/python training/extract_refiner_data.py --src kitti --units 0007 --out runs/train_smoke && \
$E/python training/train_refiner.py --data runs/train_smoke --epochs 1 --out runs/train_smoke/refiner.pt; echo \"train smoke rc=\$?\"; } > $L/train_smoke.log 2>&1"
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/train_smoke_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "train smoke launched"
