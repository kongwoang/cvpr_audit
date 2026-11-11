#!/usr/bin/env bash
# K32 brake calibration: a private CARLA server on the GPU picked by run_when_gpu_free.sh, k32_brake_calib.py, then the
# server's two processes are stopped by their exact PIDs.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_closedloop/carla_env.sh
PORT=$((30000 + RANDOM % 5000))
G=${CUDA_VISIBLE_DEVICES:-0}
$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$PORT -graphicsadapter=$G > $B/logs/k32/calib_server.log 2>&1 &
SH=$!
sleep 90
CH=$(ps -eo pid=,ppid= | awk -v p=$SH '$2 == p {print $1}')
echo "server sh $SH child $CH port $PORT gpu $G"
cd $B && python scripts/A_closedloop/k32_brake_calib.py --port $PORT
rc=$?
for p in $CH $SH; do kill $p 2>/dev/null; done
sleep 5
for p in $CH $SH; do kill -9 $p 2>/dev/null; done
echo "calib exit $rc"
