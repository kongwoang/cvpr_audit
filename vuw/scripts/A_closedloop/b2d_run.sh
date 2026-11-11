#!/usr/bin/env bash
# One Bench2Drive evaluation (leaderboard_evaluator.py of the carla_garage Bench2Drive copy) on the GPU picked by
# run_when_gpu_free.sh: usage b2d_run.sh <routes.xml> <agent.py> <agent_config> <result.json> <save_dir> [port]
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_closedloop/carla_env.sh
ROUTES=$1; AGENT=$2; CONF=$3; OUT=$4; SAVE=$5; PORT=${6:-$((20000 + RANDOM % 20000))}
mkdir -p $(dirname $OUT) $SAVE
export SAVE_PATH=$SAVE ROUTES=$ROUTES DATAGEN=0
cd $WORK_DIR
python $LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py --routes=$ROUTES --repetitions=1 --track=SENSORS \
  --checkpoint=$OUT --agent=$AGENT --agent-config=$CONF --debug=0 --resume=True --port=$PORT \
  --traffic-manager-port=$((PORT + 2000)) --gpu-rank=${CUDA_VISIBLE_DEVICES:-0}
