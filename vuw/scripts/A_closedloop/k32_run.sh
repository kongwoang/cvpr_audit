#!/usr/bin/env bash
# One K32 Bench2Drive evaluation (b2d_run.sh + the traffic-manager seed): usage k32_run.sh <routes.xml> <result.json> <save_dir>
# with the AEB_* variables of aeb_pilot.py in the environment (AEB_SEED = traffic-manager seed and latency seed).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_closedloop/carla_env.sh
ROUTES=$1; OUT=$2; SAVE=$3; PORT=$((20000 + RANDOM % 20000))
mkdir -p $(dirname $OUT) $SAVE
export SAVE_PATH=$SAVE ROUTES=$ROUTES DATAGEN=0
[ -z "${AEB_BRAKE_MAP:-}" ] && [ -f $B/results/k32/brake_map.json ] && export AEB_BRAKE_MAP="$(cat $B/results/k32/brake_map.json)"
cd $WORK_DIR
python $LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py --routes=$ROUTES --repetitions=1 --track=SENSORS \
  --checkpoint=$OUT --agent=$B/scripts/A_closedloop/aeb_pilot.py --agent-config=none --debug=0 --resume=True --port=$PORT \
  --traffic-manager-port=$((PORT + 2000)) --traffic-manager-seed=${AEB_SEED:-0} --gpu-rank=${CUDA_VISIBLE_DEVICES:-0}
