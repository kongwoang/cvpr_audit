# source this: CARLA 0.9.15 + Bench2Drive (carla_garage copy, PDM-Lite expert) inside the workspace; HOME redirected so
# UE4 / CARLA config and logs stay in it.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
deactivate 2>/dev/null
. $B/envs/carla/bin/activate
export HOME=$B/.home XDG_CONFIG_HOME=$B/.home/.config
export CARLA_ROOT=$B/tools/carla_0.9.15
export WORK_DIR=$B/third_party/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=$WORK_DIR/scenario_runner LEADERBOARD_ROOT=$WORK_DIR/leaderboard
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$WORK_DIR/leaderboard/team_code:$B/code/CVPR/src:$B/code/risk-aware-perception/src:$B/scripts/common
export CHALLENGE_TRACK_CODENAME=SENSORS IS_BENCH2DRIVE=True PLANNER_TYPE=traj DEBUG_CHALLENGE=0 REPETITIONS=1 RESUME=True
