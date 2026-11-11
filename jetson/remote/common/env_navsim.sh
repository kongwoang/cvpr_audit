# source this: NAVSIM v1.1 on the nuPlan env (python 3.9, nuplan-devkit 1.2); data / maps / outputs inside the workspace
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
       NAVSIM_EXP_ROOT=$B/results/navsim_exp NAVSIM_DEVKIT_ROOT=$B/third_party/navsim_v1.1
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
