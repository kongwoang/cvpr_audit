#!/usr/bin/env bash
# K30 V8a navhard smoke: 6 tokens through detect1 (GPU) -> field (CPU) -> detect2 (GPU) -> split, folders *_smoke
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 B=$B SPEC_CROP_SCALE=0.5 K30_NH_TAG=_smoke
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
S=$B/scripts/A_navsim/k30_navhard.py
python $S --stage detect1 --shard 0/1000 --max_tokens 6 && \
CUDA_VISIBLE_DEVICES="" python $S --stage field --shard 0/1 --max_tokens 6 && \
python $S --stage detect2 --shard 0/1 && python $S --stage split
echo "$(date +%T) K30 NAVHARD SMOKE exit $?"
