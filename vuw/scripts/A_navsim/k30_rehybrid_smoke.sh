#!/usr/bin/env bash
# K30: fixed-tube hybrid_pdm, smoke on one navtest log (parity with dets_k30 on straight-route tokens), then the navhard
# smoke's detect2 + split again with the shared hybrid_dets
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env_navsim.sh
export NAVSIM_EXP_ROOT=$B/results/navsim_exp SPEC_CROP_SCALE=0.5 B=$B
cd $B/code/risk-aware-perception
python $B/scripts/A_navsim/k30_nav_hybrid.py --stage rehybrid --only_log 2021.05.25.14.16.10_veh-35_01690_02183
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 B=$B SPEC_CROP_SCALE=0.5 K30_NH_TAG=_smoke
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
rm -rf $NAVSIM_EXP_ROOT/navhard_d2_smoke $NAVSIM_EXP_ROOT/navhard_dets_smoke
python $B/scripts/A_navsim/k30_navhard.py --stage detect2 --shard 0/1 && python $B/scripts/A_navsim/k30_navhard.py --stage split
echo "$(date +%T) K30 REHYBRID SMOKE exit $?"
