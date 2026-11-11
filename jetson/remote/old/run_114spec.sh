#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env_nuplan.sh
export HOME=$B/.home
cd $B/code/risk-aware-perception
python $B/scripts/A_nuplan/nuplan_spec_match.py --data_root $RAP_NUPLAN_ROOT --map_root $RAP_NUPLAN_MAPS --n_overlays 0
