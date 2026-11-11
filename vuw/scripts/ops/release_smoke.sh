#!/usr/bin/env bash
# Release smoke: the export directory's run_desp.py / run_wholeframe.py on one unit per dataset ($1 = python to use).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
R=$B/tmp/desp_release
PY=${1:-$R/../desp_env/bin/python}
L=$B/logs/release; mkdir -p $L
c="cd $R && { $PY scripts/run_desp.py --dataset nuscenes --split test --units 2 --out runs/sub_nusc > $L/smoke_desp_nusc.log 2>&1; \
$PY scripts/run_desp.py --dataset av2 --split test --units 1 --out runs/sub_av2 > $L/smoke_desp_av2.log 2>&1; \
$PY scripts/run_desp.py --dataset argoverse_hd --split test --units 1 --out runs/sub_ahd > $L/smoke_desp_ahd.log 2>&1; \
$PY scripts/run_wholeframe.py --dataset nuscenes --split test --units 2 --long-sides 960 --out runs/subwf_nusc > $L/smoke_wf_nusc.log 2>&1; \
$PY scripts/run_wholeframe.py --dataset av2 --split test --units 1 --long-sides 960 --out runs/subwf_av2 > $L/smoke_wf_av2.log 2>&1; \
$PY scripts/run_wholeframe.py --dataset argoverse_hd --split test --units 1 --long-sides 960 --out runs/subwf_ahd > $L/smoke_wf_ahd.log 2>&1; echo SMOKE DONE >> $L/smoke_desp_nusc.log; }"
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/smoke_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "$(date +%T) release smoke launched with $PY"
