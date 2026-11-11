#!/usr/bin/env bash
# Release gate C in the FRESH environment (CVPR/tmp/desp_env from requirements.txt): unit tests, every script's --help,
# reproduce_main.sh --subset. Logs: logs/release/gate_*.log
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
R=$B/tmp/desp_release
E=$B/tmp/desp_env/bin
L=$B/logs/release; mkdir -p $L
cd $R
$E/python -m pytest -q tests/test_core.py > $L/gate_unit.log 2>&1; echo "unit rc=$?" >> $L/gate_unit.log
: > $L/gate_help.log
for s in scripts/*.py training/*.py; do $E/python $s --help > /dev/null 2>&1; echo "$s rc=$?" >> $L/gate_help.log; done
for s in scripts/*.sh; do bash $s --help > /dev/null 2>&1; echo "$s rc=$?" >> $L/gate_help.log; done
c="cd $R && PATH=$E:\$PATH bash scripts/reproduce_main.sh --subset > $L/gate_subset.log 2>&1; echo \"subset rc=\$?\" >> $L/gate_subset.log"
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/gate_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "gate launched"
