#!/usr/bin/env bash
# K30 V8a: NAVSIM v2 metric caches (the installed v2.2 devkit, third_party/navsim; nuPlan env) for navtest and for navhard
# two-stage, after the V6 (ii) timing run (so it does not perturb the timings) and the navhard download.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
until grep -qs "K30 PROBE TIME DONE" $L/probe_time_done.log && grep -qs "NAVHARD DOWNLOAD DONE" $L/navhard_download.log; do sleep 300; done
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
for sp in navtest navhard_two_stage; do
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py train_test_split=$sp \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_v2_$sp worker=single_machine_thread_pool worker.use_process_pool=true \
    worker.max_workers=12 > $L/v2cache_$sp.log 2>&1
  echo "$(date +%T) $sp exit $?" >> $L/v2cache_done.log
done
echo "$(date +%T) K30 V2 CACHE DONE" >> $L/v2cache_done.log
