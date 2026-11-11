#!/usr/bin/env bash
# K30 V8a: navtest EPDMS of native (dets_k30) and of the fixed-tube hybrid_pdm (dets_k30b). The first attempts failed in Hydra
# ("Ambiguous value" for a comma-separated +agent.dets_dirs); both folders hold complete records, so one folder each is enough.
# Runs after the navhard chain, so only one 12-process scorer runs at a time.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
until grep -qs "K30 NAVHARD DONE" $L/v2score_done.log; do sleep 300; done
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" B=$B
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
run() {  # $1 branch, $2 dets dir, $3 tag
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py train_test_split=navtest \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_v2_navtest agent=constant_velocity_agent \
    "agent._target_=k30_nav_v2agent.PerceptionPDMAgent" +agent.branch=$1 "+agent.dets_dirs=$2" \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=12 \
    experiment_name=k30_v2_$3 output_dir=$NAVSIM_EXP_ROOT/k30_v2/navtest_$3 > $L/v2score_navtest_$3.log 2>&1
  echo "$(date +%T) navtest $3 exit $?" >> $L/v2score_done.log
}
run native dets_k30 native
run hybrid_pdm dets_k30b hybrid_pdm_fixed2
echo "$(date +%T) K30 V2 NAVTEST2 DONE" >> $L/v2score_done.log
