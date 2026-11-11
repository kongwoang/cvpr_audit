#!/usr/bin/env bash
# K30 V8a: NAVSIM v2 EPDMS on navtest (one stage; the devkit's run_pdm_score_one_stage.py, metric_cache_v2_navtest) of the
# perception-in-the-loop PDM-Closed branches (k30_nav_v2agent.PerceptionPDMAgent). reference / cheap / full960 / spec_pdm from
# the stored final detections; hybrid_pdm / native once the V6 (i) detect stage has written dets_k30 for every log.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" B=$B
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
run() {  # $1 branch, $2 dets dirs
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py train_test_split=navtest \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_v2_navtest agent=constant_velocity_agent \
    "agent._target_=k30_nav_v2agent.PerceptionPDMAgent" +agent.branch=$1 "+agent.dets_dirs=$2" \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=12 \
    experiment_name=k30_v2_$1 output_dir=$NAVSIM_EXP_ROOT/k30_v2/navtest_$1 > $L/v2score_navtest_$1.log 2>&1
  echo "$(date +%T) navtest $1 exit $?" >> $L/v2score_done.log
}
for br in reference cheap full960 spec_pdm; do run $br dets_final; done
until [ "$(ls $NAVSIM_EXP_ROOT/dets_k30/*.pkl 2>/dev/null | wc -l)" -ge "$(ls $NAVSIM_EXP_ROOT/dets_final/*.pkl | wc -l)" ]; do sleep 600; done
for br in hybrid_pdm native; do run $br dets_k30,dets_final; done
echo "$(date +%T) K30 V2 NAVTEST DONE" >> $L/v2score_done.log
