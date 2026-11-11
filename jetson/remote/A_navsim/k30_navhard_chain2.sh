#!/usr/bin/env bash
# K30 V8a navhard scoring (replaces the scoring part of k30_navhard_chain.sh, whose devkit aggregation crashed on failed
# scenarios): k30_navhard_score.py per branch (raw rows saved first; failed tokens left out of the aggregation), then
# k30_navhard_agg.py (own valid tokens and the common token set). Writes "K30 NAVHARD DONE" for k30_nav_v2score2.sh.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$B/results/navsim_exp \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" B=$B
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
for br in ${BRANCHES:-reference cheap full960 native spec_pdm hybrid_pdm}; do
  [ -f $NAVSIM_EXP_ROOT/k30_v2/navhard_$br/raw_rows.pkl ] && continue
  python $B/scripts/A_navsim/k30_navhard_score.py train_test_split=navhard_two_stage \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_v2_navhard_two_stage agent=constant_velocity_agent \
    "agent._target_=k30_nav_v2agent.PerceptionPDMAgent" +agent.branch=$br "+agent.dets_dirs=navhard_dets" \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=12 \
    experiment_name=k30_v2_navhard_$br output_dir=$NAVSIM_EXP_ROOT/k30_v2/navhard_$br > $L/v2score_navhard2_$br.log 2>&1
  echo "$(date +%T) navhard2 $br exit $?" >> $L/v2score_done.log
done
python $B/scripts/A_navsim/k30_navhard_agg.py > $L/navhard_agg.log 2>&1
echo "$(date +%T) K30 NAVHARD DONE" >> $L/v2score_done.log
