#!/usr/bin/env bash
# K30 V6 (i) / V8a after the tube fix: rehybrid (GPU, 2 shards) -> [after the old nav chain] hybrid_pdm PDMS on dets_k30b
# (12 CPU shards) -> merge with pdms_k30 -> summary (pdms_summary_k30m.csv) -> [after the navtest v2 chain] EPDMS of
# hybrid_pdm on dets_k30b, in sequence (k30_navhard_chain.sh runs on its own and scores after this one).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
X=$B/results/navsim_exp
E=". $B/scripts/common/env_navsim.sh; export NAVSIM_EXP_ROOT=$X SPEC_CROP_SCALE=0.5 B=$B; cd $B/code/risk-aware-perception;"
for i in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/rehybrid_$i.log bash -c "$E python $B/scripts/A_navsim/k30_nav_hybrid.py --stage rehybrid --shard $i/2" &
  sleep 200
done
wait
until grep -qs "K30 NAV CHAIN DONE" $L/nav_pdms_summary.log; do sleep 300; done
P="$E export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_k30b NAVSIM_PDMS_OUT=$X/pdms_k30b NAVSIM_TAG=_k30b \
NAVSIM_MAIN=hybrid_pdm NAVSIM_BRANCHES=hybrid_pdm;"
for i in $(seq 0 11); do bash -c "$P CUDA_VISIBLE_DEVICES='' python $B/scripts/A_navsim/navsim_pdms.py --stage run --shard $i/12" > $L/nav_pdms_b_$i.log 2>&1 & done
wait
bash -c "$E python $B/scripts/A_navsim/k30_nav_merge.py" > $L/nav_pdms_merge.log 2>&1
bash -c "$E export NAVSIM_PDMS_OUT=$X/pdms_k30m NAVSIM_TAG=_k30m NAVSIM_MAIN=spec_pdm NAVSIM_BRANCHES=spec_pdm,hybrid_pdm,native,hybrid_pdm_tubebug; \
  python $B/scripts/A_navsim/navsim_pdms.py --stage summary" >> $L/nav_pdms_merge.log 2>&1
echo "$(date +%T) K30 NAV REHYBRID PDMS DONE" >> $L/nav_pdms_merge.log
until grep -qs "K30 V2 NAVTEST DONE" $L/v2score_done.log; do sleep 300; done
. $B/scripts/common/env_nuplan.sh
export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=$X \
       NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" B=$B
export PYTHONPATH=$B/scripts/A_navsim:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
cd $B/code/risk-aware-perception
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py train_test_split=navtest \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_v2_navtest agent=constant_velocity_agent \
  "agent._target_=k30_nav_v2agent.PerceptionPDMAgent" +agent.branch=hybrid_pdm "+agent.dets_dirs=dets_k30b,dets_final" \
  worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=12 \
  experiment_name=k30_v2_hybrid_pdm_fixed output_dir=$NAVSIM_EXP_ROOT/k30_v2/navtest_hybrid_pdm_fixed > $L/v2score_navtest_hybrid_pdm_fixed.log 2>&1
echo "$(date +%T) navtest hybrid_pdm_fixed exit $?" >> $L/v2score_done.log
