#!/usr/bin/env bash
# Phase 1 of the VUW layout clean-up: scripts go to scripts/<topic>/; the old root paths become symlinks so that the
# chains that are still running keep resolving them (phase 2 removes the symlinks once nothing runs).
set -e
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B
place () {   # place <topic> <file>...
  local t=$1; shift
  mkdir -p scripts/$t
  for f in "$@"; do
    [ -e "$f" ] || continue
    [ -L "$f" ] && continue
    mv "$f" scripts/$t/ && ln -s scripts/$t/$f $f
  done
}
place common   env.sh env_nuplan.sh run_when_gpu_free.sh install_nuplan_env.sh train_refiner_remote.py \
               nusc_refiner_crop.py export_refiner_onnx.py
place data     download_av2.sh extract_lidar.sh nuplan_extract_watch.sh nusc_detect_vuw.py detect_loop.sh \
               nusc_detect_ft.py nusc_full_tables.py refiner_data_v2.py
place A_method nusc_full_eval.py run_eval_dev.sh run_eval_dev_v3.sh confirm_chain.sh nusc_spec_diag.py \
               nusc_spec_ttc.py nusc_spec_inspect.py nusc_ablate.py run_ablate.sh nusc_eval_planb.py run_planb.sh \
               after_confirm_planb.sh baselines_eq.py ftdraft_chain.sh refiner_v2_chain.sh
place A_nuplan nuplan_spec_match.py nuplan_spec_real.py nuplan_spec_counterfactual.py nuplan_spec_cells.py \
               run_115spec.sh run_nuplan_all.sh run_nuplan_real.sh
place A_ext    d72_lidar.py d72_chain.sh c_lidar2.py e_gate.py d_attack.py d68_build.py d68_train.py d68_eval.py \
               d68_chain.sh d68_eval_chain.sh d75_build.py d75_chain.sh
place B_churn  nusc_models_vuw.py d5_analysis.py d5_chain.sh d5_redo.sh d25_congruent.py d25b_congruent.py \
               b_congruent.py b_chain.sh
place old      diag_spec_nusc.py launch_refiners.sh run_114.sh run_114spec.sh run_eval_old.sh
mkdir -p archive
for f in code_bundle.tgz nuplan_bundle.tgz; do [ -f $f ] && mv $f archive/; done
[ -f yolov8s.sha256 ] && mv yolov8s.sha256 models/
echo "organized; root now:"; ls
