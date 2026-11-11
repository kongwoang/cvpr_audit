# AUDIT_ENTRYPOINTS — where to start reading (one page)

Two code roots: `vuw/` (GPU server: the evaluation code that produced every accuracy table) and `jetson/` (Jetson AGX
Xavier kit: TensorRT fast path and timing harnesses; `jetson/remote/` is the kit's mirror of `vuw/scripts/`). The paper
calls the method DGSP; in code and handoffs it is **DESP v6-S** (`v6S`). F = whole frame u960 + C1 (+ refiner v3a).

| topic | file:line (paths relative to `vuw/` unless marked) |
|---|---|
| **DGSP loop** (frozen braking loop: rounds, field crop, support crops, 95 % `covered` stop, merge) | `code/CVPR/src/vtf/speculative.py:199` `run` (loop body 215–240); general-consumer version `code/CVPR/src/vtf/spec_general.py:61` `run` |
| test-time driver of every pipeline on the test splits (K28 phase 3: ours, basewf, baseheur, instr) | `scripts/A_method/k28_p3.py:117` `main`; per-pipeline loop wrapper `:390` `loop`; v6-S settings `:61` dict `A`, chain diagnostics `:82` `CHAIN` |
| **field** (decision-sensitivity probe grid → image footprint) | `code/CVPR/src/vtf/speculative.py:33` `field_crop`; `code/CVPR/src/vtf/spec_general.py:48` `footprint`; grid `SP.R_GRID / SP.Y_GRID` in speculative.py |
| **support** (detections the action rests on) | `code/CVPR/src/vtf/speculative.py:23` `support`; crop of a box `:48` `box_crop` |
| **guard** (truncation re-detection with grown crops) | `code/CVPR/src/vtf/speculative.py:75` `truncated`, `:103` `grow`, `:115` `guarded`, `:157` `guarded_batch` |
| **merge** (crop detections replace draft objects whose centre lies in any crop) | `code/CVPR/src/vtf/speculative.py` docstring l.5 and `run` body; `centres_in` `:55`; coverage stop `covered` `:65` |
| **lift — geometry** (monocular range / lateral from boxes + calibration) | `code/risk-aware-perception/src/rap/mono.py:191` `predicted_geometry` |
| **lift — looming** (learned temporal cue C1, same-scale history s1/s2 per pipeline) | `scripts/A_method/k28_p3.py:298` `lift_c`, history update `:341` `push`; model + training `scripts/A_method/k28_ttrain.py`, data `k28_tdata.py` |
| **lift — range refiner v3a** (applied once after the loop, to verified support objects only) | `scripts/A_method/k28_p3.py:348` `refine`; training `scripts/A_method/k28_train.py` (+ `k28_refdata.py`, `scripts/common/train_refiner_remote.py`) |
| **planner** (required deceleration → discrete action; thresholds) | `code/risk-aware-perception/src/rap/planner.py:26` `PlannerParams`, `:52` `required_decel`, `:83` `discrete_action` |
| **loss J / G** | per-frame J: `scripts/A_method/k28_p3.py:237` `cost` → `rap.planner.decision_cost` (previous action = the cheap pipeline's); G: `scripts/A_method/k28_p3_analyze.py:58` `G` |
| annotated demand / decisive object | `scripts/A_method/k30_gt.py` (annotations only, no images) |
| **episodes** (missed / false braking episodes) | `scripts/A_method/k31_episodes.py` (`runs`, `unit_stats`, `analyse`); extended rows `k42_2_episodes.py` |
| **bootstrap** (paired unit bootstrap, 2,000, seed 0) | `scripts/A_method/k24_analyze.py:179` `Boot`; `k31_episodes.py` `analyse`; `k39_6_fidelity.py` / `k42_3_fidelity_v3a.py` `main` |
| fidelity (main Tab. 1) | `scripts/A_method/k42_3_fidelity_v3a.py` (= `k39_6_fidelity.py` with `|none`→`|v3a`) |
| **baseline runners** | whole frame + heuristics: `k28_p3.py --part basewf / baseheur`; published / object-level (AdaScale, SAHI, FOVEA, UFP, AnA, counterfactual): `scripts/A_method/k33_eval.py`, `code/CVPR/src/vtf/k33pub.py`, `k33sel.py`; routers / Chanakya: `scripts/A_method/k33_rc.py`; dev selection `k33_select.py` |
| **timing harness** (Jetson, TensorRT) | `jetson/scripts/64_k29_latency.py` (main Tab. 6 latency, K29), `jetson/experiments/k38_deadline_20261026/scripts/88_k38.py`, `jetson/experiments/k39_jetson_20261027/scripts/89_k39.py`; energy tables `jetson/scripts/68_k30_j_tables.py`; TRT engines `jetson/src/vtf/trtdyn.py` |
| frozen configuration (text) | `jetson/docs/method_freeze_draft.md` §2n (v6-S); release config `vuw/results/latency_opt/release_cpu_tests/desp/config.py` |
| written protocols / results | `jetson/docs/paper/handoff/<TASK>/{PREREG,REPORT,STATUS}.md`; kit summary `jetson/docs/paper/0*.md` |
