# Paper kit — "Check Your Blind Spots: Decision-Exact Perception by Probing the Planner with Unseen Objects"

CVPR 2027 submission (deadline 2026-11-16). This folder is the **only authoritative source** for an agent that writes the
paper. Everything here is curated from the raw research history (`docs/directions_log.md`, chronological, contains
superseded and exploratory numbers — do NOT cite numbers from it unless this kit points there).

## Reading order
1. `01_method.md` — what the method is (final definitions, per-benchmark configurations, theory pointers, code paths).
2. `02_setup.md` — benchmarks, splits (dev vs test per benchmark), metrics, protocol (pre-registration, test-once).
3. `03_results.md` — every final number, organised by the claim it supports, with 95% CIs and source files.
4. `04_baselines.md` — the baseline set, which code each uses (official vs re-implementation), fidelity checks.
5. `05_limitations_and_negative.md` — results that go against the method, and limitations to state.
6. `06_pending.md` — experiments still running (fill these sections when they finish; numbers there are NOT final).
7. `related_work/` — two literature surveys (novelty assessment; SOTA/baseline audit) for the related-work section.

Other docs kept outside this folder: `docs/paper_theory.md` (formal core: loop, proposition, proof, necessity
examples — the method section's theory), `docs/method_freeze_draft.md` (dated pre-registrations: cite for the protocol),
`docs/baseline_fidelity.md`, `docs/directions_log.md` (raw history). Superseded documents are in `docs/archive/`.

## Data access for the writer
- All summary tables (CSV) of the VUW server are mirrored locally in `results/vuw_summaries/` with the same relative
  paths: a source written as `VUW: results/av2/eval_summary__val_v4_final.csv` in 03_results.md is
  `results/vuw_summaries/av2/eval_summary__val_v4_final.csv` here. `results/INDEX.md` maps numbers to run folders.
  Files named d*.csv, b_congruent*, c_lidar2, e_gate, d_attack* are exploratory directions — not part of the paper.
- Per-frame pickles (needed only to recompute CIs or draw new per-frame figures) stay on the VUW server
  (`ssh VUW`, /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/...), reachable only from this Jetson; read-only use.

## Ground rules for the writer
- **Positioning.** The decision-value metric G comes from the authors' DEEP benchmark, now public on arXiv
  (https://arxiv.org/pdf/2609.35910): cite it as a public benchmark, but **define G in the paper itself** (the paper must
  stand even if DEEP is not accepted). Do not import text or claims from the DEEP paper beyond the metric definition.
- **Honesty.** Report negative results as listed in `05_*` (KITTI native-resolution, NAVSIM tie, latency on small
  frames, dev→test non-transfer). Every claim must cite a number in `03_results.md`.
- **Terminology.** Paper name of the method: "decision-exact speculative perception" (internal names: v3 = frozen
  braking configuration; v4-lite = AV2 configuration; spec_pdm = NAVSIM configuration). u<N> = whole frame at long side
  N px with the same COCO YOLOv8s ("uniform"). "cheap" = YOLOv8s at long side 320 (the draft; G = 0 by definition).
- **Numbers format.** G is a share of the all-cheap loss (higher is better, can be negative). CIs: paired bootstrap over
  scenes / logs / sequences, 2,000 draws, 95%.
