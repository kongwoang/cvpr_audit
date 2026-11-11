**Is ours final (with refiner) above every "+ refiner" control?** No; on no dataset is it significantly above all of them.
- **nuScenes new test.** Ours final has the highest G (0.170). It is significantly above only random crops + refiner and support only + refiner. Every other "+ refiner" interval contains 0:
  - native + refiner 0.167 (at 10× ours' px);
  - temporal_crit + refiner 0.152;
  - field only + refiner 0.118;
  - traj_support + refiner 0.093.
- **AV2 val, ego mask.**
  - **Above:** ours final (0.122) is significantly above random + refiner, u960 + refiner [+0.006, +0.069] and field only + refiner.
  - **Tied:** support only + refiner (0.122), uncertainty + refiner (0.118), ego-corridor + refiner (0.113) and temporal_crit + refiner (0.104).
  - **Borderline:** traj_support + refiner [−0.001, +0.041] and native + refiner [−0.000, +0.124].
- **Argoverse-HD val.** **The ego-corridor tube + refiner is significantly ABOVE ours final**: 0.122 vs 0.091, ours − it [−0.066, −0.001]. Support only + refiner (0.127) and uncertainty + refiner (0.094) are also higher (n.s.). Ours is significantly above u960 + refiner, native + refiner and field only + refiner.
- **Reading.** The refiner lifts nearly every pipeline. With it, planner-driven selection is clearly better than random crops and the field alone, but not demonstrably better than image- or corridor-driven crops at equal crop AREA. Note that those crops use 20–80% more tensor px than ours.

# K14: "+ refiner" controls (post-hoc analysis; frozen configurations, nothing tuned; AV2 with the K10 ego mask)

**Setup**
- **Refiner rule.** Each pipeline gets the dataset's refiner (v1 nuScenes, v2 AV2 / Argoverse-HD). It is applied exactly as for ours: to the objects of the pipeline's final scene that are in the support of its action AND come from a high-resolution pass. For whole frame, that is every object.
- **Crop heuristics** are budgeted at ours' per-frame crop AREA (strict fill, as in instr_eval). traj_support = support crops + the ego-corridor tube filling the rest; ego-corridor = the tube alone.
- **Parity.** Every no-refiner action / cost and ours' with-refiner cost reproduce the instrumented runs bit for bit (`parity_J_full_ref_vs_instr_maxabs` = 0).
- **Statistics.** Paired bootstrap over scenes / logs / sequences, 2,000 resamples. Argoverse-HD: frames with 3D labels only (24 seq., 15,006 frames).
- **Script:** `remote/A_method/k14_controls.py`, tables `k14_analyze.py`.

**(1) Paper Table 2, empty cells.** AV2 masked; heuristics WITHOUT refiner at ours' masked crop area (the K13(f) runs). Ours final minus:
- uncertainty (G 0.084): **[−0.016, +0.092]**;
- temporal_crit (0.053): **[+0.019, +0.120]**;
- random (0.002): **[+0.066, +0.175]**.

**(2)–(4) All pipelines with and without refiner.** Columns of `refiner_controls_{nusc,av2,ahd}.csv`:
- `ci_lo/ci_hi`: ours final minus the method WITH refiner;
- `*_no_refiner`: minus the method without refiner;
- tensor px mean / p95.

Support only / field only come from instr_eval (nuScenes, AV2) and K13 (Argoverse-HD). Figure: `refiner_controls.pdf/png`.

| method | nuScenes G ref (px) | ours − it | AV2 masked G ref (px) | ours − it | A-HD G ref (px) | ours − it |
|---|---|---|---|---|---|---|
| **ours** | **0.170** (0.100) | | **0.122** (0.053) | | **0.091** (0.070) | |
| uncertainty | −0.062 (0.133) | [−0.014, +0.448] | 0.118 (0.081) | [−0.041, +0.051] | 0.094 (0.124) | [−0.122, +0.132] |
| temporal_crit | 0.152 (0.129) | [−0.013, +0.052] | 0.104 (0.084) | [−0.025, +0.065] | 0.057 (0.126) | [−0.007, +0.073] |
| random | −0.058 (0.185) | [+0.044, +0.385] | 0.034 (0.092) | [+0.042, +0.137] | 0.006 (0.179) | [−0.023, +0.201] |
| traj_support | 0.093 (0.139) | [−0.018, +0.198] | 0.101 (0.072) | [−0.001, +0.041] | 0.089 (0.113) | [−0.023, +0.022] |
| ego-corridor tube | −0.030 (0.137) | [−0.005, +0.469] | 0.113 (0.066) | [−0.018, +0.040] | **0.122** (0.112) | **[−0.066, −0.001]** |
| u960 | 0.045 (0.384) | [−0.029, +0.321] | 0.086 (0.232) | [+0.006, +0.069] | 0.025 (0.253) | [+0.018, +0.122] |
| native | 0.167 (1.031) | [−0.081, +0.137] | 0.061 (1.012) | [−0.000, +0.124] | 0.003 (1.013) | [+0.005, +0.187] |
| support only | −0.107 (0.050) | [+0.018, +0.498] | 0.122 (0.041) | [−0.050, +0.055] | 0.127 (0.056) | [−0.159, +0.106] |
| field only | 0.118 (0.091) | [−0.077, +0.181] | −0.035 (0.037) | [+0.115, +0.204] | −0.066 (0.043) | [+0.100, +0.224] |

**(3) Whole frame + refiner on megapixel frames.**
- On KITTI, u1248 + refiner was the best pipeline. Here native + refiner matches ours on nuScenes (0.167 vs 0.170) at 10× the px.
- It is lower on AV2 masked (0.061) and on Argoverse-HD (0.003; ours above).
- u960 + refiner is below ours everywhere (significantly on AV2 and Argoverse-HD).

**Without refiner** (`G_no_refiner`):
- nuScenes: ours (0.135) is significantly above uncertainty, temporal_crit, random, traj_support and the corridor tube.
- AV2 masked: ours (0.031) is below uncertainty (0.084) and support only (0.083).
- Argoverse-HD: ours (0.039) is below support only (0.116), the corridor tube (0.071) and uncertainty (0.067).
