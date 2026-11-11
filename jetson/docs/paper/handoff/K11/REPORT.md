**Result:** the decision-identical implementation reaches 25.3 ms on nuScenes (u960 17.8 ms) and 42.0 ms on AV2 (u1536 41.1 ms). It is NOT faster than u960 on nuScenes; on AV2 it is 0.9 ms behind u1536, at equal G.

# K11: faster implementation of ours on the Jetson (2026-10-05)

## Setup
- **Board and frames:** Jetson AGX Xavier, clocks locked, TensorRT FP16. Same frames as K5: nuScenes 85 old DEEP scenes (3,176 timed frames) and AV2 10 val logs (586 timed frames).
- **Script:** `scripts/58_large_latency_fast.py` (md5 52fd4e4e…). It imports 55 unchanged. Crops run on ceil32 canvases with the dynamic engine (`src/vtf/trtdyn.py`, md5 fffccc45…), and post-processing is GPU-side and bit-identical (`src/vtf/gpupost.py`).
- **Windows:** exclusive Jetson windows coordinated with the DEEP session (`scripts/k11_window.sh` 03:34–04:08, `k11_window2.sh` 04:30–05:02).
- **Discarded run:** an earlier window (21:34–21:54, 4 Oct) overlapped DEEP's GPU detection. Its results are kept only as `*_OVERLAPPED` and are not used.
- **Parity:** share of timed frames whose decision cost equals that of the K5 follow-up reference run (`scripts/57 --gpupost --dyncrops`) on the same frames.

## Steps (`latency_fast.csv`, `parity.csv`, `engine_microtiming.csv`)

| Step | Dataset | ours mean / p95 / p99 ms | engine calls / frame | parity | G (timed frames) |
|---|---|---|---|---|---|
| reference: K5 follow-up (with breakdown syncs) | nuScenes | 30.2 / 63.8 / 82.3 | 3.14 | — | 0.173 |
| **phase 1: no breakdown syncs** (decision-identical) | nuScenes | **25.3** / 53.2 / 69.2 | 3.14 | **100%** | 0.173 |
| reference | AV2 | 53.0 / 106.8 / 142.0 | 4.61 | — | 0.545 |
| **phase 1: no breakdown syncs** | AV2 | **42.0** / 88.5 / 117.3 | 4.61 | **100%** | 0.545 |
| phase 1 + batched refiner (dynamic-batch engine) | AV2 | 42.2 / 86.3 / 115.0 | 4.48 | 100% | 0.545 |

Whole-frame passes in the same runs:
- nuScenes: u640 12.0, u960 17.8, u1280 23.9, native 33.4 ms.
- AV2: u960 23.0, u1280 32.0, u1536 41.1, native 64.9 ms.

### Phase 2: new configurations, NOT decision-identical
- **What changed:** wider support context (`--ctx`) and/or a field-crop margin (`--fmargin`), to reduce the sequential guard re-detections.
- **Where selected:** on the board's nuScenes frames. These are the 85 old DEEP scenes, which serve as nuScenes' dev data in this project.

| Configuration (nuScenes dev frames) | ours ms (mean / median) | calls | G | parity vs frozen |
|---|---|---|---|---|
| frozen (ctx 2.0, margin 0) | 25.3 / 23.1 | 3.14 | 0.173 | 100% |
| ctx 2.5 | 25.8 / 23.6 | 3.15 | 0.152 | 98.1% |
| field margin 0.15 | 24.3 / 19.6 | 2.90 | 0.173 | 96.6% |
| ctx 2.5 + field margin 0.15 | 24.7 / 19.6 | 2.91 | 0.198 | 96.4% |

- None of the variants reaches u960's 17.8 ms mean; their medians, at 19.6 ms, come close.
- **The dev winner on G is ctx 2.5 + margin 0.15.**
- **Test evaluation (post-hoc analysis, run once)** on nuScenes new test, 228 scenes / 9,181 frames: `remote/A_method/k11_test.py` (md5 d542ab5a…), file `phase2_test_nusc.csv`.
  - Setting: COCO cheap draft and COCO crops, refiner v1, v_close 10. The frozen v3 is rerun in the same script; it matches the confirmatory run on 99.7% of frames.

  | Configuration (nuScenes new test) | G with refiner | G without | network px |
  |---|---|---|---|
  | frozen v3 | 0.170 | 0.135 | 0.100 |
  | ctx 2.5 + field margin 0.15 | 0.195 | 0.165 | 0.125 |

  - New − frozen (with refiner): **+0.025 [−0.028, +0.077]**, not significant, at 25% more detector pixels.
  - **Keep the frozen configuration as the reported one.** The new one is a latency-oriented variant: about 0.6 ms faster on the Jetson dev frames, with a G gain that transferred in the point estimate but is not significant.

### Engine micro-timing (trtexec, GPU compute only)
| Canvas | static engine (ms) | dynamic engine (ms) |
|---|---|---|
| 96×288 | 2.17 | 2.77 |
| 320×320 | 3.38 | 3.86 |
| 640×640 | 8.77 | 10.13 |

The dynamic engine is 14–28% slower per call.

## Reading
1. **Phase 1 is decision-identical and gives the latency to report:**
   - nuScenes 25.3 ms at G 0.173: between u1280 (23.9 ms, G 0.143) and native (33.4 ms, G 0.175);
   - AV2 42.0 ms at G 0.545: u1536 41.1 ms at G 0.543.
   - Ours is on or above the whole-frame latency–G front, but not faster than the equal-G passes (u640 / u960 on nuScenes).
2. **The first breakdown measurements (K5) carried about 5–11 ms of instrumentation syncs.**
   - These are now removed for ours and for the whole-frame passes alike. Those passes also got 1–4 ms faster, so the comparison stays fair.
3. **What did not help:**
   - batching the refiner (one call for all support objects);
   - wider crops (fewer calls, but no lower mean).
4. **Remaining levers, not done:**
   - static engines for the most frequent crop canvases (about 1–2 ms; parity must be re-checked, since a different engine can change FP16 numerics);
   - reusing draft features.
   - The structural cost is about 3 sequential engine calls per frame at about 5–6 ms each including overhead.
5. **AV2 caveat:** AV2 G values here are unmasked (see K10).
