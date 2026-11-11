**Result** (post-hoc analysis; NAVSIM navtest; CPU: Intel Xeon Gold 6244, 1 thread per process)
- **Cost of the frozen spec_pdm.** 50.2 PDM-Closed calls per frame (p95 56): 45.1 probe calls, 4.0 support calls and 1 base call. The wall clock is 14.9 s per frame (p95 19.2 s), about 295 ms per call.
- **Same output, much cheaper.** Building the planner state once per frame and reusing identical ego roll-outs gives 6.2 s per frame. Output poses are bit-identical on 100/100 checked frames, and field / support parity is 1.000 on 1,000 frames. With the calls spread over 8 worker processes it takes **1.36 s per frame (11× faster)**; with 16 workers, 1.28 s.
- **Cheaper approximations cost no measurable PDMS** on the full navtest (12,146 frames; frozen 0.8945):
  - 5×5 grid: −0.0002 [−0.0009, +0.0003];
  - 3×3 grid: −0.0002 [−0.0013, +0.0006];
  - cells in the corridor of the current plan: −0.0001 [−0.0006, +0.0003].

  The 3×3 grid and the corridor variant need about 14 instead of 50 calls.
- **Interacting objects are very rare.** Of 9,449 tested pairs, 6 move the plan only when removed together (0.06%; 6 / 1,000 frames). Of 139,675 tested probe + removal combinations, 11 do (0.008%; 10 / 1,000 frames). Adding these objects to the support changes the PDMS on **0 of 1,000 frames**, so the single-object support misses nothing that matters here.
- **Bench2Drive (K6).** The perception field is the closed-form braking consumer: **0 planner queries per step**. PDM-Lite runs once per step in every branch. System time per step is 199 ms for ours vs 182 cheap, 189 u960, 195 u1600 and 152 privileged (CARLA included).

# K20: cost of planner probing (NAVSIM PDM-Closed)

**Setup**
- **Subset:** 1,000 navtest frames drawn with `numpy.random.default_rng(0)` from the frozen field's tokens, sorted.
- **Timing:** 12 single-thread processes on a shared 32-thread host. Full navtest is used for every PDMS number.
- **Scripts:** `remote/A_navsim/k20_pdm.py` (cost, pairs, base plans), `k20_fast.py` (identical-output implementations), `k20_variants.py` (approximations), `k20_pairfield.py` + `k20_pairdet.py` (pair-augmented support), `k20_analyze.py`.
- **Re-run check:** the call-by-call re-run of the frozen field reproduces the stored field exactly (sensitive cells and support, 1,000/1,000).

**(1) Where a call's time goes** (`pdm_cost.csv`; shares of a probe / support call):

| stage | share |
|---|---|
| NAVSIM scenario + planner-input construction | 0.43 |
| proposal generation (IDM unroll) | 0.19 |
| scoring | 0.12 |
| ego roll-out simulation | 0.11 |
| proposal manager (centerline, route) | 0.08 |
| observation update | 0.02 |
| other | 0.05 |

**(2) Identical-output implementations** (`pdm_variants.csv`, figure `pdm_cost.pdf/png`).
- **Why the state can be built once.** The planner's history buffer holds one state: the current ego state and objects. So the scenario, route, drivable area and proposal paths are the same for every call of a frame. The fast version builds them once and per call converts only the modified annotations into tracked objects (as NAVSIM does).
- **Roll-out reuse.** It reuses the simulated ego roll-outs when the IDM proposals are bit-identical to an earlier call's. That happens on 40.7 of ~50 calls per frame, because a probe that does not change the leading agent leaves the proposals unchanged.
- **Wall clock per frame:**
  - frozen 14.9 s;
  - fast 6.2 s;
  - fast with 4 / 8 / 16 workers: 2.07 / 1.36 / 1.28 s (200 frames).

**(3) Approximations** (derived from the frozen 7×7 field, no new probing; support unchanged; then spec_pdm detections and PDMS on navtest).

| variant | calls / frame | ms / frame* | PDMS | minus frozen [CI] | IoU of sensitive footprint |
|---|---|---|---|---|---|
| frozen 7×7 | 50.2 | 14,908 | 0.8945 | | 1 |
| 5×5 | 26.2 | 7,731 | 0.8943 | −0.0002 [−0.0009, +0.0003] | 0.99 |
| 3×3 (ego lane ± one lane) | 14.1 | 4,155 | 0.8943 | −0.0002 [−0.0013, +0.0006] | 0.96 |
| corridor of the plan | 14.5 | 4,265 | 0.8944 | −0.0001 [−0.0006, +0.0003] | 0.95 |
| closed-form braking field (paper) | 0 | — | 0.8926 | | |

\* Derived as calls × the mean frozen call. The fast implementation divides these further.
- **5×5 and 3×3 definitions.** The 5×5 grid is x ∈ {8, 12, 23, 40, 55} m, y ∈ {±10.5, ±3.5, 0} m. The 3×3 grid is x ∈ {8, 23, 55}, y ∈ {0, ±3.5}.
- **Corridor variant.** It keeps the cells within 1.75 m + half a probe width of the draft plan.

**(4) Interacting objects** (`pdm_pairs.csv`): frequencies are in the first lines. Adding the 6 pairs' objects (6 frames), or additionally the probe-interacting objects (16 frames), to the support leaves every spec_pdm score unchanged (PDMS 0.8892 on the subset, 0 frames changed).

**(5) Bench2Drive** (`b2d_calls.csv`; system time per simulation step summed over the three K6 runs, about 655 routes per branch). The perception field of ours is `vtf.speculative.run` with the closed-form braking consumer; there is no PDM-Lite query in perception.
