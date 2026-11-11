# K38-1 — in-loop deadline execution on the Jetson (replaces the K37-2 approximation)

Jetson AGX Xavier, exclusive window. Attempt 2: 2026-10-26 16:46 (kit time); MAXN, locked clocks, fan 255, all
restored; source and engine hashes unchanged; no engine builds.
- **Runtime.** K29 fast path (trt32) + K35 timing-note runtime (FAST_RECOVERY helper, crop staging).
- **Harness.** experiments/k38_deadline_20261026: 88_k38.py md5 ef7114b7 (rebuilt after the attempt-1 fix);
  deadline.py ad5b77f5 (verbatim loops + stop / mark hooks).
- **Report script.** scripts/k38_1_report.py.
- **Protocol.** K38-1_protocol.md, registered before the window, including the 40-log AV2 list. Everything is post-hoc.

## What ran
- **Rows**, in one process per repetition, each with its own history; three staged repetitions:
  - v6-S and B2K-A at D = ∞;
  - v6-S and B2K-A in-loop under D = 50 and 66.7 ms;
  - u960 + C1 + v3a.
- **In-loop rule (DESP, B2K-A).** Before each round, stop if elapsed + median(next round) + median(refiner + action) >
  D.
  - DESP returns the merged scene of the last completed round; B2K-A stops as at its cap.
  - Medians (medians.json, sha256 ebbecacd…) were measured on the dev prefixes in attempt 1, with no stopping, before
    any test deadline run, and reused hash-checked:
    - nuScenes v6-S round 1: 18.6 ms; AV2: see the file;
    - finish ≈ 0.2 ms.
  - The rule cannot prevent a round that itself overruns, so the deadline-miss shares below are real.
- **Applied action.** If a row's latency > D, the frame keeps the row's last on-time action (hold), for every row.
  - "v6-S (hold)" / "B2K-A (hold)" are the D = ∞ rows of the same runs under the hold rule.
  - u960's computation does not depend on D, so its hold row is exact.
- **Frames.** nuScenes test 20 scenes (805 timed frames); AV2 test, 40 logs (2,968 timed frames, the first 200 warm-up as
  in K29). There are 15 (nuScenes) and 42 (AV2) required episodes per repetition.

## Checks (checks.json)
- **K35 timing-note parity.** Staged vs reference runtime, D = ∞ rows and u960, all timed frames: smoke and full run on
  both datasets, 0 differing actions.
- **D = ∞ parity with K34-3** jetson_frames.csv: v6-S and B2K-A, every repetition, 805 / 586 common frames, 0 differing.
- **Cost.** Recomputing every stored J reproduces it to 4e-15. The stored a_gt equals K30 GT to 9e-16.

## Results (k38_1_rows.csv; paired row − DESP@D in k38_1_diffs.csv; 3 repetitions pooled)

| dataset | D | row | G [95% CI] | deadline miss | stopped early | missed / req. (×3) | false / 1,000 | mean / p95 ms |
|---|---|---|---|---|---|---|---|---|
| nuScenes | 50 | **DESP@50 (in-loop)** | 0.354 [0.13, 0.50] | 6.4% | 39.9% | 21 / 45 | 1.24 | 20.1 / 52.7 |
| nuScenes | 50 | B2K-A@50 (in-loop) | **0.380** [0.21, 0.51] | 3.9% | 4.1% | 19 / 45 | 1.66 | 19.9 / 48.0 |
| nuScenes | 50 | v6-S (hold) | 0.330 [0.07, 0.48] | 8.6% | – | 27 / 45 | 2.48 | 21.0 / 57.4 |
| nuScenes | 50 | B2K-A (hold) | 0.375 | 3.8% | – | 18 / 45 | 1.24 | 19.8 / 47.8 |
| nuScenes | 50 | u960 + C1 + v3a (hold) | 0.386 [0.24, 0.48] | 0.5% | – | 18 / 45 | 4.97 | 22.6 / 31.1 |
| nuScenes | 66.7 | **DESP@66.7** | 0.378 | 1.4% | 16.9% | 18 / 45 | 2.48 | 20.8 / 55.2 |
| nuScenes | 66.7 | B2K-A@66.7 | 0.391 | 0.5% | 0.4% | 18 / 45 | 3.73 | 19.9 / 48.1 |
| nuScenes | 66.7 | v6-S (hold) | 0.378 | 1.7% | – | 18 / 45 | 2.48 | |
| nuScenes | 66.7 | u960 (hold) | 0.386 | 0.0% | – | 18 / 45 | 4.97 | |
| nuScenes | ∞ | v6-S / B2K-A / u960 | 0.380 / 0.382 / 0.386 | – | – | 18 / 18 / 18 | 3.73 / 3.73 / 4.97 | 21.0 / 19.8 / 22.6 mean |
| AV2 (40 logs) | 50 | **DESP@50 (in-loop)** | −0.025 [−0.52, 0.30] | 13.4% | 59.1% | 29 / 126 | 6.29 | 29.2 / 61.9 |
| AV2 | 50 | B2K-A@50 (in-loop) | 0.119 [−0.14, 0.41] | 19.4% | 23.0% | 31 / 126 | 3.71 | 30.6 / 60.1 |
| AV2 | 50 | v6-S (hold) | −0.208 [−0.67, 0.21] | 20.9% | – | 37 / 126 | 4.38 | 32.5 / 76.3 |
| AV2 | 50 | B2K-A (hold) | −0.160 | 21.3% | – | 36 / 126 | 3.37 | 31.6 / 64.9 |
| AV2 | 50 | u960 + C1 + v3a (hold) | **0.298** [0.15, 0.45] | 3.0% | – | 18 / 126 | 6.40 | 32.1 / 47.0 |
| AV2 | 66.7 | **DESP@66.7** | 0.312 [0.20, 0.45] | 4.3% | 39.4% | 24 / 126 | 6.29 | 30.3 / 64.3 |
| AV2 | 66.7 | B2K-A@66.7 | 0.328 [0.21, 0.46] | 4.0% | 5.9% | 21 / 126 | 5.62 | 31.4 / 64.6 |
| AV2 | 66.7 | v6-S (hold) | 0.313 | 8.5% | – | 24 / 126 | 5.05 | |
| AV2 | 66.7 | u960 (hold) | **0.382** [0.23, 0.51] | 0.3% | – | 18 / 126 | 7.08 | |
| AV2 | ∞ | v6-S / B2K-A / u960 | 0.297 / 0.304 / 0.380 | – | – | 24 / 21 / 18 | 6.07 / 6.07 / 7.08 | 32.5 / 31.6 / 32.1 mean; 76.3 / 64.9 / 47.0 p95 |

Cheap: G 0, 12 / 45 and 12 / 126 missed, false 8.7 and 39.1 per 1,000, 7.4 / 10.1 ms.

Paired differences vs DESP@D (k38_1_diffs.csv), selected:
- **AV2, D = 50.**
  - v6-S hold − DESP@50: **−0.182 [−0.355, −0.032]**. The in-loop rule recovers 0.18 G over holding.
  - B2K-A@50 − DESP@50: +0.145 [−0.25, +0.73]; false episodes **−2.58 [−5.21, −0.33]**.
  - u960 hold − DESP@50: +0.323 [−0.004, +0.771].
- **AV2, D = 66.7.** DESP@66.7 ≈ v6-S hold in G (0.000), with **+1.24 [0.23, 2.46]** false episodes per 1,000.
  u960 hold +0.070 [−0.106, +0.233].
- **nuScenes, D = 50.** DESP@50 vs v6-S hold +0.024 (n.s.). B2K-A@50 +0.026 (n.s.), u960 hold +0.032 (n.s.).

## Reading
1. **The in-loop rule works as intended at tight deadlines.**
   - At D = 50 on AV2 it lifts DESP from −0.208 (hold) to −0.025 (significant), cutting late frames from 20.9% to 13.4%.
   - On nuScenes it lifts DESP from 0.330 to 0.354 (n.s.).
   - The remaining late frames are rounds that overrun on their own; round 1 has a heavy tail, p95 ≈ 50 ms on nuScenes.
2. **B2K-A is the best DESP form under a deadline.**
   - nuScenes: at 50 ms it reaches 0.380, which is its D = ∞ value, with 3.9% late frames. At 66.7 ms it gives 0.391.
   - AV2: 0.119 at 50 ms; 0.328 at 66.7 ms, above its D = ∞ value of 0.304.
3. **On this Jetson, u960 + C1 + v3a is not slower than DESP.**
   - The single 960-px whole-frame engine runs at a mean of 22.6 / 32.1 ms and a p95 of 31.1 / 47.0 ms. Its p95 is below
     both v6-S (57.4 / 76.3 ms) and B2K-A (47.8 / 64.9 ms).
   - Under deadlines it is therefore almost never late: 0.0% / 0.3% of frames at 66.7 ms, 0.5% / 3.0% at 50 ms.
   - It ties DESP / B2K-A on nuScenes and leads on AV2: 0.298 vs 0.119 at 50 ms and 0.382 vs 0.328 at 66.7 ms. Both
     differences are positive but n.s., given 40 logs.
   - Its cost is network px: 0.38 / 0.23, i.e. 4–5× DESP's.
4. **The 40-log AV2 set differs from K29's 8 logs.** At D = ∞, v6-S's G is 0.297 vs u960's 0.380; on the K29 frames it
   was 0.419 vs 0.323. The 40-log ordering agrees with the full AV2 test split on VUW (v6-S 0.294, u960 + C1 + v3a 0.338).
   K29's eight logs were favourable to v6-S.

## Consequence for the latency claim (for the writing session)
- Wall-clock deadline robustness **does not favour DESP over u960 + C1 + v3a** on the Jetson AGX Xavier with the
  current runtime. The crop loop's tail (multiple small engine executions and rounds) costs more time than one
  960-px whole-frame pass, although it processes 4–5× fewer pixels.
- Supported claims:
  - DESP decides with 4–5× fewer network px at equal G on nuScenes;
  - the bounded in-loop form (B2K-A@D) meets 15–20 Hz deadlines on nuScenes with no G loss;
  - an in-loop deadline rule is better than holding a late action;
  - on AV2, whole-frame C1 + v3a is ahead in G under deadlines (n.s.).
- A "faster than whole-frame on the edge" claim is not supported by these numbers.
- Per-pixel compute (px, energy) remains DESP's advantage. K30-J2 energy is in handoff/K30/J/j2_energy.csv.
