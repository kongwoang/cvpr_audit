# K32 STATUS — closed-loop AEB with estimated states (CARLA 0.9.15 + Bench2Drive, VUW)

Time box: the dev gate (step 2) must pass by **2026-10-26**, otherwise stop and report. Frozen v6-S and baselines; no
tuning of any perception pipeline; no new training.

## Pre-registration (2026-10-19, before any run)

### 1. Agent ("AEBPilot", scripts/A_closedloop/aeb_pilot.py, subclass of the carla_garage PDM-Lite AutoPilot of K6)
- **Route following (lateral control and target speed)**: PDM-Lite unchanged for map, route, speed limits, junction
  speed, traffic lights and stop signs (as K6). Actors are invisible to it: its world proxy returns no vehicle.*,
  walker.* or static.prop.* actors, and its route-obstacle scenario handling (CarlaDataProvider.active_scenarios:
  bypass of accidents / parked cars / construction etc.) is disabled. Traffic lights and stop signs stay visible.
- **Longitudinal safety**: the paper's braking controller (rap.planner, PlannerParams() defaults: thr_decel 1.0, thr_hard
  3.5 m/s^2, DECELERATE 2.5, HARD-BRAKE 6.0 m/s^2, corridor, range limit, stand-off, reaction time as in the paper) on the
  branch's scene. Commanded deceleration = max(route follower's, controller's): when the controller's action is DECELERATE
  or HARD-BRAKE, throttle = 0 and brake = max(PDM-Lite's brake, b(a_cmd)), where b is the brake-pedal value that gives the
  commanded deceleration for the ego vehicle, from a calibration on dev (straight road, no actors, brake in {0.1 … 1.0}
  at 5 / 10 / 15 m/s; brake = 1 if a_cmd exceeds the measured maximum). PDM-Lite's own anti-roll-back and unblock rules
  stay; the controller's brake overrides the unblock throttle.
- **Scene of a perception branch**: front camera 1600 x 900, FOV 70, K6's CAM_FRONT pose, every simulation tick
  (20 Hz, dt = 0.05 s as K6). YOLOv8s (COCO, frozen), conf >= 0.25. Detections -> the paper's monocular lift
  (rap.mono.predicted_geometry; CARLA pinhole intrinsics from the FOV; camera height 1.6 m; the paper's class height
  priors) -> (range, lateral min / max, TTC). TTC: cheap = the draft's box-growth TTC on its own previous frame (as the
  datasets); "+ comp." = C1 looming head (out/k28_temporal, frozen) on the branch's own same-scale history and refiner
  v3a on the support objects (frozen), unless the dev rule (2) disables them.
- **privileged-AEB**: the same controller on annotated states of every vehicle / walker / static prop whose 3D box
  projects into the front camera and lies within the detection radius: range = nearest box face along the ego x axis,
  lateral min / max in the ego frame, TTC = range / closing speed (inf if not closing). No latency.
- **Branches**: privileged-AEB; cheap (YOLOv8s 192 x 320); u960 + comp. (576 x 960); native + comp. (928 x 1600); hybrid
  + comp. (K28 tsp8: the draft's support boxes + the ego-corridor tube TUBE[8], guard, + C1, own history) at a per-frame
  crop budget equal to v6-S's crop px on the same frame (x1.0, the K30 V1a m = 1 form; v6-S's selection is run inside the
  hybrid branch only to obtain the budget); v6-S (frozen §2n: v_close 0, crops at 960 / 1600, support context 2, guard,
  <= 3 rounds, C1, refiner v3a); v6-S <= 2 calls (k_loop.run_bounded, N = 2).
- **Latency** (setting L1): the action computed from frame t is applied from tick t + ceil(latency_t / dt) on; at every
  tick the applied action is that of the newest frame whose apply tick has passed (KEEP before the first). latency_t is
  drawn (seeded per route and seed) from the branch's per-frame Jetson latencies of K29 (nuScenes test 20, trt32 fast
  path, 3 repetitions): v6-S and v6-S <= 2 conditional on the frame's number of detector calls (frames of K29 with the
  same number of calls; pooled with the next lower count when fewer than 50 frames), whole frames and the hybrid
  unconditional; cheap = the draft's latency. Setting L0: no latency. privileged-AEB runs in L0 only.
- **Shadow mode**: in every v6-S rollout, u960 + comp. is also run on the same frames (not applied); per-frame actions
  logged for agreement.

### 2. Dev gate (Bench2Drive training-split routes; never the 220 evaluation routes)
- Dev routes: short single-scenario routes cut from leaderboard/data/routes_training.xml (Town12) for the scenario types of
  the test list (3): the route waypoints from 80 m before to 80 m after the scenario trigger, with only that scenario;
  up to 4 routes per type (the first ones in file order); a candidate whose trigger lies within 100 m of a trigger of an
  evaluation route of the same type (bench2drive220.xml) is dropped. The list is written to results/k32/dev_routes.xml
  before any dev run.
- (a) Lift error on matched actors (IoU >= 0.3, class-compatible, in camera, within 80 m), from a log run driven by
  privileged-AEB with every branch's perception computed on the same frames: median |log(range_est / range_true)| and
  median |TTC_est - TTC_true| on closing actors with true TTC < 10 s, per branch, with and without refiner v3a and C1.
  Rule: on the matched actors pooled over the branches that use the component, if the refiner increases the median range
  error, or C1 increases the median TTC error, that component is disabled for EVERY branch.
- (b) privileged-AEB collides on <= 5% of the dev routes.
- (c) the cheap branch differs measurably from privileged-AEB on dev: the 95% paired route-bootstrap interval of
  (cheap - privileged) excludes 0 for the collision rate or for the mean per-route minimum TTC.
- Brake calibration (1) is measured on dev before (a)–(c).
- If any of (a)–(c) fails (for (a): the error table cannot be produced), stop, report, no test runs.

### 3. Test
- Routes: bench2drive220.xml restricted to (i) the official Emergency_Brake ability types (tools/ability_benchmark.py):
  BlockedIntersection, DynamicObjectCrossing, HardBreakRoute, OppositeVehicleTakingPriority,
  OppositeVehicleRunningRedLight, ParkingCutIn, PedestrianCrossing, ParkingCrossingPedestrian, StaticCutIn,
  VehicleTurningRoute, VehicleTurningRoutePedestrian, ControlLoss; (ii) types whose decisive actors are vehicles,
  pedestrians or bicycles and that need no lateral manoeuvre around an actor: CrossingBicycleFlow, EnterActorFlow,
  EnterActorFlowV2, HighwayCutIn, InterurbanActorFlow, InterurbanAdvancedActorFlow, MergerIntoSlowTraffic,
  MergerIntoSlowTrafficV2, NonSignalizedJunctionLeftTurn, NonSignalizedJunctionLeftTurnEnterFlow,
  NonSignalizedJunctionRightTurn, SignalizedJunctionLeftTurn, SignalizedJunctionLeftTurnEnterFlow,
  SignalizedJunctionRightTurn, T_Junction, VanillaNonSignalizedTurn, VanillaNonSignalizedTurnEncounterStopsign,
  VanillaSignalizedTurnEncounterGreenLight, VanillaSignalizedTurnEncounterRedLight, ParkingExit.
- Excluded types and why: Accident, AccidentTwoWays, ParkedObstacle, ParkedObstacleTwoWays, HazardAtSideLane,
  HazardAtSideLaneTwoWays, VehicleOpensDoorTwoWays, InvadingTurn, YieldToEmergencyVehicle, SequentialLaneChange,
  HighwayExit — they require a lateral manoeuvre around an actor, which the actor-blind route follower cannot make, so
  every branch would end blocked; ConstructionObstacle, ConstructionObstacleTwoWays — decisive objects are cones /
  signs (non-COCO) and a lateral bypass is required. The resulting route list is written to results/k32/test_routes.xml
  before any test run (route ids listed in REPORT).
- Branches as (1); settings L0 and L1 (privileged-AEB L0 only); 3 seeds (traffic-manager seed 0, 1, 2; latency draws
  seeded by (route, seed)).

### 4. Metrics (per branch and setting)
Collision rate per route (share of routes with >= 1 collision with a vehicle / walker / static object, from the
leaderboard infractions) and per km driven; mean ego speed at collision; share of routes with minimum TTC < 1 s (TTC to
the annotated actors in the ego corridor, per tick); braking-onset delay relative to privileged-AEB on the same scenario
trigger (seconds from privileged-AEB's first >= DECELERATE tick after the trigger to the branch's first such tick, same
route and seed); unnecessary hard-brake episodes per km (maximal runs of applied HARD-BRAKE while privileged-AEB's action
on the annotated states of the same tick is < HARD-BRAKE; runs closer than 1 s are one episode); route completion and
route time; flicker (applied action changes per 100 ticks); network px per frame and modelled latency per frame.
Shadow: per-frame agreement v6-S vs u960 + comp. and run lengths of disagreement. Paired route bootstrap (2,000, seed 0;
unit = route, seeds averaged within a route) of v6-S and v6-S <= 2 minus every other branch, per setting.

### 5. Report order
Gate outcome; test table (L0, L1); every row where v6-S is not better, with the mechanism (lift error, latency,
non-COCO object, …).

## Log
- 2026-10-19: K32 received. Pre-registration above written before any code or run.
- 2026-10-19: amendments before any dev or test run (no result seen):
  (1) privileged-AEB uses vehicles and walkers only (not static props): the static props of the included scenarios are
      ControlLoss debris and cones next to the lane, which would stop privileged-AEB for good; perception branches see
      only COCO classes anyway.
  (2) dev lift error (2a): measured on the final scene of every branch in the privileged-driven log run; range error on
      the matched actors whose range the refiner changes (refined support objects), with vs without the refiner on the
      same detections; TTC error on the matched actors detected in a high-resolution pass, C1 TTC vs the box-growth TTC of
      the same detections (C1 off = box growth on the branch's previous final scene, as the plain K28 pipelines). Also
      reported on all matched actors.
  (3) refiner v3a re-saved without numpy-2 objects for the CARLA env (out/resnet18_v3a/image_refiner_np1.pt; state-dict
      md5 5d47aea4 identical to image_refiner.pt; no training).
  (4) L1 latency samples: handoff/K32/latency_samples.csv (= results/k32/latency_samples.csv), from K29 test-window runs
      r{1,2,3}a/b (trt32; cheap, v6-S with rounds, v6-S <= 2 with loop calls, u960 / native + C1 + ref) and K30 J1 j1x
      r1-3 (hybrid tsp8 at v6-S's x1.0 budget); warm-up frames excluded; 2,415 frames per branch.
- 2026-10-19: routes written (k32_routes.py): results/k32/test_routes.xml 155 routes (types of §3), dev_routes.xml 95
  routes (24 types, 4 each except MergerIntoSlowTrafficV2 3; ParkingExit none: its triggers are off the densified route),
  routes_meta.csv. Code: aeb_pilot.py, k32_pipes.py, k32_brake_calib.py, k32_run.sh, k32_dispatch.sh.
- 2026-10-19 (VUW ~19:00): brake calibration DONE (results/k32/brake_calib.csv): the pedal-to-deceleration map is not
  linear (10 m/s: 0.7 -> 3.9, 0.8 -> 5.7, 0.9 -> 9.0 m/s^2; 15 m/s: 1.0 -> 5.2 m/s^2 over the first second); by the
  pre-registered rule (speed-averaged, interpolated) brake_map.json = {2.5: 0.443, 6.0: 0.866}.
- 2026-10-19: smoke on 2 dev routes (privileged + all branches logged; v6-S L0 with u960 shadow): both routes complete,
  0 collisions, logs complete; 0.35–0.45 s wall time per tick (CARLA dominates; all six branches add little).
- Amendment before the dev gate runs (seen: smoke only): the per-tick minimum TTC counts actors whose near face is ahead of
  the ego's front bumper (2.5 m ahead of the vehicle origin) and that overlap the controller's corridor (+-1.2 m):
  TTC = (near - 2.5 m) / closing speed. Actors alongside the ego (near face behind the bumper) are excluded; the smoke
  showed them as TTC 0 (vehicles passing next to the ego in a junction); contact is measured by the collision metric.
- 2026-10-19 (VUW 20:05): operational: VUW was CPU-bound (load 44 / 32 cores; ~15 min per dev route). The dev jobs are
  split per host: VUW keeps the four privileged-logall chunks (jobs_dev_vuw.txt), VUW1 (idle, GPUs 1-2 free) runs the four
  cheap chunks (jobs_dev_vuw1.txt). The old single dispatcher was stopped by PID; its running evaluators continue.
- 2026-10-20 (VUW 00:10): dev runs DONE (privileged logall and cheap, 95 / 95 routes each); k32_dev_gate.py run.
  **GATE NOT PASSED.**
  - (a) Table produced. The rule disables both components:
    - the refiner raises the median range error on the refined actors (0.0645 -> 0.0741, pooled);
    - C1 raises the median TTC error (3.64 s vs 1.95 s for box growth).
  - (b) **Fails**: privileged-AEB collides on 39 / 95 dev routes (41%; limit 5%).
  - (c) Passes: cheap − privileged mean minimum TTC +0.34 s [0.14, 0.56]; collision rate −0.11 [−0.21, 0.00].
  - Mechanism of (b), from the 73 vehicle collisions of the privileged run (per-tick log at the collision point):
    - 70% happen with hard braking commanded within the previous 2 s;
    - the ego speed at contact is mostly 0–0.9 m/s (40% below 0.5 m/s), at TTC ≈ 0.0–0.06 s.
    - The pattern is creeping into a stopped or slow obstacle, plus vehicles sweeping into a stopped ego at junctions.
    - The braking controller's demand v^2 / (2 (r − r0)) falls to KEEP at low speed. PDM-Lite, blind to actors,
      commands throttle again, so the ego creeps into the obstacle: the closed loop has no standstill hold.
    - A minority are at speed with no braking demand (red-light runners and crossing vehicles outside the camera
      corridor), which an AEB cannot avoid.
  - Cheap collides less (31%) because its route completion is lower (76.6 vs 86.3): it stops earlier for phantoms.
  - By the registration (§2): no test runs. Reported to the user together with a dev-only fix proposal (standstill hold);
    the gate deadline is 2026-10-26. Tables: handoff/K32/dev_gate.csv, dev_lift_error.csv, dev_routes_outcome.csv.

### Amendment A (2026-10-20, VUW 00:30; approved by the user after the failed gate; written BEFORE any re-run)
- **A1, actuator, identical for all branches: standstill hold.**
  - Trigger: ego speed < 0.5 m/s and the branch's own scene has an object in the ego corridor (half-width 1.2 m) within
    r0 + 3 m = 5 m. The range is the one the branch's braking controller uses: annotated near range for privileged-AEB,
    the lifted range for the perception branches.
  - Then the agent commands full brake (1.0, throttle 0). It keeps it until no such object is in the branch's scene for
    10 consecutive frames.
  - The braking controller's decisions and parameters are unchanged.
  - Under latency L1, the hold reads the corridor flag of the newest applicable frame, the same queue as the action.
  - The hold state is logged per frame. A spurious object that holds the ego is a real cost of that branch; timeouts and
    blocked routes are counted.
- **A2, components.** Refiner and C1 are off for every branch, by rule (a) of the first gate.
  - Looming cue of every perception branch: the box-growth TTC of DEEP's mono lift on the branch's previous final scene
    (k32_pipes with C1 off; the cue (a) compared C1 against). This is not K28's T4 scale-space estimate. Privileged-AEB
    uses the annotated TTC.
  - K32 is reported as "selection with the geometric lift", not as the full v6-S.
- **Collision log.** Every contact of the ego bounding box with a vehicle / walker is logged with the ego speed and a type:
  - ego front into an object in the corridor;
  - front contact outside the corridor;
  - side impact;
  - rear impact (by an actor outside the corridor).
  The leaderboard's collision infractions are matched to the contacts by actor id.
- **Gate re-run (dev2).** Same 95 dev routes, L0, seed 0. Branches: privileged-AEB (without the per-branch log run: (a)
  is decided) and cheap, both with A1 and A2. Thresholds unchanged:
  - (b) privileged-AEB collides on ≤ 5% of dev routes;
  - (c) cheap differs measurably from privileged-AEB.
  Output: results/k32/dev2/, tables dev2_*.csv.
- **One further amendment is allowed, only if** after A1 the remaining privileged collisions are side or rear impacts by
  actors outside the corridor (not preventable by longitudinal control). In that case those scenario types are removed
  from the test list, decided on dev and listed before any test run, threshold unchanged. No other change.
- If the gate still fails: stop K32, write the REPORT with the mechanism, no test runs, and move the CPUs to K33. The
  deadline 2026-10-26 is unchanged.
- 2026-10-20 (VUW 00:23): amendment A implemented:
  - aeb_pilot.py (md5 ffe25c88): A1 hold + contact log;
  - k32_dev_gate.py (md5 01f7e558): --dev dev2 --priv privileged --no_a, plus collision typing.
  Job files: jobs_dev2_vuw.txt (privileged / cheap c0, c1) and jobs_dev2_vuw1.txt (c2, c3); AEB_C1=0 AEB_REF=0.
  Smoke first: one dev route (9000), privileged and cheap (jobs_smoke2.txt, VUW, MAXJ 2).
- 2026-10-20 (VUW 00:40): smoke2 OK on route 9000 (privileged and cheap: no traceback; hold and contact logs written).
  - Both branches: one vehicle contact, a firetruck at 10 m/s typed "side" (OppositeVehicleRunningRedLight), then
    blocked; no hold was needed.
  - dev2 dispatchers started: VUW jobs_dev2_vuw.txt (privileged / cheap c0, c1), VUW1 jobs_dev2_vuw1.txt (c2, c3),
    MAXJ 4 each.

### Amendment B (2026-10-20, VUW ~04:50; user-approved after the dev2 analysis; written BEFORE any B run)
- **Finding: the range origin was incompatible with the controller.**
  - In CARLA every branch's range is measured from the ego's centre: privileged = near face in the actor's ego-frame
    corners; perception = z of the lift with cam_to_ego = CAM_POSE relative to the vehicle origin.
  - The ego's front is 2.45 m ahead of its centre (bounding-box half-length; contacts at near ≈ 2.0–2.5 m confirm it).
  - So the stand-off r0 = 2 m meant "stop with the centre 2 m before the object", i.e. the front 0.45 m inside it. This
    is the first gate's creeping mechanism; A1 covered only the standstill part.
- **B:** every branch's range is counted from the ego's front for the braking controller, i.e. r − 2.45 m.
  - Implemented as stand-off r0 + 2.45 m in the PlannerParams of the agent and of the perception pipes. The closing
    speed r / ttc is origin-independent.
  - The controller's decision rule, thresholds, decelerations and reaction time are unchanged.
  - The A1 hold radius r0 + 3 m is therefore also counted from the front.
  - Env AEB_FRONT=2.45, logged per route.
- **Runs.** The cheap dev2 runs still in progress are stopped (invalid under B). Kept: dev2 results, tables and
  amendment A's analysis.
- **dev3 re-run:** same 95 dev routes, L0, seed 0, A1 + A2 + B.
  - privileged-AEB first; if (b) ≤ 5% passes, then cheap for (c).
  - The one allowed further amendment stays as written in amendment A: if the remaining privileged collisions are side
    or rear impacts by actors outside the corridor, drop those scenario types (decided on dev, listed before any test
    run). Thresholds unchanged; deadline 2026-10-26.
- 2026-10-20 (VUW ~05:25): B implemented (aeb_pilot.py md5 a0047601, k32_pipes.py md5 944c1dc4: PlannerParams stand-off +
  AEB_FRONT; logged per route).
  - dev2 stopped: the VUW dispatcher and the running cheap_c0 job tree (exact PIDs).
  - State at the stop: cheap c1–c3 complete, c0 at 16 / 24. The dev2 results are kept for amendment A's record.
  - dev3 privileged launched: jobs_dev3_vuw.txt (c0, c1) and jobs_dev3_vuw1.txt (c2, c3); AEB_FRONT=2.45, A1, A2.
- **B2** (2026-10-20 VUW ~05:45; part of B, following the user's "check what else can be made correct"; before the
  restart): privileged-AEB's in-camera actors were cut at PDM-Lite's config.detection_radius = 50 m. That is a route
  follower setting, not the controller's: the controller's range is max_range = 80 m, and the perception branches have
  no such cut. Privileged-AEB now takes in-camera vehicles / walkers within max_range (80 m from the ego centre).
  - The dev3 privileged run had done 5 routes. It is stopped (exact PIDs), its outputs moved to dev3_stopped_b2, and it
    restarts with B + B2.
  - Other checks, no change:
    - the controller's v ≤ 0.1 → KEEP (covered by A1);
    - HARD = 6 m/s² commanded (the paper's u(a), kept);
    - camera FOV 70°;
    - the latency queue is reset per route;
    - throttle is zeroed whenever braking is applied.
- 2026-10-20 (VUW 05:38): dev3 privileged restarted with B + B2 (aeb_pilot.py md5 8649681d) on both hosts.

### Amendment C (2026-10-20 VUW ~06:20; under the user's instruction to keep finding incompatibilities and run; written before any cheap / perception run with it)
- **Finding (dev2 cheap and a diagnostic route).**
  - The cheap branch commanded HARD on 18% of all ticks; 86% of those had a_gt = KEEP.
  - Diagnostic dev route 9089 (cheap, B, A1, A2), logging the decisive object and saving frames: 1,032 of 4,000 ticks
    were phantom HARD.
    - The object is a real car ahead (median 31.6 m); the ego is nearly stopped (median 0.37 m/s).
    - The braking demand comes from the closing term in 99.9% of them: estimated TTC median 0.95 s, i.e. a fictitious
      closing speed of ~33 m/s.
  - Cause: the box-growth TTC of DEEP's lift is taken between consecutive ticks, dt = 0.05 s at 20 Hz. A 1–2 px jitter
    of a ~50 px box becomes a few-% size change, hence TTC ≈ 1 s. The datasets the lift and the consumer were set on run
    at 2 Hz (nuScenes keyframes, dt 0.5 s) or 5–10 Hz.
  - Rule (a) only measured TTC errors on closing actors, so it could not see these false closings.
- **C:** every perception branch's box-growth TTC (cheap draft, and the branches' own scenes since C1 is off) is taken
  against the scene 10 ticks back, with dt = 0.5 s = nuScenes's frame interval (AEB_TTC_TICKS=10).
  - Fewer ticks of history at a route start → the oldest kept scene, with its true dt.
  - Privileged-AEB uses the annotated TTC and is unaffected, so the running dev3 privileged jobs stand.
  - The cheap dev3 jobs and the test jobs set AEB_TTC_TICKS=10.
- Code: k32_pipes.py (AEB_TTC_TICKS; default 1 = the old behaviour). Check first: the diagnostic route re-run with C.
- 2026-10-20 (VUW ~06:35): C checked on the diagnostic route 9089 (cheap, dev):

  | run | phantom HARD | route |
  |---|---|---|
  | without C | 1,032 / 4,000 ticks (25.8%) | stuck, timed out at RC 47% |
  | with C (AEB_TTC_TICKS=10) | 3 / 493 ticks (0.6%) | completed, RC 100% |

  - With C, of the 152 ticks that required braking, the branch braked on 118.
  - C stands as registered: k32_pipes.py md5 5f4ddfd6; k32_jobs_test.py writes AEB_TTC_TICKS=10.

### Amendment D (2026-10-20 VUW ~08:30; chosen by the user; written before (b) is computed on the complete dev3 run)
- **Finding (dev3 privileged, 75 / 95 routes so far).**
  - Collisions on 19 routes: 11 side, 3 rear, 10 front.
  - Every front contact is a late corridor entry. Privileged-AEB's first braking demand (true states) comes 0.15–0.65 s
    before contact, at 7–9 m/s, while stopping needs ≥ 1.3 s. Examples: a pedestrian emerging between parked cars
    (DynamicObjectCrossing, ParkingCrossingPedestrian); a vehicle crossing while the ego turns at a junction.
  - No further harness incompatibility was found. These are limits of a reactive straight-corridor AEB behind an
    actor-blind route follower.
- **D: (b) counts only collisions preventable by longitudinal control.** A leaderboard collision (matched to the
  agent's contact log by actor id) is non-preventable if:
  - its contact type is side or rear; or
  - the demand lead time L < v / 6 + 0.4 s.
    - L = from the onset of the last continuous run of the privileged demand a_gt ≥ DECELERATE before contact, to the
      contact.
    - v = the ego speed at that onset.
    - 6 m/s² is HARD's commanded deceleration; 0.4 s is the controller's reaction time.
  - Any other collision, including one without a matching contact, is preventable.
- **Gate (b):** privileged-AEB has a preventable collision on ≤ 5% of the 95 dev routes. (c) is unchanged. No scenario
  type is dropped.
- **Test:** every branch reports collisions split into preventable / non-preventable by the same rule. a_gt is logged
  by every run.
- 2026-10-20 (VUW 13:10): dev3 privileged DONE (95 / 95; dispatchers complete at 08:35 / 08:40). Gate (b) under D,
  computed by k32_dev_gate.py --dev dev3 --priv privileged --no_a --preventable (tables dev3_collisions_preventable.csv
  and dev3_gate.csv):
  - collisions on 19 / 95 routes (20%); preventable on 1 / 95 (1.1%). **(b) PASSES.**
  - The only preventable one is route 9055: a collision without a matching contact record, counted preventable
    conservatively. All 10 front contacts have demand lead 0.15–0.65 s < v / 6 + 0.4 s.
  - Cheap dev3 launched for (c): jobs_dev3c_vuw.txt (c0, c1), jobs_dev3c_vuw1.txt (c2, c3); A1, A2, B, B2, C.

### Amendment E — short-horizon outcome analysis (2026-10-20 VUW ~14:00; proposed by the user, idea from DEEP's R2; written before any test run)
- **E1, observational, in every test rollout.**
  - Logging: every perception branch now also logs the cheap draft's action on the same frame (a_shadow_cheap; the draft
    is computed by every branch anyway). aeb_pilot.py md5 405e5eaa.
  - Decision-change event of branch X at tick t: X's frame action ≠ the draft's shadow action. It is *correct* if
    |a_X − a_gt| < |a_draft − a_gt| (a_gt = the privileged requirement on the same tick), *wrong* if >.
  - Outcome over the next H ∈ {1, 2, 3} s (20 H ticks), all from the true states:
    - minimum corridor TTC (the min-TTC definition);
    - a front contact within H, and whether it is preventable (D);
    - the maximum a_gt;
    - the deceleration achieved;
    - whether the ego stops before the decisive actor.
  - Reported per branch: the counts of correct / wrong changes; and the outcome after correct changes vs after wrong
    changes vs after frames without change but with a_gt ≥ DECELERATE.
  - Cross-branch, matched by route, seed and setting: the k-th required episode of a route (privileged a_gt ≥ DECELERATE)
    is aligned across branches. Per episode: onset delay, the outcome over H after onset, and a front contact within H.
  - Paired route bootstrap (2,000, seed 0): v6-S − each branch.
- **E2, forked counterfactuals (as R2).** At a decision-change tick t of a v6-S rollout:
  - restart the route and replay the logged controls up to t;
  - then run v6-S's control vs the cheap draft's control (both with the same perception warm-up) for H = 3 s, and
    compare the same outcomes.
  - This needs CARLA to reproduce a route exactly. It is checked first (two identical privileged runs of dev route 9008,
    jobs_det.txt), and E2 is registered in detail only if the check passes.
- 2026-10-20 (VUW ~14:00): determinism check for E2 FAILED.
  - Two identical privileged runs of dev route 9008 (same seed and settings) diverge from tick 13: max ego position
    difference 4.8 m, speed difference 4.6 m/s; a_gt identical on only 69% of ticks.
  - CARLA (traffic manager / scenario / physics in this setup) does not reproduce a route, so replay-based forks would put
    the two arms in different worlds. E2 is replaced by E2'.
- **E2', model-based forks**, registered before any test run.
  - At each decision-change event of a perception branch X at tick t (E1), take the true states logged at t:
    - per actor ahead (≤ 8 within 60 m, |lateral| < 8 m): near, far, lat_min, lat_max, closing speed, lateral rate
      (rec["act"]; aeb_pilot.py md5 0fb97972);
    - the ego speed.
  - Two arms over H = 3 s (dt 0.05 s):
    - arm X applies the commanded decelerations of X's applied actions on ticks t … t + 60;
    - arm D applies those of the draft's shadow actions on the same ticks.
    - Under KEEP the ego holds its speed.
    - Actors move at constant along-ego velocity (v_t − closing) and lateral rate.
  - Outcome per arm:
    - front contact: an actor overlapping the corridor (half-width 1.2 m) with gap ≤ 2.45 m;
    - the minimum gap and minimum TTC to corridor actors;
    - the final speed;
    - the braking spent (Σ u·dt).
  - Reported per branch and paired per event: share of events where X's decision avoids a contact that D's has, causes
    one D avoids, or makes no difference; the change in minimum gap and in braking spent; route bootstrap
    (2,000, seed 0).
  - Limitation (stated in the report): open-loop; the draft's later actions are its shadow actions on X's trajectory.

### Amendment F (2026-10-20 VUW ~16:40; chosen by the user; written before (c) is recomputed)
- **Gate dev3 as registered** (k32_dev_gate.py --preventable; dev3_gate.csv):
  - (b) passes: preventable 1 / 95.
  - (c) fails on the two metrics I had registered: collision rate cheap − privileged −0.054 [−0.130, +0.011]; mean
    per-route min TTC +0.05 s [−0.10, +0.20].
  - Mechanism: almost every remaining collision is non-preventable (D) and so independent of perception. The preventable
    ones are 1 / 95 for both branches.
- **F:** the user's spec says "(c) cheap differs measurably from privileged". (c) now passes if any K32 outcome metric
  registered for the test has a 95% paired route-bootstrap interval of (cheap − privileged) excluding 0. The metrics:
  - collision rate; preventable collision rate;
  - mean per-route min TTC;
  - missed required episodes per route (a run of the privileged demand a_gt ≥ DECELERATE without an applied
    action ≥ DECELERATE);
  - unnecessary hard-brake onsets per route (applied HARD rising edge with a_gt = KEEP).
  - Computed by k32_gate_f.py → dev3F_gate.csv, which carries the components decision (refiner and C1 off) into
    k32_jobs_test.py.
- The report states that (c) passed only after F, and that the test's discriminating evidence is expected from the
  braking-behaviour metrics and E1 / E2', not from collisions.
- 2026-10-20 (VUW 16:29): F computed (k32_gate_f.py; dev3F_c.csv, dev3F_gate.csv). **GATE PASSED** (b and c):
  - (c) on missed required episodes per route: +0.674 [0.463, 0.905];
  - (c) on unnecessary hard-brake onsets per route: +12.2 [8.5, 16.7];
  - collision, preventable collision and min TTC: n.s.
- **TEST LAUNCHED** (once): k32_jobs_test.py with K32_GATE=dev3F_gate.csv → 155 routes, 312 jobs. Every line carries
  AEB_C1=0 AEB_REF=0 AEB_FRONT=2.45 AEB_TTC_TICKS=10; aeb_pilot.py md5 0fb97972 (E1 / E2' logging).
  - Dispatchers: jobs_test_vuw.txt on VUW and jobs_test_vuw1.txt on VUW1, MAXJ 4 each.
- 2026-10-20 (VUW 16:36 / 16:52): dispatchers restarted (only the dispatcher PIDs; running jobs untouched) with MAXJ 6, then
  8 on VUW and 7 on VUW1 (CPU about 3 cores per job; GPU about 6.5 GB per job). Test jobs are 20 routes each and take
  about 2.5 h, so 312 jobs need about 60 h at 13–15 concurrent jobs.
- 2026-10-20 (VUW 17:40): test analysis k32_test.py rewritten for amendments D, E (E1 / E2') and F (no test number read).
  - Outputs: test_{per_route,collisions,summary,intervals,shadow}.csv, e1_events.csv, e1_episodes.csv, e1_intervals.csv,
    e2_forks.csv, e2_intervals.csv.
  - Check on dev3 (privileged + cheap, symlinked as a test layout, outputs deleted afterwards): it reproduces
    dev3F_c.csv: cheap − privileged collision −0.053; preventable 0; unnecessary hard onsets 12.2 per route.
  - E1 / E2' code exercised on a synthetic route (dev3 predates the shadow logging): event classes, H-window
    outcomes, episode alignment and the two-arm forward model behave as specified.
  - The preventable rule is the gate's code path: a contact after the demand run has ended counts as non-preventable.
- 2026-10-20 (VUW 18:21): VUW load 46 on 32 cores (8 K32 jobs + the K33 spur re-run + another user's job). To keep K32's
  CPU priority, the K33 spur processes and their chain shells were reniced to +10 on both hosts (exact PIDs; nothing
  killed). Test: 5 / 312 jobs complete.
- 2026-10-20 (VUW 20:25): **work sharing between the hosts** (user: "VUW1 có vẻ trống, chia sẻ sẵn K32 sang đó, tối ưu
  thời gian đi").
  - k32_dispatch.sh takes the other host's jobs file as a second argument. Every launch first claims the job (atomic mkdir
    under results/k32/claims, owner = host); a job claimed by the other host is skipped. Once its own file is launched,
    done or claimed elsewhere, a dispatcher takes the other file's unstarted, unclaimed jobs, so neither host idles while
    the other still has a queue. Jobs that had already started were pre-claimed by their host.
  - VUW: MAXJ 8, MINF 9000 (CPU-bound: about 4 cores per job). VUW1: MAXJ 9, MINF 8000 (it had about 10 idle cores
    and 9.8 GB free on GPU 1; the largest CARLA + evaluator pair seen is about 8 GB).
  - Only the dispatcher PIDs were stopped (VUW 611546; VUW1 459428 and its two subshells); running jobs were untouched.
  - Remaining at the restart: VUW 138 unstarted, VUW1 141. 22 / 312 complete.
- 2026-10-20 (VUW ~21:00): **interim look at the test, on the user's request** ("xem những cái đã chạy xem kết quả có
  khả quan không"). k32_test.py --tag interim on the routes finished so far: seed 0, 21–58 routes per branch, about 7% of
  the test. Outputs are tables/interim_*.
  - Nothing in the protocol, code or job list changes because of it. The final analysis runs once the test completes.

### Amendment G — test branches and metric hierarchy (2026-10-20 VUW ~21:40; the user's spec "K32 branches and metrics for the TEST run"; option chosen by the user: keep the running test and add; written before any G run or code change)
- **Disclosure.**
  - The spec asked to be fixed before the test started. The test started at VUW 16:29: about 25 / 312 jobs complete.
  - An interim look at the test (seed 0, 21–58 routes per branch) was taken at ~21:00 on the user's request.
  - So the metric hierarchy below is registered **after** an interim look, and the report must say so.
  - In that look, v6-S's onset delay vs privileged-AEB was +0.54 s (L0) / +0.54 s (L1), n.s.: no favourable signal
    that this choice could be exploiting.
- **G1, new branch: uncertainty crops + comp.**
  - A port of K28's unc2: crops around the draft's detections, lowest confidence first (ties: higher entropy), context
    2.0.
  - Filled into the per-frame budget = v6-S's crop px on the same frame (×1.0; v6-S's selection runs inside the branch
    only to obtain the budget, as for the hybrid).
  - The same crop machinery: crops at 960 / native, guard, K28 merge.
  - "comp." = what rule (a) left on for every branch: T4 looming, no refiner, no C1.
  - Jobs: L0 and L1 × seeds 0, 1, 2 × chunks c0–c7 (48), appended to the shared queue.
  - The running branches' code paths stay as they are; the old and new code are diffed before launch.
- **G2, latency (L1).**
  - Fit t = a + b · detector_calls + c · crop_px (OLS; crop px in megapixels at 1600 × 900) on the Jetson measurements
    behind latency_samples.csv:
    - K29 r1–r3: v6-S and v6-S ≤ 2, with their calls and px;
    - K30 J1 j1x r1–r3: hybrid tsp8S; calls = 1 if it has crops, else 0.
  - The uncertainty branch's L1 latency: t = a + b · calls + c · crop_px + ε. ε is drawn from the fit's residuals,
    seeded per route and seed like the existing draws.
  - v6-S, v6-S ≤ 2 and the hybrid keep their registered, already running, measured distributions. For them the model is
    validated and reported as a sensitivity check: fit error (RMSE, R²), and the modelled vs measured mean / p95 /
    apply-tick delay ceil(t / dt) per branch.
- **G3, metric hierarchy** (analysis only):
  - **Primary:** braking-onset delay relative to privileged-AEB on avoidable events, paired by route and seed.
    - Event = the k-th required episode of a route (run of privileged demand a_gt ≥ DECELERATE on the true states) in
      the branch's rollout, aligned with the k-th of privileged-AEB's rollout of the same route and seed.
    - Avoidable = at the episode onset, the true corridor min TTC ≥ v / 6 + 0.4 s (D's rule; no corridor TTC = avoidable).
    - Onset delay = the branch's first applied action ≥ DECELERATE after the episode onset, minus privileged-AEB's (from
      their own onsets). A miss (no such action within the episode) is censored at the episode end and also reported as a
      miss rate.
  - **Secondary:**
    - avoidable (= D-preventable) collisions per route;
    - unnecessary hard-brake episodes per km (§4);
    - the min-TTC distribution (per-route minimum; quantiles 5 / 25 / 50 %).
  - Also reported: all collisions by type (D) and every §4 / E / F metric.
  - Stratum (pre-registered here): ego speed at the event onset ≥ 8 m/s vs < 8 m/s.
  - Bootstrap: paired route × seed (unit = a (route, seed) pair present in both branches; 2,000, seed 0). The §4
    route-unit intervals are also kept.
- **G4, decision fidelity (shadow mode).**
  - Every perception branch (cheap, u960, native, hybrid, uncertainty, v6-S, v6-S ≤ 2) runs on the frames of v6-S and
    u960 rollouts; the applied action stays the rollout branch's (the existing AEB_LOGALL mechanism).
  - The test rollouts recorded no frames, so the shadow frames come from extra rollouts of v6-S and u960 with logging
    of all branches: L0, seed 0, chunks c0–c7 (16 jobs). The scope is set by cost; per-frame decisions do not depend on
    the latency setting.
  - Reported: per-frame agreement with u960 + comp., and its temporal pattern (flicker, disagreement run lengths,
    onset delay vs F).
- **Not added:** AnA-style (K33-D). Its implementation was finished, but the spec includes it only if done before the test
  starts. FOVEA, LZU, SAHI, corridor-only and random are not ported, by design (as the spec states).
- 2026-10-20 (VUW ~21:45): amendment G implemented.
  - **Code** (aeb_pilot.py md5 ff9df1c0, k32_pipes.py md5 6b3f801b): the "unc" branch and its G2 latency.
    - Every existing branch's code path is unchanged (diff reviewed). The only edits are:
      - a new `if "unc"` block;
      - "unc" added to the sets that trigger v6-S's hidden budget loop and its history push;
      - the raw draft kept in a variable, so its entropy can break ties.
    - Jobs launched before this keep the old code in memory; jobs launched after it run the new code, which behaves the
      same for those branches.
  - **G2 latency model** (scripts/k32_latency_model.py → results/k32/latency_model.json, latency_model_validation.csv):
    - The fit frames are exactly those of latency_samples.csv (sorted ms identical per branch).
    - t = 8.56 + 4.88 · calls + 116.7 · crop Mpx (ms); RMSE 8.0 ms; R² 0.82; 7,245 frames.
    - Modelled (with the residual draw) vs measured:

      | branch | mean (ms) | p95 (ms) | 1-tick apply delay |
      |---|---|---|---|
      | v6-S | 25.4 vs 21.8 | 61.7 vs 59.1 | 88.8% vs 90.6% |
      | v6-S ≤ 2 | 18.6 vs 19.7 | 48.2 vs 46.0 | 92.8% vs 96.9% |
      | hybrid | 21.7 vs 24.0 | 66.8 vs 74.9 | 87.6% vs 86.0% |

    - A modelled latency is clipped below at 5.49 ms, the smallest measured.
  - **Smoke** (dev routes 9008 + 9028, unc, L1), route 1: 1,108 ticks, no error.
    - Crops on 47% of frames; px mean 0.088.
    - Modelled latency mean 19.3 ms, p95 52.3 ms.
    - The draft shadow action (E1) is logged.
  - **G1 jobs:** 48 unc jobs appended to the shared queue (k32_jobs_g.py unc; backups jobs_test_vuw*.before_G.txt). Their
    variables are copied from the running lines: AEB_C1=0, AEB_REF=0, AEB_FRONT=2.45, AEB_TTC_TICKS=10.
  - **G4 shadow jobs:** inserted after the logall smoke passes (it is waiting for GPU memory).
  - **G3 analysis:** k32_test.py (g3_primary*.csv, intervals_route_seed, min-TTC quantiles) and k32_shadow.py (G4) are
    written. The G3 code was exercised on dev3 (cheap vs privileged): 89 units, 3,440 avoidable paired events.
    - "Avoidable" is judged on privileged-AEB's episode (the reference world), and the speed stratum uses its onset
      speed. This was fixed before any G3 computation on test data.
- 2026-10-20 (VUW ~22:00): G4 logall smoke (dev 9008, v6-S rollout): 1,160 ticks; all 7 branches logged per frame (unc
  included); no error. The pilot names the file aeb_<branch>_logall_<setting>.jsonl.
  - k32_shadow.py was checked on it, and fixed before any test data:
    - u960 braking runs closer than 1 s are merged into one episode (§4's rule);
    - the onset window starts 1 s before an episode, but not before the previous episode's end (u960 vs itself = 0).
  - The 16 shadow jobs (v6-S on VUW's file, u960 on VUW1's) were inserted first among the unstarted lines (backups
    *.before_G4.txt).
  - The queue is now 188 + 188 lines: 312 test + 48 unc + 16 shadow.
- 2026-10-20 (VUW 22:20): VUW1's dispatcher launched shadow/v6S c0–c2 from VUW's file; claims stop any job from running on
  both hosts.
  - Cause: the job files were replaced from VUW while VUW1's dispatcher was reading its own file. Over NFS the old file
    handle went stale on VUW1, so that pass ended early and the dispatcher went on to the shared-file loop. Its next pass
    re-reads its own file.
  - No duplicate run and no loss: work moved between the hosts, as the sharing allows.
  - From now on, job files are only edited on the host whose dispatcher reads them, or with both dispatchers stopped.
- 2026-10-20 (VUW 22:47): VUW's dispatcher restarted with MAXJ 10 (dispatcher PID 1155753 only), on the user's note that
  VUW's GPUs were free. Other users' jobs had ended: GPU 1 had 19 GB free and the load was 26 / 32.
- 2026-10-20 (VUW 22:55): on the user's request to fill the GPUs' memory, both dispatchers were restarted (dispatcher PIDs
  only): VUW MAXJ 11 and MINF 8000; VUW1 MAXJ 10 and MINF 8000.
  - Per job: CARLA 4.3–7.4 GB + evaluator 0.46–0.77 GB, about 8.2 GB at most.
  - Free memory at the restart: VUW 7.2 / 12.3 / 4.3 GB, VUW1 3.6 / 11.0 / 4.5 GB; so about one more job per host now,
    and more as other users' jobs end.
  - VUW's CPU is near saturation (load 35 / 32 with 8 jobs).
- 2026-10-21 (VUW 02:20): two evaluators had exited after a simulator failure, leaving incomplete checkpoints:
  - shadow/v6S_L0_s0_c0: simulator time-out on route 1, at 23:48;
  - test/hybrid_L1_s0_c3: CARLA segfault after 7 / 19 routes, at 01:40.
  The leaderboard records the crashed route as "Failed - Simulation crashed" and resumes after it. Across 1,250 test /
  shadow route records there is 1 such crash. "Failed - TickRuntime" is the route time-out, an outcome: it hits the
  same routes in every branch, as in dev3.
  - The old dispatcher would relaunch them only after a full pass over its file, i.e. hours later.
    - shadow/v6S c0 was resumed by hand on its claim owner (VUW1).
    - k32_dispatch.sh v3 makes one launch per scan, and each scan resumes this host's incomplete started jobs before
      starting new ones. Deployed on both hosts (dispatcher PIDs only; same MAXJ / MINF).
  - hourly_check.sh now flags any job whose evaluator exited more than 30 min ago with an incomplete checkpoint.
  - Status: 56 / 360 test jobs complete, 16 / 16 shadow jobs started (11 complete). Throughput over the last 8 h: about 7
    jobs per hour while the shadow (logall) jobs held many slots.
- 2026-10-21 (VUW ~03:50): **unattended finish set up** (user: "sắp hết quota tuần, khả năng bạn thiết lập để đưa kết quả
  về kit sau khi chạy xong nhé, tự theo dõi k32 nốt").
  - The kit's crontab runs remote/ops/k32_auto.sh at :07 and :37. Each run:
    - keeps one dispatcher v3 alive per host (VUW MAXJ 11, VUW1 MAXJ 10, MINF 8000); a dispatcher that logged "K32
      DISPATCH COMPLETE" is not restarted;
    - computes partial tables on VUW every 6 h;
    - computes the final tables (k32_test.py --tag test, k32_shadow.py, k32_autoreport.py) once all 376 jobs are
      complete, or after 2 h with no K32 job process on either host (a stall), and writes results/k32/ANALYSIS_DONE;
    - pulls results/k32/tables into handoff/K32/auto/ (REPORT_auto_<tag>.md, status.txt);
    - after the final pull, writes auto/FINAL_PULLED and a line here, and every later run is a no-op.
  - Log: logs/k32_auto.log on the kit.
  - Remove the crontab line after FINAL_PULLED.
  - k32_autoreport.py is a table dump only: per-branch summary, G3 primary and its intervals, secondary route × seed,
    every metric where v6-S is significantly worse, E1, E2', shadow, the latency model, collisions by type.
    REPORT.md is still to be written by hand.
  - The Claude session's hourly check was cancelled; the kit cron replaces it.
  - First test run (cron-like environment): todo 295 / 376; partial tables pulled.
- 2026-10-21 (kit 07:45–08:05): **the unattended partial analysis failed, and a route-matching flaw was found and fixed.**
  - Symptom: k32_test.py stopped in the route × seed bootstrap ("shapes (237,) (238,)") because some units had two rows.
    k32_auto.sh then retried the failed partial run every 30 min; now a failed partial run waits 6 h like a good one.
  - Cause: logged route lines were matched to routes by the nearest start point (route_of, < 30 m). 30 test route pairs
    and 9 dev pairs start < 30 m apart, so lines could land on the wrong route or on the same one twice. A resumed job
    can also log a route twice.
  - Fix: k32_test.match_lines matches each job's lines in its chunk's run order (test_routes_c<k>.xml), each to the next
    route whose start is within 30 m; a re-logged route keeps the longer line. k32_shadow.py uses the same.
    - Check over all 193 test / shadow jobs: 3,600 lines matched, 0 to a route outside the job's checkpoint records, 2
      records without a line (the two crashed routes).
    - The old matcher had 187 wrong or duplicate assignments.
    - The interim tables of 2026-10-20 ~21:00 used the old matcher; they are superseded.
  - **Dev gate re-checked with the new matcher:** (b) privileged preventable collisions 1 / 95 (unchanged); (c) cheap −
    privileged:
    - missed required episodes +0.684 [0.463, 0.916] (was +0.674 [0.463, 0.905]);
    - unnecessary hard onsets +12.24 [8.56, 16.64] (was +12.21 [8.46, 16.65]);
    - collision −0.053 [−0.116, +0.011] (unchanged).
    The gate outcome is unchanged. The old matcher had lost 3 / 95 lines per branch on dev.
  - Partial analysis re-run without error; the tables were pulled into auto/.
- 2026-10-21 (VUW1 14:53): stuck job reported by the user: test/v6S_L1_s0_c5 at 13 / 19, "Loading the world" for
  RouteScenario_24340, log silent since 04:24 (10.5 h).
  - The same route completed in 18 other jobs, so the hang was transient.
  - Evaluator 2439043 and CARLA 2439078 / 2439085 were stopped by PID (the CARLA binary needed SIGKILL). The dispatcher
    v3 resumed the job at 14:53 (attempt 2).
  - k32_auto.sh now also stops any job whose evaluator is alive but whose log has been silent ≥ 60 min, by PID: the
    evaluator, its CARLA children and grandchildren. The dispatcher then resumes it. A dry run on both hosts flagged
    nothing.

- 2026-10-21 (kit 09:15): hourly read-only health reporting added at the user's request ("mỗi giờ bạn check cho tôi").
  - Cron at :10 each hour runs remote/ops/k32_health.py; existing :07/:37 k32_auto.sh recovery stays unchanged.
  - Checks both hosts: dispatcher/evaluator counts, duplicate evaluators, incomplete orphan jobs, live evaluators silent >=45 min, checkpoint stagnation >3 h, recent log errors, SSH failures, stale automation and premature FINAL_PULLED.
  - Latest report: handoff/K32/health/latest.md (+ JSON); timestamped history and alert log: logs/k32_health/.
  - First live check: 188/376 jobs complete, 19 incomplete, 169 without checkpoint; no detected anomalies.
  - No chat notifications are configured. The reporter becomes a no-op only after it observes every registered job complete and FINAL_PULLED present.

- 2026-10-21 (kit 12:50–12:55): user-requested interim result update. Existing k32_test.py / k32_autoreport.py run with tag snapshot_20261021_1250 (CPU, nice 15, one BLAS thread); no experiment or protocol changes. 4,143 route runs, 387,262 E1 rows, 144,860 E2 forks, 45,643 episodes. Report and summary/interval CSVs mirrored into snapshot_20261021_1250/. Still partial: uncertainty test branch absent, seed 2 incomplete.

- 2026-10-21 (kit 23:25): completed-result audit of 321 jobs: 6,222 checkpoint route records, 6,220 matched usable route logs. Two missing units in test/v6Sb2_L1_s1_c3 (routes 2082, 2144) failed agent setup with CUDA OOM and are omitted from the analysis. Job completion is not sufficient to certify full route coverage. See AUDIT_20261021_2325.md and logs/k32_audit_20261021_232422/audit_verified.json. No experiment data was changed by the audit.
- 2026-10-21 (kit 23:33): user-requested snapshot analysis completed with tag snapshot_20261021_2333: 6,057 route runs, 568,557 E1 rows, 210,585 E2 forks, 67,074 episodes. Its report was subsequently pulled by the existing automation. No method/protocol changes.
- 2026-10-22 (kit 06:54–06:57): 374/376 jobs complete; last two unc jobs stalled with dead CARLA children while Python continued emitting timeouts.
  - test/unc_L0_s1_c0 on cuda12: checkpoint 10/20, CARLA crashed during simulation; original evaluator 3917949 stopped by exact checkpoint/PID verification.
  - test/unc_L0_s2_c4 on cuda13: no checkpoint, CARLA render-thread startup timeout; original evaluator 3675014 stopped similarly.
  - Original logs and existing checkpoint copied to timestamped recovery backups before SIGTERM. Dispatcher retains responsibility for retry; no checkpoint reset or route outcome replacement.
  - remote/ops/k32_auto.sh now also recognizes its evaluator's zombie CARLA child when no live CARLA serves the evaluator's RPC port. This covers repeated timeout output that prevented the old >=60-minute-silent-log condition from firing. Existing cron schedule remains in place. Shell syntax and six process-table fixtures passed (dead/healthy/unrelated child, live simulator guard, exact port matching, no CARLA child).
  - Read-only checkpoint status scan now finds five zero-duration agent-setup failures: the two v6Sb2 routes above; unc_L0_s1_c0 routes 1711 and 2127; unc_L0_s1_c3 route 2082. These are missing simulation observations even if job progress later reaches its total. The new unc failures' exact setup cause has not yet been audited. Do not equate 376/376 job completion with a complete usable dataset until these five units are accounted for.
  - Kit 07:01 update: unc_L0_s2_c4 is producing simulation ticks after restart. unc_L0_s1_c0's retry on GPU 1 failed setup for route 11755 with CUDA OOM while a separate Python process occupied about 16.3 GiB; dispatcher next launched it on GPU 0 at 07:00:43. Checkpoint is 11/20, but the increase includes that failed setup, not another usable route. Total known zero-duration setup failures is now six. Other workloads were not modified.

- 2026-10-22 (kit 09:34): all 376 registered jobs finished (7,285 checkpoint records). Six infrastructure-failed route units were rerun with original route XML, seed and configuration, then validated and merged into three original jobs. Four retries completed the route; two unc retries ended with legitimate Agent got blocked outcomes, retained unchanged. All original files were backed up before replacement; official checkpoint global statistics were recomputed. Remote provenance: results/k32/repair/20261022_complete/MERGED.json; local copy: repair/MERGED.json.
  - A worker claim race produced two valid attempts for unc seed 1 route 11755. The first valid attempt in chronological order is selected, independent of metrics; both raw attempts are preserved. See repair/DUPLICATE_ATTEMPT_REVIEW.json. Worker now rechecks DONE after acquiring the claim; a regression check simulating completion between the pending snapshot and claim passed.
  - Full post-merge data audit is running; finalization remains guarded until route coverage and data consistency are verified.
- 2026-10-22 09:45:53 (kit time): **K32 test complete** (jobs not complete: 0). Final tables were pulled by k32_auto.sh into auto/ (REPORT_auto_test.md). REPORT.md is still to be written by hand.

- 2026-10-22 (kit 09:46): full post-merge audit passed: 6,975/6,975 test and 310/310 shadow route logs; no structural/data-consistency errors. Final analysis and local pull succeeded. Final test table has 6,975 unique keys and all 45 configuration × seed groups contain 155 routes; shadow has 2,170 route × branch rows. Eight final table/report hashes match the server. All six infrastructure failures are repaired, with provenance and original backups retained. See AUDIT_20261022_FINAL.md, repair/COMPLETE.json and auto/REPORT_auto_test.md. Distance-normalized metrics retain 185 near-zero-distance observations; no outcome-based exclusion.
- 2026-10-25: **K32 REPORT.md written (FINAL)** from the final tables pulled on 2026-10-22 (auto/), as requested on
  2026-10-22. Paper kit (03_results.md, 06_pending.md) updated.
