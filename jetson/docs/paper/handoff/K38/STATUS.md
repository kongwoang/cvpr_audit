# K38 STATUS

| item | state |
|---|---|
| K38-1 in-loop deadlines (Jetson) | DONE 2026-10-26 (K38-1/REPORT.md) |
| K38-2 probe changes together (VUW) | DONE 2026-10-26 (K38-2/REPORT.md) |

## Log
- 2026-10-26: K38-1 harness.
  - deadline.py = verbatim speculative.run / bounded_keep plus stop / mark hooks; 88_k38.py from the K35-4 K29 harness.
  - Protocol and the AV2 40-log list registered (K38-1_protocol.md).
  - Dry run (10 frames): the hooks work. Round 1 can itself take ~55 ms, so deadline overruns are expected and are
    reported.
- 2026-10-26: K38-2.
  - k30_probe.py (sha256 e7cdb7b3…): v6-S + v_close 10 + extended grid + PED / CAR / TRUCK probes.
  - Smoke scene-0110: base rows equal K28 (0 differences); px 0.096 → 0.221.
  - Test launched 22:37 NZ.
- 2026-10-26 16:40 (kit): K38-1 window, attempt 1.
  - Dev calibration done: medians.json, sha256 ebbecacd…, copied to K38-1/.
    - nuScenes v6-S round 1: 18.6 ms; B2K-A calls: 8.2 / 12.4 ms.
  - Smoke parity nuScenes: 0 differences.
  - The window failed at the AV2 smoke: a NameError in the av2t40 loader (no test deadline run had started).
  - Fixed, dry-run OK. Attempt 2 reuses the fixed medians (hash-checked) and restarts from the smoke. Attempt-1 logs:
    logs_attempt1/.
- 2026-10-26: **K38-1 DONE.**
  - Parity: staged vs reference 0; D = ∞ vs K34-3 0.
  - The in-loop rule beats hold at D = 50 on AV2: +0.18 G (significant). B2K-A@D is the best DESP form.
  - u960 + C1 + v3a has a lower Jetson p95 (31 / 47 ms) than v6-S (57 / 76 ms). It ties on nuScenes and leads on AV2
    under deadlines (n.s.).
  - So a wall-clock latency advantage over whole-frame is NOT supported; px / compute remains the advantage.
- 2026-10-26: **K38-2 DONE.** nuScenes high-tercile coverage 0.80 → 0.997 at ×2.4 px; ΔG +0.031 / +0.031 (n.s.) / −0.092 (significant, Argoverse-HD). Checks 0. **All K38 items complete.**
- 2026-10-26: **Code-release task received** (the user's TASK_ID "K38"; kept in handoff/K38/release/ because K38-1 / K38-2
  already exist).
  - Remote github.com/anonymous221206/DESP exists and is empty (git ls-remote: no refs).
  - The kit's gh is logged in as anonymous221206 (active). The push will use HTTPS via gh, configured repo-locally.
- 2026-10-27 (release): package desp/ written (a refactored port of the frozen code path, kit /tmp/desp_release;
  VUW export copy CVPR/tmp/desp_release).
  - Unit tests: 102 passed (synthetic; includes the locality theorem with an oracle detector).
  - First regression smoke, workspace venv, vs the frozen records: 0 differences on all 1,060 subset frames.
    - DESP: action, no-refiner action, draft action, px, rounds, crops, guard counts, and crop rectangles
      (nuScenes vs the live-draft reference; AV2 and Argoverse-HD vs K28).
    - u960 + C1 + v3a: action, no-refiner action, px.
  - Reference runs: nuScenes test live-draft instr (4 shards) and the --capture subset (relcap), launched 2026-10-27 04:57
    VUW. The first launch of the live-draft run had silently failed under zsh word splitting; it was relaunched through a
    bash script.
  - Fresh venv from requirements.txt built (CVPR/tmp/desp_env).
- 2026-10-27 (release): gate C, fresh environment (built from requirements.txt).
  - Unit tests 102 passed. Every script's --help returns 0 (training/common.py and kitti.py are modules, not entry
    points).
  - reproduce_main.sh --subset is queued; the training-pipeline smoke (1–2 units, 1 epoch) is queued.
  - First anonymity grep: only '@' (Python operators / decorators) and 'jetson' (the task-specified time_jetson.py);
    no names, paths, hosts or task IDs.
- 2026-10-27 (release): **gate C PASSED** (fresh env). reproduce_main.sh --subset rc 0 (≈ 9 min). Its per-frame outputs
  vs tests/regression/expected.pkl (md5 e85c1a35…): nuScenes 80 / 80, AV2 79 / 79, Argoverse-HD 901 / 901 frames
  identical (actions, crops, grown crops, rounds, px, u960 actions / px); 0 differing. expected.pkl itself = the relcap
  capture, cross-checked against the live-draft (nuScenes) / K28 (AV2, AHD) records, 0 differences.
  - Full-split G from the reference records: nuScenes live draft v6-S 0.178 (u960 + C1 + v3a 0.077; paper 0.211 / 0.112
    with the TRT draft cache); AV2 0.294 (0.338); AHD 0.418 (0.297). README filled.
  - Training smoke attempt 1 failed: two refiner split files not synced to VUW. Attempt 2 failed: VUW has KITTI images
    only for sequences 0007–0012, so the first two KITTI units gave 0 samples (empty arrays with the wrong rank).
    Fixed: extract_refiner_data.py now raises if a KITTI image dir is missing, keeps the array rank when empty, and has
    --units. Smoke attempt 3 uses KITTI 0010 0011 (train).
  - Anonymity grep (full term list + IPs, /vol/, /tmp/, ssh, DEEP, github.com): hits reviewed by hand. All benign:
    '@' (matrix operator / decorators); "bootstrap" (matches 'rap\b'); "recs." (matches 'ecs\.');
    opencv-python==5.0.0.93 (matches the IP regex); DEEP cited only as arXiv:2609.35910 (no repository URL); jetson
    only as the device name in time_jetson.py / export_tensorrt.py / README; github.com only in the release URL, the
    Ultralytics asset URL and s5cmd. No images, PDFs or notebooks; expected.pkl has no path strings.
- 2026-10-27 (release): **RELEASE DONE.** Full pytest in the fresh env: 105 passed (1,060 / 1,060 identical).
  Training smoke rc 0 (KITTI 0007; train_looming now saves the last epoch on an empty hold-out).
  Commit 1c0bb32e471ab80bf37d70272ca0c90627d662ca "Initial release"; author and committer Anonymous only.
  Pushed via HTTPS as anonymous221206 to an empty remote. Release v1.0: looming_c1.pt, refiner_v3a.pt, MD5SUMS.
  Public clone → fresh env → weights from the Release → pytest 105 passed, 1,060 / 1,060 identical.
  REPORT: handoff/K38/release/REPORT.md.
