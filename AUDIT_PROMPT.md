# Independent code and pipeline audit — DGSP (CVPR 2027 submission)

## Your role
You are an independent senior reviewer of research code. Audit the repository at tag `frozen-v6S-audit` (branch `audit/2026-11-11`) together with the attached `main.pdf` and `supplementary.pdf`. **Read only.** Do not edit, refactor, re-tune, or re-run on test data. Your output is a findings report. Start from `AUDIT_MANIFEST.md` and `AUDIT_ENTRYPOINTS.md`.

The goal is the truth about the method and its evaluation, not a favourable verdict. A confirmed absence of problems is a valid result; so is a finding that weakens the paper. Do not invent issues to appear thorough, and do not soften real ones.

## Context in five lines
DGSP runs a cheap detector at 320 px, asks the planner where an unseen object would change the action (decision-sensitivity field) and which detections the action rests on (support), verifies those crops at 960-px scale, and iterates. A "lift" (monocular geometry, learned looming, range refiner applied once after the loop to verified support objects) turns boxes into states; every baseline gets the same lift. The reference F is whole-frame u960 with that lift. Known facts you should not rediscover but should explain: (1) whole-frame u512 is as faithful to F as DGSP at similar pixels, faster, and uses less energy; (2) DGSP misses more required-braking episodes than u320 at the deployed threshold; (3) the refiner never changes a KEEP frame (support is empty at KEEP) and lowers the action far more often than it raises it; (4) agreement with F drops to 64–79 % on frames where F's action changes; (5) G of u640 on nuScenes is negative while u512 is positive.

## What to audit (answer each; cite file:line)
1. **Correctness of the loop vs. the paper.** Field evaluation (grid, probe, projection, footprint), support rule, crop construction and context, the 95 % `covered` stopping test, merge/dedup, truncation guard, round cap. List every place where code and Sec. 3 / Supp. S1–S2 differ, and whether the paper states it.
2. **Refiner.** Where and on which objects it runs, for DGSP and for each baseline; training data and any leakage into dev/test; whether its asymmetry (fact 3) is inherent or an implementation choice; what a symmetric, principled variant would be and what it would cost. Do not implement it.
3. **Temporal estimation.** How looming / TTC history is kept for objects detected in crops vs. whole frames, across changing crop sets and scales; association rules; what happens on first appearance. Is fact (4) explained by history resets? Is any baseline advantaged or handicapped by the history rule?
4. **Crop verification and locality.** Preprocessing differences between crop and whole-frame inference (resize, letterbox/padding to 32, normalization, confidence/NMS thresholds, class filtering, coordinate mapping, clipping at image borders). Anything that makes V(R) ≠ F|_R beyond what the paper lists.
5. **Baseline fairness.** For u512/u640/u960/native, AdaScale, routers, Chanakya-style, SAHI, FOVEA, UFP, AnA-style, counterfactual, crop heuristics, support + fixed tube: same detector weights and thresholds? same lift and history rule? tuning grid and what split it used? pixel accounting and caps computed identically? Any baseline that is a weakened re-implementation? Any that receives DGSP's support without disclosure?
6. **Evaluation validity.** Loss and G (Eq. 5; note the "previous action = cheap pipeline" convention), annotated demand and decisive object, episode definition, bootstrap units and pairing, Holm/BH usage, dev/test/held-out separation in code (search for any test-split read inside a selection or training script), seeds, and whether each number in main Tab. 1, 2, 3, 6 is reproduced by the script the manifest names. Explain fact (5).
7. **Efficiency.** Timing harness fairness (warm-up, engine shapes, batching, what is included in latency and energy for each row), sources of DGSP's tail latency, and concrete, action-preserving optimizations with an estimate of the gain.
8. **Algorithmic opportunities.** At most five, ranked, each with: mechanism, why the evidence in the repo suggests it helps (cite stored results), expected effect on fidelity / missed / false / latency, what it does to Theorem 1's conditions, and the smallest dev-only experiment that would test it. In particular assess: why u512 matches DGSP (is it the 80 m cut-off and corridor geometry, the lift dominating G, or selection errors?), and whether there is a regime, definable without looking at test outcomes, where selective verification should beat uniform downscaling.

## Rules
- Every finding needs evidence: file path and line, or a stored result file and row. Mark anything you could not verify as "unverified".
- Classify each finding: **Bug** (code does not do what the authors intend), **Protocol risk** (validity, leakage, fairness), **Paper–code mismatch**, **Methodological opportunity**, or **Checked, no issue**.
- Severity: Critical (changes a headline number or claim), Major, Minor. For bugs, estimate which tables are affected and in which direction; do not guess magnitudes.
- Keep bug fixes separate from method changes. Never propose selecting anything on test or held-out data. The frozen configuration and stored results stay as they are; proposals are for dev-only studies under a pre-registered protocol.
- No rewriting of the paper; you may note where a claim is not supported by the code or results.

## Deliverables
1. `AUDIT_REPORT.md`: one-paragraph verdict; table of findings (ID, class, severity, location, evidence, effect, recommended action, effort); the eight sections above; a list of what you checked and found correct.
2. `REPRO_CHECK.md`: for each main-paper table, the script and inputs that produce it and whether your reading of the code agrees with the reported protocol.
3. `PROPOSALS.md`: the ranked opportunities of item 8 with their dev-only test designs.
4. `QUESTIONS.md`: what you need from the authors to resolve unverified items.
