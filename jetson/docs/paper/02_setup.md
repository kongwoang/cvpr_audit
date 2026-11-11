# Experimental setup

## Decision value G (define in the paper)
For a consumer (planner) and a perception pipeline X, per frame the decision cost J (DEEP's cost: wrong action vs the
action under ground-truth objects, with action-change/history terms) is computed; the all-cheap pipeline is the
reference. G(X) = Σ_frames (J_cheap − J_X) / Σ_frames J_cheap. Positive = fewer/cheaper wrong decisions than the cheap
draft; negative = escalation made decisions worse. Also reported: "harmed share" (frames where X's cost exceeds the
cheap cost, among affected frames). Action history = the cheap pipeline's previous action (DEEP's convention).
Consumers: braking controller (DEEP's; max-composable), Planner B (argmin over candidate trajectories), PKL's planner.
Geometry: "mono" = monocular ego-frame lift of each 2D box; "oracle" = ground-truth range for matched boxes.

## Benchmarks and splits (tuning only on dev; test run once)
| benchmark | data | dev (tuning) | test (reported) | metric |
|---|---|---|---|---|
| nuScenes | CAM_FRONT keyframes, 2 Hz | new val, 152 scenes | **new test, 228 scenes / 9,181 frames** (scenes outside DEEP's 85) | G (braking, mono) |
| Argoverse 2 sensor | ring_front_center 1550×2048, 5 Hz | AV2 train (700 logs; confirmatory 3) | **AV2 val, 150 logs / 11,818 frames** | G |
| KITTI tracking | image_02, DEEP's split | train + val, 15 seq / 4,472 frames | test, 6 seq / 3,536 frames (wide CIs) | G in 4 DEEP cells |
| NAVSIM v1.1 | CAM_F0, PDM-Closed privileged on map/route | OpenScene mini (= nuPlan-mini logs) | **navtest, 12,146 scenes** | PDMS |
| nuPlan-mini | CAM_F0, 1,440 states | (same logs as NAVSIM dev → dev evidence only) | — | NAVSIM-v1 PDMS |
| Argoverse-HD (pending) | 1920×1200 @ 30 Hz, 24 seq / 15,062 frames | — (AV2 config unchanged) | val | AP (3 road-user groups), G, sAP |
| Bench2Drive (pending) | CARLA 0.9.15, 220 routes, front camera 1600×900 | — (nuScenes config unchanged) | 220 routes | Driving Score, Success Rate |

Perception-in-the-loop convention for planners (NAVSIM, nuPlan, Bench2Drive): eligible GT objects projected into the
front camera are kept only if matched by a detection (Hungarian, class-compatible, IoU ≥ 0.3); unmatched in-camera
objects are removed; unmatched detections become static objects on the ground plane (class-median size); objects
outside the camera stay privileged. (Follows DEEP's nuPlan protocol.)

## Protocol
- Pre-registration with dated amendments: `docs/method_freeze_draft.md` (§2b v3 freeze before the nuScenes
  confirmatory run; §2c AV2 val; §2d AV2 train; §3 per-benchmark configurations with dev evidence and test outcome).
- Code md5 of the frozen files logged before confirmatory runs (results/frozen_v3/ holds the archived code).
- Every detector is the same COCO-pretrained YOLOv8s unless marked "fine-tuned" (fine-tuned table: trained on the
  benchmark's train split only).
- Statistics: paired bootstrap over scenes / logs / sequences, 2,000 draws, 95% intervals of (ours − baseline).
- Hardware: training / evaluation on 3× RTX A5000 (shared server); deployment latency on Jetson AGX Xavier, TensorRT
  FP16, clocks locked (`jetson_clocks`), image decode excluded.
