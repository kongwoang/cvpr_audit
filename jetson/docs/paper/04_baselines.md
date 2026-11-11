# Baselines

| baseline | what it is | code | where evaluated | fidelity check |
|---|---|---|---|---|
| uniform u320…u2048 | whole frame at long side N, same YOLOv8s | ours (rap.detect) | all | — |
| AdaScale (Chin et al., MLSys'19) | per-frame scale regressor (ResNet-18 classifier over {320…960}) | re-implementation (original Caffe/R-FCN) | nuScenes, AV2, NAVSIM | — |
| Budget-Adaptive Routing (Geng et al.) | learned frame router cheap↔strong (ORIC, MobileNetV2-Lite / XGBoost, budget arbiter) | official bgt-ada @ 6669ab0; skip_V / cond_V (decision-value targets) are our additions | nuScenes (3 seeds), AV2, NAVSIM | VOC table reproduced within ±0.01 |
| FOVEA (Thavamani et al., ICCV'21) | KDE saliency warp (KDE_I from the draft, KDE_S dataset prior), frozen and fine-tuned | official grid generator; detector YOLOv8s | nuScenes, AV2, KITTI, NAVSIM, nuPlan | FOVEA-ft AP50 ≈ uniform-ft (.524 vs .522) as in their paper |
| LZU (Thavamani et al., CVPR'23) | zoom/unzoom of features | official fixed_grid / invert_grid around YOLOv8s | nuScenes (frozen + ft, 2 seeds) | LZU-ft > uniform control AP50 (.500 vs .497) |
| SAHI (Akyon et al.) | sliced inference + full pass | official library | nuScenes, AV2, KITTI, NAVSIM, nuPlan | small-object recall up as claimed |
| UFP / UFPMP-Det | cluster crops packed into a mosaic, fine detector | official UFPMP-Det-Tools; MP-Det-style fine-tune of YOLOv8s | nuScenes (frozen + ft) | needs mosaic-trained detector (as UFPMP) |
| ESOD (Liu et al., TIP) | sparse feature-level zoom with SAM pseudo-masks | official repo (YOLOv5s, 1536) | nuScenes (2 seeds), AV2 | VisDrone: −3 to −4 AP vs paper with released weights, ordering reproduced |
| ZoomDet (2026) | learned zoom on YOLOv8-s (mmyolo) | official zoomdet_yolo (736² square input) | nuScenes (2 seeds), AV2 | — |
| uncertainty / temporal-criticality / random crops | equal-budget crop heuristics in our machinery | ours | nuScenes, AV2, KITTI, NAVSIM, nuPlan | — |

Compute notes: SAHI on KITTI = 1.89× native pixels (≈10× ours), 117 ms torch vs 22 ms for u960; LZU ≈ u960 latency;
UFP 76–101 ms; bgt-ada router alone 11.7 ms (Jetson, see directions_log "Jetson latency of the baselines").
Full provenance table: docs/baseline_fidelity.md ("Code provenance per baseline").
