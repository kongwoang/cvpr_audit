# Refiner geometry — stored-scene feasibility diagnostic

All 152 nuScenes/300 AV2 development units; baseline raw/deployed actions and costs reproduced exactly.
No inference or main-source change. Five prespecified rows, all retained.
Paired unit bootstrap 2000 seed0, pointwise 95% intervals in intervals.csv/contrasts.csv.

| Dataset | Counterfactual | Missed / episodes | False/1000 | Insufficient loss/frame | G |
|---|---|---:|---:|---:|---:|
| av2 | deployed geometry, fixed closing velocity | 60/366 | 7.9020 | 0.3879 | 0.3012 |
| av2 | deployed v3a | 59/366 | 7.8597 | 0.3798 | 0.3087 |
| av2 | lateral only | 48/366 | 11.7473 | 0.3296 | 0.3190 |
| av2 | range only | 56/366 | 8.5358 | 0.3609 | 0.2799 |
| av2 | raw | 44/366 | 12.5079 | 0.3132 | 0.2760 |
| nusc | deployed geometry, fixed closing velocity | 40/99 | 3.2696 | 0.3859 | 0.3224 |
| nusc | deployed v3a | 40/99 | 2.9426 | 0.3874 | 0.3212 |
| nusc | lateral only | 39/99 | 5.7218 | 0.3915 | 0.3144 |
| nusc | range only | 40/99 | 3.1061 | 0.3872 | 0.3206 |
| nusc | raw | 39/99 | 6.5392 | 0.3911 | 0.3130 |

Range-only and lateral-only can violate camera ray/depth coupling. These isolate mechanisms;
favorable decision metrics alone do not establish accurate geometry or a deployable remedy.
No generic whole-frame controls, selected method, or untouched confirmatory evidence is established.
Unit attribution counts overlap and do not prove that a raw support detection is the true decisive object.

The fixed-closing-velocity row scales finite TTC by refined/raw depth. It preserves the algebraic ratio z/TTC before the controller clips TTC/relative velocity; it does not guarantee invariance of the clipped controller term. Infinite TTC remains infinite.
