# Remaining missed episodes — development attribution

All 152 nuScenes/300 AV2 units. Required episode definitions reproduce the completed screen exactly.
Flags describe the baseline scene and can overlap. A flag in any required frame is not the cause of an episode miss.
GT matching identifies the annotated decisive object in the scene; there is no GT track-ID identity audit.
The remaining-after-recovery population still uses BASELINE scene flags, not unrecorded recovery-scene geometry.

| Dataset | Episodes | Baseline miss | Recovery miss | Recovered | Introduced | Baseline-only vs cheap | Cheap-only vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| av2 | 366 | 59 | 49 | 10 | 0 | 36 | 7 |
| nusc | 99 | 40 | 38 | 2 | 0 | 12 | 6 |

These counts distinguish net improvement from recovery/harm transitions. No new method is selected.
Detailed overlapping mechanisms: mechanism_counts.csv. Required-frame counts: required_keep_frames.csv.
