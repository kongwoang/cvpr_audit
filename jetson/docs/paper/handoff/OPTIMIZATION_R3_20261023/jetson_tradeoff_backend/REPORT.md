# R3 cross-backend common-frame check

512 timed frames per dataset; first repetition only. Original timeline/history prefixes retained in both implementations.
VUW Torch detector and requested crop canvases differ from Jetson TensorRT detector/padded canvases. This measures differences, not equivalence.
The within-Jetson port/encoder checks and three-repetition decision check are separate. Ground-truth equality is checked explicitly.

| dataset | pipeline | common_frames | units | differing_actions | differing_gt | jetson_higher_actions | jetson_lower_actions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| av2_dev | compact point fallback | 512 | 8 | 3 | 0 | 3 | 0 |
| av2_dev | corroborated upper fallback | 512 | 8 | 4 | 0 | 4 | 0 |
| av2_dev | critical extent retention | 512 | 8 | 6 | 0 | 5 | 1 |
| av2_dev | learned missing history | 512 | 8 | 3 | 0 | 3 | 0 |
| av2_dev | round1 combined | 512 | 8 | 3 | 0 | 3 | 0 |
| av2_dev | v6-S | 512 | 8 | 1 | 0 | 1 | 0 |
| nusc_val | compact point fallback | 512 | 13 | 2 | 0 | 1 | 1 |
| nusc_val | corroborated upper fallback | 512 | 13 | 3 | 0 | 2 | 1 |
| nusc_val | critical extent retention | 512 | 13 | 2 | 0 | 1 | 1 |
| nusc_val | learned missing history | 512 | 13 | 3 | 0 | 2 | 1 |
| nusc_val | round1 combined | 512 | 13 | 2 | 0 | 1 | 1 |
| nusc_val | v6-S | 512 | 13 | 2 | 0 | 0 | 2 |

Use full development VUW results for quality and Jetson runs for runtime feasibility. Changes in this prefix cannot be attributed solely to the new readout if baseline backend decisions also differ.
