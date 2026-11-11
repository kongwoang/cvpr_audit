# Overnight results (auto-collected 2026-09-27 03:33:24)

Raw tables copied from the logs; interpretations live in docs/directions_log.md. Runs: results/raw/.

## logs/speculative_KITTI_mono_r10.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002236_speculative_KITTI_mono_r10
               cell split  rmin  mean_pixel_cost  decision_agrees_with_full  agrees_on_frames_where_full_flips  G_all_full  G_speculative  G_frames_oracle_equal_pixels  G_frames_random_equal_pixels
0  KITTI mono brake   all  10.0           0.0468                     0.9422                             0.7918      0.5168         0.5286                        0.4695                        0.0242
1  KITTI mono brake  test  10.0           0.0511                     0.9021                             0.6732      0.1109         0.1169                        0.1751                        0.0057
2   KITTI mono traj   all  10.0           0.0538                     0.9931                             0.9204      0.1278         0.1111                        0.1781                        0.0069
3   KITTI mono traj  test  10.0           0.0582                     0.9915                             0.8324     -0.0241        -0.0301                        0.0528                       -0.0014
```

## logs/speculative_KITTI_mono_r5.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002237_speculative_KITTI_mono_r5
               cell split  rmin  mean_pixel_cost  decision_agrees_with_full  agrees_on_frames_where_full_flips  G_all_full  G_speculative  G_frames_oracle_equal_pixels  G_frames_random_equal_pixels
0  KITTI mono brake   all   5.0           0.2163                     0.9668                             0.8776      0.5168         0.5282                        0.6211                        0.1118
1  KITTI mono brake  test   5.0           0.2303                     0.9409                             0.7971      0.1109         0.1284                        0.2794                        0.0255
2   KITTI mono traj   all   5.0           0.2483                     0.9974                             0.9690      0.1278         0.1133                        0.1781                        0.0317
3   KITTI mono traj  test   5.0           0.2618                     1.0000                             1.0000     -0.0241        -0.0241                        0.0528                       -0.0063
```

## logs/speculative_nuScenes_mono_r10.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_004742_speculative_nuScenes_mono_r10
                  cell split  rmin  mean_pixel_cost  decision_agrees_with_full  agrees_on_frames_where_full_flips  G_all_full  G_speculative  G_frames_oracle_equal_pixels  G_frames_random_equal_pixels
0  nuScenes mono brake   all  10.0           0.0476                     0.9870                             0.8388      0.1647         0.1534                        0.2776                        0.0078
1  nuScenes mono brake  test  10.0           0.0528                     0.9812                             0.6842      0.1649         0.0639                        0.2494                        0.0087
```

## logs/spec_refine_KITTI.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_004901_spec_refine_KITTI_mono
               cell split  G_all_full  G_speculative  G_speculative_plus_refiner
0  KITTI mono brake   all      0.5168         0.5291                      0.5393
1  KITTI mono brake  test      0.1109         0.1209                      0.1336
2   KITTI mono traj   all      0.1278         0.1111                      0.3914
3   KITTI mono traj  test     -0.0241        -0.0301                      0.0953
```

## logs/realspec_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_011533_realspec_KITTI_mono
NvMapMemAllocInternalTagged: 1074810371 error 12
NvMapMemHandleAlloc: error 0
NvMapMemAllocInternalTagged: 1074810371 error 12
NvMapMemHandleAlloc: error 0
NvMapMemAllocInternalTagged: 1074810371 error 12
NvMapMemHandleAlloc: error 0
               cell           variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  KITTI mono brake  discovery+pivots   all      0.5168              0.5193              0.0606           0.7434
1  KITTI mono brake  discovery+pivots  test      0.1109              0.1658              0.0649           0.7149
```

## logs/realspec_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_025247_realspec_KITTI_oracle
                 cell           variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  KITTI oracle brake  discovery+pivots   all      0.6786              0.6747              0.0561           0.7199
1  KITTI oracle brake  discovery+pivots  test      0.2687              0.2840              0.0587           0.6702
```

## logs/realspec_nodisc_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_012434_realspec_nodisc_KITTI_mono
               cell  variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  KITTI mono brake  _nodisc   all      0.5168              0.0026              0.0207           0.2642
1  KITTI mono brake  _nodisc  test      0.1109              0.0223              0.0251           0.2378
```

## logs/realspec_nodisc_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_010608_realspec_nodisc_nuScenes_mono
                  cell  variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  nuScenes mono brake  _nodisc   all      0.1647              0.0717              0.0110           0.0989
1  nuScenes mono brake  _nodisc  test      0.1649              0.0375              0.0146           0.1005
```

## logs/realspec_nopiv_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_012748_realspec_nopiv_KITTI_mono
               cell variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  KITTI mono brake  _nopiv   all      0.5168              0.5091              0.0399           0.4791
1  KITTI mono brake  _nopiv  test      0.1109              0.1294              0.0398           0.4771
```

## logs/realspec_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_005847_realspec_nuScenes_mono
                  cell           variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  nuScenes mono brake  discovery+pivots   all      0.1647              0.1130              0.0677           0.7464
1  nuScenes mono brake  discovery+pivots  test      0.1649              0.0709              0.0745           0.7853
```

## logs/realspec_nuScenes_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_030119_realspec_nuScenes_oracle
                    cell           variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  nuScenes oracle brake  discovery+pivots   all      0.1401              0.1944              0.0643           0.7302
1  nuScenes oracle brake  discovery+pivots  test      0.1422              0.0870              0.0734           0.7812
```

## logs/realspec_s960_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_024355_realspec_s0.773_KITTI_mono
               cell  variant split  G_all_full  G_speculative_real  net_pixels_vs_full  crops_per_frame
0  KITTI mono brake  _s0.773   all      0.5168              0.5423              0.1253           0.7434
1  KITTI mono brake  _s0.773  test      0.1109              0.1820              0.1344           0.7149
```

## logs/realcrop_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001750_realcrop_KITTI_mono
               cell split     n  G_all_full  G_region_proxy  G_region_real_crop  mean_crop_net_pixels_vs_full640
0  KITTI mono brake   all  8008      0.5168          0.5021              0.5091                           0.0399
1  KITTI mono brake  test  3536      0.1109          0.0975              0.1294                           0.0398
2   KITTI mono traj   all  8008      0.1278          0.1087              0.1347                           0.0514
3   KITTI mono traj  test  3536     -0.0241         -0.0237             -0.0132                           0.0470
```

## logs/realcrop_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_013445_realcrop_KITTI_oracle
                 cell split     n  G_all_full  G_region_proxy  G_region_real_crop  mean_crop_net_pixels_vs_full640
0  KITTI oracle brake   all  8008      0.6786          0.6584              0.6543                           0.0399
1  KITTI oracle brake  test  3536      0.2687          0.2091              0.2435                           0.0398
2   KITTI oracle traj   all  8008      0.6834          0.6279              0.5772                           0.0515
3   KITTI oracle traj  test  3536      0.5193          0.3091              0.3074                           0.0473
```

## logs/ensemble_bootstrap.log

```text
6        KITTI mono brake      ens_exist  0.030           -0.174            0.180         -0.336          0.148
7      KITTI oracle brake         random  0.139            0.000            0.000         -0.385          0.062
8      KITTI oracle brake      ego speed  0.584            0.027            0.535         -0.304          0.531
9      KITTI oracle brake  DEEP gate GBM  0.226           -0.062            0.385          0.000          0.000
10     KITTI oracle brake        ens_all  0.532            0.063            0.481         -0.256          0.401
11     KITTI oracle brake      ens_range  0.016           -0.225            0.139         -0.505         -0.052
12     KITTI oracle brake       ens_miss  0.611           -0.054            0.591         -0.340          0.555
13     KITTI oracle brake      ens_exist  0.059           -0.210            0.183         -0.491          0.091
14    nuScenes mono brake         random  0.132            0.000            0.000         -0.575         -0.100
15    nuScenes mono brake      ego speed  0.099           -0.264            0.130         -0.728         -0.095
16    nuScenes mono brake  DEEP gate GBM  0.482            0.100            0.575          0.000          0.000
17    nuScenes mono brake        ens_all  0.426           -0.028            0.482         -0.362          0.174
18    nuScenes mono brake      ens_range  0.493           -0.143            0.561         -0.470          0.279
19    nuScenes mono brake       ens_miss  0.091           -0.200            0.048         -0.749         -0.101
20    nuScenes mono brake      ens_exist  0.341           -0.056            0.631         -0.351          0.217
21  nuScenes oracle brake         random  0.148            0.000            0.000         -0.497          0.300
22  nuScenes oracle brake      ego speed  0.240           -0.243            0.304         -0.592          0.477
23  nuScenes oracle brake  DEEP gate GBM  0.155           -0.300            0.497          0.000          0.000
24  nuScenes oracle brake        ens_all  0.428           -0.148            0.691         -0.237          0.625
25  nuScenes oracle brake      ens_range  0.307           -0.246            0.525         -0.468          0.577
26  nuScenes oracle brake       ens_miss  0.304           -0.147            0.370         -0.503          0.567
27  nuScenes oracle brake      ens_exist  0.487           -0.083            0.722         -0.139          0.699
```

## logs/ensemble_KITTI.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002658_ensemble_sens_KITTI
  lift sigma 0.203; P(real|conf) [0.52 0.69 0.88 0.95]; recall gain [0.033 0.036 0.088 0.173 0.256 0.269 0.287 0.54  0.493 0.272]
                 cell      v  voi_brake  ens_range  ens_exist  ens_miss  ens_all
0    KITTI mono brake  0.415      0.423      0.009      0.030     0.439    0.386
1  KITTI oracle brake  0.584      0.599      0.016      0.059     0.611    0.532
```

## logs/ensemble_nuScenes.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_005228_ensemble_sens_nuScenes
  lift sigma 0.260; P(real|conf) [0.42 0.6  0.78 0.93]; recall gain [0.    0.011 0.044 0.117 0.149 0.163 0.206 0.184 0.167 0.215]
                    cell      v  voi_brake  ens_range  ens_exist  ens_miss  ens_all
0    nuScenes mono brake  0.099      0.128      0.493      0.341     0.091    0.426
1  nuScenes oracle brake  0.240      0.223      0.307      0.487     0.304    0.428
```

## logs/learned_refiner.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001306_learned_refiner_KITTI
  refiner fitted on 14605 boxes; train |log err| lift 0.145 -> refined 0.060
  pivot |log range error| by split:
split                                
test        0.143      0.138    0.070
train       0.113      0.092    0.033
val         0.156      0.138    0.046
               cell split  G_all_full  G_pivot_full_geometry  G_pivot_learned_refiner
0  KITTI mono brake   all      0.5168                 0.0127                   0.0240
1  KITTI mono brake  test      0.1109                 0.0071                   0.0235
2   KITTI mono traj   all      0.1278                 0.0034                   0.3897
3   KITTI mono traj  test     -0.0241                -0.0380                   0.2583
```

## logs/learned_refiner_nuScenes.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_011020_learned_refiner_nuScenes
  refiner fitted on 6043 boxes; train |log err| lift 0.233 -> refined 0.089
  pivot |log range error| by split:
split                                
test        0.181      0.184    0.113
train       0.185      0.192    0.079
val         0.162      0.156    0.101
                  cell split  G_all_full  G_pivot_full_geometry  G_pivot_learned_refiner
0  nuScenes mono brake   all      0.1647                 0.0150                   0.0452
1  nuScenes mono brake  test      0.1649                 0.0126                   0.0134
```

## logs/region960_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001840_region_full_960_KITTI_mono
               cell split     n  mean_crop_area  share_frames_with_crop  G_full_all_frames  G_frames_random_at_b  G_frames_oracle_at_b  G_dsf_region  G_static_region  dsf_region_harm_share  frame_harm_share
0  KITTI mono brake   all  8008          0.0267                  0.4791             0.5168                0.0138                0.3968        0.5052           0.1200                 0.2475            0.3422
1  KITTI mono brake  test  3536          0.0266                  0.4771             0.1109                0.0030                0.1198        0.0961           0.0439                 0.4087            0.4467
2   KITTI mono traj   all  8008          0.0343                  0.6178             0.1278                0.0044                0.1778        0.1408          -0.0147                 0.4618            0.4838
3   KITTI mono traj  test  3536          0.0314                  0.5653            -0.0241               -0.0008                0.0528       -0.0259          -0.0582                 0.5273            0.5780
```

## logs/region_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260926_235456_region_KITTI_mono
               cell split     n  mean_crop_area  share_frames_with_crop  G_full_all_frames  G_frames_random_at_b  G_frames_oracle_at_b  G_dsf_region  G_static_region  dsf_region_harm_share  frame_harm_share
0  KITTI mono brake   all  8008          0.0267                  0.4791             0.5168                0.0138                0.3968        0.5021           0.1200                 0.2452            0.3422
1  KITTI mono brake  test  3536          0.0266                  0.4771             0.1109                0.0030                0.1198        0.0975           0.0439                 0.3950            0.4467
2   KITTI mono traj   all  8008          0.0343                  0.6178             0.1278                0.0044                0.1778        0.1087          -0.0147                 0.4983            0.4838
3   KITTI mono traj  test  3536          0.0314                  0.5653            -0.0241               -0.0008                0.0528       -0.0237          -0.0582                 0.5462            0.5780
```

## logs/region_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001841_region_KITTI_oracle
                 cell split     n  mean_crop_area  share_frames_with_crop  G_full_all_frames  G_frames_random_at_b  G_frames_oracle_at_b  G_dsf_region  G_static_region  dsf_region_harm_share  frame_harm_share
0  KITTI oracle brake   all  8008          0.0267                  0.4791             0.6786                0.0181                0.5046        0.6584           0.1922                 0.1279            0.2461
1  KITTI oracle brake  test  3536          0.0266                  0.4771             0.2687                0.0072                0.1878        0.2091           0.1542                 0.1969            0.3545
2   KITTI oracle traj   all  8008          0.0343                  0.6191             0.6834                0.0235                0.5060        0.6279           0.1233                 0.0293            0.0302
3   KITTI oracle traj  test  3536          0.0316                  0.5690             0.5193                0.0164                0.5206        0.3091           0.2994                 0.1270            0.0800
```

## logs/region_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001710_region_nuScenes_mono
                  cell split     n  mean_crop_area  share_frames_with_crop  G_full_all_frames  G_frames_random_at_b  G_frames_oracle_at_b  G_dsf_region  G_static_region  dsf_region_harm_share  frame_harm_share
0  nuScenes mono brake   all  3376          0.0352                  0.6475             0.1647                0.0058                0.2680        0.0598           0.0958                 0.4697            0.3956
1  nuScenes mono brake  test   955          0.0373                  0.6848             0.1649                0.0062                0.2494        0.0062           0.0671                 0.5000            0.3860
```

## logs/regionR320_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002600_region_../rtdetr_kitti/rt_cheap_320_KITTI_mono
               cell split     n  mean_crop_area  share_frames_with_crop  G_full_all_frames  G_frames_random_at_b  G_frames_oracle_at_b  G_dsf_region  G_static_region  dsf_region_harm_share  frame_harm_share
0  KITTI mono brake   all  8008          0.0267                  0.4791             0.5168                0.0138                0.3968        0.4453           0.1200                 0.3927            0.3422
1  KITTI mono brake  test  3536          0.0266                  0.4771             0.1109                0.0030                0.1198        0.1117           0.0439                 0.4559            0.4467
2   KITTI mono traj   all  8008          0.0343                  0.6178             0.1278                0.0044                0.1778        0.1213          -0.0147                 0.4835            0.4838
3   KITTI mono traj  test  3536          0.0314                  0.5653            -0.0241               -0.0008                0.0528        0.0168          -0.0582                 0.5461            0.5780
```

## logs/secondlook_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000750_secondlook_KITTI_mono
                                                0                 1                2                3
cell                             KITTI mono brake  KITTI mono brake  KITTI mono traj  KITTI mono traj
split                                         all              test              all             test
G_all_full                                 0.5168            0.1109           0.1278          -0.0241
G_second_look_K1                           0.0361            0.0169           0.0038           -0.038
mean_pixel_cost_K1                         0.0245            0.0316           0.0294            0.041
G_frames_random_equal_pixels_K1            0.0127            0.0035           0.0038           -0.001
G_frames_oracle_equal_pixels_K1             0.386            0.1325           0.1744           0.0528
G_second_look_K2                           0.0344             0.013           0.0025          -0.0377
mean_pixel_cost_K2                         0.0304            0.0415           0.0353           0.0506
G_frames_random_equal_pixels_K2            0.0157            0.0046           0.0045          -0.0012
G_frames_oracle_equal_pixels_K2            0.4139            0.1552           0.1779           0.0528
G_second_look_K1_existence_only            0.0031            0.0049           0.0003           0.0001
```

## logs/secondlook_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000751_secondlook_KITTI_oracle
                                                  0                   1                  2                  3
cell                             KITTI oracle brake  KITTI oracle brake  KITTI oracle traj  KITTI oracle traj
split                                           all                test                all               test
G_all_full                                   0.6786              0.2687             0.6834             0.5193
G_second_look_K1                             0.0115              0.0177            -0.0042             0.0177
mean_pixel_cost_K1                           0.0201              0.0248             0.0222             0.0295
G_frames_random_equal_pixels_K1              0.0137              0.0067             0.0151             0.0153
G_frames_oracle_equal_pixels_K1              0.4614              0.1801             0.3311             0.5206
G_second_look_K2                             0.0113              0.0161            -0.0024             0.0177
mean_pixel_cost_K2                           0.0249              0.0333             0.0273             0.0383
G_frames_random_equal_pixels_K2              0.0169              0.0089             0.0186             0.0199
G_frames_oracle_equal_pixels_K2              0.4933              0.2172             0.4057             0.5206
G_second_look_K1_existence_only              0.0047              0.0093             0.0008             0.0002
```

## logs/secondlook_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000930_secondlook_nuScenes_mono
                                                   0                    1
cell                             nuScenes mono brake  nuScenes mono brake
split                                            all                 test
G_all_full                                    0.1647               0.1649
G_second_look_K1                              0.1273               0.0578
mean_pixel_cost_K1                            0.0224               0.0257
G_frames_random_equal_pixels_K1               0.0037               0.0042
G_frames_oracle_equal_pixels_K1               0.2309               0.2405
G_second_look_K2                              0.1322               0.0684
mean_pixel_cost_K2                            0.0256               0.0278
G_frames_random_equal_pixels_K2               0.0042               0.0046
G_frames_oracle_equal_pixels_K2               0.2458               0.2439
G_second_look_K1_existence_only               0.0144               0.0053
```

## logs/secondlook_nuScenes_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_003403_secondlook_nuScenes_oracle
                                                     0                      1
cell                             nuScenes oracle brake  nuScenes oracle brake
split                                              all                   test
G_all_full                                      0.1401                 0.1422
G_second_look_K1                                0.0227                  0.029
mean_pixel_cost_K1                              0.0189                 0.0249
G_frames_random_equal_pixels_K1                 0.0026                 0.0035
G_frames_oracle_equal_pixels_K1                 0.2221                 0.1919
G_second_look_K2                                 0.032                 0.0322
mean_pixel_cost_K2                              0.0217                 0.0271
G_frames_random_equal_pixels_K2                  0.003                 0.0038
G_frames_oracle_equal_pixels_K2                 0.2301                 0.1919
G_second_look_K1_existence_only                 0.0238                 0.0151
```

## logs/reuse_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000903_reuse_KITTI_mono
                                    0                 1
cell                 KITTI mono brake  KITTI mono brake
split                             all              test
G_all_full                     0.5168            0.1109
G_every2_plus_carry            0.4641            0.1116
G_every2_no_carry              0.2795            0.0793
G_random_rate_1/2              0.2584            0.0555
G_every3_plus_carry            0.3969            0.0837
G_every3_no_carry              0.1594            0.0212
G_random_rate_1/3              0.1723             0.037
G_every5_plus_carry            0.3763            0.1065
G_every5_no_carry               0.096            0.0278
G_random_rate_1/5              0.1034            0.0222
```

## logs/reuse_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000904_reuse_KITTI_oracle
                                      0                   1
cell                 KITTI oracle brake  KITTI oracle brake
split                               all                test
G_all_full                       0.6786              0.2687
G_every2_plus_carry              0.6147              0.2353
G_every2_no_carry                 0.352              0.1313
G_random_rate_1/2                0.3393              0.1343
G_every3_plus_carry              0.5394              0.1929
G_every3_no_carry                0.2154              0.0665
G_random_rate_1/3                0.2262              0.0896
G_every5_plus_carry              0.4889              0.1759
G_every5_no_carry                0.1233              0.0538
G_random_rate_1/5                0.1357              0.0537
```

## logs/reuse_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_003933_reuse_nuScenes_mono
                                       0                    1
cell                 nuScenes mono brake  nuScenes mono brake
split                                all                 test
G_all_full                        0.1647               0.1649
G_every2_plus_carry               0.1137               0.1633
G_every2_no_carry                 0.1217               0.1445
G_random_rate_1/2                 0.0823               0.0825
G_every3_plus_carry                0.082               0.0029
G_every3_no_carry                 0.0401               0.0438
G_random_rate_1/3                 0.0549                0.055
G_every5_plus_carry               0.0424              -0.0917
G_every5_no_carry                 0.0355               0.0191
G_random_rate_1/5                 0.0329                0.033
```

## logs/range_amb_KITTI.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001052_range_ambiguity_KITTI
                                              0                 1
cell                           KITTI mono brake  KITTI mono brake
split                                       all              test
G_all_full                               0.5168            0.1109
frames_with_pivot                        0.3854            0.3973
frames_ambiguous                         0.0377            0.0467
G_refine_pivot_always                    0.1093             0.166
G_refine_only_if_ambiguous               0.0183            0.0162
G_refine_if_ambiguous_noisy10             0.017            0.0149
```

## logs/range_amb_nuScenes.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_004410_range_ambiguity_nuScenes
                                                 0                    1
cell                           nuScenes mono brake  nuScenes mono brake
split                                          all                 test
G_all_full                                  0.1647               0.1649
frames_with_pivot                           0.2737               0.2995
frames_ambiguous                            0.0616               0.0869
G_refine_pivot_always                       0.2825               0.1813
G_refine_only_if_ambiguous                   0.065               0.1052
G_refine_if_ambiguous_noisy10               0.0662               0.1081
```

## logs/stakes_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000021_stakes_KITTI_mono
                                                      0                1
cell                                   KITTI mono brake  KITTI mono traj
t_pivot_full                                       0.45             0.25
t_pivot_cheap                                      0.25             0.25
G_full (DEEP)                                    0.1109          -0.0241
G_full + stakes rule                             0.0819          -0.0241
G_cheap + stakes rule (no escalation)              -0.0             -0.0
G_full t=0.25                                    0.1109          -0.0241
G_full t=0.3                                      0.099          -0.0241
G_full t=0.35                                     0.097           -0.024
G_full t=0.4                                     0.0914           -0.024
G_full t=0.45                                    0.0819          -0.0273
G_full t=0.5                                     0.0801          -0.0273
G_full t=0.6                                     0.0562          -0.0334
G_full t=0.7                                     0.0242          -0.0668
```

## logs/stakes_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_001326_stakes_nuScenes_mono
                                                         0
cell                                   nuScenes mono brake
t_pivot_full                                           0.4
t_pivot_cheap                                         0.45
G_full (DEEP)                                       0.1649
G_full + stakes rule                                0.1885
G_cheap + stakes rule (no escalation)              -0.0005
G_full t=0.25                                       0.1649
G_full t=0.3                                        0.1771
G_full t=0.35                                       0.1781
G_full t=0.4                                        0.1885
G_full t=0.45                                       0.1737
G_full t=0.5                                        0.1794
G_full t=0.6                                        0.1628
G_full t=0.7                                        0.1232
```

## logs/bayes_KITTI_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000406_bayes_KITTI_mono
  P(real) cheap:
 [[0.5  0.29 0.6  0.53 0.41]
 [0.5  0.89 0.81 0.68 0.6 ]
 [0.5  0.75 0.94 0.91 0.83]
 [0.5  0.5  0.98 0.97 0.94]] 
  full:
 [[0.4  0.37 0.3  0.36 0.33]
 [0.32 0.62 0.52 0.47 0.47]
 [0.36 0.84 0.8  0.68 0.64]
 [0.5  0.95 0.97 0.93 0.89]]
               cell split consumer  all_cheap_loss_vs_threshold_consumer  all_full_loss_vs_threshold_cheap  affected  harmed_share  harm_ratio
0  KITTI mono brake   all      thr                                 1.000                             0.483      2075         0.342       0.168
1  KITTI mono brake   all    bayes                                 0.990                             0.472      2097         0.341       0.166
2  KITTI mono brake  test      thr                                 1.000                             0.889       976         0.447       0.603
3  KITTI mono brake  test    bayes                                 0.985                             0.864      1010         0.432       0.575
```

## logs/bayes_KITTI_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000407_bayes_KITTI_oracle
  P(real) cheap:
 [[0.5  0.29 0.6  0.53 0.41]
 [0.5  0.89 0.81 0.68 0.6 ]
 [0.5  0.75 0.94 0.91 0.83]
 [0.5  0.5  0.98 0.97 0.94]] 
  full:
 [[0.4  0.37 0.3  0.36 0.33]
 [0.32 0.62 0.52 0.47 0.47]
 [0.36 0.84 0.8  0.68 0.64]
 [0.5  0.95 0.97 0.93 0.89]]
                 cell split consumer  all_cheap_loss_vs_threshold_consumer  all_full_loss_vs_threshold_cheap  affected  harmed_share  harm_ratio
0  KITTI oracle brake   all      thr                                 1.000                             0.321      1296         0.246       0.080
1  KITTI oracle brake   all    bayes                                 0.994                             0.311      1421         0.234       0.077
2  KITTI oracle brake  test      thr                                 1.000                             0.731       584         0.354       0.305
3  KITTI oracle brake  test    bayes                                 0.986                             0.705       634         0.331       0.287
```

## logs/bayes_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002207_bayes_nuScenes_mono
  P(real) cheap:
 [[0.5  0.33 0.28 0.36 0.54]
 [0.5  0.5  0.39 0.56 0.66]
 [0.5  0.5  0.59 0.74 0.81]
 [0.5  0.5  0.2  0.94 0.93]] 
  full:
 [[0.25 0.01 0.13 0.38 0.42]
 [0.25 0.05 0.22 0.54 0.58]
 [0.5  0.09 0.37 0.67 0.72]
 [0.5  0.2  0.5  0.87 0.88]]
                  cell split consumer  all_cheap_loss_vs_threshold_consumer  all_full_loss_vs_threshold_cheap  affected  harmed_share  harm_ratio
0  nuScenes mono brake   all      thr                                 1.000                             0.835       273         0.396       0.407
1  nuScenes mono brake   all    bayes                                 0.933                             0.748       284         0.391       0.365
2  nuScenes mono brake  test      thr                                 1.000                             0.835        57         0.386       0.339
3  nuScenes mono brake  test    bayes                                 0.909                             0.811        66         0.439       0.595
```

## logs/headroom_iou03_nuScenes_mono.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_000436_headroom_iou0.3_nuScenes_mono
  3376 frames written
```

## logs/headroom_iou03_nuScenes_oracle.log

```text
  run: /home/kongwoang/research/CVPR/results/raw/20260927_002700_headroom_iou0.3_nuScenes_oracle
  3376 frames written
```

## logs/ensemble_bootstrap.log

```text
6        KITTI mono brake      ens_exist  0.030           -0.174            0.180         -0.336          0.148
7      KITTI oracle brake         random  0.139            0.000            0.000         -0.385          0.062
8      KITTI oracle brake      ego speed  0.584            0.027            0.535         -0.304          0.531
9      KITTI oracle brake  DEEP gate GBM  0.226           -0.062            0.385          0.000          0.000
10     KITTI oracle brake        ens_all  0.532            0.063            0.481         -0.256          0.401
11     KITTI oracle brake      ens_range  0.016           -0.225            0.139         -0.505         -0.052
12     KITTI oracle brake       ens_miss  0.611           -0.054            0.591         -0.340          0.555
13     KITTI oracle brake      ens_exist  0.059           -0.210            0.183         -0.491          0.091
14    nuScenes mono brake         random  0.132            0.000            0.000         -0.575         -0.100
15    nuScenes mono brake      ego speed  0.099           -0.264            0.130         -0.728         -0.095
16    nuScenes mono brake  DEEP gate GBM  0.482            0.100            0.575          0.000          0.000
17    nuScenes mono brake        ens_all  0.426           -0.028            0.482         -0.362          0.174
18    nuScenes mono brake      ens_range  0.493           -0.143            0.561         -0.470          0.279
19    nuScenes mono brake       ens_miss  0.091           -0.200            0.048         -0.749         -0.101
20    nuScenes mono brake      ens_exist  0.341           -0.056            0.631         -0.351          0.217
21  nuScenes oracle brake         random  0.148            0.000            0.000         -0.497          0.300
22  nuScenes oracle brake      ego speed  0.240           -0.243            0.304         -0.592          0.477
23  nuScenes oracle brake  DEEP gate GBM  0.155           -0.300            0.497          0.000          0.000
24  nuScenes oracle brake        ens_all  0.428           -0.148            0.691         -0.237          0.625
25  nuScenes oracle brake      ens_range  0.307           -0.246            0.525         -0.468          0.577
26  nuScenes oracle brake       ens_miss  0.304           -0.147            0.370         -0.503          0.567
27  nuScenes oracle brake      ens_exist  0.487           -0.083            0.722         -0.139          0.699
```

## Crop engine timings (trtexec, FP16, Jetson AGX Xavier)

```text
engines/time_288x960.log: min = 7.40845 ms, max = 7.47705 ms, mean = 7.43803 ms, median = 7.43872 ms, percentile(90%) = 7.45581 ms, percentile(95%) = 7.46216 ms, percentile(99%) = 7.47168 ms
engines/time_64x192.log: min = 1.98901 ms, max = 16.7373 ms, mean = 2.24657 ms, median = 2.01672 ms, percentile(90%) = 2.72021 ms, percentile(95%) = 3.18457 ms, percentile(99%) = 3.52075 ms
engines/time_96x288.log: min = 2.11426 ms, max = 2.52954 ms, mean = 2.12385 ms, median = 2.12354 ms, percentile(90%) = 2.1272 ms, percentile(95%) = 2.12842 ms, percentile(99%) = 2.13062 ms
engines/time_96x320.log: min = 2.1936 ms, max = 2.87524 ms, mean = 2.21063 ms, median = 2.20947 ms, percentile(90%) = 2.21851 ms, percentile(95%) = 2.22217 ms, percentile(99%) = 2.25476 ms
engines/time_cheap320.log: min = 2.18823 ms, max = 2.24768 ms, mean = 2.19863 ms, median = 2.19824 ms, percentile(90%) = 2.203 ms, percentile(95%) = 2.20459 ms, percentile(99%) = 2.20844 ms
engines/time_full640.log: min = 3.80688 ms, max = 3.90527 ms, mean = 3.83058 ms, median = 3.82935 ms, percentile(90%) = 3.8396 ms, percentile(95%) = 3.84619 ms, percentile(99%) = 3.87939 ms
```
