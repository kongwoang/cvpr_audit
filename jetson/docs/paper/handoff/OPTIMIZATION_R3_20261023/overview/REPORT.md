# Mechanism-repair experiments — round 3 (complete)

Updated 2026-10-24 07:42:11 NZDT

Main source/weights/thresholds and K34 remain frozen. All changes are sandbox-only.
New policies use training/REF-VAL and CFG-DEV only. Multiple adaptive rounds: exploratory, not confirmatory.
Paired unit bootstrap: 2,000, seed0, pointwise95% CI. G uses the prior cheap-action jerk convention.
Requested detector pixels do not include encoder/head work and are not Jetson latency.

## Complete development datasets

Every executed row from a completed dataset is listed below, including no-refiner and weaker alternatives.
Full comparisons/intervals: all_complete_rows.csv and all_complete_contrasts.csv. Partial units are excluded from this table.

| Suite | Dataset | Pipeline | Missed / episodes | False episodes /1000 | Insufficient loss/frame | G | Detector px/native |
|---|---|---|---:|---:|---:|---:|---:|
| dev | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| dev | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| dev | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| dev | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| dev | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| dev | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| dev | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| dev | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| dev | nusc | mixed-history C1 / none | 39/99 | 8.174 | 0.37243 | 0.341 | 0.07444 |
| dev | nusc | mixed-history C1 | 40/99 | 4.087 | 0.36918 | 0.349 | 0.07444 |
| dev | nusc | mixed-history + recovery / none | 37/99 | 8.337 | 0.37058 | 0.343 | 0.07497 |
| dev | nusc | mixed-history + recovery | 38/99 | 4.250 | 0.36655 | 0.353 | 0.07497 |
| dev | nusc | cheap TTC transfer / none | 35/99 | 11.607 | 0.36626 | 0.325 | 0.07496 |
| dev | nusc | cheap TTC transfer | 36/99 | 7.847 | 0.36221 | 0.335 | 0.07496 |
| dev | nusc | critical extent retention / none | 37/99 | 7.847 | 0.36326 | 0.353 | 0.07784 |
| dev | nusc | critical extent retention | 38/99 | 3.923 | 0.35876 | 0.364 | 0.07784 |
| dev | nusc | motion-conditioned field / none | 37/99 | 7.520 | 0.37270 | 0.338 | 0.07863 |
| dev | nusc | motion-conditioned field | 38/99 | 3.760 | 0.36870 | 0.348 | 0.07863 |
| dev | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| dev | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| dev | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| dev | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| dev | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| dev | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| dev | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| dev | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| dev | av2 | mixed-history C1 / none | 34/366 | 14.494 | 0.29563 | 0.287 | 0.05040 |
| dev | av2 | mixed-history C1 | 48/366 | 9.888 | 0.36037 | 0.324 | 0.05040 |
| dev | av2 | mixed-history + recovery / none | 34/366 | 14.071 | 0.29528 | 0.289 | 0.05143 |
| dev | av2 | mixed-history + recovery | 48/366 | 9.465 | 0.36021 | 0.326 | 0.05143 |
| dev | av2 | cheap TTC transfer / none | 29/366 | 20.410 | 0.28615 | 0.256 | 0.05138 |
| dev | av2 | cheap TTC transfer | 43/366 | 15.973 | 0.34843 | 0.300 | 0.05138 |
| dev | av2 | critical extent retention / none | 34/366 | 13.395 | 0.29257 | 0.290 | 0.05315 |
| dev | av2 | critical extent retention | 49/366 | 8.958 | 0.36107 | 0.321 | 0.05315 |
| dev | av2 | motion-conditioned field / none | 34/366 | 14.114 | 0.30055 | 0.282 | 0.05758 |
| dev | av2 | motion-conditioned field | 50/366 | 9.592 | 0.36699 | 0.317 | 0.05758 |
| learned | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| learned | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| learned | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| learned | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| learned | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| learned | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| learned | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| learned | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| learned | nusc | learned missing history / none | 32/99 | 9.318 | 0.35148 | 0.360 | 0.07499 |
| learned | nusc | learned missing history | 35/99 | 5.395 | 0.35213 | 0.365 | 0.07499 |
| learned | nusc | learned geometric / none | 32/99 | 8.991 | 0.35094 | 0.362 | 0.07497 |
| learned | nusc | learned geometric | 36/99 | 5.395 | 0.35279 | 0.363 | 0.07497 |
| learned | nusc | learned geometric / adapt | 34/99 | 7.193 | 0.35133 | 0.360 | 0.07497 |
| learned | nusc | learned decision / none | 32/99 | 9.155 | 0.35265 | 0.361 | 0.07497 |
| learned | nusc | learned decision | 36/99 | 5.722 | 0.35398 | 0.362 | 0.07497 |
| learned | nusc | learned decision / adapt | 34/99 | 7.030 | 0.35554 | 0.363 | 0.07497 |
| learned | nusc | upper-quantile missing history / none | 26/99 | 16.348 | 0.32659 | 0.372 | 0.07504 |
| learned | nusc | upper-quantile missing history | 29/99 | 11.280 | 0.32998 | 0.375 | 0.07504 |
| learned | nusc | uncertainty query / none | 30/99 | 10.299 | 0.33794 | 0.372 | 0.07732 |
| learned | nusc | uncertainty query | 34/99 | 6.866 | 0.33960 | 0.374 | 0.07732 |
| learned | nusc | uncertainty query / adapt | 32/99 | 8.664 | 0.33833 | 0.371 | 0.07732 |
| learned | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| learned | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| learned | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| learned | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| learned | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| learned | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| learned | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| learned | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| learned | av2 | learned missing history / none | 30/366 | 14.114 | 0.28165 | 0.304 | 0.05142 |
| learned | av2 | learned missing history | 44/366 | 9.930 | 0.34595 | 0.343 | 0.05142 |
| learned | av2 | learned geometric / none | 30/366 | 13.564 | 0.28380 | 0.304 | 0.05142 |
| learned | av2 | learned geometric | 44/366 | 9.254 | 0.35024 | 0.340 | 0.05142 |
| learned | av2 | learned geometric / adapt | 51/366 | 9.212 | 0.35150 | 0.340 | 0.05142 |
| learned | av2 | learned decision / none | 30/366 | 13.353 | 0.28414 | 0.304 | 0.05142 |
| learned | av2 | learned decision | 45/366 | 9.043 | 0.35100 | 0.340 | 0.05142 |
| learned | av2 | learned decision / adapt | 54/366 | 7.141 | 0.41106 | 0.400 | 0.05142 |
| learned | av2 | upper-quantile missing history / none | 28/366 | 16.142 | 0.27468 | 0.303 | 0.05143 |
| learned | av2 | upper-quantile missing history | 41/366 | 12.001 | 0.33718 | 0.344 | 0.05143 |
| learned | av2 | uncertainty query / none | 23/366 | 15.043 | 0.27188 | 0.314 | 0.05272 |
| learned | av2 | uncertainty query | 37/366 | 10.818 | 0.33867 | 0.350 | 0.05272 |
| learned | av2 | uncertainty query / adapt | 44/366 | 10.818 | 0.34146 | 0.348 | 0.05272 |
| calibrated | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| calibrated | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| calibrated | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| calibrated | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| calibrated | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| calibrated | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| calibrated | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| calibrated | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| calibrated | nusc | upper-quantile missing history / none | 26/99 | 19.617 | 0.32445 | 0.361 | 0.07505 |
| calibrated | nusc | upper-quantile missing history | 29/99 | 14.386 | 0.32818 | 0.369 | 0.07505 |
| calibrated | nusc | uncertainty query / none | 30/99 | 10.626 | 0.33649 | 0.373 | 0.07827 |
| calibrated | nusc | uncertainty query | 34/99 | 7.030 | 0.33905 | 0.374 | 0.07827 |
| calibrated | nusc | uncertainty query / adapt | 32/99 | 8.991 | 0.33698 | 0.373 | 0.07827 |
| calibrated | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| calibrated | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| calibrated | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| calibrated | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| calibrated | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| calibrated | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| calibrated | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| calibrated | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| calibrated | av2 | upper-quantile missing history / none | 25/366 | 19.438 | 0.27068 | 0.297 | 0.05144 |
| calibrated | av2 | upper-quantile missing history | 36/366 | 15.635 | 0.33128 | 0.340 | 0.05144 |
| calibrated | av2 | uncertainty query / none | 24/366 | 15.128 | 0.26839 | 0.316 | 0.05379 |
| calibrated | av2 | uncertainty query | 38/366 | 10.860 | 0.33487 | 0.353 | 0.05379 |
| calibrated | av2 | uncertainty query / adapt | 45/366 | 10.691 | 0.33783 | 0.351 | 0.05379 |
| lift | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| lift | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| lift | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| lift | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| lift | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| lift | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| lift | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| lift | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| lift | nusc | joint geometric lift | 28/99 | 8.828 | 0.33817 | 0.363 | 0.07490 |
| lift | nusc | joint decision lift | 28/99 | 8.010 | 0.34392 | 0.372 | 0.07498 |
| lift | nusc | upper quantile + extent retention / none | 26/99 | 16.021 | 0.30503 | 0.398 | 0.07894 |
| lift | nusc | upper quantile + extent retention | 29/99 | 11.280 | 0.30914 | 0.401 | 0.07894 |
| lift | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| lift | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| lift | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| lift | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| lift | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| lift | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| lift | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| lift | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| lift | av2 | joint geometric lift | 49/366 | 9.719 | 0.34710 | 0.330 | 0.05196 |
| lift | av2 | joint decision lift | 52/366 | 6.212 | 0.41519 | 0.395 | 0.05236 |
| lift | av2 | upper quantile + extent retention / none | 27/366 | 16.015 | 0.26234 | 0.310 | 0.05341 |
| lift | av2 | upper quantile + extent retention | 40/366 | 11.916 | 0.32567 | 0.349 | 0.05341 |
| persist | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| persist | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| persist | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| persist | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| persist | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| persist | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| persist | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| persist | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| persist | nusc | upper-quantile missing history / none | 26/99 | 19.617 | 0.32445 | 0.361 | 0.07505 |
| persist | nusc | upper-quantile missing history | 29/99 | 14.386 | 0.32818 | 0.369 | 0.07505 |
| persist | nusc | upper quantile + extent retention / none | 26/99 | 19.454 | 0.30051 | 0.389 | 0.07908 |
| persist | nusc | upper quantile + extent retention | 29/99 | 14.550 | 0.30504 | 0.396 | 0.07908 |
| persist | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| persist | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| persist | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| persist | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| persist | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| persist | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| persist | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| persist | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| persist | av2 | upper-quantile missing history / none | 25/366 | 19.438 | 0.27068 | 0.297 | 0.05144 |
| persist | av2 | upper-quantile missing history | 36/366 | 15.635 | 0.33128 | 0.340 | 0.05144 |
| persist | av2 | upper quantile + extent retention / none | 24/366 | 19.269 | 0.25841 | 0.303 | 0.05357 |
| persist | av2 | upper quantile + extent retention | 35/366 | 15.466 | 0.31979 | 0.344 | 0.05357 |
| iddev | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| iddev | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| iddev | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| iddev | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| iddev | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| iddev | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| iddev | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| iddev | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| iddev | nusc | upper-quantile missing history / none | 26/99 | 16.348 | 0.32659 | 0.372 | 0.07504 |
| iddev | nusc | upper-quantile missing history | 29/99 | 11.280 | 0.32998 | 0.375 | 0.07504 |
| iddev | nusc | identity gate + uncertain fallback / none | 26/99 | 16.348 | 0.32654 | 0.372 | 0.07504 |
| iddev | nusc | identity gate + uncertain fallback | 29/99 | 11.444 | 0.32952 | 0.376 | 0.07504 |
| low | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| low | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| low | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| low | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| low | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| low | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| low | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| low | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| low | nusc | learned missing history / none | 32/99 | 9.318 | 0.35148 | 0.360 | 0.07499 |
| low | nusc | learned missing history | 35/99 | 5.395 | 0.35213 | 0.365 | 0.07499 |
| low | nusc | upper-quantile missing history / none | 26/99 | 16.348 | 0.32659 | 0.372 | 0.07504 |
| low | nusc | upper-quantile missing history | 29/99 | 11.280 | 0.32998 | 0.375 | 0.07504 |
| low | nusc | point fallback + extent retention / none | 32/99 | 9.155 | 0.33682 | 0.378 | 0.07835 |
| low | nusc | point fallback + extent retention | 35/99 | 5.231 | 0.33700 | 0.383 | 0.07835 |
| low | nusc | observed-history point fallback / none | 36/99 | 8.501 | 0.36808 | 0.345 | 0.07496 |
| low | nusc | observed-history point fallback | 38/99 | 4.577 | 0.36453 | 0.354 | 0.07496 |
| low | nusc | corroborated upper fallback / none | 31/99 | 9.482 | 0.35103 | 0.358 | 0.07498 |
| low | nusc | corroborated upper fallback | 33/99 | 5.231 | 0.35122 | 0.363 | 0.07498 |
| low | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| low | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| low | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| low | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| low | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| low | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| low | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| low | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| low | av2 | learned missing history / none | 30/366 | 14.114 | 0.28165 | 0.304 | 0.05142 |
| low | av2 | learned missing history | 44/366 | 9.930 | 0.34595 | 0.343 | 0.05142 |
| low | av2 | upper-quantile missing history / none | 28/366 | 16.142 | 0.27468 | 0.303 | 0.05143 |
| low | av2 | upper-quantile missing history | 41/366 | 12.001 | 0.33718 | 0.344 | 0.05143 |
| low | av2 | point fallback + extent retention / none | 29/366 | 14.029 | 0.26957 | 0.311 | 0.05327 |
| low | av2 | point fallback + extent retention | 43/366 | 9.804 | 0.33489 | 0.348 | 0.05327 |
| low | av2 | observed-history point fallback / none | 34/366 | 13.860 | 0.29071 | 0.294 | 0.05142 |
| low | av2 | observed-history point fallback | 49/366 | 9.465 | 0.35522 | 0.333 | 0.05142 |
| low | av2 | corroborated upper fallback / none | 30/366 | 14.283 | 0.28052 | 0.303 | 0.05142 |
| low | av2 | corroborated upper fallback | 44/366 | 10.015 | 0.34384 | 0.343 | 0.05142 |
| compact | nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.000 | 0.04267 |
| compact | nusc | v6-S / none | 39/99 | 6.539 | 0.39114 | 0.313 | 0.08037 |
| compact | nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.321 | 0.08037 |
| compact | nusc | round1 combined / none | 37/99 | 7.847 | 0.37248 | 0.341 | 0.07497 |
| compact | nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.351 | 0.07497 |
| compact | nusc | round1 combined / half | 37/99 | 5.722 | 0.37236 | 0.346 | 0.07497 |
| compact | nusc | round1 combined / speed | 38/99 | 4.741 | 0.36787 | 0.350 | 0.07497 |
| compact | nusc | round1 combined / speedhalf | 38/99 | 4.577 | 0.36769 | 0.350 | 0.07497 |
| compact | nusc | learned missing history / none | 32/99 | 9.318 | 0.35148 | 0.360 | 0.07499 |
| compact | nusc | learned missing history | 35/99 | 5.395 | 0.35213 | 0.365 | 0.07499 |
| compact | nusc | compact point fallback / none | 33/99 | 8.991 | 0.36358 | 0.348 | 0.07498 |
| compact | nusc | compact point fallback | 33/99 | 5.231 | 0.35924 | 0.359 | 0.07498 |
| compact | nusc | compact observed-history fallback / none | 36/99 | 8.174 | 0.37148 | 0.339 | 0.07497 |
| compact | nusc | compact observed-history fallback | 37/99 | 4.250 | 0.36760 | 0.349 | 0.07497 |
| compact | av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.000 | 0.02581 |
| compact | av2 | v6-S / none | 44/366 | 12.508 | 0.31324 | 0.276 | 0.05556 |
| compact | av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.309 | 0.05556 |
| compact | av2 | round1 combined / none | 34/366 | 13.438 | 0.30131 | 0.286 | 0.05141 |
| compact | av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.319 | 0.05141 |
| compact | av2 | round1 combined / half | 43/366 | 10.522 | 0.32631 | 0.329 | 0.05141 |
| compact | av2 | round1 combined / speed | 51/366 | 8.663 | 0.37813 | 0.310 | 0.05141 |
| compact | av2 | round1 combined / speedhalf | 48/366 | 8.958 | 0.37192 | 0.317 | 0.05141 |
| compact | av2 | learned missing history / none | 30/366 | 14.114 | 0.28165 | 0.304 | 0.05142 |
| compact | av2 | learned missing history | 44/366 | 9.930 | 0.34595 | 0.343 | 0.05142 |
| compact | av2 | compact point fallback / none | 34/366 | 14.114 | 0.29641 | 0.288 | 0.05141 |
| compact | av2 | compact point fallback | 48/366 | 9.634 | 0.36168 | 0.325 | 0.05141 |
| compact | av2 | compact observed-history fallback / none | 34/366 | 14.029 | 0.29669 | 0.289 | 0.05141 |
| compact | av2 | compact observed-history fallback | 49/366 | 9.508 | 0.36216 | 0.325 | 0.05141 |

## Training-data and model audit

785 original training/holdout units were regenerated (385 nuScenes,400 AV2), retaining partial and unmatched histories.
Every finite-target original C1 example was uniquely matched to regenerated crops by exact image bytes: 15,360 nuScenes and81,605 AV2.
Known full crop-chain GT identity switches: nuScenes train306/3077 andholdout115/609; AV2 train1210/21861 andholdout263/3973.
These are annotation-matching diagnostics, not manually certified identity errors for every chain. The regenerated current-object matching population includes incomplete chains, unlike the original sampler; some matched targets differ. Alignment CSVs retain all differences.
Masked adapters freeze the original image encoder. Geometric and decision-aware heads, raw/calibrated quantiles, a separate pair-identity classifier and every selection epoch are retained.
Holdouts already influenced model/epoch selection. Quantile calibration is empirical; no conformal/safety guarantee. Model-audit results concern matched objects, not end-to-end coverage or episode performance.

## Mechanism and tradeoff checks

On completed nuScenes, point fallback rescued3 episodes and lost0 versus round1; all3 have decisive-object braking evidence. Upper-quantile fallback rescued9 and lost0, but one has no decisive-object evidence.
The crossing-truck upper-quantile rescue brakes at the earlier matched frame but still misses the peak-demand frame: L1 episode rescue is not adequate hard braking.
Auxiliary identity logits initially separated switches poorly. The separate classifier meets the predeclared feasibility gate on nuScenes only; AV2 was not deployed as an identity gate.
Support-only adapter refinement and joint estimation inside the loop are distinct experiments and are both retained.
The low-tradeoff suite tests temporal-evidence-only fallback, corroborated upper estimates and point+extent persistence.

## Execution and reproducibility

Authoritative scripts/protocols: experiments/optimization_r3_20261023/. Each worker stores hashes, actual exit status and atomic unit files; only original import-path failures were repaired/retried.
Initial AV2 import failure and a CPU audit duplicate-key error are documented in execution_notes.md; original attempts/logs are preserved.
GPU launches use the unchanged run_when_gpu_free.sh, SHARE=1 MIN_FREE_MB=10000, two workers per group. VUW/VUW1 share one workspace.
Jetson feasibility is separately isolated in experiments/optimization_r3_jetson_20261024, using the prior fixed development manifests and existing engines.
No new test-set selection, main-code promotion or closed-loop safety claim.
