#!/usr/bin/env python
"""K32 amendment G2 (handoff/K32/STATUS.md): latency model t = a + b * detector_calls + c * crop_px (OLS; crop px in
megapixels at 1600 x 900) on the Jetson frames behind results/k32/latency_samples.csv (K29 r1-r3 a: v6-S "ours@trt32" with
calls_ours@trt32, v6-S <= 2 "ours_b2@trt32" with calls_loop_ours_b2@trt32; K30 J1 j1x r1-r3: hybrid "tsp8S+C1@trt32",
calls = 1 if it has crops). Warm-up frames excluded; nusc_test20 only (1600 x 900).
The modelled latency of a frame is clipped below at min_ms, the smallest measured latency of the fit frames.
Validation: per branch, measured vs modelled (model + a residual drawn from the pooled fit residuals, seed 0) mean, p50,
p95 and the apply-tick delay ceil(t / 50 ms) distribution.
-> results/k32_latency_model.json (a, b, c, rmse, r2, residuals) and results/k32_latency_model_validation.csv
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
NATIVE = 1600 * 900
rows = []
for d in sorted(glob.glob(str(R / "raw" / "*_k29_timing_nusc_test20_v6S_r[123]a"))):
    f = pd.DataFrame(pd.read_pickle(Path(d) / "frames.pkl"))
    f = f[~f["warm"].astype(bool)]
    for br, ms, calls, px in (("v6S", "ms_ours@trt32", "calls_ours@trt32", "px_ours@trt32"),
                              ("v6Sb2", "ms_ours_b2@trt32", "calls_loop_ours_b2@trt32", "px_ours_b2@trt32")):
        rows.append(pd.DataFrame({"branch": br, "run": Path(d).name, "unit": f.unit, "frame": f.frame, "ms": f[ms].astype(float),
                                  "calls": f[calls].astype(float), "crop_mpx": (f[px] - f.px_cheap).astype(float) * NATIVE / 1e6}))
for d in sorted(glob.glob(str(R / "raw" / "*_k30j_timing_nusc_test20_v6S_j1x_r[123]"))):
    f = pd.DataFrame(pd.read_pickle(Path(d) / "frames.pkl"))
    f = f[~f["warm"].astype(bool)]
    nm = "tsp8S+C1@trt32"
    rows.append(pd.DataFrame({"branch": "hybrid", "run": Path(d).name, "unit": f.unit, "frame": f.frame, "ms": f[f"ms_{nm}"].astype(float),
                              "calls": (f[f"ncrops_{nm}"] > 0).astype(float), "crop_mpx": (f[f"px_{nm}"] - f.px_cheap).astype(float) * NATIVE / 1e6}))
D = pd.concat(rows, ignore_index=True).dropna()
# the same frames as latency_samples.csv (check)
if len(sys.argv) > 1:
    S = pd.read_csv(sys.argv[1])
    for br in ("v6S", "v6Sb2", "hybrid"):
        a, b = D[D.branch == br].ms.sort_values().to_numpy(), S[S.branch == br].ms.sort_values().to_numpy()
        print(f"check {br}: fit frames {len(a)}, latency_samples {len(b)}, max |sorted diff| {np.abs(a - b).max() if len(a) == len(b) else 'n/a'}")
X = np.c_[np.ones(len(D)), D.calls, D.crop_mpx]
beta, *_ = np.linalg.lstsq(X, D.ms.to_numpy(), rcond=None)
pred = X @ beta
res = D.ms.to_numpy() - pred
r2 = 1 - (res ** 2).sum() / ((D.ms - D.ms.mean()) ** 2).sum()
rng = np.random.default_rng(0)
val = []
for br, g in D.groupby("branch"):
    m = X[D.branch == br] @ beta + rng.choice(res, (D.branch == br).sum())
    t_meas, t_mod = np.ceil(g.ms.to_numpy() / 50 - 1e-9), np.ceil(m / 50 - 1e-9)
    val.append({"branch": br, "frames": len(g), "rmse_ms": float(np.sqrt((res[(D.branch == br).to_numpy()] ** 2).mean())),
                "mean_meas": g.ms.mean(), "mean_model": m.mean(), "p50_meas": g.ms.median(), "p50_model": np.median(m),
                "p95_meas": g.ms.quantile(.95), "p95_model": np.quantile(m, .95),
                **{f"ticks{k}_meas": float((t_meas == k).mean()) for k in (1, 2, 3)},
                **{f"ticks{k}_model": float((t_mod == k).mean()) for k in (1, 2, 3)}})
V = pd.DataFrame(val)
json.dump({"a": float(beta[0]), "b_per_call": float(beta[1]), "c_per_mpx": float(beta[2]), "rmse_ms": float(np.sqrt((res ** 2).mean())),
           "r2": float(r2), "frames": len(D), "min_ms": float(D.ms.min()), "residuals": [round(float(x), 3) for x in res]},
          open(R / "k32_latency_model.json", "w"))
V.to_csv(R / "k32_latency_model_validation.csv", index=False)
print({"a": beta[0], "b": beta[1], "c": beta[2], "rmse": float(np.sqrt((res ** 2).mean())), "r2": r2, "frames": len(D)})
print(V.round(3).to_string(index=False))
