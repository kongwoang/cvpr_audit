#!/usr/bin/env python
"""K40 (private helper): per-frame records of the published methods in their official form (tab:official) ->
records/<dataset>_official.parquet (nuScenes, AV2): J of AdaScale / FOVEA / SAHI from their evaluation pickles (J_c
must equal J_cheap), per-frame net px of AdaScale, and the per-frame J (no refiner / refiner) and px of every router
of k28_analyze_test2.routers_v6 matched to v6-S (Budget-Adaptive Routing, decision-value router, oracle; Argoverse-HD: the
routers only).
usage: k40_export_official.py <out_dir>"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def slug(s):
    """short neutral router names: bar_<estimator> / dvr_<estimator> / oracle, suffix _comp (+ components)"""
    comp = "_comp" if s.endswith("+ components") else ""
    if s.startswith("oracle"):
        return "oracle" + comp
    est = re.search(r": (\S+) @", s).group(1)
    return ("bar_" if "Budget-Adaptive" in s else "dvr_") + est + comp


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    T2 = mod("k28t2", B / "scripts/A_method/k28_analyze_test2.py")
    K24A, KA = T2.K24A, T2.KA
    GF = dict(KA.GF)
    gfp = T2.REAL / "results/k28/phase0/gflops_640.csv"
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    S = B / "k10_masked"
    for ds in ("nusc", "av2", "ahd"):
        d, why = K24A.merged(ds, "main", "supp")
        assert d is not None, why
        for part in ("ours", "abl"):
            p, n = T2.load(ds, part)
            keep = ["unit", "frame", "J_cheap"] + [c for c in p.columns if c != "J_cheap" and c.startswith(("J_", "a_", "px_"))]
            p = p[keep].rename(columns={c: f"{c}@{part}" for c in keep[3:] if c in d.columns})
            d = d.merge(p, on=["unit", "frame"], how="left", suffixes=("", f"_{part}"))
        px = d["px_v6S"]
        target = float((px * GF["yolov8s"]).mean()) / GF["yolov8s"]
        out = d[["unit", "frame", "J_cheap"]].copy()
        out["unit"] = out.unit.astype(str)
        for lab, e in T2.routers_v6(d, ds, {"v6-S": target}).items():
            s = slug(lab)
            out[f"J_{s}|none"], out[f"J_{s}|v3a"] = np.asarray(e["Jn"], float), np.asarray(e["Jr"], float)
            out[f"px_{s}"] = float(e["px"])
        srcs = ({"ada": (B / "results/adascale/eval_newtest.pkl", "seq"), "fovea": (B / "results/fovea/eval_newtest.pkl", "seq"),
                 "sahi": (B / "results/sahi/eval_newtest.pkl", "seq"),
                 # fine-tuned detectors (two seeds): FOVEA-ft / u960-ft / v3 verifier, LZU, UFP, ESOD, ZoomDet
                 **{f"{n}@s{k}": (B / "results" / n.split("_")[0] / (f + ("_s1" if k else "") + ".pkl"), "seq")
                    for n, f in (("fovea_ft", "eval_ft_newtest"), ("lzu", "eval_newtest"), ("ufp_ft", "eval_ft_newtest"),
                                 ("esod", "eval_newtest"), ("zoomdet", "eval_newtest")) for k in (0, 1)}} if ds == "nusc" else
                {"ada": (S / "results/av2_sota/ada_eval_test.pkl", "log"), "base": (S / "results/av2/baselines__val.pkl", "log")} if ds == "av2" else {})
        for nm, (pth, k) in srcs.items():
            o = pd.read_pickle(pth).rename(columns={k: "unit"})
            o["unit"] = o.unit.astype(str)
            cols = [c for c in o.columns if (c.startswith(("J_", "tpx_")) and c != "J_c")] + (["net_px"] if "net_px" in o else [])
            m = out[["unit", "frame", "J_cheap"]].merge(o[["unit", "frame", "J_c"] + cols], on=["unit", "frame"], how="left", validate="1:1")
            assert np.allclose(m.J_c.to_numpy(float), m.J_cheap.to_numpy(float)), (ds, nm, "J_c differs from J_cheap")
            for c in cols:
                name = c.replace("J_esod", "J_zoomdet") if nm.startswith("zoomdet") else c     # ZoomDet's pickle reused the column name
                sfx = nm[nm.index("@"):] if "@" in nm else ""
                out[f"official.{name}{sfx}" if c != "net_px" else f"official.{nm}_net_px"] = m[c].to_numpy()
        f = od / f"{NAME[ds]}_official.parquet"
        out.sort_values(["unit", "frame"]).to_parquet(f, index=False, compression="zstd")
        print(ds, out.shape, [c for c in out.columns if "budget" in c or "official" in c][:30], flush=True)


if __name__ == "__main__":
    main()
