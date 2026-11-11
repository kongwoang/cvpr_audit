#!/usr/bin/env python
"""D63 / D53: a hard per-frame crop budget (for the p95 latency) and which crops to verify first.

The proved loop (exact-locality proxy, KITTI mono braking, all units) with at most `cap` crops per frame; when a
round asks for more, crops are taken in priority order:
  order     as the loop generates them (field first, then support in index order)
  deadline  earliest decision deadline first: a crop's deadline is the time the ego needs to reach its nearest range
            (support object: its range; field: the nearest sensitive cell), i.e. range / max(v, 1)  (D63, EDF)
  demand    highest demand first (support: its own demand; field: the largest demand among its sensitive cells)
Reports decision value (share of the all-cheap loss) and agreement with the full mode for cap in {1, 2, 3, inf}.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import dsf, runmeta, speculative as SP

V_CLOSE = 10.0
pp, cp = P.PlannerParams(), P.CostParams()


def loop(ch, fu, v, cam, W, H, cap, prio, lift):
    crops, full = [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8"), "idx": np.zeros(0, int)}
    scene = dict(ch)
    for _ in range(3):
        cand = []                                           # (crop, deadline, demand)
        sens = dsf.insertion_grid(scene["mono"], v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, V_CLOSE)
        fc = SP.field_crop(scene["mono"], v, cam, W, H, pp, V_CLOSE)
        if fc is not None and not SP.covered(fc, crops):
            rr = np.meshgrid(SP.R_GRID, SP.Y_GRID, indexing="ij")[0][sens]
            dem = dsf._insert_demand(rr, v, pp, V_CLOSE)
            cand.append((fc, rr.min() / max(v, 1.0), float(np.max(dem))))
        _, a_obj = P.required_decel(*scene["mono"], v, pp)
        for k in SP.support(scene["mono"], v, pp):
            bc = SP.box_crop(scene["xyxy"][k], W, H)
            if not SP.covered(bc, crops + [c[0] for c in cand]):
                cand.append((bc, float(scene["mono"][0][k]) / max(v, 1.0), float(a_obj[k])))
        if not cand:
            break
        left = cap - len(crops)
        if left <= 0:
            break
        if prio == "deadline":
            cand.sort(key=lambda c: c[1])
        elif prio == "demand":
            cand.sort(key=lambda c: -c[2])
        new = [c[0] for c in cand[:left]]
        crops += new
        m = SP.centres_in(fu["xyxy"], crops)
        full = {"xyxy": fu["xyxy"][m], "conf": fu["conf"][m], "coarse": fu["coarse"][m]}
        out = SP.centres_in(ch["xyxy"], crops)
        fz = lift(full)
        scene = {"xyxy": np.r_[ch["xyxy"][~out], full["xyxy"]].reshape(-1, 4),
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(ch["mono"], fz))}
    return scene, len(crops)


def main():
    dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    rows = []
    for s in sorted(set(dv.seq)):
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        prev_act = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            sc = c.scalars(i)
            W, H = float(sc["img_w"]), float(sc["img_h"])

            def pack(cache, idx):
                d, gg = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= 0.25
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "mono": tuple(np.asarray(gg[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))}

            ch, fu = pack(c, i), pack(f, f.index[fr])

            def lift(full):
                idx = [int(np.flatnonzero((fu["xyxy"] == b).all(1))[0]) for b in full["xyxy"]]
                return tuple(a[idx] for a in fu["mono"])

            act = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            a_c, a_f = act(ch["mono"]), act(fu["mono"])
            rec = {"seq": s, "frame": fr, "split": None, "J_c": cost(a_c), "J_f": cost(a_f)}
            for cap in (1, 2, 3, 99):
                for prio in (("order", "deadline", "demand") if cap < 99 else ("order",)):
                    scene, n = loop(ch, fu, v, cam, W, H, cap, prio, lift)
                    a = act(scene["mono"])
                    rec[f"J_{cap}_{prio}"], rec[f"agree_{cap}_{prio}"], rec[f"n_{cap}_{prio}"] = cost(a), int(a == a_f), n
            rows.append(rec)
            prev_act = a_c
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows).merge(dv[["seq", "frame", "split"]], on=["seq", "frame"], suffixes=("_x", ""))
    d.to_pickle(runmeta.FINAL / "d63_crop_budget_kitti.pkl")
    out = []
    for sp in ("all", "test"):
        q = d if sp == "all" else d[d.split == "test"]
        L = q.J_c.sum()
        for col in [c for c in d.columns if c.startswith("J_") and c not in ("J_c", "J_f")]:
            key = col[2:]
            out.append({"split": sp, "cap_prio": key, "G": (q.J_c - q[col]).sum() / L, "agree_full": q[f"agree_{key}"].mean(),
                        "crops_mean": q[f"n_{key}"].mean(), "crops_p95": q[f"n_{key}"].quantile(0.95)})
        out.append({"split": sp, "cap_prio": "uniform full", "G": (q.J_c - q.J_f).sum() / L, "agree_full": 1.0})
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / "d63_crop_budget_kitti.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
