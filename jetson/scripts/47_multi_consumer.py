#!/usr/bin/env python
"""D61: several consumers share one verification budget — braking and Planner B on the same frame.

KITTI mono, all units, exact-locality proxy (cached FULL detections inside crops). Loops:
  brake-only      crops from the braking field + support (vtf.spec_general.Brake)
  planB-only      crops from Planner B's field + pivots (vtf.spec_general.PlannerB)
  union           each round takes the union of both consumers' crops (one shared scene, both consumers served)
Each consumer is scored on the scene every loop produces, so the table shows what a consumer loses when the crops
were chosen for the other one, and what serving both costs in pixels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as PB
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import runmeta, spec_general as SG, speculative as SP

pp, cp = P.PlannerParams(), P.CostParams()


class Union:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def field(self, t, v, rg, yg):
        return self.a.field(t, v, rg, yg) | self.b.field(t, v, rg, yg)

    def support(self, t, v):
        return np.union1d(self.a.support(t, v), self.b.support(t, v)).astype(int)


def union_area(crops, W, H):
    if not crops:
        return 0.0
    m = np.zeros((int(H) // 4 + 1, int(W) // 4 + 1), bool)
    for c in crops:
        m[int(c[1]) // 4:int(np.ceil(c[3])) // 4 + 1, int(c[0]) // 4:int(np.ceil(c[2])) // 4 + 1] = True
    return m.mean()


def main():
    dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    rows = []
    brake = SG.Brake(pp, 10.0)
    planb = SG.PlannerB()
    pb, cb = planb.pb, planb.cb
    for s in sorted(set(dv.seq)):
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        prev_a, prev_p = None, None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gt = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            a_gt, _ = P.required_decel(*gt, v, pp)
            sc = c.scalars(i)
            W, H = float(sc["img_w"]), float(sc["img_h"])

            def pack(cache, idx):
                d, gg = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= 0.25
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "mono": tuple(np.asarray(gg[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))}

            ch, fu = pack(c, i), pack(f, f.index[fr])

            def detect_crop(crop):
                m = SP.centres_in(fu["xyxy"], [crop])
                return {"xyxy": fu["xyxy"][m], "conf": fu["conf"][m], "coarse": fu["coarse"][m]}

            def lift(full):
                idx = [int(np.flatnonzero((fu["xyxy"] == b).all(1))[0]) for b in full["xyxy"]]
                return tuple(a[idx] for a in fu["mono"])

            Jb = lambda t: P.decision_cost(P.discrete_action(P.required_decel(*t, v, pp)[0], pp), a_gt, prev_a, pp, cp)["J"]  # noqa: E731
            Jp = lambda t: PB.executed_cost(PB.plan(*t, v, prev_p, pb, cb), *gt, v, prev_p, pb, cb)["J"]                   # noqa: E731
            planb.prev = prev_p
            rec = {"seq": s, "frame": fr, "Jb_c": Jb(ch["mono"]), "Jb_f": Jb(fu["mono"]), "Jp_c": Jp(ch["mono"]),
                   "Jp_f": Jp(fu["mono"])}
            for name, cons in (("brake", brake), ("planb", planb), ("union", Union(brake, planb))):
                scene, crops, _ = SG.run(ch, v, cam, W, H, detect_crop, lift, cons, guard=False)
                rec.update({f"Jb_{name}": Jb(scene["mono"]), f"Jp_{name}": Jp(scene["mono"]),
                            f"px_{name}": union_area(crops, W, H), f"n_{name}": len(crops)})
            rows.append(rec)
            prev_a = P.discrete_action(P.required_decel(*ch["mono"], v, pp)[0], pp)
            prev_p = PB.plan(*ch["mono"], v, prev_p, pb, cb)
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows).merge(dv[["seq", "frame", "split"]], on=["seq", "frame"])
    d.to_pickle(runmeta.FINAL / "d61_multi_consumer_kitti.pkl")
    out = []
    for sp in ("all", "test"):
        q = d if sp == "all" else d[d.split == "test"]
        for name in ("f", "brake", "planb", "union"):
            out.append({"split": sp, "loop": {"f": "uniform full 640"}.get(name, name),
                        "G_brake": (q.Jb_c - q[f"Jb_{name}"]).sum() / q.Jb_c.sum(),
                        "G_planB": (q.Jp_c - q[f"Jp_{name}"]).sum() / q.Jp_c.sum(),
                        "pixels": q[f"px_{name}"].mean() if name != "f" else 1.0,
                        "crops": q[f"n_{name}"].mean() if name != "f" else np.nan})
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / "d61_multi_consumer_kitti.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
