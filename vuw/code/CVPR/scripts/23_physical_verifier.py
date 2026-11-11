#!/usr/bin/env python
"""D46 (exploratory): a physical-consistency verifier of cheap-decision pivots vs re-detection (KITTI, test split)."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from rap.cache import DetCache
from rap.mono import box_iou
from rap.paths import CACHE
from vtf import runmeta

rows = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / "*_verifier_accuracy_KITTI" / "verifier_rows.pkl")))[-1])
dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
dv = dv[dv.track == "KITTI"]
split = dict(zip(dv.seq, dv.split))
piv = rows[rows["pivot"]].pivot_table(index=["seq", "frame", "obj", "real", "cheap_conf", "box_h"],
                                      columns="verifier", values="best_conf").reset_index()
feat, cache = [], {}
for r in piv.itertuples():
    c = cache.get(r.seq) or cache.setdefault(r.seq, DetCache(Path(CACHE) / "det" / "cheap_320" / f"{r.seq}.npz"))
    i = c.index[r.frame]
    d, g, sc = c.det(i), c.geo(i), c.scalars(i)
    k = np.flatnonzero(d["conf"] >= 0.25)[r.obj]
    box = d["xyxy"][k].astype(float)

    def prev_iou(j):
        if i - j < 0:
            return 0.0
        dp = c.det(i - j)
        return float(box_iou(box[None], dp["xyxy"].astype(float)).max()) if len(dp["conf"]) else 0.0
    zg, zh = float(g["z_ground"][k]), float(g["z_height"][k])
    feat.append(dict(zg=zg, zh=zh, lzz=abs(np.log(max(zg, .5) / max(zh, .5))), bottom=box[3] / sc["img_h"],
                     top=box[1] / sc["img_h"], xc=(box[0] + box[2]) / 2 / sc["img_w"], w=(box[2] - box[0]) / sc["img_w"],
                     aspect=(box[3] - box[1]) / max(box[2] - box[0], 1), p1=prev_iou(1), p2=prev_iou(2), p3=prev_iou(3),
                     ent=float(d["binent"][k]), cls=float({"vehicle": 0, "person": 1, "cyclist": 2}[str(d["coarse"][k])])))
F = pd.concat([piv.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
F["split"] = F.seq.map(split)
phys = ["cheap_conf", "box_h", "zg", "zh", "lzz", "bottom", "top", "xc", "w", "aspect", "p1", "p2", "p3", "ent", "cls"]
tr, te = F[F.split != "test"], F[F.split == "test"]
out = []
for name, cols in (("re-detection (Y640, Y960, R640)", ["Y640", "Y960", "R640"]), ("physical consistency", phys),
                   ("physical + re-detection", phys + ["Y640", "Y960", "R640"])):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, min_samples_leaf=20,
                                       random_state=0).fit(tr[cols], ~tr.real)
    p, ptr = m.predict_proba(te[cols])[:, 1], m.predict_proba(tr[cols])[:, 1]
    for fr_ in (0.005, 0.01, 0.02, 0.05):
        thr = np.quantile(ptr[tr.real.values], 1 - fr_)
        say = p > thr
        out.append({"verifier": name, "target_real_called_phantom": fr_, "test_pivots": len(te),
                    "test_phantoms": int((~te.real).sum()), "real_called_phantom": float(np.mean(say[te.real.values])),
                    "phantoms_caught": float(np.mean(say[~te.real.values]))})
res = pd.DataFrame(out)
res.to_csv(runmeta.FINAL / "d46_physical_verifier.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())
