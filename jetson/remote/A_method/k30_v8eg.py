#!/usr/bin/env python
"""K30 V8e / V8g (post-hoc, CPU; pre-registered in handoff/K30/STATUS.md).
V8e: frames of nuScenes new test / AV2 val (ego mask) with a non-COCO object in the ego corridor (results/k30/corner,
     k30_corner.py). Per pipeline (cheap, u960 + C1 + v3a, native + C1 + v3a, hybrid tsp8@S + C1 + v3a, v6-S): G on those
     frames (K28 phase 3 instr records; paired unit bootstrap vs v6-S) and the recall of those objects by any output box
     (class-agnostic IoU >= 0.3; the merged scene at conf >= 0.25 of the V7 apdump outputs, results/k30/apdump).
V8g: CODA base-val-1500's nuScenes images (sample tokens) that fall in nuScenes new test. A CODA box (2D only) is in the
     corridor if its ground-contact lift (bottom edge on the ground plane with the scene's camera height; lateral extent
     from its left / right edges) overlaps |y| <= 1.2 m at 0 < z <= 80 m. Recall of every CODA box and of the in-corridor
     ones by each pipeline's output boxes (class-agnostic IoU >= 0.3), and G of each pipeline on the CODA frames.
results/k30/tables/v8e_corner.csv, v8g_coda.csv
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rap.mono import box_iou

B = Path(os.environ.get("B_REAL", os.environ["B"]))
R30 = B / "results" / "k30"
NWS = {"nusc": 1600, "av2": 2048}
PIPES = {"cheap": ("cheap", "a_cheap", "J_cheap"), "u960 + C1 + v3a": ("wf960+C1", None, "J_wf960+C1|v3a"),
         "native + C1 + v3a": ("wf{n}+C1", None, "J_wf{n}+C1|v3a"), "hybrid tsp8@S + C1 + v3a": ("tsp8@S+C1", None, "J_tsp8@S+C1|v3a"),
         "v6-S": ("v6S", None, "J_v6S|v3a")}


def load_instr(ds):
    rows = []
    for f in sorted(glob.glob(str(B / "results" / "k28" / "p3" / f"{ds}_test_instr" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    return d[d["gt"].astype(bool)].reset_index(drop=True)


def load_dets(ds):
    dets = {}
    for f in sorted(glob.glob(str(R30 / "apdump" / ds / "*.pkl"))):
        x = pickle.load(open(f, "rb"))
        u = Path(f).stem
        for (fr, nm), o in x["dets"].items():
            dets[(u, fr, nm)] = np.asarray(o["sc"]["xyxy"], float).reshape(-1, 4)
    return dets


def G(L, J):
    return float((L.sum() - J.sum()) / L.sum()) if L.sum() > 0 else np.nan


def boot_diff(units, L, Ja, Jb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq, inv = np.unique(np.asarray(units), return_inverse=True)
    l, a, b = (np.bincount(inv, np.asarray(x, float), len(uq)) for x in (L, Ja, Jb))
    v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in (rng.integers(0, len(uq), len(uq)) for _ in range(n))]
    return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def recall_rows(tag, ds, objs, dets, nat):
    out = []
    for lab, (key, _, _) in PIPES.items():
        k = key.format(n=nat)
        hit, n = 0, 0
        for (u, fr), grp in objs.groupby(["unit", "frame"]):
            pb = dets.get((u, int(fr), k))
            if pb is None:
                continue
            gb = grp[["x1", "y1", "x2", "y2"]].to_numpy(float)
            n += len(gb)
            if len(pb):
                hit += int((box_iou(gb, pb).max(1) >= 0.3).sum())
        out.append({"item": tag, "dataset": ds, "pipeline": lab, "objects": n, "recall_iou0.3": hit / n if n else np.nan})
    return out


def main():
    out = []
    # ---------------- V8e
    for ds in ("nusc", "av2"):
        cf = R30 / "corner" / f"{ds}.pkl"
        if not cf.exists():
            continue
        c = pd.read_pickle(cf)
        cc = c[c.in_corridor]
        frames = cc[["unit", "frame"]].drop_duplicates()
        d = load_instr(ds)
        x = d.merge(frames, on=["unit", "frame"])
        nat = NWS[ds]
        L = x.J_cheap.to_numpy(float)
        for lab, (key, _, jc) in PIPES.items():
            if lab == "cheap":
                continue
            J = x[jc.format(n=nat)].to_numpy(float)
            r = {"item": "V8e", "dataset": ds, "pipeline": lab, "frames": len(x), "units": x.unit.nunique(), "G": G(L, J)}
            if lab != "v6-S":
                pt, lo, hi = boot_diff(x.unit, L, x["J_v6S|v3a"].to_numpy(float), J)
                r.update({"v6S_minus": pt, "ci_lo": lo, "ci_hi": hi})
            out.append(r)
        dets = load_dets(ds)
        if dets:
            out += recall_rows("V8e recall of in-corridor corner-case objects", ds, cc, dets, nat)
            for cls, g_ in cc.groupby("cls"):
                out += [dict(r, cls=cls) for r in recall_rows("V8e recall by class", ds, g_, dets, nat)]
    # ---------------- V8g (CODA, nuScenes)
    cod = B / "datasets" / "coda" / "CODA" / "base-val-1500"
    if cod.exists():
        from rap.cache import Camera
        from rap.nusc import NuScenesDB
        from rap.paths import NUSCENES_TRAINVAL
        idx = json.load(open(cod / "nuscenes_indices.json"))
        cj = json.load(open(cod / "corner_case.json"))
        cat = {c_["id"]: c_["name"] for c_ in cj["categories"]}
        imgs = {im["id"]: im["file_name"] for im in cj["images"] if im["file_name"] in idx}
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
        test = set(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
        tok2 = {}
        for sc in db.scenes:
            nm = db.scene_name(sc)
            if nm in test:
                for fi, tk in enumerate(db.samples(sc)):
                    tok2[tk] = (nm, fi)
        rows = []
        for a in cj["annotations"]:
            fn = imgs.get(a["image_id"])
            if fn is None or idx[fn] not in tok2:
                continue
            nm, fi = tok2[idx[fn]]
            x1, y1, w, h = a["bbox"]
            cam = Camera("nuScenes", nm)
            P2 = cam.calib.P2
            fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
            vb = y1 + h
            z = fy * cam.cam_h / (vb - cy) if vb > cy + 1 else np.inf
            ya, yb = -(x1 - cx) * z / fx, -(x1 + w - cx) * z / fx
            inc = bool(np.isfinite(z) and 0 < z <= 80 and min(ya, yb) < 1.2 and max(ya, yb) > -1.2)
            rows.append({"unit": nm, "frame": fi, "cls": cat[a["category_id"]], "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h,
                         "z_ground": z, "in_corridor": inc})
        co = pd.DataFrame(rows)
        out.append({"item": "V8g", "dataset": "nusc", "pipeline": "(CODA frames in nuScenes new test)",
                    "frames": int(co[["unit", "frame"]].drop_duplicates().shape[0]) if len(co) else 0, "objects": len(co),
                    "objects_in_corridor": int(co.in_corridor.sum()) if len(co) else 0})
        if len(co):
            dets = load_dets("nusc")
            if dets:
                out += recall_rows("V8g recall of all CODA objects", "nusc", co, dets, 1600)
                out += recall_rows("V8g recall of in-corridor CODA objects", "nusc", co[co.in_corridor], dets, 1600)
            d = load_instr("nusc")
            x = d.merge(co[["unit", "frame"]].drop_duplicates(), on=["unit", "frame"])
            L = x.J_cheap.to_numpy(float)
            for lab, (key, _, jc) in PIPES.items():
                if lab != "cheap":
                    out.append({"item": "V8g G on the CODA frames", "dataset": "nusc", "pipeline": lab, "frames": len(x),
                                "G": G(L, x[jc.format(n=1600)].to_numpy(float))})
            co.to_csv(R30 / "tables" / "v8g_coda_objects.csv", index=False)
    od = R30 / "tables"
    od.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(out)
    o.to_csv(od / "v8eg_corner_coda.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(3).to_string(index=False))
    print("K30 V8EG DONE")


if __name__ == "__main__":
    main()
