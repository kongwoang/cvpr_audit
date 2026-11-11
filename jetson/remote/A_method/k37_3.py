#!/usr/bin/env python
"""K37-3 (post-hoc): why AnA-style iterated (ana05hinfi) recovers more natural misses than v6-S.
Population (= K33 tab:missed, k33_report.missed_natural): on the evaluated test frames, the decision-relevant annotated
objects that the cheap draft misses (found_cheap False) and u960 + comp finds (found_wf960 True); recovery of a row =
found_<row> on that object (K33 records, frozen matching).
Classes:
  (i)  low-confidence draft box: a box of the frozen K33 pass-1 draft (YOLOv8s at the cheap mode, conf >= 0.05, i.e. AnA's
       draft threshold) with conf in [0.05, 0.25) and IoU >= 0.3 with the object. These boxes were never stored, so the
       pass-1 draft is RE-RUN here once, detector only (user-approved 2026-10-26), on the population frames; check: its
       boxes with conf >= 0.25 must equal the stored draft (nuScenes cache / K30 apdump cheap scene).
  (ii) the object's box centre inside one of v6-S's executed crops (K30 match crops_v6S): inside a field crop
       (field_v6S), else inside a support / guard crop, else outside.
usage: k37_3.py detect <ds>   (GPU; -> results/k37/k3/lo_<ds>.pkl)
       k37_3.py report        (CPU; -> results/k37/k3/{k37_3_objects.csv, k37_3_classes.csv, checks.json})
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
OUT = REAL / "results" / "k37" / "k3"


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, B / "scripts" / "A_method" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


KR = _load("k33r", "k33_report.py")
from rap.mono import box_iou                                                     # noqa: E402


def population(ds):
    """[(unit, frame, rel_index)] of the natural-miss objects + their found flags (cheap / wf960 / v6S / ana05hinfi)"""
    d = KR.frames(ds)
    out = []
    for i in range(len(d)):
        fc = d.found_cheap.iat[i]
        if not (isinstance(fc, (list, np.ndarray)) and len(fc)):
            continue
        fc, fu = np.asarray(fc, bool), np.asarray(d.found_wf960.iat[i], bool)
        fv, fa = np.asarray(d.found_v6S.iat[i], bool), np.asarray(d.found_ana05hinfi.iat[i], bool)
        for k in np.flatnonzero((~fc) & fu):
            out.append({"dataset": ds, "unit": str(d.unit.iat[i]), "frame": int(d.frame.iat[i]), "rel": int(k), "n_rel": len(fc),
                        "found_v6S": bool(fv[k]), "found_ana": bool(fa[k])})
    return pd.DataFrame(out)


MODE = {"nusc": ("ns_c320", 192, 320), "av2": ("av2_c320", 320, 256), "ahd": ("ahd_c320", 224, 320)}


def detect(ds):
    import cv2
    from rap.detect import Mode, TwoFidelityDetector
    K24 = _load("k24t", "k24_test.py")
    pop = population(ds)
    need = set(zip(pop.unit, pop.frame))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    CH = Mode(*MODE[ds])
    res = {}
    for unit, cam, gen in K24.units(ds):
        if not any(u == str(unit) for u, _ in need):
            continue
        for fr, path, v, g, cached in gen():
            if (str(unit), int(fr)) not in need:
                continue
            img = cv2.imread(path)
            lo = det.detect(img, replace(CH, conf=0.05))[0]
            res[(str(unit), int(fr))] = {"xyxy": np.asarray(lo["xyxy"], float).reshape(-1, 4), "conf": np.asarray(lo["conf"], float),
                                         "coarse": np.asarray(lo["coarse"])}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"lo_{ds}.pkl", "wb"))
    print("K37-3 DETECT", ds, len(res), "frames of", len(need))


def nusc_drafts(need):
    """the cached nuScenes drafts of the needed (unit, frame), in one pass over the units"""
    K24 = _load("k24t", "k24_test.py")
    units, out = {u for u, _ in need}, {}
    for u, cam, gen in K24.units("nusc"):
        if str(u) in units:
            out[str(u)] = {int(f): c for f, p, v, g, c in gen() if (str(u), int(f)) in need}
    return out


def stored_draft(ds, unit, fr, cache):
    if ds == "nusc":
        c = cache[unit][fr]
        return np.asarray(c["xyxy"], float).reshape(-1, 4), np.asarray(c["conf"], float)
    if unit not in cache:
        cache[unit] = pickle.load(open(REAL / "results" / "k30" / "apdump" / ds / f"{unit}.pkl", "rb"))["dets"]
    sc = cache[unit][(fr, "cheap")]["sc"]
    return np.asarray(sc["xyxy"], float).reshape(-1, 4), np.asarray(sc["conf"], float)


def report():
    checks, objs = {}, []
    for ds in ("nusc", "av2", "ahd"):
        pop = population(ds)
        lo = pickle.load(open(OUT / f"lo_{ds}.pkl", "rb"))
        gt = pd.read_pickle(REAL / "results" / "k30" / "gt" / f"{ds}.pkl").set_index(["unit", "frame"])
        mt = {}
        for f in sorted((REAL / "results" / "k30" / "match" / ds).glob("*.pkl")):
            for r in pickle.load(open(f, "rb")):
                mt[(str(r["unit"]), int(r["frame"]))] = (r["crops_v6S"], r["field_v6S"])
        cache, eq, nfr = (nusc_drafts(set(zip(pop.unit, pop.frame))) if ds == "nusc" else {}), 0, 0
        for (u, fr) in sorted(set(zip(pop.unit, pop.frame))):
            sx, scf = stored_draft(ds, u, fr, cache)
            L = lo[(u, fr)]
            hi = L["conf"] >= 0.25
            nfr += 1
            same = hi.sum() == len(sx) and (len(sx) == 0 or np.allclose(np.sort(L["xyxy"][hi], 0), np.sort(sx, 0), atol=1e-3))
            eq += int(same)
        checks[f"{ds}_frames"] = nfr
        checks[f"{ds}_rerun_draft_ge025_equals_stored"] = eq
        for r in pop.itertuples():
            g = gt.loc[(r.unit, r.frame)]
            rb = np.asarray(g["rel_box"], float).reshape(-1, 4)
            assert len(rb) == r.n_rel, (ds, r.unit, r.frame, len(rb), r.n_rel)
            box = rb[r.rel]
            L = lo[(r.unit, r.frame)]
            band = (L["conf"] >= 0.05) & (L["conf"] < 0.25)
            iou = box_iou(box[None], L["xyxy"][band])[0] if band.any() else np.zeros(0)
            crops, field = mt[(r.unit, r.frame)]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            inside = lambda cs: any(c[0] <= cx <= c[2] and c[1] <= cy <= c[3] for c in cs)          # noqa: E731
            fld = [np.asarray(c, float) for c in (field or [])]
            region = "field crop" if inside(fld) else ("support / guard crop" if inside([np.asarray(c, float) for c in (crops or [])]) else "outside")
            objs.append({"dataset": ds, "unit": r.unit, "frame": r.frame, "rel": r.rel, "lowconf_draft_box": bool((iou >= 0.3).any()),
                         "max_iou_lowconf": float(iou.max()) if len(iou) else 0.0, "v6S_region": region,
                         "found_v6S": r.found_v6S, "found_ana": r.found_ana})
    O = pd.DataFrame(objs)
    O.to_csv(OUT / "k37_3_objects.csv", index=False)
    rows = []
    for ds, g0 in O.groupby("dataset", sort=False):
        for key, g in [("all", g0)] + [(f"(i) low-conf draft box = {b}", x) for b, x in g0.groupby("lowconf_draft_box")] + \
                [(f"(ii) {k}", x) for k, x in g0.groupby("v6S_region")] + \
                [(f"(i)={b} x (ii) {k}", x) for (b, k), x in g0.groupby(["lowconf_draft_box", "v6S_region"])]:
            rows.append({"dataset": ds, "class": key, "objects": len(g), "recovery_v6S": g.found_v6S.mean(), "recovery_AnA": g.found_ana.mean(),
                         "AnA_minus_v6S": g.found_ana.mean() - g.found_v6S.mean(), "only_AnA": int((g.found_ana & ~g.found_v6S).sum()),
                         "only_v6S": int((g.found_v6S & ~g.found_ana).sum())})
    pd.DataFrame(rows).to_csv(OUT / "k37_3_classes.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(pd.DataFrame(rows).round(3).to_string()); print(checks)


if __name__ == "__main__":
    detect(sys.argv[2]) if sys.argv[1] == "detect" else report()
