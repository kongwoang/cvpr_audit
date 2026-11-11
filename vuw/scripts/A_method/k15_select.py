#!/usr/bin/env python
"""K15 frame candidates for Figure 1 (CPU). From the instrumented base runs (nuScenes new test; AV2 val with the ego
mask, K10 shadow tree): frames where a decision-relevant annotated object is missed by the cheap draft (natural miss),
is NOT recovered by the uncertainty-crop heuristic, IS recovered by ours, ours (final, with refiner) takes the required
action, and the required action is not KEEP. For each candidate: number of annotated objects in the image (box height
>= 10 px) that are NOT decision-relevant (salient but irrelevant), mean image brightness (daylight proxy), ego speed.
usage: k15_select.py  ->  results/k15/candidates.csv
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402


def cands(root, ds):
    rows = []
    for f in sorted(glob.glob(str(root / "results" / "instr" / f"{ds}_test_base__*.pkl"))):
        for r in pickle.load(open(f, "rb")):
            if r["act_gt"] < 1 or r["n_relevant"] == 0 or "found_cheap" not in r:
                continue
            fc, fu, ff = (np.asarray(r[k], bool) for k in ("found_cheap", "found_uncertainty", "found_full"))
            hit = ~fc & ~fu & ff
            if hit.any() and r["a_full_ref"] == r["act_gt"]:
                rows.append({"dataset": ds, "unit": r["unit"], "frame": r["frame"], "v": r["v"], "act_gt": r["act_gt"],
                             "a_cheap": r["a_cheap"], "a_uncertainty": r["a_uncertainty"], "a_full_ref": r["a_full_ref"],
                             "a_u960": r["a_u960"], "a_native": r["a_native"], "n_relevant": r["n_relevant"],
                             "missed_idx": int(np.flatnonzero(hit)[0]), "px_full": r["px_full"]})
    return pd.DataFrame(rows)


def main():
    pp = P.PlannerParams()
    out = []
    for ds, root in (("nusc", B), ("av2", B / "k10_masked")):
        c = cands(root, ds)
        print(ds, len(c), "candidate frames in", c.unit.nunique() if len(c) else 0, "units", flush=True)
        want = set(zip(c.unit, c.frame))
        units = IE.units_nusc("test") if ds == "nusc" else IE.units_av2("test")
        extra = {}
        for unit, cam, frames in units:
            if unit not in set(c.unit):
                continue
            for fr, path, v, g, _ in frames():
                if (unit, fr) not in want:
                    continue
                img = cv2.imread(path)
                a, a_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
                rel = np.asarray(a_obj) >= pp.thr_decel
                relidx = np.flatnonzero(rel)
                k = c[(c.unit == unit) & (c.frame == fr)].iloc[0]
                mi = relidx[int(k.missed_idx)]
                extra[(unit, fr)] = {"path": path, "n_gt": len(g), "n_irrelevant": int((~rel).sum()),
                                     "n_irrelevant_vehicle": int(((~rel) & (np.asarray(g["coarse"]) == "vehicle")).sum()),
                                     "n_irrelevant_person": int(((~rel) & (np.asarray(g["coarse"]) != "vehicle")).sum()),
                                     "brightness": float(img.mean()) if img is not None else np.nan,
                                     "missed_cls": str(np.asarray(g["coarse"])[mi]), "missed_range": float(np.asarray(g["long_near"])[mi]),
                                     "missed_box": [float(np.asarray(g[q])[mi]) for q in ("x1", "y1", "x2", "y2")]}
        for (unit, fr), e in extra.items():
            out.append({**c[(c.unit == unit) & (c.frame == fr)].iloc[0].to_dict(), **e})
    res = pd.DataFrame(out).sort_values(["dataset", "n_irrelevant", "brightness"], ascending=[True, False, False])
    (B / "results" / "k15").mkdir(parents=True, exist_ok=True)
    res.to_csv(B / "results" / "k15" / "candidates.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns=["path", "missed_box"]).groupby("dataset").head(12).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
