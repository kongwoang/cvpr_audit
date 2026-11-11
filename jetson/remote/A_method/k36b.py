#!/usr/bin/env python
"""K36b (post-hoc, CPU only, frozen v6-S; no detector run; fixed frame rules). Reuses k36_f1 (draft reconstruction,
tube / footprint geometry, BEV style), whose checks passed in K36.
F1  nuScenes test, time order = scene name, then frame index; stored records results/k28/p3/nusc_test_instr:
    (a) K36's scene-0110 frame 0 (draft KEEP), re-rendered with the support crops (none at KEEP);
    (b) first frame with a_cheap = DECELERATE and ego speed in [8, 12] m/s;
    (c) first frame with a_cheap = HARD-BRAKE and ego speed in [8, 12] m/s;
    (d) first frame with ego speed < 2.5 m/s and a_cheap = KEEP.
    Field = the draft-round braking field (dsf.insertion_grid on the draft scene, v_close 0: cells where one inserted probe
    raises the draft action to the next level; empty at HARD-BRAKE); footprint = k_loop.field_crop_g; support crops = v6-S's
    round-1 support crops of the draft action (k_loop.box_crop_ctx, context SP.CTX, objects SP.support); tube = hybrid
    tsp8@S's tube rectangle (as K36). Support-crop area share = area of the union of the support crops / image area.
F2  all three test splits, evaluated (labelled) frames, grouped by draft action x ego-speed bin (< 2.5, 2.5-8, 8-12,
    > 12 m/s): frames; share with a non-empty draft-round field; mean footprint area share (0 when the field is empty;
    also the mean over non-empty fields); mean tube crop area share (0 when no tube is formed); mean stored px of v6-S
    and hybrid tsp8@S; paired unit bootstrap (2,000, seed 0) of the mean v6-S minus hybrid px per group.
    Draft: nuScenes cached; AV2 / Argoverse-HD the stored K30 apdump cheap scene lifted on the previous frame's draft
    (AV2 with the K10 ego mask y >= 1830, as every AV2 test run); checked per frame against the stored a_cheap.
usage: k36b.py   -> results/k36b/{panels/, f1_panels.csv, f1_cells.csv, f2_groups.csv, f2_frames.csv.gz, checks.json}
"""
from __future__ import annotations

import json
import pickle
import shutil

import cv2
import numpy as np
import pandas as pd

import k36_f1 as M
from k34_common import bootstrap, interval

P, SP, KL, IE, K24, dsf = M.P, M.SP, M.KL, M.IE, M.K24, M.dsf
OUT = M.REAL / "results" / "k36b"
PP0, RG, YG = M.PP0, M.RG, M.YG
NAMES = {0: "KEEP", 1: "DECELERATE", 2: "HARD-BRAKE"}
BINS = [(0, 2.5, "<2.5"), (2.5, 8, "2.5-8"), (8, 12, "8-12"), (12, np.inf, ">12")]
Y0 = 1830.0


def masked_geometry(dd, prev, cam, R, t, dt, mask):
    gk = M.predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, prev, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
    mono = [np.asarray(gk[x], float).copy() for x in ("z", "lat_min", "lat_max", "ttc")]
    if mask and len(dd["xyxy"]):                                   # = k10_masked_run.predicted_geometry
        b = np.asarray(dd["xyxy"], float).reshape(-1, 4)
        m = (b[:, 3] - np.maximum(b[:, 1], Y0)).clip(0, None) / np.maximum(b[:, 3] - b[:, 1], 1e-6) >= 0.5
        mono[0][m] = np.inf; mono[3][m] = np.inf
    return tuple(mono)


def union_area(crops, W, H):
    if not crops:
        return 0.0
    m = np.zeros((H, W), bool)
    for c in crops:
        x1, y1, x2, y2 = (int(round(float(q))) for q in c)
        m[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = True
    return float(m.mean())


def frame_quantities(G, dd, v, rec, W, H):
    mono = dd["mono"]
    sens = dsf.insertion_grid(mono, v, RG, YG, SP.OBJ_W, PP0, 0.0)
    fp = KL.field_crop_g(mono, v, G.cam, W, H, PP0, (0.0,), RG, YG, (KL.CAR,))
    sup = SP.support(mono, v, PP0)
    sup_crops = [KL.box_crop_ctx(dd["xyxy"][k], W, H, SP.CTX) for k in sup]
    crop51 = round(float(rec["cpx_v51S"]) * W * H)
    _, used = G.fill_px([SP.box_crop(dd["xyxy"][k], W, H) for k in sup], crop51)
    trect, thalf = G.tube(crop51 - used) if crop51 - used > 0 else (None, None)
    tcells = G.in_rect(trect)
    rr = np.meshgrid(RG, YG, indexing="ij")[0]
    return dict(sens=sens, fp=fp, sup_crops=sup_crops, trect=trect, thalf=thalf, tcells=tcells,
                n_sens=int(sens.sum()), rmax=float(rr[sens].max()) if sens.any() else 0.0,
                fp_share=0.0 if fp is None else float((fp[2] - fp[0]) * (fp[3] - fp[1]) / (W * H)),
                tube_share=0.0 if trect is None else float((trect[2] - trect[0]) * (trect[3] - trect[1]) / (W * H)),
                in_tube=float((sens & tcells).sum() / sens.sum()) if sens.any() else np.nan)


def units(ds):
    if ds == "nusc":
        return IE.units_nusc("test")
    if ds == "av2":
        return IE.units_av2("test")
    return K24.units("ahd")


SIZE = {"nusc": (1600, 900), "av2": (2048, 1550), "ahd": (1920, 1200)}


def dataset_frames(ds, checks, keep_frames=None):
    """per evaluated frame: draft action, speed, field / footprint / tube quantities, stored px. keep_frames: also return
    the per-frame drafts of these (unit, frame) keys (for F1 rendering)"""
    ins = M.recs(M.REAL / "results" / "k28" / "p3" / f"{ds}_test_instr")
    W, H = SIZE[ds]
    rows, kept, mism, nfr = [], {}, 0, 0
    for unit, cam, gen in units(ds):
        if unit not in ins:
            continue
        G = M.Geo(cam, W, H, ds if ds != "av2" else "nusc")
        G.scale = 960 / {"nusc": 1600, "av2": 2048, "ahd": 1920}[ds]
        R, t = cam.transform("ego")
        dt = float(cam.dt)
        ap = None if ds == "nusc" else pickle.load(open(M.REAL / "results" / "k30" / "apdump" / ds / f"{unit}.pkl", "rb"))["dets"]
        prev = None
        for fr, path, v, g, cached in gen():
            fr = int(fr)
            if fr not in ins[unit]:
                continue
            rec = ins[unit][fr]
            if ds == "nusc":
                dd = {k: cached[k] for k in ("xyxy", "conf", "coarse")}
                dd["mono"] = tuple(np.asarray(q, float) for q in cached["mono"])
            else:
                sc = ap[(fr, "cheap")]["sc"]
                dd = {"xyxy": sc["xyxy"], "conf": sc["conf"], "coarse": sc["coarse"]}
                dd["mono"] = masked_geometry(dd, prev, cam, R, t, dt, ds == "av2")
                prev = {"xyxy": dd["xyxy"], "coarse": dd["coarse"]}
            v = float(rec["v"])
            a_c = M.act(dd["mono"], v)
            nfr += 1
            mism += int(a_c != int(rec["a_cheap"]))
            if keep_frames is not None and (unit, fr) in keep_frames:
                kept[(unit, fr)] = (dd, path, cam)
            if not bool(rec["gt"]):
                continue
            q = frame_quantities(G, dd, v, rec, W, H)
            rows.append({"dataset": ds, "unit": unit, "frame": fr, "a_cheap": int(rec["a_cheap"]), "v": v,
                         "field_nonempty": q["n_sens"] > 0, "sensitive_cells": q["n_sens"], "footprint_share": q["fp_share"],
                         "tube_share": q["tube_share"], "tube_formed": q["trect"] is not None,
                         "share_sensitive_in_tube": q["in_tube"], "px_v6S": float(rec["px_v6S"]), "px_tsp8S": float(rec["px_tsp8@S+C1"])})
    checks[f"{ds}_frames_checked"] = nfr
    checks[f"{ds}_a_cheap_mismatches"] = mism
    return pd.DataFrame(rows), kept


def f2_groups(d):
    out = []
    d = d.copy()
    d["speed_bin"] = pd.cut(d.v, [b[0] for b in BINS] + [np.inf], right=False, labels=[b[2] for b in BINS])
    for (ds, a, sb), g in d.groupby(["dataset", "a_cheap", "speed_bin"], observed=True):
        r = {"dataset": ds, "draft_action": NAMES[a], "speed_bin": sb, "frames": len(g), "units": g.unit.nunique(),
             "share_field_nonempty": g.field_nonempty.mean(), "mean_footprint_share": g.footprint_share.mean(),
             "mean_footprint_share_nonempty": g.footprint_share[g.field_nonempty].mean() if g.field_nonempty.any() else np.nan,
             "share_tube_formed": g.tube_formed.mean(), "mean_tube_share": g.tube_share.mean(),
             "mean_share_sensitive_in_tube": g.share_sensitive_in_tube.mean(),
             "mean_px_v6S": g.px_v6S.mean(), "mean_px_tsp8S": g.px_tsp8S.mean(), "px_v6S_minus_tsp8S": (g.px_v6S - g.px_tsp8S).mean()}
        u, C = bootstrap(g.unit.to_numpy())
        su = (g.px_v6S - g.px_tsp8S).groupby(g.unit).sum().reindex(u).to_numpy()
        nu = g.groupby("unit").size().reindex(u).to_numpy()
        iv = interval((C @ su) / (C @ nu))
        r["ci_lo"], r["ci_hi"] = iv["ci_lo"], iv["ci_hi"]
        out.append(r)
    return pd.DataFrame(out)


def render(tag, key, dd, path, cam, rec, out):
    img = cv2.imread(path); H, W = img.shape[:2]
    G = M.Geo(cam, W, H, "nusc")
    v = float(rec["v"])
    q = frame_quantities(G, dd, v, rec, W, H)
    nm = f"{tag}_nusc_{key[0]}_{key[1]:04d}_brake_x1"
    M.bev(q["sens"], v, dd["mono"], out, nm + "_bev", [(G.rect_outline(q["trect"]), "--")] if q["trect"] is not None else [])
    M.K8.img_panel(img, ([(q["fp"], M.K8.ORANGE, "-", 3)] if q["fp"] is not None else []) + [(c, M.K8.BLUE, "-", 2) for c in q["sup_crops"]]
                   + ([(q["trect"], "black", "--", 2)] if q["trect"] is not None else []), out, nm + "_img")
    row = {"panel": nm, "unit": key[0], "frame": key[1], "draft_action": NAMES[int(rec["a_cheap"])], "v6S_action": NAMES[int(rec["a_v6S|v3a"])],
           "ego_speed_mps": v, "sensitive_cells": q["n_sens"], "farthest_sensitive_range_m": q["rmax"],
           "footprint_xyxy": None if q["fp"] is None else [round(float(x), 1) for x in q["fp"]], "footprint_area_share": q["fp_share"],
           "support_crops": len(q["sup_crops"]), "support_crop_area_share": union_area(q["sup_crops"], W, H),
           "tube_xyxy": None if q["trect"] is None else [round(float(x), 1) for x in q["trect"]], "tube_half_width_m": q["thalf"],
           "tube_area_share": q["tube_share"], "share_sensitive_in_tube": q["in_tube"],
           "px_v6S": float(rec["px_v6S"]), "px_tsp8S": float(rec["px_tsp8@S+C1"]), "image": path}
    cells = [{"panel": nm, "range_m": rg, "lateral_m": yg, "sensitive": bool(q["sens"][i, j]), "in_tube": bool(q["tcells"][i, j])}
             for i, rg in enumerate(RG) for j, yg in enumerate(YG) if q["sens"][i, j] or q["tcells"][i, j]]
    return row, cells


def main():
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    checks = {}
    ins = M.recs(M.REAL / "results" / "k28" / "p3" / "nusc_test_instr")
    order = [(u, f) for u in sorted(ins) for f in sorted(ins[u])]
    rules = {"a": lambda r: False,
             "b": lambda r: int(r["a_cheap"]) == P.DECELERATE and 8.0 <= float(r["v"]) <= 12.0,
             "c": lambda r: int(r["a_cheap"]) == P.HARD_BRAKE and 8.0 <= float(r["v"]) <= 12.0,
             "d": lambda r: float(r["v"]) < 2.5 and int(r["a_cheap"]) == P.KEEP}
    sel = {"a": ("scene-0110", 0)}
    for k in "bcd":
        sel[k] = next(x for x in order if rules[k](ins[x[0]][x[1]]))
    d_n, kept = dataset_frames("nusc", checks, set(sel.values()))
    rows, cells = [], []
    for tag in "abcd":
        key = sel[tag]
        dd, path, cam = kept[key]
        r, c = render(tag, key, dd, path, cam, ins[key[0]][key[1]], OUT / "panels")
        r["rule"] = {"a": "K36 frame (v6-S KEEP, 8-12 m/s, no draft object in the corridor <= 70 m)",
                     "b": "first a_cheap = DECELERATE, 8-12 m/s", "c": "first a_cheap = HARD-BRAKE, 8-12 m/s",
                     "d": "first ego speed < 2.5 m/s, a_cheap = KEEP"}[tag]
        rows.append(r); cells += c
    d = pd.concat([d_n] + [dataset_frames(ds, checks)[0] for ds in ("av2", "ahd")], ignore_index=True)
    pd.DataFrame(rows).to_csv(OUT / "f1_panels.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "f1_cells.csv", index=False)
    d.to_csv(OUT / "f2_frames.csv.gz", index=False)
    g = f2_groups(d)
    g.to_csv(OUT / "f2_groups.csv", index=False)
    checks["frames_per_dataset"] = d.groupby("dataset").size().to_dict()
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(pd.DataFrame(rows).drop(columns=["image"]).T.to_string())
    print(g.round(4).to_string()); print(checks)
    print("K36b DONE")


if __name__ == "__main__":
    main()
