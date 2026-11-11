#!/usr/bin/env python
"""K36-F1 / F2 (post-hoc, CPU only, frozen v6-S; no detector run, no selection on outcomes).
Draft-round sensitivity fields on stored test frames, k28_render / k8_render BEV style (PDF + PNG, no text), with the
hybrid's ego-corridor tube (tsp8@S = TUBE[8]: near 10 m, far 70 m, 'fill' width grown within the S budget) outlined.
Draft scene: nuScenes = the cached draft (with its mono geometry, as every K28 / K30 run); Argoverse-HD = the stored K30
apdump cheap scene (conf >= 0.25) lifted with predicted_geometry on the previous frame's draft (as k30_eval2's draft);
both are checked by recomputing a_cheap on every frame of the unit against the stored K28 instr record.
Frame selection (fixed, in time order = unit name, then frame index):
  nuScenes test   first frame with stored a_v6S|v3a = KEEP, ego speed in [8, 12] m/s and no draft object inside the
                  corridor (|lateral extent| overlapping +-corridor_half_w) at range <= 70 m.
  Argoverse-HD    first frame with |kappa| > 0.01 (K31 item 3 curvature, results/k31/r3/ahd_curv), stored a_v6S|v3a =
  val (curved)    KEEP and no draft object inside the straight corridor (range <= 70 m) nor inside the curved
                  corridor (path coordinates, arc length <= 70 m).
Panels: nuScenes braking consumer at ego speed x0.5 / x1 / x1.5 (dsf.insertion_grid with the scaled speed, probe
SP.OBJ_W, v_close 0, as K30 V8d); TTC-warning consumer (instr_eval.TTCWarn(pp, 0), K9 / K31) at x1; Argoverse-HD:
the K31 curved consumer's field (k31_eval.curved_grid, probe KL.CAR) with the straight and the path-following tubes.
Footprint (crop that would be requested): k_loop.field_crop_g (braking), spec_general.footprint (TTC warning), the same
formula on the curved field (curved consumer, as K31's module substitution).
Tube geometry: the frame's exact tube rectangle (k28_p3 / k31_eval `tube` / `tube_c`, budget = v5.1-S crop px of the
frame from the stored record minus the support crops filled first). A BEV cell is "in the tube" iff its ground-contact
point, under the method's own projection (u = cx - fx y / r, v = cy + fy h / r), lies inside the tube rectangle.
usage: k36_f1.py   -> results/k36/{panels/, k36_cells.csv, k36_panels.csv, k36_selection.csv, k36_f2.csv, checks.json}
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, B / "scripts" / "A_method" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


K8 = _load("k8", "k8_render.py")
K31 = _load("k31", "k31_eval.py")
IE, KL, K24 = K31.IE, K31.KL, K31.K24
from rap import planner as P                                                     # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import dsf                                                              # noqa: E402
from vtf import spec_general as SG                                               # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = REAL / "results" / "k36"
NWS = {"nusc": 1600, "ahd": 1920}
NEAR, FAR = K31.TUBE[8][0], K31.TUBE[8][1]
assert K31.TUBE[8][2] == "fill"
PP0 = P.PlannerParams()
CW = PP0.corridor_half_w
RG, YG = np.asarray(SP.R_GRID, float), np.asarray(SP.Y_GRID, float)


def recs(folder):
    out = {}
    for f in sorted(glob.glob(str(folder / "*.pkl"))):
        rr = pickle.load(open(f, "rb"))
        rr = rr["rows"] if isinstance(rr, dict) else rr
        out[Path(f).stem] = {int(r["frame"]): r for r in rr}
    return out


class Geo:
    """camera constants + the k28_p3 / k31_eval crop helpers (ckey at scale 960 / NW, canvas, fill_px, tube, tube_c)"""

    def __init__(self, cam, W, H, ds):
        P2 = cam.calib.P2
        self.fx, self.fy, self.cx, self.cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        self.h, self.W, self.H, self.cam = cam.cam_h, W, H, cam
        self.scale = 960 / NWS[ds]

    def canvas(self, cr):
        x1, y1 = int(cr[0]), int(cr[1])
        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
        xe, ye = min(x2, self.W), min(y2, self.H)
        return max(32, int(math.ceil((ye - y1) * self.scale / 32)) * 32) * max(32, int(math.ceil((xe - x1) * self.scale / 32)) * 32)

    def fill_px(self, cands, bud):
        sel, used = [], 0
        for c in cands:
            cv_ = self.canvas(c)
            if used + cv_ <= bud:
                sel.append(c); used += cv_
        return sel, used

    def tube(self, bud):
        vb_n = min(self.cy + self.fy * self.h / NEAR, self.H)
        top = max(self.cy + self.fy * self.h / FAR - self.fy * 1.6 / FAR, 0)
        best, hm = None, None
        for half_m in np.arange(0.5, 12.01, 0.25):
            half = self.fx * half_m / NEAR
            c = np.array([max(self.cx - half, 0), top, min(self.cx + half, self.W), vb_n])
            if self.canvas(c) > bud:
                break
            best, hm = c, float(half_m)
        return best, hm

    def tube_c(self, bud, k):
        if abs(k) < 1e-6:
            return self.tube(bud)
        ss = np.arange(NEAR, FAR + 1e-6, 2.0)
        R_ = 1.0 / k
        px_, py_ = R_ * np.sin(ss / R_), R_ * (1 - np.cos(ss / R_))
        nx_, ny_ = -np.sin(ss / R_), np.cos(ss / R_)
        best, hm = None, None
        for half_m in np.arange(0.5, 12.01, 0.25):
            X_ = np.r_[px_ + half_m * nx_, px_ - half_m * nx_]
            Y_ = np.r_[py_ + half_m * ny_, py_ - half_m * ny_]
            ok = X_ > 1.0
            if not ok.any():
                break
            X_, Y_ = X_[ok], Y_[ok]
            u_ = self.cx - self.fx * Y_ / X_
            vb_ = self.cy + self.fy * self.h / X_
            vt_ = vb_ - self.fy * 1.6 / X_
            c = np.array([max(u_.min(), 0), max(vt_.min(), 0), min(u_.max(), self.W), min(vb_.max(), self.H)])
            if c[2] - c[0] < 8 or c[3] - c[1] < 8 or self.canvas(c) > bud:
                break
            best, hm = c, float(half_m)
        return best, hm

    def footprint(self, sens, w, h):
        if not sens.any():
            return None
        rr, yy = np.meshgrid(RG, YG, indexing="ij")
        rr, yy = rr[sens], yy[sens]
        vb = self.cy + self.fy * self.h / rr
        uc, hw = self.cx - self.fx * yy / rr, self.fx * w / (2 * rr)
        return np.array([max((uc - hw).min(), 0), max((vb - self.fy * h / rr).min(), 0), min((uc + hw).max(), self.W),
                         min(vb.max(), self.H)])

    def in_rect(self, rect):
        """BEV cells whose ground-contact point projects inside the image rectangle"""
        if rect is None:
            return np.zeros((len(RG), len(YG)), bool)
        rr, yy = np.meshgrid(RG, YG, indexing="ij")
        u, v = self.cx - self.fx * yy / rr, self.cy + self.fy * self.h / rr
        return (u >= rect[0]) & (u <= rect[2]) & (v >= rect[1]) & (v <= rect[3])

    def rect_outline(self, rect, rmax=80.0):
        """BEV preimage outline of an image rectangle (ground contact), clipped to the plotted range"""
        if rect is None:
            return None
        x1, y1, x2, y2 = rect
        r0 = self.fy * self.h / (y2 - self.cy) if y2 > self.cy else 0.5
        r1 = min(self.fy * self.h / (y1 - self.cy), rmax) if y1 > self.cy else rmax
        r = np.linspace(max(r0, 0.5), r1, 200)
        lo, hi = (self.cx - x2) * r / self.fx, (self.cx - x1) * r / self.fx
        return np.r_[lo, hi[::-1], lo[:1]], np.r_[r, r[::-1], r[:1]]


def in_corridor_straight(mono, rmax=70.0):
    z, lo, hi, _ = (np.asarray(a, float) for a in mono)
    return (z <= rmax) & (lo < CW) & (hi > -CW)


def in_corridor_curved(mono, k, smax=70.0):
    s, lo, hi, _ = K31.curved(mono, k)
    return (np.asarray(s) <= smax) & (np.asarray(lo) < CW) & (np.asarray(hi) > -CW)


def act(mono, v):
    return P.discrete_action(P.required_decel(*mono, v, PP0)[0], PP0)


def bev(sens, v, mono, out, name, tubes, corridor=None):
    """k8_render.bev_panel + tube outlines (black dashed: straight tube; black dotted: path-following tube) and, for the
    curved consumer, the path-following corridor (grey) instead of the straight one"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    dr = np.diff(np.r_[RG, RG[-1] + 15]); dy = 1.5
    for i, r in enumerate(RG):
        for j, y in enumerate(YG):
            ax.add_patch(plt.Rectangle((y - dy / 2, r - dr[i] / 2), dy, dr[i], facecolor=K8.ORANGE if sens[i, j] else "none",
                                       edgecolor="#cccccc", linewidth=0.5, alpha=0.8 if sens[i, j] else 1))
    if corridor is None:
        for x in (-CW, CW):
            ax.plot([x, x], [0, 80], color=K8.GREY, linewidth=1)
    else:
        for xs, ys in corridor:
            ax.plot(xs, ys, color=K8.GREY, linewidth=1)
    for (xs, ys), ls in tubes:
        ax.plot(xs, ys, color="black", linestyle=ls, linewidth=1.2)
    ax.add_patch(plt.Rectangle((-0.9, -4.5), 1.8, 4.5, facecolor="black"))
    ax.arrow(0, 0, 0, max(v, 0.5) * 1.0, width=0.15, color="black")
    z, lmin, lmax, _ = mono
    for q in range(len(z)):
        ax.add_patch(plt.Rectangle((lmin[q], z[q]), lmax[q] - lmin[q], 1.5, facecolor=K8.BLUE, alpha=0.8))
    ax.set_xlim(-8, 8); ax.set_ylim(-6, 80); ax.set_aspect("auto")
    K8.save(fig, out, name)


def curved_corridor(k, smax=80.0):
    s = np.linspace(0, smax, 200)
    if abs(k) < 1e-6:
        return [(np.full_like(s, -CW), s), (np.full_like(s, CW), s)]
    R_ = 1.0 / k
    out = []
    for d in (-CW, CW):
        x = (R_ - d) * np.sin(s / R_)
        y = R_ - (R_ - d) * np.cos(s / R_)
        out.append((y, x))                       # plot axes: lateral (left +) horizontal, forward vertical
    return out


def curved_band(k, half_m):
    """the path-following tube's band: arc length NEAR..FAR, |offset| <= half_m (outline, plot axes)"""
    s = np.linspace(NEAR, FAR, 100)
    R_ = 1.0 / k
    pts = []
    for d, ss in ((half_m, s), (-half_m, s[::-1])):
        x = (R_ - d) * np.sin(ss / R_)
        y = R_ - (R_ - d) * np.cos(ss / R_)
        pts.append((y, x))
    ys = np.r_[pts[0][0], pts[1][0], pts[0][0][:1]]
    xs = np.r_[pts[0][1], pts[1][1], pts[0][1][:1]]
    return ys, xs


def nusc_unit(unit):
    for u, cam, gen in IE.units_nusc("test"):
        if u == unit:
            return cam, gen
    raise KeyError(unit)


def ahd_unit(unit):
    for u, cam, gen in K24.units("ahd"):
        if u == unit:
            return cam, gen
    raise KeyError(unit)


def main():
    (OUT / "panels").mkdir(parents=True, exist_ok=True)
    checks, sel_rows, cell_rows, panel_rows = {}, [], [], []
    # ------------------------------------------------------------------ nuScenes test
    ins = recs(REAL / "results" / "k28" / "p3" / "nusc_test_instr")
    chosen = None
    for unit in sorted(ins):
        cand = [f for f, r in sorted(ins[unit].items()) if int(r["a_v6S|v3a"]) == P.KEEP and 8.0 <= float(r["v"]) <= 12.0]
        if not cand:
            continue
        cam, gen = nusc_unit(unit)
        drafts = {}
        for fr, path, v, g, cached in gen():
            drafts[int(fr)] = (cached, path, v)
        mism = sum(int(act(drafts[f][0]["mono"], float(ins[unit][f]["v"])) != int(ins[unit][f]["a_cheap"])) for f in drafts if f in ins[unit])
        for f in cand:
            ch = drafts[f][0]
            n_in = int(in_corridor_straight(ch["mono"]).sum())
            sel_rows.append({"dataset": "nusc", "unit": unit, "frame": f, "v": float(ins[unit][f]["v"]), "draft_in_corridor_70m": n_in,
                             "selected": n_in == 0})
            if n_in == 0:
                chosen = (unit, f, cam, drafts, mism)
                break
        if chosen:
            break
    unit, f, cam, drafts, mism = chosen
    checks["nusc_draft_a_cheap_mismatches_in_unit"] = mism
    ch, path, v0 = drafts[f]
    r = ins[unit][f]
    assert abs(float(v0) - float(r["v"])) < 1e-6
    img = cv2.imread(path); H, W = img.shape[:2]
    G = Geo(cam, W, H, "nusc")
    mono = tuple(np.asarray(q, float) for q in ch["mono"])
    crop51 = round(float(r["cpx_v51S"]) * W * H)
    sup = SP.support(mono, float(v0), PP0)
    _, used = G.fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in sup], crop51)
    trect, thalf = G.tube(crop51 - used) if crop51 - used > 0 else (None, None)
    tube_cells = G.in_rect(trect)
    stem = f"nusc_{unit}_{f:04d}"
    panels = [(f"brake_x{s}", dsf.insertion_grid(mono, float(v0) * s, RG, YG, SP.OBJ_W, PP0, 0.0), float(v0) * s, KL.CAR, s)
              for s in (0.5, 1.0, 1.5)]
    ttc = IE.TTCWarn(PP0, 0.0)
    panels.append(("ttcwarn_x1", ttc.field(mono, float(v0), RG, YG), float(v0), (SP.OBJ_W, SP.OBJ_H), 1.0))
    for name, sens, v, probe, s in panels:
        fp = G.footprint(sens, *probe)
        if name.startswith("brake"):
            ref = KL.field_crop_g(mono, v, cam, W, H, PP0, (0.0,), RG, YG, (KL.CAR,))
            assert (fp is None and ref is None) or np.allclose(fp, ref), name
        else:
            ref = SG.footprint(sens, cam, W, H, RG, YG)
            assert (fp is None and ref is None) or np.allclose(fp, ref), name
        nm = f"{stem}_{name}"
        bev(sens, v, mono, OUT / "panels", nm + "_bev", [(G.rect_outline(trect), "--")] if trect is not None else [])
        K8.img_panel(img, ([(fp, K8.ORANGE, "-", 3)] if fp is not None else []) + ([(trect, "black", "--", 2)] if trect is not None else []),
                     OUT / "panels", nm + "_img")
        rr = np.meshgrid(RG, YG, indexing="ij")[0]
        panel_rows.append({"panel": nm, "dataset": "nuScenes test", "unit": unit, "frame": f, "consumer": name, "speed_scale": s,
                           "ego_speed_mps": v, "sensitive_cells": int(sens.sum()), "farthest_sensitive_range_m": float(rr[sens].max()) if sens.any() else 0.0,
                           "footprint_xyxy": None if fp is None else [round(float(x), 1) for x in fp],
                           "footprint_area_share": 0.0 if fp is None else float((fp[2] - fp[0]) * (fp[3] - fp[1]) / (W * H)),
                           "tube_xyxy": None if trect is None else [round(float(x), 1) for x in trect], "tube_half_width_m_at_near": thalf,
                           "tube_cells": int(tube_cells.sum()),
                           "share_sensitive_in_tube": float((sens & tube_cells).sum() / sens.sum()) if sens.any() else np.nan,
                           "image": path})
        for i, rg in enumerate(RG):
            for j, yg in enumerate(YG):
                if sens[i, j] or tube_cells[i, j]:
                    cell_rows.append({"panel": nm, "range_m": rg, "lateral_m": yg, "sensitive": bool(sens[i, j]), "in_tube": bool(tube_cells[i, j])})
    # ------------------------------------------------------------------ Argoverse-HD val, curved
    ins = recs(REAL / "results" / "k28" / "p3" / "ahd_test_instr")
    cur = recs(REAL / "results" / "k31" / "r3" / "ahd_curv")
    chosen = None
    for unit in sorted(ins):
        cand = [fr for fr, x in sorted(cur[unit].items()) if abs(float(x["kappa"])) > 0.01 and int(ins[unit][fr]["a_v6S|v3a"]) == P.KEEP]
        if not cand:
            continue
        ap = pickle.load(open(REAL / "results" / "k30" / "apdump" / "ahd" / f"{unit}.pkl", "rb"))["dets"]
        cam, gen = ahd_unit(unit)
        R, t = cam.transform("ego")
        dt = float(cam.dt)
        prev, drafts, mism = None, {}, 0
        for fr, path, v, g, cached in gen():
            sc = ap[(int(fr), "cheap")]["sc"]
            dd = {"xyxy": sc["xyxy"], "coarse": sc["coarse"], "conf": sc["conf"]}
            gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, prev, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
            dd["mono"] = tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            prev = {"xyxy": dd["xyxy"], "coarse": dd["coarse"]}
            drafts[int(fr)] = (dd, path, float(v), g)
            mism += int(act(dd["mono"], float(v)) != int(ins[unit][int(fr)]["a_cheap"]))
        for fr in cand:
            dd = drafts[fr][0]
            k_ = float(cur[unit][fr]["kappa"])
            n_s, n_c = int(in_corridor_straight(dd["mono"]).sum()), int(in_corridor_curved(dd["mono"], k_).sum())
            sel_rows.append({"dataset": "ahd", "unit": unit, "frame": fr, "v": drafts[fr][2], "kappa": k_,
                             "draft_in_corridor_70m": n_s, "draft_in_curved_corridor_70m": n_c, "selected": n_s == 0 and n_c == 0})
            if n_s == 0 and n_c == 0:
                chosen = (unit, fr, cam, drafts, mism, k_)
                break
        if chosen:
            break
    unit, f, cam, drafts, mism, k_ = chosen
    checks["ahd_draft_a_cheap_mismatches_in_unit"] = mism
    dd, path, v0, _ = drafts[f]
    r = ins[unit][f]
    img = cv2.imread(path); H, W = img.shape[:2]
    G = Geo(cam, W, H, "ahd")
    mono = dd["mono"]
    crop51 = round(float(r["cpx_v51S"]) * W * H)
    _, used = G.fill_px([SP.box_crop(dd["xyxy"][q], W, H) for q in SP.support(mono, v0, PP0)], crop51)
    srect, shalf = G.tube(crop51 - used) if crop51 - used > 0 else (None, None)
    supc = K31.curved_support(mono, v0, PP0, k_)
    _, usedc = G.fill_px([SP.box_crop(dd["xyxy"][q], W, H) for q in supc], crop51)
    crect, chalf = G.tube_c(crop51 - usedc, k_) if crop51 - usedc > 0 else (None, None)
    # check of the tube reconstruction: K31's stored flags 'decisive object centre in the straight / curved tube' on
    # every frame of the unit with a decisive object (curved requirement)
    flag_mis, flag_n = 0, 0
    for fr, x in sorted(cur[unit].items()):
        if int(x.get("decc", -1)) < 0 or "decc_in_straight_tube" not in x:
            continue
        d2, _, v2, g2 = drafts[fr]
        k2 = float(x["kappa"])
        c51 = round(float(ins[unit][fr]["cpx_v51S"]) * W * H)
        _, u2 = G.fill_px([SP.box_crop(d2["xyxy"][q], W, H) for q in SP.support(d2["mono"], v2, PP0)], c51)
        s2 = G.tube(c51 - u2)[0] if c51 - u2 > 0 else None
        _, u3 = G.fill_px([SP.box_crop(d2["xyxy"][q], W, H) for q in K31.curved_support(d2["mono"], v2, PP0, k2)], c51)
        c2 = G.tube_c(c51 - u3, k2)[0] if c51 - u3 > 0 else None
        bdec = np.stack([g2["x1"], g2["y1"], g2["x2"], g2["y2"]], 1).astype(float)[int(x["decc"]):int(x["decc"]) + 1]
        fs = bool(SP.centres_in(bdec, [s2])[0]) if s2 is not None else False
        fc = bool(SP.centres_in(bdec, [c2])[0]) if c2 is not None else False
        flag_n += 1
        flag_mis += int(fs != bool(x["decc_in_straight_tube"])) + int(fc != bool(x["decc_in_curved_tube"]))
    checks["ahd_tube_flag_frames"], checks["ahd_tube_flag_mismatches"] = flag_n, flag_mis
    sens = K31.curved_grid(k_)(mono, v0, RG, YG, KL.CAR[0], PP0, 0.0)
    fp = G.footprint(sens, *KL.CAR)
    s_cells, c_cells = G.in_rect(srect), G.in_rect(crect)
    nm = f"ahd_{unit}_{f:04d}_curved"
    tubes = ([(G.rect_outline(srect), "--")] if srect is not None else []) + ([(curved_band(k_, chalf), ":")] if crect is not None else [])
    bev(sens, v0, mono, OUT / "panels", nm + "_bev", tubes, corridor=curved_corridor(k_))
    K8.img_panel(img, ([(fp, K8.ORANGE, "-", 3)] if fp is not None else []) + ([(srect, "black", "--", 2)] if srect is not None else [])
                 + ([(crect, "black", ":", 2)] if crect is not None else []), OUT / "panels", nm + "_img")
    rr = np.meshgrid(RG, YG, indexing="ij")[0]
    sgrid, _ = K31.path_sd(*np.meshgrid(RG, YG, indexing="ij"), k_)
    panel_rows.append({"panel": nm, "dataset": "Argoverse-HD val", "unit": unit, "frame": f, "consumer": "curved_brake_x1", "speed_scale": 1.0,
                       "ego_speed_mps": v0, "kappa_per_m": k_, "sensitive_cells": int(sens.sum()),
                       "farthest_sensitive_range_m": float(rr[sens].max()) if sens.any() else 0.0,
                       "farthest_sensitive_arclength_m": float(sgrid[sens].max()) if sens.any() else 0.0,
                       "footprint_xyxy": None if fp is None else [round(float(x), 1) for x in fp],
                       "footprint_area_share": 0.0 if fp is None else float((fp[2] - fp[0]) * (fp[3] - fp[1]) / (W * H)),
                       "tube_xyxy": None if srect is None else [round(float(x), 1) for x in srect], "tube_half_width_m_at_near": shalf,
                       "curved_tube_xyxy": None if crect is None else [round(float(x), 1) for x in crect], "curved_tube_half_width_m": chalf,
                       "tube_cells": int(s_cells.sum()), "curved_tube_cells": int(c_cells.sum()),
                       "share_sensitive_in_tube": float((sens & s_cells).sum() / sens.sum()) if sens.any() else np.nan,
                       "share_sensitive_in_curved_tube": float((sens & c_cells).sum() / sens.sum()) if sens.any() else np.nan,
                       "image": path})
    for i, rg in enumerate(RG):
        for j, yg in enumerate(YG):
            if sens[i, j] or s_cells[i, j] or c_cells[i, j]:
                cell_rows.append({"panel": nm, "range_m": rg, "lateral_m": yg, "sensitive": bool(sens[i, j]), "in_tube": bool(s_cells[i, j]),
                                  "in_curved_tube": bool(c_cells[i, j])})
    # ------------------------------------------------------------------ F2
    x = recs(REAL / "results" / "k28" / "p3" / "nusc_test_instr")["scene-0122"][7]
    f2 = {"unit": "scene-0122", "frame": 7, "action_required": x["act_gt"], "v6S": x["a_v6S|v3a"],
          "unc2S_C1_v3a": x["a_unc2@S+C1|v3a"], "native_C1_v3a": x["a_wf1600+C1|v3a"],
          "px_v6S": x["px_v6S"], "px_unc2S": x["px_unc2@S+C1"]}
    pd.DataFrame(sel_rows).to_csv(OUT / "k36_selection.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "k36_cells.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(OUT / "k36_panels.csv", index=False)
    pd.DataFrame([f2]).to_csv(OUT / "k36_f2.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(pd.DataFrame(panel_rows).drop(columns=["image"]).to_string()); print(f2); print(checks)
    print("K36 F1 DONE")


if __name__ == "__main__":
    main()
