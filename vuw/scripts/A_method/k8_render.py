#!/usr/bin/env python
"""K8 qualitative panels. usage: k8_render.py <frames.csv> <out_dir>
frames.csv: dataset (nusc|av2), split (test), unit, frame, category, reason.
For each frame the sequence of its unit is re-run from the start (exact lift / scene histories), with ours' final
configuration (nuScenes v3 + refiner v1, AV2 v4-lite + refiner v2) and the loop of instr_eval.run_variant with capture
hooks (identical decisions to speculative.run; asserted). Four panels per frame, separate files, PDF (vector overlays
on the image) + PNG, native resolution, no text:
  (a) frame + draft detections (blue; dashed if later removed)
  (b) bird's-eye view: ground grid with sensitive cells (orange), ego, corridor (grey lines), draft objects (blue)
  (c) frame + field footprint (orange), support crops (blue solid), guard-grown crops (grey)
  (d) frame + final merged detections (red; draft boxes kept in blue) + ground truth (green thin)
Also writes examples.csv: actions of cheap / ours / u960 / native, required (GT) action, crops, rounds, network px.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from vtf import dsf                                                              # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c"


def run_capture(cheap, v, cam, W, H, detect_crop, lift, pp, v_close):
    """speculative.run (full configuration) with capture of field crops, support crops, grown crops, sensitive grid."""
    cap = {"field": [], "support": [], "grown": [], "grid": None}
    gdet0 = SP.guarded(detect_crop, W, H)

    def gdet(c):
        d, ex = gdet0(c)
        cap["grown"] += ex
        return d, ex
    crops, grown, full = [], [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds = dict(cheap), 0
    for rd in range(3):
        new = []
        if rd == 0:
            cap["grid"] = dsf.insertion_grid(scene["mono"], v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, v_close)
        fc = SP.field_crop(scene["mono"], v, cam, W, H, pp, v_close, None, SP.OBJ_H)
        if fc is not None and not SP.covered(fc, crops):
            new.append(fc); cap["field"].append(fc)
        for k in SP.support(scene["mono"], v, pp):
            bc = SP.box_crop(scene["xyxy"][k], W, H)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc); cap["support"].append(bc)
        if not new:
            break
        rounds += 1
        for c in new:
            d, ex = gdet(c)
            grown += ex
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops += new
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable"); iou = box_iou(full["xyxy"], full["xyxy"]); keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            full = {k: full[k][np.sort(np.array(keep))] for k in full}
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                 "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                 "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)],
                 "removed_draft": out}
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    scene.setdefault("removed_draft", np.zeros(len(cheap["conf"]), bool))
    return scene, crops + grown, rounds, cap


def save(fig, out, name):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=72, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def img_panel(img, boxes, out, name):
    """boxes: list of (xyxy, colour, linestyle, linewidth)."""
    import matplotlib.pyplot as plt
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img[:, :, ::-1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    for (x1, y1, x2, y2), col, ls, lw in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linestyle=ls, linewidth=lw))
    save(fig, out, name)


def bev_panel(grid, v, pp, mono, out, name):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    rg, yg = SP.R_GRID, SP.Y_GRID
    dr = np.diff(np.r_[rg, rg[-1] + 15]); dy = 1.5
    for i, r in enumerate(rg):
        for j, y in enumerate(yg):
            ax.add_patch(plt.Rectangle((y - dy / 2, r - dr[i] / 2), dy, dr[i], facecolor=ORANGE if grid is not None and grid[i, j] else "none",
                                       edgecolor="#cccccc", linewidth=0.5, alpha=0.8 if grid is not None and grid[i, j] else 1))
    for x in (-pp.corridor_half_w, pp.corridor_half_w):
        ax.plot([x, x], [0, 80], color=GREY, linewidth=1)
    ax.add_patch(plt.Rectangle((-0.9, -4.5), 1.8, 4.5, facecolor="black"))              # ego
    ax.arrow(0, 0, 0, max(v, 0.5) * 1.0, width=0.15, color="black")                       # speed arrow, 1 s of travel
    z, lmin, lmax, _ = mono
    for k in range(len(z)):
        ax.add_patch(plt.Rectangle((lmin[k], z[k]), lmax[k] - lmin[k], 1.5, facecolor=BLUE, alpha=0.8))
    ax.set_xlim(-8, 8); ax.set_ylim(-6, 80); ax.set_aspect("auto")    # lateral axis stretched ~4x for legibility
    save(fig, out, name)


def main():
    import torch
    frames = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    for (ds, split, unit), want in frames.groupby(["dataset", "split", "unit"]):
        want = want.set_index("frame")
        if ds == "nusc":
            units = IE.units_nusc(split); V_CLOSE, SCALE = 10.0, 0.6
            Mc = None; M9 = Mode("f576x960", 576, 960); Mn = Mode("ns_u1600", 928, 1600)
        else:
            units = IE.units_av2(split); V_CLOSE, SCALE = 0.0, 960 / 2048
            Mc = Mode("av2_c320", 320, 256); M9 = Mode("av2_f960", 960, 768); Mn = Mode("av2_u2048", 2048, 1568)
        for u, cam, gen in units:
            if u != unit:
                continue
            R, t = cam.transform("ego"); hist = {}; prev_act = None; prev_cheap = None
            for fr, path, v, g, cheap_cached in gen():
                img = cv2.imread(path); H, W = img.shape[:2]
                a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
                lift = lambda dd, pv: tuple(np.asarray(predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))  # noqa: E731
                act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)       # noqa: E731
                keep = lambda d: {"xyxy": d["xyxy"][d["conf"] >= 0.25].astype(float), "conf": d["conf"][d["conf"] >= 0.25].astype(float), "coarse": d["coarse"][d["conf"] >= 0.25]}  # noqa: E731
                if cheap_cached is not None:
                    cheap = cheap_cached
                else:
                    cheap = keep(det.detect(img, Mc)[0]); cheap["mono"] = lift(cheap, prev_cheap)
                tpx = [0.0]

                def detect_crop(cr):
                    x1, y1 = int(cr[0]), int(cr[1])
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32); nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                    tpx[0] += nh * nw
                    d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0]); d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                    return d
                scene, crops, rounds, cap = run_capture({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H,
                                                        detect_crop, lambda full: lift(full, hist.get("scene")), pp, V_CLOSE)
                if fr in want.index:
                    d9 = keep(det.detect(img, M9)[0]); dn = keep(det.detect(img, Mn)[0])
                    a9 = act(lift(d9, hist.get("u960"))); an = act(lift(dn, hist.get("native")))
                    stem = f"{ds}_{unit}_{fr:04d}"
                    rm = scene["removed_draft"]
                    img_panel(img, [(b, BLUE, "--" if rm[i] else "-", 5) for i, b in enumerate(cheap["xyxy"])], out, stem + "_a_draft")
                    bev_panel(cap["grid"], v, pp, cheap["mono"], out, stem + "_b_bev")
                    img_panel(img, [(c, ORANGE, "-", 6) for c in cap["field"]] + [(c, BLUE, "-", 6) for c in cap["support"]] +
                              [(c, GREY, "-", 4) for c in cap["grown"]], out, stem + "_c_crops")
                    gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                    img_panel(img, [(b, GREEN, "-", 2) for b in gtb] + [(b, RED if scene["from_full"][i] else BLUE, "-", 5) for i, b in enumerate(scene["xyxy"])],
                              out, stem + "_d_final")
                    rows.append({"dataset": ds, "split": split, "unit": unit, "frame": fr, "timestamp_or_path": path, "category": want.loc[fr, "category"],
                                 "reason": want.loc[fr, "reason"], "ego_speed_mps": round(v, 2), "action_required_gt": P.discrete_action(a_gt, pp),
                                 "action_cheap": act(cheap["mono"]), "action_ours_noref": act(scene["mono"]), "action_u960": a9, "action_native": an,
                                 "crops": len(crops), "field_crops": len(cap["field"]), "support_crops": len(cap["support"]), "guard_crops": len(cap["grown"]),
                                 "rounds": rounds, "network_px": round(((192 * 320 if ds == "nusc" else 320 * 256) + tpx[0]) / (W * H), 4)})
                for nm, m in (("u960", M9), ("native", Mn)):
                    if fr + 1 in want.index or fr + 2 in want.index:                  # histories only needed near targets
                        hist[nm] = keep(det.detect(img, m)[0])
                hist["scene"] = {"xyxy": scene["xyxy"]}
                prev_act = act(cheap["mono"]); prev_cheap = {"xyxy": cheap["xyxy"], "coarse": cheap["coarse"]} if cheap_cached is None else None
                if fr >= want.index.max():
                    break
            break
    pd.DataFrame(rows).to_csv(out / "examples.csv", index=False)
    print("K8 RENDER DONE", len(rows))


if __name__ == "__main__":
    main()
