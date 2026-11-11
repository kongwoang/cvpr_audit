#!/usr/bin/env python
"""K15 — Figure 1 panels on real frames: three ways of spending high-resolution perception on ONE frame.
usage: k15_render.py <frames.csv> <out_dir>      (AV2 through k10_masked_run.py: ego mask as every AV2 number)
frames.csv: dataset (nusc|av2), unit, frame, reason.  Each unit is re-run from its start (exact lift / scene
histories) with the frozen configurations (nuScenes v3 + refiner v1, AV2 v4-lite + refiner v2; k8_render.run_capture,
identical to speculative.run). Five panels per frame, separate files, PDF (vector overlays) + PNG at native resolution,
no text, ground truth green thin, colours of K8:
  (1) draft: draft detections (blue), the missed decision-relevant object (dashed red)
  (2) whole frame at native resolution: all its detections (red), image tinted lightly orange
  (3) uncertainty crops at ours' crop area (instr_eval's rule): crops (orange), their detections (red)
  (4) ours: field footprint (orange), support crops (blue), guard-grown crops (grey), verified detections (red)
  (5) bird's-eye view (K8 style): ego, corridor, sensitive cells (orange), draft objects (blue), missed object (dashed red)
panels.csv: per frame and pipeline, action taken, action required (consumer on the annotated scene), network px.
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
_k = importlib.util.spec_from_file_location("k8", B / "scripts" / "A_method" / "k8_render.py")
K8 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K8)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
BLUE, ORANGE, GREY, RED, GREEN = K8.BLUE, K8.ORANGE, K8.GREY, K8.RED, K8.GREEN


def img_panel(img, boxes, out, name, tint=False):
    import matplotlib.pyplot as plt
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img[:, :, ::-1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    if tint:
        ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor=ORANGE, alpha=0.15, edgecolor="none"))
    for (x1, y1, x2, y2), col, ls, lw in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linestyle=ls, linewidth=lw))
    K8.save(fig, out, name)


def bev_panel(grid, v, pp, mono, missed, out, name):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    rg, yg = SP.R_GRID, SP.Y_GRID
    dr = np.diff(np.r_[rg, rg[-1] + 15]); dy = 1.5
    for i, r in enumerate(rg):
        for j, y in enumerate(yg):
            on = grid is not None and grid[i, j]
            ax.add_patch(plt.Rectangle((y - dy / 2, r - dr[i] / 2), dy, dr[i], facecolor=ORANGE if on else "none",
                                       edgecolor="#cccccc", linewidth=0.5, alpha=0.8 if on else 1))
    for x in (-pp.corridor_half_w, pp.corridor_half_w):
        ax.plot([x, x], [0, 80], color=GREY, linewidth=1)
    ax.add_patch(plt.Rectangle((-0.9, -4.5), 1.8, 4.5, facecolor="black"))
    ax.arrow(0, 0, 0, max(v, 0.5) * 1.0, width=0.15, color="black")
    z, lmin, lmax, _ = mono
    for k in range(len(z)):
        if np.isfinite(z[k]):
            ax.add_patch(plt.Rectangle((lmin[k], z[k]), lmax[k] - lmin[k], 1.5, facecolor=BLUE, alpha=0.8))
    if missed is not None:
        mz, mlo, mhi = missed
        ax.add_patch(plt.Rectangle((mlo, mz), mhi - mlo, 1.5, fill=False, edgecolor=RED, linestyle="--", linewidth=2))
    ax.set_xlim(-8, 8); ax.set_ylim(-6, 80); ax.set_aspect("auto")
    K8.save(fig, out, name)


def main():
    import torch
    frames = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    for (ds, unit), want in frames.groupby(["dataset", "unit"]):
        want = want.set_index("frame")
        ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev,
                        weights_only=False)
        rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
        fmu, fsd = ck["fmu"], ck["fsd"]
        if ds == "nusc":
            units = IE.units_nusc("test"); VC, SCALE, Mc, cpx = 10.0, 0.6, None, 192 * 320
            M9, Mn = Mode("f576x960", 576, 960), Mode("ns_u1600", 928, 1600)
        else:
            units = IE.units_av2("test"); VC, SCALE, Mc, cpx = 0.0, 960 / 2048, Mode("av2_c320", 320, 256), 320 * 256
            M9, Mn = Mode("av2_f960", 960, 768), Mode("av2_u2048", 2048, 1568)
        for u, cam, gen in units:
            if u != unit:
                continue
            R, t = cam.transform("ego"); hist = {}; prev_cheap = None
            for fr, path, v, g, cheap_cached in gen():
                img = cv2.imread(path); H, W = img.shape[:2]; NATIVE = W * H
                a_gt, a_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))

                def lift(dd, pv):
                    gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

                def act(tt):
                    return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

                def keep(d):
                    k = d["conf"] >= IE.THR
                    return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                            "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}
                if cheap_cached is not None:
                    cheap = cheap_cached
                else:
                    cheap = keep(det.detect(img, Mc)[0]); cheap["mono"] = lift(cheap, prev_cheap)
                tpx = [0.0]
                cdets = []

                def detect_crop(cr):
                    x1, y1 = int(cr[0]), int(cr[1])
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32); nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                    tpx[0] += nh * nw
                    d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0]); d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                    cdets.append(d["xyxy"])
                    return {k: d[k] for k in ("xyxy", "conf", "coarse")}

                def refine(scene):
                    refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                    for p_ in SP.support(scene["mono"], v, pp):
                        if not scene["from_full"][p_]:
                            continue
                        x1, y1, x2, y2 = scene["xyxy"][p_]
                        zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                                cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                        zg, zh = float(zz["z_ground"][0]), float(zz["z_height"][0])
                        f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                                       float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                                       float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
                        f = ((f - fmu) / fsd).astype(np.float32)
                        x = torch.from_numpy(crop64(img, scene["xyxy"][p_])).permute(2, 0, 1)[None].float().to(dev) / 255.0
                        with torch.no_grad():
                            corr = float(np.exp(rnet(x, torch.from_numpy(f).to(dev)).item()))
                        for a in refined[:3]:
                            a[p_] *= corr
                    return tuple(refined)
                base = {k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}
                scene, crops, rounds, cap = K8.run_capture(base, v, cam, W, H, detect_crop, lambda full: lift(full, hist.get("scene")), pp, VC)
                px_ours = (cpx + tpx[0]) / NATIVE
                # uncertainty crops at ours' crop area (strict), merged once with the guard
                budget = sum(IE.area(c) for c in crops)
                order = np.lexsort((-np.asarray(cheap.get("entropy", np.zeros(len(cheap["conf"])))), cheap["conf"]))
                sel = IE.fill_strict([SP.box_crop(cheap["xyxy"][k], W, H) for k in order], budget)
                tpx[0] = 0.0; cdets.clear()
                sc_u, cr_u = IE.merge_once(base, sel, SP.guarded(detect_crop, W, H), lambda full: lift(full, hist.get("scene_unc")))
                px_unc = (cpx + tpx[0]) / NATIVE
                d9 = keep(det.detect(img, M9)[0]); dn = keep(det.detect(img, Mn)[0])
                a9 = act(lift(d9, hist.get("u960"))); an = act(lift(dn, hist.get("native")))
                if fr in want.index:
                    gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                    gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
                    rel = np.flatnonzero(np.asarray(a_obj) >= pp.thr_decel) if len(g) else np.zeros(0, int)
                    fc = IE.matched(gtb[rel], gtc[rel], cheap["xyxy"], cheap["coarse"]) >= 0
                    fu = IE.matched(gtb[rel], gtc[rel], sc_u["xyxy"], sc_u["coarse"]) >= 0
                    fo = IE.matched(gtb[rel], gtc[rel], scene["xyxy"], scene["coarse"]) >= 0
                    mi = [rel[i] for i in range(len(rel)) if not fc[i] and not fu[i] and fo[i]]
                    mi = mi[0] if mi else (rel[int(np.flatnonzero(~fc)[0])] if (~fc).any() else None)
                    stem = f"{ds}_{unit}_{fr:04d}"
                    gt_boxes = [(b, GREEN, "-", 2) for b in gtb]
                    missed = [(gtb[mi], RED, "--", 5)] if mi is not None else []
                    img_panel(img, gt_boxes + [(b, BLUE, "-", 5) for b in cheap["xyxy"]] + missed, out, stem + "_1_draft")
                    img_panel(img, gt_boxes + [(b, RED, "-", 5) for b in dn["xyxy"]], out, stem + "_2_wholeframe", tint=True)
                    img_panel(img, gt_boxes + [(c, ORANGE, "-", 6) for c in cr_u] +
                              [(b, RED, "-", 5) for i, b in enumerate(sc_u["xyxy"]) if sc_u["from_full"][i]], out, stem + "_3_uncertainty")
                    img_panel(img, gt_boxes + [(c, ORANGE, "-", 6) for c in cap["field"]] + [(c, BLUE, "-", 6) for c in cap["support"]] +
                              [(c, GREY, "-", 4) for c in cap["grown"]] +
                              [(b, RED, "-", 5) for i, b in enumerate(scene["xyxy"]) if scene["from_full"][i]], out, stem + "_4_ours")
                    ms = (float(np.asarray(g["long_near"])[mi]), float(np.asarray(g["lat_min"])[mi]), float(np.asarray(g["lat_max"])[mi])) if mi is not None else None
                    bev_panel(cap["grid"], v, pp, cheap["mono"], ms, out, stem + "_5_bev")
                    req = P.discrete_action(a_gt, pp)
                    base_r = {"dataset": ds, "unit": unit, "frame": fr, "image": path, "reason": want.loc[fr, "reason"],
                              "ego_speed_mps": round(v, 2), "action_required": req,
                              "missed_class": gtc[mi] if mi is not None else "", "missed_range_m": ms[0] if ms else np.nan}
                    for pl, a_, px in (("cheap draft", act(cheap["mono"]), cpx / NATIVE), ("whole frame native", an, Mn.net_h * Mn.net_w / NATIVE),
                                       ("whole frame u960", a9, M9.net_h * M9.net_w / NATIVE), ("uncertainty crops", act(sc_u["mono"]), px_unc),
                                       ("ours without refiner", act(scene["mono"]), px_ours), ("ours", act(refine(scene)), px_ours)):
                        rows.append({**base_r, "pipeline": pl, "action": a_, "network_px": round(px, 4)})
                hist["scene"] = {"xyxy": scene["xyxy"]}; hist["scene_unc"] = {"xyxy": sc_u["xyxy"]}
                hist["u960"], hist["native"] = d9, dn
                prev_cheap = {"xyxy": cheap["xyxy"], "coarse": cheap["coarse"]} if cheap_cached is None else None
                if fr >= want.index.max():
                    break
            break
    pd.DataFrame(rows).to_csv(out / "panels.csv", index=False)
    print("K15 RENDER DONE", len(rows))


if __name__ == "__main__":
    main()
