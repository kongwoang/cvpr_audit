#!/usr/bin/env python
"""Baselines on Argoverse 2 sensor VAL (the 150 logs of confirmatory 2), every method with the same frozen COCO YOLOv8s.

Per frame, exactly as av2_eval.py (cheap 320x256 draft, braking, monocular ego-frame lift, DEEP's cost, cheap action
history; the cheap cost is asserted equal to confirmatory 2), plus:
  sahi_s047 / sahi_s10   SAHI (official library): 640x640 slices, overlap 0.2, + standard full-image prediction, each
                         slice detected at 320 px (~ v3's crop magnification 960/2048 = 0.469) / 640 px
  fovea_kdei960          FOVEA (official grid generator, KDE_I from the current cheap draft) warped to 960x768
  uncertainty / temporal_crit / random   crop heuristics of baselines_eq.py with v3's machinery (guard, merge, crop
                         scale 0.469), strictly within v3's per-frame pixel budget from confirmatory 2
Compared with u960, v3, v3 + refiner of confirmatory 2; paired log bootstrap (2,000).
"""
from __future__ import annotations

import argparse
import math
import os
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts" / "A_method"))
import baselines_eq as BE                                                          # noqa: E402
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import av2 as AV                                                        # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

THR = 0.25
M_CHEAP = Mode("av2_c320", 320, 256)
SAHI_COARSE = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}
CROP_ARMS = ("uncertainty", "temporal_crit", "random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_logs", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    GG = FB["GG"]
    BE.STRICT = True
    ROOT = B / "datasets" / "av2" / "sensor" / "val"
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v3_confirm2.pkl")
    logs = sorted(ref.log.unique())[: args.max_logs or None]
    coco = str(B / "models" / "yolov8s.pt")
    det = TwoFidelityDetector(coco, device="cuda:0", backend="torch")
    sahi = {k: AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=coco, confidence_threshold=THR,
                                                  device="cuda:0", image_size=sz) for k, sz in (("sahi_s047", 320), ("sahi_s10", 640))}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for li, log in enumerate(logs):
        L = AV.AV2Log(ROOT, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        W, H = L.W, L.H
        scale = 960 / max(W, H)
        gen = GG.PlainKDEGrid(output_shape=(768, 960), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                              anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
        meta = [{"pad_shape": (H, W, 3)}]
        rl = ref[ref.log == log].set_index("frame")
        arms = list(sahi) + ["fovea_kdei960"] + list(CROP_ARMS)
        prev = {k: None for k in arms + ["c"]}
        prev_state = {k: None for k in CROP_ARMS}
        prev_act = None

        def lift(dd, pv):
            g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                    "entropy": d["entropy"][k].astype(float) if "entropy" in d else np.zeros(int(k.sum()))}

        for fr in range(len(L.frame_ts)):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None or fr not in rl.index:
                continue
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            ch = keep(det.detect(img, M_CHEAP)[0])
            ch["mono"] = lift(ch, prev["c"])
            a_c = act(ch["mono"])
            rec = {"log": log, "frame": fr, "J_c": cost(a_c)}
            # SAHI
            for k, sm in sahi.items():
                res = get_sliced_prediction(img[:, :, ::-1].copy(), sm, slice_height=640, slice_width=640,
                                            overlap_height_ratio=0.2, overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
                ob = [o for o in res.object_prediction_list if o.category.id in SAHI_COARSE]
                dd = {"xyxy": np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4),
                      "coarse": np.array([SAHI_COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
                rec[f"J_{k}"] = cost(act(lift(dd, prev[k])))
                prev[k] = dd
            # FOVEA KDE_I
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                grid = gen(x, meta, [torch.tensor(ch["xyxy"], dtype=torch.float32, device="cuda").reshape(-1, 4)])
                wimg = F.grid_sample(x, grid, align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
            d = keep(det.detect(wimg, Mode("fov768x960", 768, 960))[0])
            bb = GG.unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W)).cpu().numpy() \
                if len(d["conf"]) else np.zeros((0, 4))
            dd = {"xyxy": bb.astype(float), "coarse": d["coarse"]}
            rec["J_fovea_kdei960"] = cost(act(lift(dd, prev["fovea_kdei960"])))
            prev["fovea_kdei960"] = dd
            # crop heuristics within v3's pixel budget
            bud = float(rl.loc[fr, "px"]) * W * H

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                return {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}
            gdet = SP.guarded(detect_crop, W, H)
            bxc = ch["xyxy"]
            for arm in CROP_ARMS:
                if arm == "random":
                    cands = []
                    for _ in range(40):
                        if len(bxc):
                            b = bxc[rng.integers(len(bxc))]
                            w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                        else:
                            w_, h_ = W / 6, H / 6
                        x0, y0 = rng.uniform(0, W - w_), rng.uniform(0, H - h_)
                        cands.append(SP.box_crop(np.array([x0, y0, x0 + w_, y0 + h_]), W, H))
                elif arm == "uncertainty":
                    order = np.lexsort((-ch["entropy"], ch["conf"]))
                    cands = [SP.box_crop(bxc[k], W, H) for k in order]
                else:
                    cands = []
                    if prev_state[arm] is not None:
                        pxy, pmono = prev_state[arm]
                        if len(pxy):
                            _, dem = P.required_decel(*pmono, v, pp)
                            order = np.lexsort((pmono[0], -np.asarray(dem)))
                            cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                        pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                crops = BE.fill(cands, bud)
                scene = BE.merge_once(ch, crops, gdet, lambda full, arm=arm: lift(full, prev[arm]))
                rec[f"J_{arm}"] = cost(act(scene["mono"]))
                rec[f"px_{arm}"] = sum(BE.area(q) for q in crops) / (W * H)
                prev[arm] = {"xyxy": scene["xyxy"]}
                prev_state[arm] = (np.asarray(scene["xyxy"], float).reshape(-1, 4), tuple(np.asarray(a, float) for a in scene["mono"]))
            rows.append(rec)
            prev["c"], prev_act = ch, a_c
        print(f"  [{li + 1}/{len(logs)}] {log}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "av2"
    d.to_pickle(out / f"baselines__val{args.tag}.pkl")
    m = d.merge(ref[["log", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref", "px"]], on=["log", "frame"], suffixes=("", "_ref"))
    print(f"cheap cost equal to confirmatory 2 on {np.mean(np.isclose(m.J_c, m.J_c_ref)):.4f} of {len(m)} frames")
    Lc = m.J_c_ref.to_numpy()
    V = {k: (m.J_c_ref - m[f"J_{k}"]).to_numpy() for k in arms}
    V.update({"u960": (m.J_c_ref - m.J_u960).to_numpy(), "v3": (m.J_c_ref - m.J_spec).to_numpy(),
              "v3_ref": (m.J_c_ref - m.J_spec_ref).to_numpy()})
    uq = m.log.unique(); idx = [np.flatnonzero(m.log.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / Lc.sum(), "harmed": float(np.mean(v_ < -1e-9)),
             "px": float(m[f"px_{k}"].mean()) if f"px_{k}" in m else (float(m.px.mean()) if k.startswith("v3") else np.nan)}
        for rk in ("v3", "v3_ref"):
            dr = [(V[rk][t_].sum() - v_[t_].sum()) / Lc[t_].sum() for t_ in boots]
            r[f"{rk}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / f"baselines_summary__val{args.tag}.csv", index=False)
    print(res.to_string(index=False))
    print("AV2 BASELINES DONE")


if __name__ == "__main__":
    main()
