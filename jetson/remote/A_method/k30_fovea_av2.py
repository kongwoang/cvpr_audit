#!/usr/bin/env python
"""K30 V5 (post-hoc): the best published method on AV2 val with the ego mask, FOVEA KDE_I warped to 960x768 (K28 phase 3
section 4: G -0.023, the highest prior-work G there), re-run with av2_baselines.py's code unchanged (SAHI and the crop
heuristics of that script dropped), storing per frame what the stored run did not: the action, the lifted geometry (z,
lat_min, lat_max, ttc) of FOVEA's and of the cheap draft's objects, the continuous a_gt, and the previous cheap action.
Run through k10_masked_run.py (B = the masked shadow tree) as the stored masked run. Parity: J_c and J_fovea_kdei960 equal
the stored <masked tree>/results/av2/baselines__val.pkl. Output: <B_REAL>/results/k30/prior/av2_fovea_kdei960.pkl
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
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    GG = FB["GG"]
    BE.STRICT = True
    ROOT = B / "datasets" / "av2" / "sensor" / "val"
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v3_confirm2.pkl")
    logs = sorted(ref.log.unique())[: args.max_logs or None]
    coco = str(B / "models" / "yolov8s.pt")
    det = TwoFidelityDetector(coco, device="cuda:0", backend="torch")
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
        arms = ["fovea_kdei960"]
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
            rec = {"log": log, "frame": fr, "v": v, "J_c": cost(a_c), "a_c": a_c, "a_gt": float(a_gt), "prev_cheap": prev_act,
                   "geo_c": ch["mono"]}
            # FOVEA KDE_I
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                grid = gen(x, meta, [torch.tensor(ch["xyxy"], dtype=torch.float32, device="cuda").reshape(-1, 4)])
                wimg = F.grid_sample(x, grid, align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
            d = keep(det.detect(wimg, Mode("fov768x960", 768, 960))[0])
            bb = GG.unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W)).cpu().numpy() \
                if len(d["conf"]) else np.zeros((0, 4))
            dd = {"xyxy": bb.astype(float), "coarse": d["coarse"]}
            tt_ = lift(dd, prev["fovea_kdei960"])
            rec["a_fovea_kdei960"] = act(tt_)
            rec["J_fovea_kdei960"] = cost(rec["a_fovea_kdei960"])
            rec["geo_fovea_kdei960"] = tt_
            prev["fovea_kdei960"] = dd
            rows.append(rec)
            prev["c"], prev_act = ch, a_c
        print(f"  [{li + 1}/{len(logs)}] {log}", flush=True)
    d = pd.DataFrame(rows)
    out = Path(os.environ.get("B_REAL", B)) / "results" / "k30" / "prior"
    out.mkdir(parents=True, exist_ok=True)
    d.to_pickle(out / f"av2_fovea_kdei960{args.tag}.pkl")
    st = pd.read_pickle(B / "results" / "av2" / "baselines__val.pkl")
    m = d.merge(st[["log", "frame", "J_c", "J_fovea_kdei960"]], on=["log", "frame"], suffixes=("", "_stored"))
    print("frames", len(d), "merged", len(m), "J_c identical", float(np.isclose(m.J_c, m.J_c_stored).mean()),
          "J_fovea identical", float(np.isclose(m.J_fovea_kdei960, m.J_fovea_kdei960_stored).mean()), flush=True)
    print("K30 PRIOR FOVEA AV2 DONE", flush=True)


if __name__ == "__main__":
    main()
