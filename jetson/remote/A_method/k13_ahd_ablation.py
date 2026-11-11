#!/usr/bin/env python
"""K13(a) — necessity ablation on Argoverse-HD val (post-hoc analysis; no ego part visible, so no mask): the K3 variants
of ours with the AV2 configuration (v4-lite: v_close 0, crops at 960/1920 = 0.5, guard, merged-scene history, <= 3
rounds) + refiner v2: full, field only, support only, pivots, one round, no guard, draft history (instr_eval.run_variant /
speculative.run, unchanged). Per frame: decision cost without / with refiner, tensor detector px (draft canvas + every
crop canvas incl. guard re-detections), crops, rounds, and the action of F = u960 (agreement with pi(F)).
usage: k13_ahd_ablation.py [--shard i/n] | --analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k13_ahd_ablation"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
VARIANTS = {"full": None, "field_only": {"support_mode": "none"}, "support_only": {"use_field": False},
            "pivots": {"support_mode": "pivots"}, "one_round": {"max_rounds": 1}, "no_guard": {"guard": False}, "draft_hist": {}}


def run(args):
    import torch
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    si, sn = (int(x) for x in args.shard.split("/"))
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / "resnet18_v2" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    VC, SCALE, CH = 0.0, 960 / 1920, Mode("ahd_c320", 224, 320)
    M9 = Mode("ahd_u960", 608, 960)
    OUT.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(HOOD.units("ahd")):
        if ui % sn != si or (OUT / f"{unit}.pkl").exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, prev_cheap_boxes, rows = {}, None, None, None, []
        for fr, path, v, g, extra in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "gt": extra != "nogt"}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
            ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            a_c = act(ch["mono"]); rec["a_cheap"], rec["J_cheap"] = a_c, cost(a_c)
            d9 = keep(det.detect(img, M9)[0]); a9 = act(lift(d9, hist.get("u960"))); hist["u960"] = d9
            rec["a_u960"], rec["J_u960"] = a9, cost(a9)
            tpx = [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                tpx[0] += nh * nw
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d

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
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            for vn, cfg in VARIANTS.items():
                pv = prev_cheap_boxes if vn == "draft_hist" else hist.get(f"scene_{vn}")
                tpx[0] = 0.0
                if cfg is None:
                    scene, crops, rounds = SP.run(base, v, cam, W, H, detect_crop, lambda full, pv=pv: lift(full, pv), pp,
                                                  v_close=VC, guard=True)
                else:
                    scene, crops, rounds = IE.run_variant(base, v, cam, W, H, detect_crop, lambda full, pv=pv: lift(full, pv), pp, VC, **cfg)
                a = act(scene["mono"]); ar = act(refine(scene))
                rec.update({f"a_{vn}": a, f"J_{vn}": cost(a), f"a_{vn}_ref": ar, f"J_{vn}_ref": cost(ar),
                            f"px_{vn}": (CH.net_h * CH.net_w + tpx[0]) / (W * H), f"crops_{vn}": len(crops), f"rounds_{vn}": rounds})
                hist[f"scene_{vn}"] = {"xyxy": scene["xyxy"]}
            prev_cheap_boxes = {"xyxy": ch["xyxy"]}
            prev_act = a_c
            rows.append(rec)
        with open(OUT / f"{unit}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  {unit}: {len(rows)} frames", flush=True)
    print("K13 AHD ABLATION RUN DONE", flush=True)


def analyze(args):
    rows = []
    for p in sorted(OUT.glob("seq*.pkl")):
        rows += pickle.load(open(p, "rb"))
    d = pd.DataFrame(rows)
    d = d[d["gt"].astype(bool)]
    L = d.J_cheap.to_numpy()
    uq = d.unit.unique(); idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    full_r = L - d.J_full_ref.to_numpy(); full_n = L - d.J_full.to_numpy()
    out = []
    for vn in VARIANTS:
        Vn = L - d[f"J_{vn}"].to_numpy(); Vr = L - d[f"J_{vn}_ref"].to_numpy()
        r = {"dataset": f"Argoverse-HD val ({len(uq)} seq, {len(d)} frames with 3D labels)", "variant": vn,
             "G_no_refiner": Vn.sum() / L.sum(), "G_refiner": Vr.sum() / L.sum(),
             "tensor_px_mean": d[f"px_{vn}"].mean(), "tensor_px_p95": d[f"px_{vn}"].quantile(.95),
             "crops_per_frame": d[f"crops_{vn}"].mean(), "agreement_u960": float((d[f"a_{vn}"] == d.a_u960).mean())}
        if vn != "full":
            dr = [(full_r[b].sum() - Vr[b].sum()) / L[b].sum() for b in boots]
            dn = [(full_n[b].sum() - Vn[b].sum()) / L[b].sum() for b in boots]
            r.update({"ci_lo_full_minus_ref": np.percentile(dr, 2.5), "ci_hi_full_minus_ref": np.percentile(dr, 97.5),
                      "ci_lo_full_minus_noref": np.percentile(dn, 2.5), "ci_hi_full_minus_noref": np.percentile(dn, 97.5)})
        out.append(r)
    out.append({"dataset": out[0]["dataset"], "variant": "u960 (F)", "G_no_refiner": (L - d.J_u960.to_numpy()).sum() / L.sum(),
                "agreement_u960": 1.0})
    out.append({"dataset": out[0]["dataset"], "variant": "cheap", "G_no_refiner": 0.0,
                "agreement_u960": float((d.a_cheap == d.a_u960).mean())})
    res = pd.DataFrame(out); res.to_csv(OUT / "ablation_ahd.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
