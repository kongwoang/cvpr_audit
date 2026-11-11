#!/usr/bin/env python
"""K12(a) — locality decomposition of the agreement with pi(F) (F = u960) on AV2 val, first 50 logs (post-hoc analysis).

For v_close in {0 (the AV2 configuration), 10 (the nuScenes one)}, ours without refiner (the loop of instr_eval: guard,
merged-scene history, <= 3 rounds, crops at 960/2048) is run three ways on the same frames:
  real   — the crops are detected by the crop detector (as deployed),
  loc    — exact locality: each crop returns F's own detections whose centres lie inside it (as instr_eval's K2.5),
  locF   — exact locality + F's previous detections as the lift history (F's ttc).
Agreement = share of frames whose braking action equals F's; interval = log bootstrap (2,000).
usage: k12_locality.py [--units 50] [--shard i/n] | --analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import math
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
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k12_locality"
SCALE = 960 / 2048


def run(args):
    si, sn = (int(x) for x in args.shard.split("/"))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    pp = P.PlannerParams()
    Mc, M9 = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768)
    OUT.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, frames) in enumerate(IE.units_av2("test")):
        if ui >= args.units:
            break
        if ui % sn != si or (OUT / f"{unit}.pkl").exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_c, rows = {}, None, []
        for fr, path, v, g, _ in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)
            ch = keep(det.detect(img, Mc)[0]); ch["mono"] = lift(ch, prev_c)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            F = keep(det.detect(img, M9)[0])
            rec = {"unit": unit, "frame": fr, "a_cheap": act(ch["mono"]), "a_F": act(lift(F, hist.get("F")))}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d

            def detect_crop_F(cr):
                k = SP.centres_in(F["xyxy"], [cr]) if len(F["xyxy"]) else np.zeros(0, bool)
                return {"xyxy": F["xyxy"][k], "conf": F["conf"][k], "coarse": F["coarse"][k]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            for vc in (0.0, 10.0):
                for mode, dc in (("real", detect_crop), ("loc", detect_crop_F), ("locF", detect_crop_F)):
                    key = f"{mode}_vc{vc:g}"
                    pv = hist.get("F") if mode == "locF" else hist.get(key)
                    sc, crops, _ = SP.run(base, v, cam, W, H, dc, lambda full, pv=pv: lift(full, pv), pp, v_close=vc, guard=True)
                    rec[f"a_{key}"] = act(sc["mono"])
                    rec[f"crops_{key}"] = len(crops)
                    hist[key] = {"xyxy": sc["xyxy"]}
            hist["F"] = F
            rows.append(rec)
        with open(OUT / f"{unit}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K12 LOCALITY RUN DONE", flush=True)


def analyze(args):
    rows = []
    for p in sorted(OUT.glob("*.pkl")):
        rows += pickle.load(open(p, "rb"))
    d = pd.DataFrame(rows)
    uq = d.unit.unique(); idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    for col, lab in [("a_cheap", "cheap draft")] + [(f"a_{m}_vc{vc}", f"{nm}, v_close {vc}") for vc in (0, 10)
                                                    for m, nm in (("real", "real crops (ours, no refiner)"),
                                                                  ("loc", "exact locality (F's objects inside the crops)"),
                                                                  ("locF", "exact locality + F's ttc history"))]:
        agree = (d[col].to_numpy() == d.a_F.to_numpy()).astype(float)
        bs = [agree[b].mean() for b in boots]
        r = {"dataset": f"AV2 val, first {len(uq)} logs ({len(d)} frames)", "variant": lab, "agreement_with_u960": agree.mean(),
             "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)}
        cc = col.replace("a_", "crops_")
        if cc in d:
            r["crops_per_frame"] = d[cc].mean()
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "locality_av2.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=50)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
