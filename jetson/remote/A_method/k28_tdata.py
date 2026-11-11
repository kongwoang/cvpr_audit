#!/usr/bin/env python
"""K28 family C — training data for the learned temporal cue (method_freeze_draft §2j; training data only, no dev /
test unit). Units:
  --src av2   AV2 REF-TRAIN / REF-VAL logs (results/k23/av2_split.csv; ego mask through k10_masked_run.py), every frame
              run, every 2nd frame sampled; split = the log's role
  --src nusc  nuScenes new TRAIN scenes (cached draft, instr_eval.units_nusc("train")), every keyframe sampled; split =
              "val" for scenes with md5(name)[0] in "01" (about 1/8), else "train"
Per frame: the v5.1-S loop (k_loop.run_g, v_close 0, context 2, crops at 960 px long side / nuScenes 0.6, guard, the
T4+T2_1 lift on its own same-scale history) and u960 (own same-scale history). A sample is a box of the final scene that
came from a crop (kind crop) or a u960 box (kind u960), within 80 m and 2 m of the ego corridor, with a same-scale
3-observation chain (t-2, t-1, t; the pairing of T4): the three 64x64 crops (each frame's box in its own image), the 12
box features at t (as the refiner), the height ratios h_t / h_{t-1}, h_{t-1} / h_{t-2}, the T4 rate, dt, and the
target r = max(-range_rate, 0) / long_near of the matched annotation (class-compatible IoU >= 0.3, Hungarian; NaN when
unmatched or the annotation has no range rate)  ->  data/tdata__<src>_<i>of<n>.npz
usage: k28_tdata.py --src av2|nusc [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_k = importlib.util.spec_from_file_location("k27", B / "scripts" / "A_method" / "k27_dev.py")
K27 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K27)
IE, KL = K27.IE, K27.KL
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def chains(X, s1, s2):
    """same-scale 3-observation chains of T4: (i, j1, j2) for boxes X with a partner in s1 whose partner is in s2"""
    if s1 is None or s2 is None or not len(s1) or not len(s2) or not len(X):
        return []
    j1 = K27.assoc_idx(X, s1)
    a12 = K27.assoc_idx(s1, s2)
    return [(i, int(j1[i]), int(a12[j1[i]])) for i in range(len(X)) if j1[i] >= 0 and a12[j1[i]] >= 0]


def units(src, max_units=0):
    if src == "nusc":
        for i, (s, cam, gen) in enumerate(IE.units_nusc("train")):
            if max_units and i >= max_units:
                return
            yield s, cam, gen, ("val" if hashlib.md5(s.encode()).hexdigest()[0] in "01" else "train"), 1
        return
    from vtf import av2 as AV
    sp = pd.read_csv(REAL / "results" / "k23" / "av2_split.csv")
    root = B / "datasets" / "av2" / "sensor" / "train"
    logs = sorted(sp[sp.role.isin(["REF-TRAIN", "REF-VAL"])].itertuples(index=False), key=lambda r: r.log)
    for i, r in enumerate(logs):
        if max_units and i >= max_units:
            return
        L = AV.AV2Log(root, r.log)

        def frames(L=L):
            geom, speeds = L.geometry(), L.speeds()
            for fr in range(len(L.frame_ts)):
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, None
        yield r.log, L, frames, ("train" if r.role == "REF-TRAIN" else "val"), 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, choices=["av2", "nusc"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp = P.PlannerParams()
    pol = K27.parse("T4+T2_1")
    if args.src == "nusc":
        CH, SCALE, F, MS = None, 0.6, Mode("f576x960", 576, 960), 0.6
    else:
        CH, SCALE, F, MS = Mode("av2_c320", 320, 256), 960 / 2048, Mode("av2_f960", 960, 768), 960 / 2048
    acc = {k: [] for k in ("crops", "feats", "hr", "rT4", "dt", "target", "split", "seq", "kind")}
    for ui, (unit, cam, gen, split, every) in enumerate(units(args.src, args.max_units)):
        if ui % sn != si:
            continue
        R, t = cam.transform("ego")
        dt = float(cam.dt)
        st = {}
        prev_c, imgs = None, []
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]

            def geom(dd, pv=None):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                xe, ye = min(x2, W), min(y2, H)
                nw = max(32, int(math.ceil((xe - x1) * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil((ye - y1) * SCALE / 32)) * 32)
                d = keep(det.detect(img[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d

            def lift(name, sc):
                h = st.get(name, {})

                def f(full):
                    z, lo, hi, _ = geom(full)
                    tt, _ = K27.ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"])
                    return z, lo, hi, tt
                return f

            def push(name, xyxy):
                h = st.setdefault(name, {})
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)
            if cached is not None:
                ch = cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = geom(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            v_ = float(v)
            sc, crops, grown, rounds, conv = KL.run_g(base, v_, cam, W, H, detect_crop, lift("crop", SCALE), pp, v_closes=(0.0,))
            ff = np.asarray(sc["from_full"], bool)
            Xc = np.asarray(sc["xyxy"], float).reshape(-1, 4)[ff]
            mc = tuple(np.asarray(a, float)[ff] for a in sc["mono"])
            cc, oc = np.asarray(sc["conf"])[ff], np.asarray(sc["coarse"])[ff]
            d9 = keep(det.detect(img, F)[0])
            m9 = geom(d9)
            sample = (fr % every == 0) and len(imgs) == 2
            if sample:
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
                for kind, X, mono, conf, coarse, sc_ in (("crop", Xc, mc, cc, oc, SCALE), ("u960", d9["xyxy"], m9, d9["conf"], d9["coarse"], MS)):
                    h = st.get(kind, {})
                    chn = chains(X, h.get("s1"), h.get("s2"))
                    if not chn:
                        continue
                    z, lo, hi, _ = mono
                    chn = [c for c in chn if np.isfinite(z[c[0]]) and 0 < z[c[0]] <= 80 and hi[c[0]] > -pp.corridor_half_w - 2
                           and lo[c[0]] < pp.corridor_half_w + 2]
                    if not chn:
                        continue
                    idx = np.array([c[0] for c in chn])
                    mm = IE.matched(gtb, gtc, X[idx], np.asarray(coarse)[idx]) if len(gtb) else -np.ones(0, int)
                    gt_of = -np.ones(len(idx), int)
                    for gi, dj in enumerate(mm):
                        if dj >= 0:
                            gt_of[dj] = gi
                    zz = predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(coarse)[idx]}, None, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                    tT4, _ = K27.ttc_ss(X[idx], h["s1"], h["s2"], sc_, dt, None, True)
                    for n, (i, j1, j2) in enumerate(chn):
                        x1, y1, x2, y2 = X[i]
                        b1, b2 = h["s1"][j1], h["s2"][j2]
                        hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                        zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                        acc["crops"].append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                        acc["feats"].append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf[i]),
                                             float(CLS[str(coarse[i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5)))])
                        acc["hr"].append([hh[0] / max(hh[1], 1e-3), hh[1] / max(hh[2], 1e-3)])
                        acc["rT4"].append(float(1 / tT4[n]) if np.isfinite(tT4[n]) and tT4[n] > 0 else 0.0)
                        acc["dt"].append(dt)
                        gi = gt_of[n]
                        has = (bool(np.asarray(g["has_rate"])[gi]) if "has_rate" in (getattr(getattr(g, "dtype", None), "names", None) or list(getattr(g, "columns", [])))
                               else bool(np.isfinite(np.asarray(g["range_rate"])[gi]))) if gi >= 0 else False
                        if has:
                            rr, ln = float(np.asarray(g["range_rate"])[gi]), float(np.asarray(g["long_near"])[gi])
                            acc["target"].append(max(-rr, 0.0) / max(ln, 0.5))
                        else:
                            acc["target"].append(np.nan)
                        acc["split"].append(split); acc["seq"].append(str(unit)[:16]); acc["kind"].append(kind)
            push("crop", Xc)
            push("u960", d9["xyxy"])
            imgs = (imgs + [img])[-2:]
        print(f"  [{ui}] {unit} ({split}): {len(acc['target'])} samples so far", flush=True)
    np.savez_compressed(REAL / "data" / f"tdata__{args.src}_{si}of{sn}.npz",
                        crops=np.array(acc["crops"], np.uint8).reshape(-1, 3, 64, 64, 3),
                        feats=np.array(acc["feats"], np.float32).reshape(-1, 12), hr=np.array(acc["hr"], np.float32).reshape(-1, 2),
                        rT4=np.array(acc["rT4"], np.float32), dt=np.array(acc["dt"], np.float32),
                        target=np.array(acc["target"], np.float32), split=np.array(acc["split"]), seq=np.array(acc["seq"]),
                        kind=np.array(acc["kind"]))
    print("K28 TDATA DONE", len(acc["target"]), flush=True)


if __name__ == "__main__":
    main()
