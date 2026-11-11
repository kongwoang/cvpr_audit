#!/usr/bin/env python
"""K11 phase 2 — test-split evaluation (post-hoc analysis) of the configuration selected on nuScenes dev frames for
latency: support context 2.5 (vtf.speculative.CTX, frozen 2.0) + field crop enlarged by 0.15 x its size on each side.
nuScenes new test (228 scenes), the confirmatory setting: cached COCO cheap draft (192x320), COCO YOLOv8s crops at 0.6,
v_close 10, guard, merged-scene history, <= 3 rounds, refiner v1 on support objects from crops. The frozen v3 is run in
the same script for parity with the confirmatory run (eval__new_test_v3_confirm.pkl).
usage: python k11_test.py [--shard i/n] | --analyze
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
OUT = B / "results" / "k11_test"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
CFGS = {"frozen": (2.0, 0.0), "ctx2.5_fm0.15": (2.5, 0.15)}


def run(si, sn):
    import cv2
    import torch
    from rap import planner as P
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import predicted_geometry
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL
    from vtf import speculative as SP
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / "resnet18" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"); ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    pp, cp = P.PlannerParams(), P.CostParams()
    W0, H0 = 1600, 900
    _fc = SP.field_crop
    margin = [0.0]

    def field_crop(*a, **k):
        c = _fc(*a, **k)
        if c is None or margin[0] == 0:
            return c
        w, h = c[2] - c[0], c[3] - c[1]
        return np.array([max(c[0] - margin[0] * w, 0), max(c[1] - margin[0] * h, 0), min(c[2] + margin[0] * w, a[3]), min(c[3] + margin[0] * h, a[4])])
    SP.field_crop = field_crop
    OUT.mkdir(parents=True, exist_ok=True)
    for k_, s in enumerate(sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)):
        if k_ % sn != si or (OUT / f"{s}.pkl").exists():
            continue
        rng = np.random.default_rng(0)
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev_scene, rows = None, {n: None for n in CFGS}, []
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731

            def lift(dd, pv):
                gg = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            tpx = [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
                tpx[0] += nh * nw
                dk, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = dk["conf"] >= 0.25
                return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]}
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
            for name, (ctx, fm) in CFGS.items():
                SP.CTX, margin[0] = ctx, fm
                tpx[0] = 0.0
                scene, crops, _ = SP.run(cheap_d, v, cam, W0, H0, detect_crop, lambda ff, n=name: lift(ff, prev_scene[n]), pp,
                                         v_close=10.0, guard=True)
                refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                for p_ in SP.support(scene["mono"], v, pp):
                    if not scene["from_full"][p_]:
                        continue
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                            cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    zg, zh = float(zz["z_ground"][0]), float(zz["z_height"][0])
                    f = np.array([[x1 / W0, y1 / H0, x2 / W0, y2 / H0, (x2 - x1) / W0, (y2 - y1) / H0, float(scene["conf"][p_]),
                                   float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                                   float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
                    f = ((f - fmu) / fsd).astype(np.float32)
                    x = torch.from_numpy(crop64(img, scene["xyxy"][p_])).permute(2, 0, 1)[None].float().to(dev) / 255.0
                    with torch.no_grad():
                        corr = float(np.exp(rnet(x, torch.from_numpy(f).to(dev)).item()))
                    for a in refined[:3]:
                        a[p_] *= corr
                rec[f"J_{name}"], rec[f"J_{name}_ref"] = cost(act(scene["mono"])), cost(act(tuple(refined)))
                rec[f"px_{name}"] = (192 * 320 + tpx[0]) / (W0 * H0)
                rec[f"crops_{name}"] = len(crops)
                prev_scene[name] = {"xyxy": scene["xyxy"]}
            SP.CTX, margin[0] = 2.0, 0.0
            rows.append(rec)
            prev_act = a_c
        pd.DataFrame(rows).to_pickle(OUT / f"{s}.pkl")
        print(f"  {s}", flush=True)
    print("K11 TEST RUN DONE", flush=True)


def analyze():
    d = pd.concat([pd.read_pickle(p) for p in sorted(OUT.glob("scene-*.pkl"))])
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_spec", "J_spec_ref_resnet18", "J_u960"]], on=["seq", "frame"])
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    Vf = L - m.J_frozen_ref.to_numpy()
    for col, lab in (("J_frozen_ref", "frozen v3 + refiner v1 (rerun)"), ("J_ctx2.5_fm0.15_ref", "ctx 2.5 + field margin 0.15 + refiner v1"),
                     ("J_frozen", "frozen v3 (rerun)"), ("J_ctx2.5_fm0.15", "ctx 2.5 + field margin 0.15"), ("J_u960", "u960 (confirmatory)")):
        V = L - m[col].to_numpy()
        dr = [(V[b].sum() - Vf[b].sum()) / L[b].sum() for b in boots]
        pxc = col.replace("J_", "px_").replace("_ref", "")
        out.append({"method": lab, "G": V.sum() / L.sum(), "network_px": m[pxc].mean() if pxc in m else np.nan,
                    "ci_lo_minus_frozen_ref": np.percentile(dr, 2.5), "ci_hi_minus_frozen_ref": np.percentile(dr, 97.5),
                    "frames": len(m), "scenes": len(uq)})
    par = {"parity_frozen_vs_confirmatory_ref": float(np.isclose(m.J_frozen_ref, m.J_spec_ref_resnet18).mean()),
           "parity_frozen_noref_vs_confirmatory": float(np.isclose(m.J_frozen, m.J_spec).mean())}
    res = pd.DataFrame(out); res.to_csv(OUT / "k11_test_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False)); print(par)


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze()
    else:
        sh = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
        run(*(int(x) for x in sh.split("/")))
