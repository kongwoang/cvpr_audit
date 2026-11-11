#!/usr/bin/env python
"""K12(d) — the missing cell of the nuScenes fine-tuned table: v3 with the fine-tuned verifier + refiner (post-hoc
analysis, nuScenes new test, both detector seeds).

Exactly fovea_ft.py's "v3 with the fine-tuned verifier on crops" (COCO cached cheap draft, YOLOv8s fine-tuned at 960 on
the new train scenes = out/fovea_ft/plain{sfx}, crops at 0.6, guard, merged-scene history, v_close 10), then the frozen
refiner v1 (out/resnet18, nuScenes' final configuration) on the support objects from crops, as nusc_full_eval / instr.
Writes J without refiner too (parity with fovea_ft's J_v3_ftverifier) and the costs of u960-ft for the table.
usage: FT_SUFFIX= | _s1  python k12_ftref.py      then  python k12_ftref.py --analyze
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
OUT = B / "results" / "k12_ftref"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def run():
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
    sfx = os.environ.get("FT_SUFFIX", "")
    dev = torch.device("cuda:0")
    det_p = TwoFidelityDetector(str(B / "out" / "fovea_ft" / f"plain{sfx}" / "weights" / "best.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / "resnet18" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"); ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    W0, H0 = 1600, 900
    rows = []
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev_scene = None, None
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
                return tuple(np.asarray(gg[k], float) for k in ("z", "lat_min", "lat_max", "ttc"))
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
                dk, _ = det_p.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = dk["conf"] >= 0.25
                return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]}
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
            scene, crops, _ = SP.run(cheap_d, v, cam, W0, H0, detect_crop, lambda ff: lift(ff, prev_scene), pp, v_close=10.0, guard=True)
            rec["J_v3_ftverifier"] = cost(act(scene["mono"]))
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
            rec["J_v3_ftverifier_ref"] = cost(act(tuple(refined)))
            prev_scene = {"xyxy": scene["xyxy"]}
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(OUT / f"ftref_newtest{sfx}.pkl")
    print("K12 FTREF DONE", flush=True)


def analyze():
    out = []
    for sfx, seed in (("", 0), ("_s1", 1)):
        p = OUT / f"ftref_newtest{sfx}.pkl"
        if not p.exists():
            continue
        d = pd.read_pickle(p)
        o = pd.read_pickle(B / "results" / "fovea" / f"eval_ft_newtest{sfx}.pkl")
        m = d.merge(o[["seq", "frame", "J_v3_ftverifier", "J_u960_ft"]], on=["seq", "frame"], suffixes=("", "_orig"))
        parity = float(np.mean(np.isclose(m.J_v3_ftverifier, m.J_v3_ftverifier_orig)))
        L = m.J_c.to_numpy()
        uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
        rng = np.random.default_rng(0)
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        Vr = L - m.J_v3_ftverifier_ref.to_numpy()
        for col, lab in (("J_v3_ftverifier", "v3 with fine-tuned verifier"), ("J_v3_ftverifier_ref", "v3 with fine-tuned verifier + refiner v1"),
                         ("J_u960_ft", "u960-ft")):
            V = L - m[col].to_numpy()
            r = {"seed": seed, "method": lab, "G": V.sum() / L.sum(), "frames": len(m), "scenes": len(uq),
                 "parity_without_refiner_vs_fovea_ft": parity}
            if col != "J_v3_ftverifier_ref":
                dr = [(Vr[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
                r["ci_lo_ftref_minus"], r["ci_hi_ftref_minus"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "ft_refiner.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else run()
