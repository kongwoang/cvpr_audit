#!/usr/bin/env python
"""K10 part 1 / 3 — how often is a pipeline's braking action driven by a detection in the ego region (hood)?

Per frame and pipeline (unmasked, as in the main tables): the scene's objects are lifted, the action is taken; the
DRIVER is the object with the largest required deceleration when the action is not KEEP (none for KEEP). A frame is
"hood-driven" if the driver's box lies at least half inside the ego-region mask (full-width band y >= y0, the same rule
as k10_masked_run.py). Also logged: number of detections in the mask, the decision cost J (prev action = cheap's, as in
every table). Datasets / masks: AV2 val (2048 rows, y0 1830 = the hood), Argoverse-HD val (1200 rows, y0 1140; no ego
part is visible), nuScenes new test (900 rows, y0 855; no ego part is visible).
Pipelines: cheap, u640, u960, u1280, native (AV2 also u1536), ours final (AV2 / Argoverse-HD: v4-lite + refiner v2;
nuScenes: v3 + refiner v1), with the frozen configurations, exactly the loops of instr_eval / argoverse_hd_eval.
usage: k10_hood.py --dataset av2|ahd|nusc [--shard i/n] | --analyze
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import runpy
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

OUT = B / "results" / "k10_hood"
Y0 = {"av2": 1830, "ahd": 1140, "nusc": 855}
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def in_mask(xyxy, y0):
    b = np.asarray(xyxy, float).reshape(-1, 4)
    if not len(b):
        return np.zeros(0, bool)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    return (b[:, 3] - np.maximum(b[:, 1], y0)).clip(0, None) / h >= 0.5


def units(ds):
    if ds in ("av2", "nusc"):
        for u, cam, gen in (IE.units_av2("test") if ds == "av2" else IE.units_nusc("test")):
            yield u, cam, gen
        return
    from vtf import av1 as A1
    HDm = runpy.run_path(str(B / "scripts" / "A_method" / "argoverse_hd_eval.py"), run_name="ahd_lib")
    for sid, log, names, ids in HDm["seqs"]():
        L = A1.AV1Log(HDm["AV1"] / "val", log, names)

        def gen(L=L):
            geom, speeds = L.geometry(), L.speeds()
            gtf = L.gt_frames()
            for fr in range(len(names)):
                if fr not in gtf:
                    g = geom[geom["frame"] == -1] if len(geom) else geom
                else:
                    g = geom[geom["frame"] == fr] if len(geom) else geom
                    g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, (None if fr in gtf else "nogt")
        yield f"seq{sid:02d}", L, gen


def run(args):
    import torch
    import sys
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    dev = torch.device(args.device)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        VC, SCALE, CH = 10.0, 0.6, None
        M = {"u640": Mode("f384x640", 384, 640), "u960": Mode("f576x960", 576, 960), "u1280": Mode("f736x1280", 736, 1280),
             "native": Mode("ns_u1600", 928, 1600)}
    elif ds == "av2":
        VC, SCALE, CH = 0.0, 960 / 2048, Mode("av2_c320", 320, 256)
        M = {"u640": Mode("av2_f640", 640, 512), "u960": Mode("av2_f960", 960, 768), "u1280": Mode("av2_f1280", 1280, 992),
             "u1536": Mode("av2_f1536", 1536, 1184), "native": Mode("av2_u2048", 2048, 1568)}
    else:
        VC, SCALE, CH = 0.0, 960 / 1920, Mode("ahd_c320", 224, 320)
        M = {f"u{s}" if s < 1920 else "native": Mode(f"ahd_u{s}", int(math.ceil(s * 1200 / 1920 / 32)) * 32, s) for s in (640, 960, 1280, 1920)}
    y0 = Y0[ds]
    OUT.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(units(ds)):
        if ui % sn != si:
            continue
        fo = OUT / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, extra in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            has_gt = extra != "nogt"
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "gt": has_gt}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def put(name, xyxy, tt):
                a, a_obj = P.required_decel(*tt, v, pp)
                act = P.discrete_action(a, pp)
                drv = int(np.argmax(a_obj)) if act != P.KEEP and len(a_obj) else -1
                m = in_mask(xyxy, y0)
                rec[f"a_{name}"], rec[f"J_{name}"] = act, P.decision_cost(act, a_gt, prev_act, pp, cp)["J"]
                rec[f"hood_{name}"] = bool(drv >= 0 and m[drv])
                rec[f"nmask_{name}"] = int(m.sum())
                return act
            if extra is not None and not isinstance(extra, str):
                ch = extra                                                       # nuScenes cached draft (as the tables)
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            a_c = put("cheap", ch["xyxy"], ch["mono"])
            for nm, m in M.items():
                d = keep(det.detect(img, m)[0])
                put(nm, d["xyxy"], lift(d, hist.get(nm)))
                hist[nm] = d

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
            scene, crops, _ = SP.run({k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H, detect_crop,
                                     lambda full: lift(full, hist.get("scene")), pp, v_close=VC, guard=True)
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
            put("ours", scene["xyxy"], tuple(refined))
            # instrumented detector pixels of ours (draft canvas + every crop canvas incl. guard re-detections)
            rec["tpx_ours"] = ((CH.net_h * CH.net_w if CH is not None else 192 * 320) + tpx[0]) / (W * H)
            rec["crop_on_mask_ours"] = bool(any(in_mask([c], y0)[0] for c in crops))
            hist["scene"] = {"xyxy": scene["xyxy"]}
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  {ds} {unit}: {len(rows)} frames", flush=True)
    print("K10 HOOD RUN DONE", flush=True)


def analyze(args):
    out = []
    for ds in ("av2", "ahd", "nusc"):
        rows = []
        for p in sorted(OUT.glob(f"{ds}__*.pkl")):
            rows += pickle.load(open(p, "rb"))
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d = d[d["gt"].astype(bool)]
        names = [c[2:] for c in d.columns if c.startswith("a_")]
        for n in names:
            h = d[f"hood_{n}"].astype(bool)
            J = d[f"J_{n}"]
            out.append({"dataset": ds, "mask_rows_from": Y0[ds], "pipeline": n, "frames": len(d),
                        "hood_driven_frames": int(h.sum()), "hood_driven_share": h.mean(),
                        "loss_share_from_hood_frames": J[h].sum() / max(J.sum(), 1e-9),
                        "G_parity": (d.J_cheap.sum() - J.sum()) / d.J_cheap.sum(),
                        "detections_in_mask_per_frame": d[f"nmask_{n}"].mean(),
                        "frames_with_detection_in_mask": float((d[f"nmask_{n}"] > 0).mean())})
            if n == "ours":
                out[-1]["frames_with_crop_on_mask"] = float(d.crop_on_mask_ours.mean())
                if "tpx_ours" in d:
                    out[-1]["tensor_px_ours_mean"] = float(d.tpx_ours.mean()); out[-1]["tensor_px_ours_p95"] = float(d.tpx_ours.quantile(.95))
                    out[-1]["tensor_px_frames"] = int(d.tpx_ours.notna().sum())
    r = pd.DataFrame(out)
    r.to_csv(OUT / "hood_share.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["av2", "ahd", "nusc"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
