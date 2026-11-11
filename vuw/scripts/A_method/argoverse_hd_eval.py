#!/usr/bin/env python
"""Argoverse-HD val (the streaming-perception benchmark of FOVEA / LZU; Argoverse 1.1 ring_front_center 1920x1200 @ 30 Hz,
24 sequences, 15,062 frames) with frozen COCO YOLOv8s — our AV2 configuration applied unchanged (no tuning here).

Pipelines (every frame, in order, each with its own lift history): cheap 320, uniform 640 / 960 / 1280 / 1600 / 1920
(native), ours = AV2 config (v4-lite: cheap draft, braking field v_close 0, support crops at 960 / 1920 = 0.5, truncation
guard, merged-scene history, <= 3 rounds) + refiner v2.
Outputs per frame:
  * detections with conf >= 0.1 for COCO-style AP against Argoverse-HD's annotations, in three road-user groups
    (vehicle = car / bus / truck, person, two-wheeler = bicycle / motorcycle; our pipeline carries coarse classes only,
    so every method is scored on the same groups); ours = cheap boxes outside its crops ∪ de-duplicated crop boxes
  * the braking decision cost (DEEP's), on frames with 3D ground truth (labelled LiDAR sweep within 50 ms, ~10 Hz),
    cheap action history at 30 Hz — decision value G as on AV2
  * network pixels per frame (for the compute axis and the streaming simulation)
Stages: --stage run [--shard i/n] (GPU) | --stage score (CPU: AP via pycocotools, G with log bootstrap).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
HD = B / "datasets" / "argoverse_hd" / "Argoverse-HD" / "annotations"
AV1 = B / "datasets" / "argoverse1" / "argoverse-tracking"
OUT = B / "results" / "argoverse_hd"
THR, LOW = 0.25, 0.1
GROUP = {0: 1, 1: 2, 2: 0, 3: 2, 4: 0, 5: 0}             # Argoverse-HD category -> group (0 vehicle, 1 person, 2 two-wheeler)
CGROUP = {"vehicle": 0, "person": 1, "cyclist": 2}
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
UNIF = (640, 960, 1280, 1600, 1920)


def mode_for(s):
    from rap.detect import Mode
    return Mode(f"ahd_u{s}", int(math.ceil(s * 1200 / 1920 / 32)) * 32, s)


def seqs():
    d = json.loads((HD / "val.json").read_text())
    by = {}
    for im in d["images"]:
        by.setdefault(im["sid"], []).append(im)
    out = []
    for sid, ims in sorted(by.items()):
        ims = sorted(ims, key=lambda x: x["fid"])
        sd = d["seq_dirs"][sid]                              # 'val/<log>/ring_front_center'
        out.append((sid, sd.split("/")[1], [i["name"] for i in ims], [i["id"] for i in ims]))
    return out


def stage_run(args):
    import torch
    from rap import planner as P
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import box_iou, predicted_geometry
    from vtf import av1 as A1
    from vtf import speculative as SP
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    si, sn = (int(x) for x in args.shard.split("/"))
    dev = torch.device(args.device)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    ck = torch.load(B / "out" / "resnet18_v2" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    M_CHEAP = Mode("ahd_c320", 224, 320)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (sid, log, names, ids) in enumerate(seqs()):
        if k % sn != si or (OUT / f"seq{sid:02d}.pkl").exists():
            continue
        L = A1.AV1Log(AV1 / "val", log, names)
        geom, speeds = L.geometry(), L.speeds()
        gtf = L.gt_frames()
        R, t = L.transform("ego")
        W, H = L.W, L.H
        scale = 960 / max(W, H)
        prev = {s: None for s in ("c",) + UNIF}
        prev_scene, prev_act = None, None
        rows, dets = [], {}

        def lift(dd, pv):
            g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def cut(d, thr):
            k_ = d["conf"] >= thr
            return {"xyxy": d["xyxy"][k_].astype(float), "conf": d["conf"][k_].astype(float), "coarse": d["coarse"][k_]}
        for fr, (name, iid) in enumerate(zip(names, ids)):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None:
                continue
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            has_gt = fr in gtf
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            rec = {"sid": sid, "frame": fr, "image_id": iid, "gt": has_gt, "v": v}
            raw_c = det.detect(img, M_CHEAP)[0]
            ch = cut(raw_c, THR)
            ch["mono"] = lift(ch, prev["c"])
            a_c = act(ch["mono"])
            rec["J_c"] = cost(a_c)
            dets[(iid, "cheap")] = cut(raw_c, LOW)
            for s in UNIF:
                raw = det.detect(img, mode_for(s))[0]
                dd = cut(raw, THR)
                rec[f"J_u{s}"] = cost(act(lift(dd, prev[s])))
                prev[s] = dd
                dets[(iid, f"u{s}")] = cut(raw, LOW)
            low_crops = []

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                raw = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0]
                lo = cut(raw, LOW)
                lo["xyxy"] = lo["xyxy"] + np.array([x1, y1, x1, y1], float)
                low_crops.append(lo)
                d = cut(raw, THR)
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            scene, crops, rounds = SP.run(ch, v, L, W, H, detect_crop, lambda ff: lift(ff, prev_scene), pp, v_close=0.0, guard=True)
            rec["J_ours"] = cost(act(scene["mono"]))
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            for p_ in SP.support(scene["mono"], v, pp):
                if not scene["from_full"][p_]:
                    continue
                x1, y1, x2, y2 = scene["xyxy"][p_]
                zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                        L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
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
            rec["J_ours_ref"] = cost(act(tuple(refined)))
            rec["n_crops"], rec["rounds"] = len(crops), rounds
            rec["netpx_ours"] = 224 * 320 / (W * H) + sum((c[2] - c[0]) * (c[3] - c[1]) for c in crops) * scale ** 2 / (W * H)
            # AP output of ours: low-confidence cheap boxes outside the crops ∪ de-duplicated low-confidence crop boxes
            lc = cut(raw_c, LOW)
            out_ = SP.centres_in(lc["xyxy"], crops) if crops else np.zeros(len(lc["xyxy"]), bool)
            full = {k_: np.concatenate([d_[k_] for d_ in low_crops]) if low_crops else np.zeros((0, 4) if k_ == "xyxy" else 0)
                    for k_ in ("xyxy", "conf", "coarse")}
            if len(full["conf"]) > 1:
                o = np.argsort(-full["conf"], kind="stable"); iou = box_iou(full["xyxy"], full["xyxy"]); kp = []
                for i in o:
                    if not kp or iou[i, kp].max() < 0.5:
                        kp.append(i)
                full = {k_: full[k_][np.sort(np.array(kp))] for k_ in full}
            dets[(iid, "ours")] = {k_: np.concatenate([np.asarray(lc[k_])[~out_], np.asarray(full[k_])]) for k_ in ("xyxy", "conf", "coarse")}
            prev["c"], prev_scene, prev_act = ch, {"xyxy": scene["xyxy"]}, a_c
            rows.append(rec)
        with open(OUT / f"seq{sid:02d}.pkl", "wb") as fh:
            pickle.dump({"rows": rows, "dets": dets}, fh)
        print(f"  seq {sid} ({log[:8]}): {len(rows)} frames, {sum(r['gt'] for r in rows)} with 3D GT, crops/frame "
              f"{np.mean([r['n_crops'] for r in rows]):.2f}", flush=True)
    print("ARGOVERSE HD RUN DONE")


def stage_score(args):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    d = json.loads((HD / "val.json").read_text())
    gt = {"images": d["images"], "categories": [{"id": i, "name": n} for i, n in enumerate(("vehicle", "person", "two-wheeler"))],
          "annotations": [{**a, "category_id": GROUP[a["category_id"]]} for a in d["annotations"] if a["category_id"] in GROUP]}
    tmp = OUT / "gt_groups.json"
    tmp.write_text(json.dumps(gt))
    cg = COCO(str(tmp))
    rows, dets = [], {}
    for p in sorted(OUT.glob("seq*.pkl")):
        z = pickle.load(open(p, "rb"))
        rows += z["rows"]; dets.update(z["dets"])
    df = pd.DataFrame(rows)
    methods = ["cheap"] + [f"u{s}" for s in UNIF] + ["ours"]
    res = []
    for m in methods:
        out = []
        for (iid, mm), dd in dets.items():
            if mm != m:
                continue
            for b, c, k in zip(dd["xyxy"], dd["conf"], dd["coarse"]):
                out.append({"image_id": int(iid), "category_id": CGROUP[str(k)], "score": float(c),
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        E = COCOeval(cg, cg.loadRes(out), "bbox")
        E.params.imgIds = sorted({int(i) for (i, mm) in dets if mm == m})
        E.evaluate(); E.accumulate(); E.summarize()
        res.append({"method": m, "AP": E.stats[0], "AP50": E.stats[1], "AP_s": E.stats[3], "AP_m": E.stats[4], "AP_l": E.stats[5]})
    ap = pd.DataFrame(res)
    g = df[df["gt"].astype(bool)]
    Lc = g.J_c.to_numpy()
    uq = g.sid.unique(); idx = [np.flatnonzero(g.sid.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {m: (g.J_c - g[f"J_{m}"]).to_numpy() for m in [f"u{s}" for s in UNIF] + ["ours", "ours_ref"]}
    px = {f"u{s}": mode_for(s).net_h * mode_for(s).net_w / (1920 * 1200) for s in UNIF}
    px["ours"] = px["ours_ref"] = float(df.netpx_ours.mean())
    out = []
    for m, v_ in V.items():
        r = {"method": m, "G": v_.sum() / Lc.sum(), "network px": px[m]}
        if m != "ours_ref":
            dr = [(V["ours_ref"][t_].sum() - v_[t_].sum()) / Lc[t_].sum() for t_ in boots]
            r["ours_ref minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
    gres = pd.DataFrame(out)
    ap.to_csv(OUT / "summary_ap.csv", index=False)
    gres.to_csv(OUT / "summary_G.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(df)} frames ({len(g)} with 3D GT) in {df.sid.nunique()} sequences; ours crops/frame {df.n_crops.mean():.2f}")
    print(ap.round(4).to_string(index=False))
    print(gres.to_string(index=False))
    print("ARGOVERSE HD SCORE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "score"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    {"run": stage_run, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
