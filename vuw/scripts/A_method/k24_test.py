#!/usr/bin/env python
"""K24 — CONFIRMATORY run (method_freeze_draft §2i): v5.1-S / v5.1-L (primary), v5 (reference) and the baselines for
v5.1 on nuScenes new test (228 scenes), AV2 val (150 logs; ego mask through k10_masked_run.py) and Argoverse-HD val
(24 sequences; frames without 3D labels are run for the histories and flagged gt = False). Run ONCE.
Per frame:
  ours        the loop of v5 (k_loop.run_g, v_close 0, crops at 960 px long side, context 2, guard) with the lift policy
              of k27_dev: base (= v5), T4+T2_1 (= v5.1-S), T3+T4+T2_1 (= v5.1-L); nuScenes also v_close 10 base (= v3)
              refiners: none / frozen (v1 nuScenes, v2 AV2 / A-HD) / v3a
  whole frame u640 / u960 / u1280 / u1536 / native: own history; also with T4+T2_1 (same-detector history)
  heuristics  uncertainty / temporal_crit / random / traj_support / corridor (K18 protocol: per-frame crop canvases
              <= b x the reference's crop tensor px, b in {0.5, 1, 2}; reference = v5.1-S (S) and v5.1-L incl. its
              retrospective px (L)); guard, history; also with T4+T2_1; refiners none / v3a
usage: k24_test.py --dataset nusc|av2|ahd [--tag t] [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import runpy
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_k = importlib.util.spec_from_file_location("k27", B / "scripts" / "A_method" / "k27_dev.py")
K27 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K27)
IE, KL, KT = K27.IE, K27.KL, K27.KT
ttc_ss, parse = K27.ttc_ss, K27.parse
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
OURS = {"v5": "base", "v51S": "T4+T2_1", "v51L": "T3+T4+T2_1"}
TPOL = "T4+T2_1"
HEUR = ("uncertainty", "temporal_crit", "random", "traj_support", "corridor")
BUDGETS = (0.5, 1.0, 2.0)


def units(ds, max_units=0):
    if ds in ("nusc", "av2"):
        gen = IE.units_nusc("test") if ds == "nusc" else IE.units_av2("test")
    else:
        from vtf import av1 as A1
        HDm = runpy.run_path(str(B / "scripts" / "A_method" / "argoverse_hd_eval.py"), run_name="ahd_lib")

        def _g():
            for sid, log, names, ids in HDm["seqs"]():
                L = A1.AV1Log(HDm["AV1"] / "val", log, names)

                def frames(L=L, names=names):
                    geom, speeds = L.geometry(), L.speeds()
                    gtf = L.gt_frames()
                    for fr in range(len(names)):
                        if fr not in gtf:
                            g = geom[geom["frame"] == -1] if len(geom) else geom
                        else:
                            g = geom[geom["frame"] == fr] if len(geom) else geom
                            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                        yield fr, str(L.image_path(fr)), float(speeds[fr]), g, (None if fr in gtf else "nogt")
                yield f"seq{sid:02d}", L, frames
        gen = _g()
    for i, u in enumerate(gen):
        if max_units and i >= max_units:
            return
        yield u


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    nets = {}
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev, weights_only=False)
    n1 = TR.Res18(len(ck["fmu"])); n1.load_state_dict(ck["state"]); nets["frozen"] = (n1.to(dev).eval(), ck["fmu"], ck["fsd"])
    ck = torch.load(REAL / "out" / "resnet18_v3a" / "image_refiner.pt", map_location=dev, weights_only=False)
    n3 = KT.Res18x3(len(ck["fmu"])); n3.load_state_dict(ck["state"]); nets["v3a"] = (n3.to(dev).eval(), ck["fmu"], ck["fsd"])
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, cheap_px, SCALE, NW = None, 192 * 320, 0.6, 1600
        M = {"u640": Mode("f384x640", 384, 640), "u960": Mode("f576x960", 576, 960), "u1280": Mode("f736x1280", 736, 1280),
             "u1536": Mode("f864x1536", 864, 1536), "native": Mode("ns_u1600", 928, 1600)}
    elif ds == "av2":
        CH, cheap_px, SCALE, NW = Mode("av2_c320", 320, 256), 320 * 256, 960 / 2048, 2048
        M = {"u640": Mode("av2_f640", 640, 512), "u960": Mode("av2_f960", 960, 768), "u1280": Mode("av2_f1280", 1280, 992),
             "u1536": Mode("av2_f1536", 1536, 1184), "native": Mode("av2_u2048", 2048, 1568)}
    else:
        CH, cheap_px, SCALE, NW = Mode("ahd_c320", 224, 320), 224 * 320, 960 / 1920, 1920
        M = {f"u{s}" if s < 1920 else "native": Mode(f"ahd_u{s}", int(math.ceil(s * 1200 / 1920 / 32)) * 32, s)
             for s in (640, 960, 1280, 1536, 1920)}
    MS = {nm: (int(nm[1:]) if nm != "native" else NW) / NW for nm in M}
    od = REAL / "results" / "k24" / f"{ds}_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(units(ds, args.max_units)):
        if ui % sn != si:
            continue
        fo = od / f"{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        P2m = cam.calib.P2
        fx, fy, cx, cy = P2m[0, 0], P2m[1, 1], P2m[0, 2], P2m[1, 2]
        dt = float(cam.dt)
        st = {}
        prev_act, prev_c, imgs, rows = None, None, [], []
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            has_gt = not isinstance(cached, str)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "v": v, "gt": has_gt, "act_gt": P.discrete_action(a_gt, pp)}

            def geom(dd, pv=None):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

            def ckey(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                xe, ye = min(x2, W), min(y2, H)
                nw = max(32, int(math.ceil((xe - x1) * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil((ye - y1) * SCALE / 32)) * 32)
                return (x1, y1, x2, y2, nh, nw)
            cache, rcache = {}, {}

            def run_det(im, key):
                x1, y1, x2, y2, nh, nw = key
                d = keep(det.detect(im[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0])
                return {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}

            def make_detect(tpx, log=None):
                def detect_crop(cr):
                    key = ckey(cr)
                    tpx[0] += key[4] * key[5]
                    if key not in cache:
                        cache[key] = run_det(img, key)
                    out = {k: x.copy() for k, x in cache[key].items()}
                    if log is not None:
                        for b in out["xyxy"]:
                            log[tuple(b.tolist())] = key
                    return out
                return detect_crop

            def make_retro(log, used):
                def retro(box, lag):
                    key = log.get(tuple(np.asarray(box, float).tolist()))
                    if key is None or len(imgs) < lag:
                        return None
                    rk = (lag,) + key
                    used.add(rk)
                    if rk not in rcache:
                        rcache[rk] = run_det(imgs[-lag], key)["xyxy"]
                    return rcache[rk]
                return retro

            def lift_pol(name, pol, sc, retro=None):
                h = st.get(name, {})

                def f(full):
                    if pol is None:
                        return geom(full, {"xyxy": h["merged"]} if "merged" in h else None)
                    z, lo, hi, _ = geom(full)
                    tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"],
                                   retro if pol["t3"] else None)
                    return z, lo, hi, tt
                return f

            def push(name, scene):
                h = st.setdefault(name, {})
                ff = np.asarray(scene["from_full"], bool)
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
                h["merged"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)

            def refine(scene, name):
                tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                if name == "none":
                    return tuple(tt)
                sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
                if not sup:
                    return tuple(tt)
                net, fmu, fsd = nets[name]
                zz = predicted_geometry({"xyxy": scene["xyxy"][sup], "coarse": scene["coarse"][sup]}, None, cam.calib, cam.cam_h,
                                        dt, cam_to_ego=(R, t))
                Fm = []
                for n, p_ in enumerate(sup):
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                               float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                               float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
                Fm = ((np.array(Fm, np.float32) - fmu) / fsd).astype(np.float32)
                x = torch.from_numpy(np.stack([crop64(img, scene["xyxy"][p_]) for p_ in sup])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.no_grad():
                    o = net(x, torch.from_numpy(Fm).to(dev)).float().cpu().numpy().reshape(len(sup), -1)
                for n, p_ in enumerate(sup):
                    c = float(np.exp(o[n, 0]))
                    for a in tt[:3]:
                        a[p_] *= c
                return tuple(tt)

            def score(prefix, scene, refs=("none", "v3a")):
                for r_ in refs:
                    a1 = act(refine(scene, r_))
                    rec[f"a_{prefix}|{r_}"], rec[f"J_{prefix}|{r_}"] = a1, cost(a1)

            def scene_dict(dd, mono):
                return {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                        "from_full": np.ones(len(dd["conf"]), bool)}
            # ---- draft
            if isinstance(cached, dict):
                ch = cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = geom(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            # ---- whole frame (own history; + T4+T2_1)
            tp = parse(TPOL)
            for nm, md in M.items():
                d = keep(det.detect(img, md)[0])
                mono = geom(d, {"xyxy": st[nm]["merged"]} if nm in st else None)
                score(nm, scene_dict(d, mono), ("none", "frozen", "v3a"))
                tt, _ = ttc_ss(d["xyxy"], st.get(nm, {}).get("s1"), st.get(nm, {}).get("s2"), MS[nm], dt, tp["gate"], tp["t4"])
                score(f"{nm}+T4", scene_dict(d, mono[:3] + (tt,)))
                rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                push(nm, scene_dict(d, None))
            # ---- ours (v5, v5.1-S, v5.1-L; nuScenes v3 = v_close 10 base)
            crop_px = {}
            loops = [(k, a, 0.0) for k, a in OURS.items()] + ([("v3", "base", 10.0)] if ds == "nusc" else [])
            for nm, arm, vc in loops:
                tpx, log, used = [0.0], {}, set()
                sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, log),
                                                          lift_pol(nm, parse(arm), SCALE, make_retro(log, used)), pp,
                                                          v_closes=(vc,))
                rpx = sum(k_[5] * k_[6] for k_ in used)
                score(nm, sc, ("none", "frozen", "v3a"))
                crop_px[nm] = tpx[0] + rpx
                rec[f"px_{nm}"] = (cheap_px + tpx[0] + rpx) / NATIVE
                rec[f"pxretro_{nm}"], rec[f"nretro_{nm}"], rec[f"rounds_{nm}"] = rpx / NATIVE, len(used), rounds
                push(nm, sc)
            # ---- heuristics at b x the crop tensor px of v5.1-S / v5.1-L (K18 protocol)

            def canvas(cr):
                k_ = ckey(cr)
                return k_[4] * k_[5]

            def fill_px(cands, bud):
                sel, used_ = [], 0
                for c in cands:
                    cv_ = canvas(c)
                    if used_ + cv_ <= bud:
                        sel.append(c); used_ += cv_
                return sel, used_

            def tube_px(bud):
                vb_n, top = min(cy + fy * cam.cam_h / 10.0, H), max(cy + fy * cam.cam_h / 70.0 - fy * 1.6 / 70.0, 0)
                best = []
                for half_m in np.arange(0.5, 12.01, 0.25):
                    half = fx * half_m / 10.0
                    c = np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])
                    if canvas(c) > bud:
                        break
                    best = [c]
                return best
            ent = np.asarray(ch.get("entropy", np.zeros(len(ch["conf"]))))
            for ref_nm, ref_key in (("S", "v51S"), ("L", "v51L")):
                for bx in BUDGETS:
                    bud = bx * crop_px[ref_key]
                    for hn in HEUR:
                        nm = f"{hn}_{ref_nm}x{bx:g}"
                        if hn == "uncertainty":
                            sel, _ = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in np.lexsort((-ent, ch["conf"]))], bud)
                        elif hn == "temporal_crit":
                            cands = []
                            ps_ = st.get(nm, {}).get("state")
                            if ps_ is not None and len(ps_[0]):
                                pxy, pmono = ps_
                                _, dem = P.required_decel(*pmono, v, pp)
                                order = np.lexsort((pmono[0], -np.asarray(dem)))
                                cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                           pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                            sel, _ = fill_px(cands, bud)
                        elif hn == "random":
                            rr = np.random.default_rng(zlib.crc32(f"{unit}/{fr}".encode()))
                            cands = []
                            for _ in range(40):
                                if len(ch["xyxy"]):
                                    b = ch["xyxy"][rr.integers(len(ch["xyxy"]))]
                                    w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                                else:
                                    w_, h_ = 200.0, 120.0
                                h_ = min(h_, H * 0.65)
                                x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                                cands.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                            sel, _ = fill_px(cands, bud)
                        elif hn == "traj_support":
                            sup, used_ = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in SP.support(ch["mono"], v, pp)], bud) if bud > 0 else ([], 0)
                            sel = sup + (tube_px(bud - used_) if bud - used_ > 0 else [])
                        else:
                            sel = tube_px(bud) if bud > 0 else []
                        for pol_nm, pol in (("", None), ("+T4", tp)):
                            name = nm + pol_nm
                            tpx = [0.0]
                            sh, _ = IE.merge_once(base, sel, SP.guarded(make_detect(tpx), W, H), lift_pol(name, pol, SCALE))
                            score(name, sh)
                            rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                            push(name, sh)
                            if hn == "temporal_crit":
                                st[name]["state"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            imgs = (imgs + [img])[-2:]
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K24 TEST DONE", flush=True)


if __name__ == "__main__":
    main()
