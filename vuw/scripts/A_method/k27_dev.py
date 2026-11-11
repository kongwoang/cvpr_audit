#!/usr/bin/env python
"""K27 — temporal (looming) cue of crop objects, DEVELOPMENT data only (method_freeze_draft §2g). nuScenes new val
(152 scenes, cached draft) and AV2 CFG-DEV (300 train logs of results/k23/av2_split.csv; ego mask through
k10_masked_run.py). Nothing here touches a test split.
Base = v5 (§2f): the frozen loop (k_loop.run_g) at v_close 0, crops at nuScenes scale 0.6 / AV2 960 px long side,
context 2, no margin, <= 3 rounds, guard; refiners none / previous frozen (v1 nuScenes, v2 AV2) / v3a.
An ARM changes only the ttc given to the crop detections (the lift's history); the draft keeps its own ttc:
  base        the frozen lift: crop boxes paired (greedy IoU >= 0.3) with the previous frame's MERGED scene
  T1          paired only with the previous final scene's crop / guard-crop boxes (same detector scale); else ttc inf
  T2_<k>      T1 + gate: ttc kept only if (h_t - h_prev) x detector scale > k px
  T3          T1 + retrospective verification: a crop box with no same-scale match at t-1 is paired with the
              detections of the same crop canvas re-detected in the buffered frame t-1 (extra px / crops counted)
  T4          growth from 3 consecutive same-scale observations: median of the 3 pairwise rates; no ttc before
  '+'-joined combinations (T4+T2_k gates on h_t - h_{t-2}; T3+T4 re-detects in t-1 and t-2 when the chain is missing)
Per frame also: support-only loops per arm (field-raise statistics of K22(3)); nuScenes v_close 10 per arm (the
pre-registered nuScenes geometry); controls u960 (base / T2_k / T4), uncertainty crops and the ego-corridor tube at
v5's per-frame crop tensor px (base / T1 / T2_k / T4); the DIAGNOSTIC changed reference (gate on every pipeline incl.
the draft: u640 / u960 / native / ours); the base arm's action-raising crop objects (diagnosis (0)).
usage: k27_dev.py --dataset nusc|av2 [--tag t] [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_l = importlib.util.spec_from_file_location("kl", B / "scripts" / "A_method" / "k_loop.py")
KL = importlib.util.module_from_spec(_l); _l.loader.exec_module(KL)
_t = importlib.util.spec_from_file_location("kt", B / "scripts" / "A_method" / "k23_train_refiner.py")
KT = importlib.util.module_from_spec(_t); _t.loader.exec_module(KT)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import associate_prev, box_iou, predicted_geometry, ttc_from_scale  # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
KS = (1.0, 1.5, 2.0)
ALONE = ["base", "T1"] + [f"T2_{k:g}" for k in KS] + ["T3", "T4"]
COMBO = ([f"T3+T2_{k:g}" for k in KS] + [f"T4+T2_{k:g}" for k in KS] + ["T3+T4"] + [f"T3+T4+T2_{k:g}" for k in KS])
ARMS = ALONE + COMBO
CTRL = ["base", "T1"] + [f"T2_{k:g}" for k in KS] + ["T4"]


def parse(arm):
    if arm == "base":
        return None
    p = {"gate": None, "t3": False, "t4": False}
    for tok in arm.split("+"):
        if tok == "T1":
            continue
        if tok.startswith("T2_"):
            p["gate"] = float(tok[3:])
        elif tok == "T3":
            p["t3"] = True
        elif tok == "T4":
            p["t4"] = True
        else:
            raise ValueError(tok)
    return p


def assoc_idx(curr, prev, thr=0.3):
    """rap.mono.associate_prev, returning the index of the previous box (-1) instead of its height"""
    out = -np.ones(len(curr), int)
    if len(prev) == 0 or len(curr) == 0:
        return out
    iou = box_iou(curr, prev)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        for pj in np.argsort(-iou[ci]):
            if iou[ci, pj] < thr:
                break
            if pj not in taken:
                taken.add(int(pj)); out[ci] = pj
                break
    return out


def ttc_ss(X, s1, s2, sc, dt, gate=None, t4=False, retro=None):
    """ttc of boxes X (n, 4; full-image px) from same-scale history s1 (t-1) / s2 (t-2) (arrays or None) at detector
    scale sc. retro(box, lag) -> re-detected boxes of the box's crop canvas in frame t-lag (T3) or None.
    Returns ttc, source of the t-1 partner (0 none, 1 same-scale, 2 retrospective)."""
    X = np.asarray(X, float).reshape(-1, 4)
    n = len(X)
    h = X[:, 3] - X[:, 1]
    P1, src = np.full((n, 4), np.nan), np.zeros(n, int)
    j1 = assoc_idx(X, s1) if s1 is not None and len(s1) else -np.ones(n, int)
    if (j1 >= 0).any():
        P1[j1 >= 0], src[j1 >= 0] = s1[j1[j1 >= 0]], 1
    if retro is not None:
        for i in np.flatnonzero(src == 0):
            R = retro(X[i], 1)
            if R is not None and len(R):
                io = box_iou(X[i:i + 1], R)[0]; k = int(np.argmax(io))
                if io[k] >= 0.3:
                    P1[i], src[i] = R[k], 2
    h1 = P1[:, 3] - P1[:, 1]
    if not t4:
        ttc = ttc_from_scale(h, h1, dt)
        if gate is not None:
            with np.errstate(invalid="ignore"):
                ttc[~((h - h1) * sc > gate)] = np.inf
        return ttc, src
    P2 = np.full((n, 4), np.nan)
    if s1 is not None and len(s1) and s2 is not None and len(s2):
        a12 = assoc_idx(s1, s2)
        for i in np.flatnonzero(src == 1):
            if a12[j1[i]] >= 0:
                P2[i] = s2[a12[j1[i]]]
    if retro is not None:
        for i in np.flatnonzero((src > 0) & ~np.isfinite(P2[:, 0])):
            R = retro(X[i], 2)
            if R is not None and len(R):
                io = box_iou(P1[i:i + 1], R)[0]; k = int(np.argmax(io))
                if io[k] >= 0.3:
                    P2[i] = R[k]
    h2 = P2[:, 3] - P2[:, 1]
    ttc = np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(h1) & np.isfinite(h2) & (h > 1) & (h1 > 1) & (h2 > 1)
        r = np.median(np.stack([(h1 / h2 - 1) / dt, (h / h1 - 1) / dt, (h / h2 - 1) / (2 * dt)]), 0)
        ex = ok & (r > 1e-3 / dt)
        ttc[ex] = np.minimum(1 / r[ex], 1e3)
        if gate is not None:
            ttc[~((h - h2) * sc > gate)] = np.inf
    return ttc, src


def units(ds, max_units=0):
    if ds == "nusc":
        for i, u in enumerate(IE.units_nusc("val")):
            if max_units and i >= max_units:
                return
            yield u
        return
    from vtf import av2 as AV
    sp = pd.read_csv(REAL / "results" / "k23" / "av2_split.csv")
    root = B / "datasets" / "av2" / "sensor" / "train"
    for i, log in enumerate(sorted(sp[sp.role == "CFG-DEV"].log)):
        if max_units and i >= max_units:
            return
        L = AV.AV2Log(root, log)

        def frames(L=L):
            geom, speeds = L.geometry(), L.speeds()
            for fr in range(len(L.frame_ts)):
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, None
        yield log, L, frames


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
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
        CH, cheap_px, SCALE, DS = None, 192 * 320, 0.6, 320 / 1600
        M = {"u640": Mode("f384x640", 384, 640), "u960": Mode("f576x960", 576, 960), "native": Mode("ns_u1600", 928, 1600)}
        MS = {"u640": 0.4, "u960": 0.6, "native": 1.0}
    else:
        CH, cheap_px, SCALE, DS = Mode("av2_c320", 320, 256), 320 * 256, 960 / 2048, 320 / 2048
        M = {"u640": Mode("av2_f640", 640, 512), "u960": Mode("av2_f960", 960, 768), "native": Mode("av2_u2048", 2048, 1568)}
        MS = {"u640": 640 / 2048, "u960": 960 / 2048, "native": 1.0}
    POL = {a: parse(a) for a in ARMS}
    od = REAL / "results" / "k27" / f"{ds}_{args.tag}"
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
        st = {}                       # pipeline -> {"merged", "s1", "s2", "src"} (history of final scenes)
        prev_act, prev_c, prev_draft, imgs, rows, prev_state = None, None, None, [], [], {}
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "v": v, "act_gt": P.discrete_action(a_gt, pp)}
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))

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
                """lift of crop / full-image detections of pipeline `name` under policy pol (None = merged history)"""
                h = st.get(name, {})

                def f(full):
                    if pol is None:
                        return geom(full, {"xyxy": h["merged"]} if "merged" in h else None)
                    z, lo, hi, _ = geom(full)
                    tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"],
                                   retro if pol["t3"] else None)
                    return z, lo, hi, tt
                return f

            def push(name, scene, src=None):
                h = st.setdefault(name, {})
                ff = np.asarray(scene["from_full"], bool)
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
                h["merged"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)
                h["src"] = src

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

            def score(prefix, scene, refs=("none", "frozen", "v3a")):
                out = {}
                for r_ in refs:
                    tt = refine(scene, r_)
                    a1 = act(tt)
                    rec[f"a_{prefix}|{r_}"], rec[f"J_{prefix}|{r_}"] = a1, cost(a1)
                    out[r_] = tt
                return out

            def gt_of(box):
                if not len(gtb):
                    return None
                io = box_iou(np.asarray(box, float)[None], gtb)[0]; k = int(np.argmax(io))
                return k if io[k] >= 0.3 else None

            def scene_dict(dd, mono):
                return {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                        "from_full": np.ones(len(dd["conf"]), bool)}
            # ---- draft (and its gated variants for the diagnostic changed reference)
            if cached is not None:
                ch = cached
                ph = associate_prev(ch["xyxy"], prev_draft) if prev_draft is not None else np.full(len(ch["xyxy"]), np.nan)
                re_ttc = ttc_from_scale(ch["xyxy"][:, 3] - ch["xyxy"][:, 1], ph, dt) if len(ch["xyxy"]) else np.zeros(0)
                rec["draft_ttc_parity"] = bool(np.allclose(np.minimum(re_ttc, 1e3), np.minimum(ch["mono"][3], 1e3), equal_nan=True))
            else:
                ch = keep(det.detect(img, CH)[0])
                ph = associate_prev(ch["xyxy"], prev_c["xyxy"]) if prev_c is not None else np.full(len(ch["xyxy"]), np.nan)
                ch["mono"] = geom(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
                rec["draft_ttc_parity"] = True
            prev_draft = ch["xyxy"]
            hc = ch["xyxy"][:, 3] - ch["xyxy"][:, 1]
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            # ---- full-image pipelines (u640 / u960 / native): own history; u960 also with T2_k / T4
            full_img = {}
            for nm, md in M.items():
                d = keep(det.detect(img, md)[0])
                full_img[nm] = d
                mono = geom(d, {"xyxy": st[nm]["merged"]} if nm in st else None)
                score(nm, scene_dict(d, mono), ("none", "frozen", "v3a") if nm == "u960" else ("none", "v3a"))
                if nm == "u960":
                    for pol in CTRL[2:]:
                        p = parse(pol)
                        tt, _ = ttc_ss(d["xyxy"], st.get(nm, {}).get("s1"), st.get(nm, {}).get("s2"), MS[nm], dt, p["gate"], p["t4"])
                        score(f"u960:{pol}", scene_dict(d, mono[:3] + (tt,)))
                for k in KS:
                    tt, _ = ttc_ss(d["xyxy"], st.get(nm, {}).get("s1"), None, MS[nm], dt, k)
                    score(f"g{k:g}:{nm}", scene_dict(d, mono[:3] + (tt,)), ("none", "v3a"))
                rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
            d9 = full_img["u960"]
            mono9 = geom(d9, {"xyxy": st["u960"]["merged"]} if "u960" in st else None)
            a9 = act(refine(scene_dict(d9, mono9), "v3a"))
            # ---- v5 arms (+ support only; nuScenes also v_close 10)
            px_base = None
            arm_out = {}
            for vc, tagp, field_modes in ((0.0, "", (True, False)), (10.0, "vc10:", (True,))):
                if vc == 10.0 and ds != "nusc":
                    continue
                for arm in ARMS:
                    pol = POL[arm]
                    for fld in field_modes:
                        name = f"{tagp}{'' if fld else 'so:'}{arm}"
                        tpx, log, used = [0.0], {}, set()
                        retro = make_retro(log, used)
                        sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, log),
                                                                  lift_pol(name, pol, SCALE, retro), pp, v_closes=(vc,), field=fld)
                        rpx = sum(k_[5] * k_[6] for k_ in used)
                        gkeys = {ckey(gc) for gc in grown}
                        src = np.array(["draft" if not f_ else ("guard" if log.get(tuple(b.tolist())) in gkeys else "crop")
                                        for b, f_ in zip(np.asarray(sc["xyxy"], float).reshape(-1, 4), sc["from_full"])], "<U5")
                        if fld:
                            tts = score(name, sc)
                            rec[f"px_{name}"] = (cheap_px + tpx[0] + rpx) / NATIVE
                            rec[f"pxretro_{name}"], rec[f"nretro_{name}"] = rpx / NATIVE, len(used)
                            rec[f"rounds_{name}"] = rounds
                            if not tagp and arm == "base":
                                px_base = tpx[0]
                            arm_out[name] = (sc, tts, src)
                        else:
                            arm_out[name] = (sc, score(name, sc, ("v3a",)), src)
                        push(name, sc, src)
            # ---- K22(3) field-raise statistics per arm (v3a), and diagnosis (0) on the base arm
            for arm in ARMS:
                sc, tts, src = arm_out[arm]
                so = arm_out[f"so:{arm}"][0]
                tt = tts["v3a"]
                af, aso = rec[f"a_{arm}|v3a"], rec[f"a_so:{arm}|v3a"]
                fr_ = False
                if af > aso:
                    _, dem = P.required_decel(*tt, v, pp)
                    dem = np.asarray(dem, float)
                    if len(dem):
                        j = int(np.argmax(dem))
                        new = not len(so["xyxy"]) or box_iou(sc["xyxy"][j:j + 1], so["xyxy"]).max() < 0.5
                        if sc["from_full"][j] and new:
                            fr_ = True
                            rec[f"fr_est_ttc_{arm}"], rec[f"fr_est_z_{arm}"] = float(tt[3][j]), float(tt[0][j])
                            k = gt_of(sc["xyxy"][j])
                            if k is not None:
                                _, gdem = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                                rec[f"fr_gt_ttc_{arm}"] = float(np.asarray(g["ttc"])[k])
                                rec[f"fr_gt_req_{arm}"] = bool(P.discrete_action(float(np.asarray(gdem)[k]), pp) >= af)
                            else:
                                rec[f"fr_gt_req_{arm}"] = False
                rec[f"fr_{arm}"] = fr_
            sc, tts, src = arm_out["base"]
            tt = tts["v3a"]
            a_b = rec["a_base|v3a"]
            if a_b > P.KEEP:
                _, dem = P.required_decel(*tt, v, pp)
                dem = np.asarray(dem, float)
                j = int(np.argmax(dem))
                keepm = np.ones(len(dem), bool); keepm[j] = False
                lower = act(tuple(a[keepm] for a in tt)) < a_b
                t_inf = [a.copy() for a in tt]; t_inf[3][j] = np.inf
                if sc["from_full"][j] and lower:
                    X = sc["xyxy"][j]
                    # st was already pushed for this frame: the previous frame's merged scene is kept in prev_state
                    pm, psrc = prev_state["base"] if "base" in prev_state else (np.zeros((0, 4)), np.zeros(0, "<U5"))
                    full_x = np.asarray(sc["xyxy"], float)[np.asarray(sc["from_full"], bool)]
                    jj = assoc_idx(full_x, pm)
                    pos = np.flatnonzero(np.asarray(sc["from_full"], bool)).tolist().index(j)
                    pj = jj[pos]
                    psrc_j = str(psrc[pj]) if pj >= 0 else "none"
                    sc_prev = (DS if psrc_j == "draft" else SCALE)
                    rec.update({"dg": True, "dg_src_obj": str(src[j]), "dg_pair": psrc_j, "dg_h": float(X[3] - X[1]) * SCALE,
                                "dg_h_prev": float(pm[pj, 3] - pm[pj, 1]) * sc_prev if pj >= 0 else np.nan,
                                "dg_rel": float((X[3] - X[1]) / (pm[pj, 3] - pm[pj, 1]) - 1) if pj >= 0 else np.nan,
                                "dg_ttc": float(tt[3][j]), "dg_z": float(tt[0][j]), "dg_a": a_b,
                                "dg_ttc_driven": act(tuple(t_inf)) < a_b, "dg_field": rec["fr_base"], "dg_a960": a9})
                    k = gt_of(X)
                    if k is not None:
                        _, gdem = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                        rec.update({"dg_gt_ttc": float(np.asarray(g["ttc"])[k]), "dg_gt_z": float(np.asarray(g["long_near"])[k]),
                                    "dg_gt_req": bool(P.discrete_action(float(np.asarray(gdem)[k]), pp) >= a_b)})
                    if len(d9["xyxy"]):
                        io = box_iou(X[None], d9["xyxy"])[0]; m9 = int(np.argmax(io))
                        if io[m9] >= 0.5:
                            p9 = prev_state.get("u960", np.zeros((0, 4)))
                            q = assoc_idx(d9["xyxy"], p9)[m9] if len(p9) else -1
                            b9 = d9["xyxy"][m9]
                            rec.update({"dg9_has": True, "dg9_h": float(b9[3] - b9[1]) * MS["u960"], "dg9_ttc": float(mono9[3][m9]),
                                        "dg9_h_prev": float(p9[q, 3] - p9[q, 1]) * MS["u960"] if q >= 0 else np.nan,
                                        "dg9_rel": float((b9[3] - b9[1]) / (p9[q, 3] - p9[q, 1]) - 1) if q >= 0 else np.nan})
                        else:
                            rec["dg9_has"] = False
            prev_state = {"base": (st["base"]["merged"], st["base"]["src"]), "u960": d9["xyxy"]}
            # ---- controls: uncertainty crops / ego-corridor tube at v5's per-frame crop tensor px
            bud = px_base

            def canvas(cr):
                k_ = ckey(cr)
                return k_[4] * k_[5]

            def fill_px(cands):
                sel, used_ = [], 0
                for c in cands:
                    cv_ = canvas(c)
                    if used_ + cv_ <= bud:
                        sel.append(c); used_ += cv_
                return sel

            def tube_px():
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
            sels = {"unc": fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in np.lexsort((-ent, ch["conf"]))]) if bud > 0 else [],
                    "cor": tube_px() if bud > 0 else []}
            for hn, sel in sels.items():
                for pol in CTRL:
                    name = f"{hn}:{pol}"
                    tpx = [0.0]
                    sh, _ = IE.merge_once(base, sel, SP.guarded(make_detect(tpx), W, H), lift_pol(name, parse(pol), SCALE))
                    score(name, sh)
                    rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                    push(name, sh)
            # ---- DIAGNOSTIC changed reference: gate on every pipeline incl. the draft
            for k in KS:
                tt_c, _ = ttc_ss(ch["xyxy"], st.get("draft", {}).get("s1"), None, DS, dt, k)
                gmono = ch["mono"][:3] + (tt_c,)
                a_g = act(gmono)
                rec[f"J_g{k:g}:cheap"] = cost(a_g)
                name = f"g{k:g}:ours"
                tpx = [0.0]
                gbase = dict(base); gbase["mono"] = gmono
                sc, _, _, _, _ = KL.run_g(gbase, v, cam, W, H, make_detect(tpx), lift_pol(name, parse(f"T2_{k:g}"), SCALE), pp,
                                          v_closes=(0.0,))
                score(name, sc, ("none", "v3a"))
                rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                push(name, sc)
            st["draft"] = {"s1": ch["xyxy"]}
            for nm in M:
                push(nm, scene_dict(full_img[nm], None))
            imgs = (imgs + [img])[-2:]
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K27 DEV DONE", flush=True)


if __name__ == "__main__":
    main()
