#!/usr/bin/env python
"""K28 phase 1, family C — the learned temporal cue on DEVELOPMENT data only (method_freeze_draft §2j). Same units,
draft, loop and scoring as k28_dev.py. Per frame:
  v51S       v5.1-S (parity with k28_dev / K27)
  C1         v5.1-S with the T4 rate replaced by the head (out/k28_temporal/temporal_head.pt): a crop object with a
             same-scale 3-observation chain gets ttc = 1 / r_hat if r_hat > 1e-3 / dt, else inf; no chain: inf
  C2         v5.1-S with r = mean(r_hat, r_T4) (T4 rate 0 where T4 gives no ttc), the T2_1 gate kept
  u960 / u960+T4 / u960+C1 / u960+C2   the whole-frame control with the same cues on its own history (R4)
refiners none / v3a.  results/k28/<ds>_temporal_<tag>/<unit>.pkl
usage: k28_tdev.py --dataset nusc|av2 [--tag r1] [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_th = importlib.util.spec_from_file_location("k28t", B / "scripts" / "A_method" / "k28_ttrain.py")
KTT = importlib.util.module_from_spec(_th); _th.loader.exec_module(KTT)
_td = importlib.util.spec_from_file_location("k28d", B / "scripts" / "A_method" / "k28_tdata.py")
KTD = importlib.util.module_from_spec(_td); _td.loader.exec_module(KTD)
_k = importlib.util.spec_from_file_location("k27", B / "scripts" / "A_method" / "k27_dev.py")
K27 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K27)
IE, KL, KT = K27.IE, K27.KL, K27.KT
ttc_ss, parse = K27.ttc_ss, K27.parse
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
TPOL = "T4+T2_1"
CAPS = {"S": 1.0, "M": 1.5, "L": 2.5}
# family A: name -> (lift arm, v_close mode, value, ctx, crop long side, max_calls)
A = {"v51S": (TPOL, "c", 0.0, 2.0, 960, None),
     "vc5": (TPOL, "c", 5.0, 2.0, 960, None), "vc10": (TPOL, "c", 10.0, 2.0, 960, None),
     "va0.25": (TPOL, "a", 0.25, 2.0, 960, None), "va0.5": (TPOL, "a", 0.5, 2.0, 960, None),
     "va1": (TPOL, "a", 1.0, 2.0, 960, None), "T3": ("T3+T4+T2_1", "c", 0.0, 2.0, 960, None),
     "ctx2.5": (TPOL, "c", 0.0, 2.5, 960, None), "ctx3": (TPOL, "c", 0.0, 3.0, 960, None),
     "s1280": (TPOL, "c", 0.0, 2.0, 1280, None), "N2": (TPOL, "c", 0.0, 2.0, 960, 2), "N3": (TPOL, "c", 0.0, 2.0, 960, 3)}
UNC = [(c, o) for c in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0) for o in ("conf", "ent")]
TCR = [(g, n) for g in (0.0, 0.25, 0.5, 1.0) for n in (0, 3, 1)]                 # n = 0: all previous objects
TUBE = [(near, far, w) for near in (5.0, 10.0) for far in (40.0, 70.0, 100.0) for w in ("fill", "lane")]
WF = {"nusc": (448, 512, 576, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1600),
      "av2": (512, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1664, 1792, 2048)}
VER_WF = {"nusc": (448, 640, 768, 960, 1280, 1600), "av2": (512, 640, 896, 1280, 1536, 2048)}


class _Proxy:
    """vtf.speculative with support() replaced (call-scoped, k_loop's reference only)"""
    def __init__(self, mod, fn):
        self._m, self._fn = mod, fn

    def support(self, *a, **k):
        return self._fn(*a, **k)

    def __getattr__(self, n):
        return getattr(self._m, n)


def wf_mode(ds, L):
    """whole-frame mode at long side L (the K24 modes where they exist; else height / width rounded up to 32)"""
    if ds == "nusc":
        k24 = {640: (384, 640), 960: (576, 960), 1280: (736, 1280), 1536: (864, 1536), 1600: (928, 1600)}
        h, w = k24.get(L, (int(math.ceil(L * 900 / 1600 / 32)) * 32, L))
        return Mode(f"ng{h}x{w}", h, w), L / 1600
    k24 = {640: (640, 512), 960: (960, 768), 1280: (1280, 992), 1536: (1536, 1184), 2048: (2048, 1568)}
    h, w = k24.get(L, (L, int(math.ceil(L * 1550 / 2048 / 32)) * 32))
    return Mode(f"ag{h}x{w}", h, w), L / 2048


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--part", default="temporal")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    args = ap.parse_args()
    ds, part = args.dataset, args.part
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    vdet, vname = det, "yolov8s"
    if part.startswith("ver:"):
        vname = part.split(":", 1)[1]
        vdet = TwoFidelityDetector(str(B / "models" / f"{vname}.pt"), device="cuda:0", backend="torch")
    nets = {}

    def load_net(key, path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        n3 = KT.Res18x3(len(ck["fmu"])); n3.load_state_dict(ck["state"]); nets[key] = (n3.to(dev).eval(), ck["fmu"], ck["fsd"])
    load_net("v3a", REAL / "out" / "resnet18_v3a" / "image_refiner.pt")
    ckh = torch.load(REAL / "out" / "k28_temporal" / "temporal_head.pt", map_location=dev, weights_only=False)
    head = KTT.TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.to(dev).eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    if part.startswith("ver:"):
        load_net("own", REAL / "out" / f"resnet18_v3a_{vname}" / "image_refiner.pt")
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, cheap_px, NW = None, 192 * 320, 1600
    else:
        CH, cheap_px, NW = Mode("av2_c320", 320, 256), 320 * 256, 2048
    od = REAL / "results" / "k28" / f"{ds}_temporal_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(K27.units(ds, args.max_units)):
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
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "v": v, "act_gt": P.discrete_action(a_gt, pp)}

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

            def ckey(cr, scale):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                xe, ye = min(x2, W), min(y2, H)
                nw = max(32, int(math.ceil((xe - x1) * scale / 32)) * 32)
                nh = max(32, int(math.ceil((ye - y1) * scale / 32)) * 32)
                return (x1, y1, x2, y2, nh, nw)
            cache, rcache = {}, {}

            def run_det(dt_, im, key):
                x1, y1, x2, y2, nh, nw = key
                d = keep(dt_.detect(im[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0])
                return {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}

            def make_detect(tpx, scale, log=None, dt_=None, cnt=None):
                dt_ = dt_ or det

                def detect_crop(cr):
                    key = ckey(cr, scale)
                    tpx[0] += key[4] * key[5]
                    if cnt is not None:
                        cnt[0] += 1
                    ck = (id(dt_),) + key
                    if ck not in cache:
                        cache[ck] = run_det(dt_, img, key)
                    out = {k: x.copy() for k, x in cache[ck].items()}
                    if log is not None:
                        for b in out["xyxy"]:
                            log[tuple(b.tolist())] = key
                    return out
                return detect_crop

            def make_retro(log, used, dt_=None):
                dt_ = dt_ or det

                def retro(box, lag):
                    key = log.get(tuple(np.asarray(box, float).tolist()))
                    if key is None or len(imgs) < lag:
                        return None
                    rk = (lag, id(dt_)) + key
                    used.add(rk)
                    if rk not in rcache:
                        rcache[rk] = run_det(dt_, imgs[-lag], key)["xyxy"]
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

            def lift_c(name, mode, sc):
                """C1 / C2 lift of the detections of pipeline `name` (its same-scale history s1 / s2)"""
                h = st.get(name, {})

                def f(full):
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    z, lo, hi, _ = geom(full)
                    t4, _ = ttc_ss(X, h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                    ttc = np.full(len(X), np.inf) if mode == "C1" else t4.copy()
                    chn = KTD.chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                    if not chn:
                        return z, lo, hi, ttc
                    idx = np.array([c[0] for c in chn])
                    zz = predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                            cam_to_ego=(R, t))
                    cr, fe = [], []
                    for n, (i, j1, j2) in enumerate(chn):
                        b1, b2 = h["s1"][j1], h["s2"][j2]
                        x1, y1, x2, y2 = X[i]
                        zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                        hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                        cr.append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                        fe.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                                   float(CLS[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                                   float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))), dt, 1.0])
                    xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().to(dev) / 255.0
                    fin = torch.from_numpy(((np.array(fe, np.float32) - hmu) / hsd).astype(np.float32)).to(dev)
                    with torch.no_grad():
                        rh = np.clip(head(xin, fin).float().cpu().numpy(), 0, 2)
                    for n, (i, j1, j2) in enumerate(chn):
                        if mode == "C1":
                            r = rh[n]
                        else:
                            r4 = 1 / t4[i] if np.isfinite(t4[i]) and t4[i] > 0 else 0.0
                            r = (rh[n] + r4) / 2
                            hh0, hh2 = X[i][3] - X[i][1], h["s2"][j2][3] - h["s2"][j2][1]
                            if not (hh0 - hh2) * sc > 1.0:
                                ttc[i] = np.inf
                                continue
                        ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                    return z, lo, hi, ttc
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

            def loop(nm, arm, vmode, vval, ctx, side, max_calls, dt_=None, refs=("none", "v3a"), sup_fn=None, cmode=None, **kw):
                scale = side / NW
                tpx, log, used, cnt = [0.0], {}, set(), [0]
                vc = vval if vmode == "c" else vval * v
                if sup_fn is not None:
                    KL.SP = _Proxy(SP, sup_fn)
                try:
                    sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, scale, log, dt_, cnt),
                                                              (lift_c(nm, cmode, scale) if cmode else lift_pol(nm, parse(arm), scale, make_retro(log, used, dt_))), pp,
                                                              v_closes=(vc,), ctx=ctx, max_calls=max_calls, **kw)
                finally:
                    KL.SP = SP
                rpx = sum(k_[-2] * k_[-1] for k_ in used)
                score(nm, sc, refs)
                rec[f"px_{nm}"] = (cheap_px + tpx[0] + rpx) / NATIVE
                rec[f"cpx_{nm}"] = (tpx[0] + rpx) / NATIVE
                rec[f"ncrops_{nm}"], rec[f"nguard_{nm}"], rec[f"rounds_{nm}"] = len(crops), len(grown), rounds
                rec[f"ncalls_{nm}"] = cnt[0] + len(used)
                push(nm, sc)
                return tpx[0] + rpx
            # ---- draft
            if cached is not None:
                ch = cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = geom(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            tp = parse(TPOL)
            loop("v51S", TPOL, "c", 0.0, 2.0, 960, None)
            loop("C1", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1")
            loop("C2", TPOL, "c", 0.0, 2.0, 960, None, cmode="C2")
            md, ms = wf_mode(ds, 960)
            d = keep(det.detect(img, md)[0])
            mono = geom(d, {"xyxy": st["u960"]["merged"]} if "u960" in st else None)
            score("u960", scene_dict(d, mono))
            tt, _ = ttc_ss(d["xyxy"], st.get("u960", {}).get("s1"), st.get("u960", {}).get("s2"), ms, dt, tp["gate"], tp["t4"])
            score("u960+T4", scene_dict(d, mono[:3] + (tt,)))
            for cm in ("C1", "C2"):
                st[f"u960+{cm}"] = st.get("u960", {})
                score(f"u960+{cm}", scene_dict(d, lift_c(f"u960+{cm}", cm, ms)(d)))
            rec["px_u960"] = md.net_h * md.net_w / NATIVE
            push("u960", scene_dict(d, None))
            imgs = (imgs + [img])[-2:]
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K28 TDEV DONE", flush=True)


if __name__ == "__main__":
    main()
