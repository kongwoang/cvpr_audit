#!/usr/bin/env python
"""K17 / K21(1) — variants of ours with switches (post-hoc analysis; frozen configurations are the reference, nothing
tuned; AV2 with the K10 ego mask via k10_masked_run.py).  Per frame: cheap draft, F = u960 (agreement with pi(F)) and
every arm: action and decision cost without / with the dataset's refiner (v1 nuScenes, v2 AV2 / Argoverse-HD; on the
support objects from high-resolution passes, as instr_eval), tensor detector px (canvases incl. guard re-detections,
32-px rounding), crops, rounds, and which decision-relevant annotated objects the final scene contains (recovery in the
K4 controlled deletion: --setting del_all deletes the draft detections matched to them, as instr_eval).
Arms (k_loop.run_g):
  full            the frozen loop (nuScenes v_close 10, AV2 / Argoverse-HD v_close 0; car probe; grid; 3 rounds)
  vc<v>_<probe>   the field with probe <probe> in {ped 0.6x1.8, car 1.8x1.6, truck 2.6x3.6, union} at v_close <v>
  so              support only (no field crop)
  so_ctx<c>       support only with support-crop context c instead of 2 (equal compute by crop size)
  so_rand         support only + random extra crops (instr_eval's random-crop generator), after the loop, each detected
                  only while the frame's actual tensor px so far + its canvas stays <= the full arm's tensor px on that
                  frame (equal compute by extra crops; overshoot bounded by the guard growth of the last crop)
For the full / so pair the frame is classified (K17(1)) by what the field crop did: a = a field-crop detection absent
from support-only's scene matches a decision-relevant annotated object the draft missed (true discovery); b = such a
detection matches no annotated object and has the demand of full's raised action (spurious); c = the deciding object
was already in the draft (box / range changed); d = other.
usage: k17_variants.py --dataset nusc|av2|ahd --split test|val|dev --setting base|del_all --arms a,b,... [--shard i/n]
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
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
_l = importlib.util.spec_from_file_location("kl", B / "scripts" / "A_method" / "k_loop.py")
KL = importlib.util.module_from_spec(_l); _l.loader.exec_module(KL)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k17"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
PROBES = {"ped": (KL.PED,), "car": (KL.CAR,), "truck": (KL.TRUCK,), "union": (KL.PED, KL.CAR, KL.TRUCK)}


def units(ds, split):
    if ds == "ahd":
        yield from HOOD.units("ahd")
    elif ds == "nusc":
        yield from IE.units_nusc(split)
    else:
        yield from IE.units_av2(split)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--setting", default="base", choices=["base", "del_all"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds, arms = args.dataset, args.arms.split(",")
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev,
                    weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        VC, SCALE, CH, F, cheap_px = 10.0, 0.6, None, Mode("f576x960", 576, 960), 192 * 320
    elif ds == "av2":
        VC, SCALE, CH, F, cheap_px = 0.0, 960 / 2048, Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768), 320 * 256
    else:
        VC, SCALE, CH, F, cheap_px = 0.0, 960 / 1920, Mode("ahd_c320", 224, 320), Mode("ahd_u960", 608, 960), 224 * 320

    def arm_kw(name):
        if name == "full":
            return dict(v_closes=(VC,))
        if name.startswith("vc"):
            v_, pr = name[2:].split("_")
            return dict(v_closes=(float(v_),), probes=PROBES[pr])
        if name == "so" or name == "so_rand":
            return dict(field=False)
        if name.startswith("so_ctx"):
            return dict(field=False, ctx=float(name[6:]))
        raise ValueError(name)
    od = OUT / f"{ds}_{args.split}_{args.setting}{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(units(ds, args.split)):
        if ui % sn != si:
            continue
        fo = od / f"{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        rng_del = np.random.default_rng(zlib.crc32(unit.encode()))
        for fr, path, v, g, extra in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, a_gt_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            relevant = np.flatnonzero(np.asarray(a_gt_obj) >= pp.thr_decel) if len(g) else np.zeros(0, int)
            rec = {"unit": unit, "frame": fr, "v": v, "gt": not isinstance(extra, str), "a_gt": float(a_gt),
                   "act_gt": P.discrete_action(a_gt, pp), "rel_cls": gtc[relevant].tolist(),
                   "rel_range": [float(np.asarray(g["long_near"])[i]) for i in relevant] if len(relevant) else []}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt,
                                        cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

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

            def canvas(cr):
                w_, h_ = max(cr[2] - cr[0], 8), max(cr[3] - cr[1], 8)
                return max(32, int(math.ceil(w_ * SCALE / 32)) * 32) * max(32, int(math.ceil(h_ * SCALE / 32)) * 32)

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
            # ---- cheap draft
            if ds == "nusc":
                ch = {k: extra[k] for k in ("xyxy", "conf", "coarse", "mono")}
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            ch_raw = ch
            if args.setting == "del_all" and len(relevant):
                mm = IE.matched(gtb[relevant], gtc[relevant], ch["xyxy"], ch["coarse"])
                dl = [j for j in mm if j >= 0]
                rec["deleted_mask"] = np.array([j >= 0 for j in mm], bool)
                if dl:
                    k_ = np.ones(len(ch["conf"]), bool); k_[dl] = False
                    ch = {kk: (vv[k_] if not isinstance(vv, tuple) else tuple(a[k_] for a in vv)) for kk, vv in ch.items()}
            if ds != "nusc":
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c)})
            if len(relevant):
                rec["found_cheap"] = IE.matched(gtb[relevant], gtc[relevant], ch["xyxy"], ch["coarse"]) >= 0
            d9 = keep(det.detect(img, F)[0])
            a9 = act(lift(d9, hist.get("u960"))); hist["u960"] = d9
            rec.update({"a_u960": a9, "J_u960": cost(a9)})
            scenes = {}
            for an in arms:
                kw = arm_kw(an)
                if an == "so_rand":
                    budget_px = rec.get("tpx_full", rec.get(f"tpx_vc{VC:g}_car"))
                    rr = np.random.default_rng(zlib.crc32(f"{unit}/{fr}".encode()))

                    def extra_fn(crops, grown, rr=rr):
                        out = []
                        for _ in range(40):
                            if len(ch["xyxy"]):
                                b = ch["xyxy"][rr.integers(len(ch["xyxy"]))]
                                w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                            else:
                                w_, h_ = 200.0, 120.0
                            h_ = min(h_, H * 0.65)
                            x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                            out.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                        return out if budget_px else []
                    # a random crop is detected only while the ACTUAL tensor px so far (incl. guard re-detections) plus its
                    # canvas stays within full's tensor px on this frame (overshoot <= the guard growth of the last crop)
                    kw["extra_gate"] = lambda c, b_=budget_px: tpx[0] + canvas(c) <= b_
                    kw["extra"] = extra_fn
                tpx[0] = 0.0
                sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, detect_crop,
                                                          lambda full, an=an: lift(full, hist.get(f"scene_{an}")), pp, **kw)
                a, ar = act(sc["mono"]), act(refine(sc))
                rec.update({f"a_{an}": a, f"J_{an}": cost(a), f"a_{an}_ref": ar, f"J_{an}_ref": cost(ar),
                            f"px_{an}": (cheap_px + tpx[0]) / NATIVE, f"tpx_{an}": tpx[0], f"crops_{an}": len(crops),
                            f"rounds_{an}": rounds, f"conv_{an}": conv})
                if len(relevant):
                    rec[f"found_{an}"] = IE.matched(gtb[relevant], gtc[relevant], sc["xyxy"], sc["coarse"]) >= 0
                hist[f"scene_{an}"] = {"xyxy": sc["xyxy"]}
                scenes[an] = sc
            # ---- K17(1): what did the field crop do (full vs support only)?
            if "full" in scenes and "so" in scenes:
                sf, so = scenes["full"], scenes["so"]
                af, aso = rec["a_full"], rec["a_so"]
                cat = ""
                if af != aso or rec["a_full_ref"] != rec["a_so_ref"]:
                    nf = np.flatnonzero(sf["from_full"])
                    if len(nf) and len(so["xyxy"]):
                        nf = nf[box_iou(sf["xyxy"][nf], so["xyxy"]).max(1) < 0.5]
                    _, dem = P.required_decel(*sf["mono"], v, pp)
                    dem = np.asarray(dem, float)
                    thr = pp.thr_hard if af == P.HARD_BRAKE else pp.thr_decel
                    if len(relevant) and len(nf):
                        in_draft = IE.matched(gtb[relevant], gtc[relevant], ch["xyxy"], ch["coarse"]) >= 0
                        mm = IE.matched(gtb[relevant], gtc[relevant], sf["xyxy"][nf], sf["coarse"][nf])
                        if ((mm >= 0) & ~in_draft).any():
                            cat = "a"
                    if not cat and len(nf):
                        anyg = box_iou(sf["xyxy"][nf], gtb).max(1) if len(gtb) else np.zeros(len(nf))
                        if ((anyg < 0.3) & (dem[nf] >= thr)).any() and af > aso:
                            cat = "b"
                    if not cat:
                        sd, dd = (sf, dem) if af >= aso else (so, np.asarray(P.required_decel(*so["mono"], v, pp)[1], float))
                        if len(dd) and len(ch["xyxy"]):
                            j = int(np.argmax(dd))
                            if box_iou(sd["xyxy"][j:j + 1], ch["xyxy"]).max() >= 0.5:
                                cat = "c"
                    cat = cat or "d"
                rec["field_cat"] = cat
            prev_act = act(ch_raw["mono"])
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K17 VARIANTS DONE", flush=True)


if __name__ == "__main__":
    main()
