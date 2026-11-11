#!/usr/bin/env python
"""K9 follow-up — the TTC-warning consumer: dev-split tuning of its loop and a pixel-matched uncertainty control.

Configurations of ours for the TTC consumer (vtf.spec_general.run, guard, <= 3 rounds, merged-scene history, the
dataset's crop scale), a grid of
  vc  : closing speed of the field's probe object (the probe at (r, y) closes at v + vc): 0, 5, 10, 15 m/s
  m   : support margin — objects whose 1/ttc reaches (1 - m) x the threshold of the current action (they hold it) or
        of the next stronger action (they could raise it) are verified; m = 0 is K9's support exactly
Controls on the same frames: cheap, u960 (TTC action of the whole frame at 960), uncertainty crops (cheap boxes by
ascending confidence / descending entropy, guarded, one pass) filled strictly up to
  (a) ours' crop AREA (as K9; K9's default ours = vc of the braking configuration, m = 0) and
  (b) ours' NETWORK PIXELS, strictly: candidates are run one by one (guard re-detections included) and the first one
      that would exceed ours' detector pixels on that frame is discarded and filling stops.
Splits: --split dev = nuScenes new val / AV2 dev (train logs, instr_eval.units_av2('dev')) — selection happens here;
--split test = nuScenes new test / AV2 val with --configs <one config> (post-hoc analysis, after K9's confirmatory run).
--unc_mults 0.75,1,1.25,1.5,2 : the uncertainty control's G-vs-pixels curve — px-matched filling at those multiples of
ours' (default configuration's) detector pixels per frame (output tag <ds>_<split>_curve)
usage: k9_tune.py --dataset nusc|av2 --split dev|test [--configs vc:m,...] [--unc_mults ...] [--shard i/n] [--max_units N]
       k9_tune.py --analyze --dataset ... --split ...
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
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import spec_general as SG                                               # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k9_tune"
GRID = [(vc, m) for vc in (0.0, 5.0, 10.0, 15.0) for m in (0.0, 0.25, 0.5)]


class TTCWarnM(IE.TTCWarn):
    def __init__(self, pp, v_close, margin):
        super().__init__(pp, v_close)
        self.m = margin

    def support(self, t, v):
        if self.m == 0:
            return super().support(t, v)
        a = IE.c2_action(t, self.pp)
        d = IE.c2_demand(t, self.pp)
        thr = (0.0, 1 / IE.TTC_WARN, 1 / IE.TTC_BRAKE)
        keep = np.zeros(len(d), bool)
        if a > 0:
            keep |= d >= (1 - self.m) * thr[a]
        if a < 2:
            keep |= d >= (1 - self.m) * thr[a + 1]
        return np.flatnonzero(keep)


def canvas_px(c, scale):
    w, h = max(int(math.ceil(c[2])) - int(c[0]), 8), max(int(math.ceil(c[3])) - int(c[1]), 8)
    return max(32, int(math.ceil(w * scale / 32)) * 32) * max(32, int(math.ceil(h * scale / 32)) * 32)


def run(args):
    si, sn = (int(x) for x in args.shard.split("/"))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    pp = P.PlannerParams()
    if args.dataset == "nusc":
        W0, H0, VC0, SCALE = 1600, 900, 10.0, 0.6
        Mc, M9 = None, Mode("f576x960", 576, 960)
        units = IE.units_nusc("val" if args.split == "dev" else "test")
        CHEAP_PX = 192 * 320 / (W0 * H0)
    else:
        W0, H0, VC0, SCALE = 1550, 2048, 0.0, 960 / 2048
        Mc, M9 = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768)
        units = IE.units_av2("dev" if args.split == "dev" else "test")
        CHEAP_PX = 320 * 256 / (W0 * H0)
    NATIVE = W0 * H0
    configs = [tuple(float(x) for x in c.split(":")) for c in args.configs.split(",")] if args.configs else GRID
    if (VC0, 0.0) not in configs:
        configs = [(VC0, 0.0)] + configs
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.split}" + ("_curve" if args.unc_mults else "")
    mults = [float(x) for x in args.unc_mults.split(",")] if args.unc_mults else []
    for ui, (unit, cam, frames) in enumerate(units):
        if ui % sn != si or (args.max_units and ui >= args.max_units):
            continue
        fo = OUT / f"{tag}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_cheap, rows = {}, None, []
        for fr, path, v, g, cheap_cached in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            a2_gt = IE.c2_action(gt_t, pp)
            rec = {"unit": unit, "frame": fr, "v": v, "act2_gt": a2_gt}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def put(name, tt, px):
                a2 = IE.c2_action(tt, pp)
                rec[f"a2_{name}"], rec[f"J2_{name}"], rec[f"px_{name}"] = a2, float(IE.C2_COST[a2_gt, a2]), px
            if cheap_cached is not None:
                cheap = cheap_cached
            else:
                cheap = keep(det.detect(img, Mc)[0]); cheap["mono"] = lift(cheap, prev_cheap)
                prev_cheap = {"xyxy": cheap["xyxy"], "coarse": cheap["coarse"]}
            put("cheap", cheap["mono"], CHEAP_PX)
            d9 = keep(det.detect(img, M9)[0])
            put("u960", lift(d9, hist.get("u960")), M9.net_h * M9.net_w / NATIVE)
            hist["u960"] = d9
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
                return {k: d[k] for k in ("xyxy", "conf", "coarse")}
            base = {k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}
            ref_area = ref_px = None
            for vc, m in configs:
                name = f"ours_vc{vc:g}_m{m:g}"
                tpx[0] = 0.0
                sc, crops, _ = SG.run(base, v, cam, W, H, detect_crop, lambda full, n=name: lift(full, hist.get(n)),
                                      TTCWarnM(pp, vc, m), max_rounds=3, guard=True)
                put(name, sc["mono"], CHEAP_PX + tpx[0] / NATIVE)
                rec[f"crops_{name}"] = len(crops)
                hist[name] = {"xyxy": sc["xyxy"]}
                if (vc, m) == (VC0, 0.0) or (args.split == "test" and len(configs) == 2 and (vc, m) != (VC0, 0.0)):
                    if (vc, m) == (VC0, 0.0):
                        ref_area, ref_px = sum(IE.area(c) for c in crops), tpx[0]
                    else:
                        sel_area, sel_px = sum(IE.area(c) for c in crops), tpx[0]
            gdet = SP.guarded(detect_crop, W, H)
            order = np.lexsort((-np.asarray(cheap.get("entropy", np.zeros(len(cheap["conf"])))), cheap["conf"]))
            cands = [SP.box_crop(cheap["xyxy"][k], W, H) for k in order]
            budgets = [("unc_area", ref_area, "area"), ("unc_px", ref_px, "px")]
            if args.split == "test" and len(configs) == 2:
                budgets += [("unc_area_sel", sel_area, "area"), ("unc_px_sel", sel_px, "px")]
            if mults:
                budgets = [(f"unc_px_x{m:g}", ref_px * m, "px") for m in mults]
            for nm, bud, kind in budgets:
                if kind == "area":
                    sel = IE.fill_strict(cands, bud)
                    tpx[0] = 0.0
                    sc, _ = IE.merge_once(base, sel, gdet, lambda full, n=nm: lift(full, hist.get(n)))
                    used = tpx[0]
                else:
                    sel, res, used = [], {}, 0.0
                    for c in cands:
                        tpx[0] = 0.0
                        r_ = gdet(c)
                        if used + tpx[0] > bud:
                            break
                        sel.append(c); res[len(sel) - 1] = r_; used += tpx[0]
                    it = iter(range(len(sel)))
                    sc, _ = IE.merge_once(base, sel, lambda c, it=it, res=res: res[next(it)],
                                          lambda full, n=nm: lift(full, hist.get(n)))
                put(nm, sc["mono"], CHEAP_PX + used / NATIVE)
                hist[nm] = {"xyxy": sc["xyxy"]}
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  {unit}: {len(rows)} frames", flush=True)
    print("K9 TUNE RUN DONE")


def analyze(args):
    rows = []
    tag = f"{args.dataset}_{args.split}" + ("_curve" if args.unc_mults else "")
    for p in sorted(OUT.glob(f"{tag}__*.pkl")):
        rows += pickle.load(open(p, "rb"))
    d = pd.DataFrame(rows)
    names = [c[3:] for c in d.columns if c.startswith("J2_")]
    Lc = d.J2_cheap.to_numpy()
    uq = d.unit.unique(); idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    for n in names:
        V = Lc - d[f"J2_{n}"].to_numpy()
        r = {"method": n, "G": V.sum() / Lc.sum(), "network_px": d[f"px_{n}"].mean(), "frames": len(d), "units": len(uq)}
        for ref in [r_ for r_ in ("u960", "unc_px") if f"J2_{r_}" in d.columns]:
            Vr = Lc - d[f"J2_{ref}"].to_numpy()
            dr = [(V[b].sum() - Vr[b].sum()) / Lc[b].sum() for b in boots]
            r[f"ci_minus_{ref}"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
    res = pd.DataFrame(out).sort_values("G", ascending=False)
    res.to_csv(OUT / f"summary_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--unc_mults", default="")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)
