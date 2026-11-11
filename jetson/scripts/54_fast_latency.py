#!/usr/bin/env python
"""Jetson wall-clock with the engineering fixes of the latency caveat (directions_log "Jetson latency of the
baselines"): GPU letterbox / crop resizing (frame uploaded once per pipeline), all crops of a loop round and the guard's
grown crops of each step in ONE batched TensorRT call (batch engines 1 / 2 / 4 / 8 of the 96x288 crop canvas,
vtf.speculative.run(batch_round=True), identical per-crop results), one device -> host copy per call.

KITTI test split (DEEP), TensorRT FP16, image decode excluded, CUDA synchronised per pipeline, same frames; each
pipeline's braking decision value (mono, cheap action history) is computed in the same run (parity with earlier runs).
Pipelines (each timed from its own frame upload to its decision):
  cheap, u960                 GPU pre-processing; u960_cpu = the earlier CPU letterbox path (reference)
  v3_seq(+ref)                the earlier deployed path: CPU letterbox, one engine call per crop, guard b4 batches
  v3(+ref)                    frozen v3 (v_close 10, guard, scene history), fast path
  v4lite(+ref)                v_close 0 (AV2 / nuScenes-val choice), fast path
  kitti(+ref)                 the KITTI-dev configuration (braking v_close 0 ∪ Planner B field), fast path
Writes results/raw/<run>/fast_latency.pkl and prints mean / median / p95 ms and G per pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("e2e", ROOT / "scripts" / "40_e2e_v2.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
P, PB, SP, runmeta, THR = E.P, E.PB, E.SP, E.runmeta, E.THR
from rap.cache import Camera                                                     # noqa: E402
from rap.decision import KittiAdapter                                            # noqa: E402
from rap.detect import MODES, Mode                                               # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.paths import KITTI_IMAGES, MODELS                                       # noqa: E402
from rap.trt import TRTModule                                                    # noqa: E402
from vtf import spec_general as SG                                               # noqa: E402
from vtf.fastpost import FastDetector                                            # noqa: E402
from vtf.gpuprep import GPUDetect                                                # noqa: E402

FAST = ("cheap", "u960", "v3", "v3_ref", "v4lite", "v4lite_ref", "kitti", "kitti_ref")
REF = ("u960_cpu", "v3_seq", "v3_seq_ref")


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    run = runmeta.new_run("fast_latency", vars(args))
    w = str(Path(MODELS) / "yolov8s.pt")
    base = FastDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop")
    engines = {1: None}
    for b in (2, 4, 8):
        p = runmeta.ROOT / "engines" / f"yolov8s_crop_b{b}_96x288.engine"
        if p.exists():
            engines[b] = TRTModule(p)
    print("crop batch engines:", sorted(engines), flush=True)
    ref = E.TRTRefiner(runmeta.ROOT / "engines" / "image_refiner.engine", runmeta.ROOT / "engines" / "norm.npz")
    m_cheap, m_960, m_canvas = MODES["cheap_320"], Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    # uniform at KITTI's native width (no upsampling): the latency-matched whole-frame baseline
    m_1248 = Mode("384x1248", 384, 1248) if (runmeta.ROOT / "engines" / "yolov8s_crop_384x1248.engine").exists() else None
    pp, cp = P.PlannerParams(), P.CostParams()
    pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
    G = GPUDetect()
    G.warm(crop, m_canvas, engines)
    dv = pd.read_csv(E.DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    seqs = sorted(set(dv[dv.split == args.split].seq))
    rows, n = [], 0
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        hist = {}
        prev_act, prev_plan = None, None
        for fr in sorted(int(f) for f in dv[dv.seq == s].frame):
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            H, W = img.shape[:2]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            rec = {"seq": s, "frame": fr}

            def lift(xyxy, coarse, pv):
                gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def J(tt):
                return P.decision_cost(P.discrete_action(P.required_decel(*tt, v, pp)[0], pp), a_gt, prev_act, pp, cp)["J"]

            def keep(d):
                k = d["conf"] >= THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def refine(scene, sup):
                refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                for p_ in sup:
                    if scene["from_full"][p_]:
                        corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], cam, R, t, W, H)
                        for a in refined[:3]:
                            a[p_] *= corr
                return tuple(refined)

            def cheap_fast():
                G.frame(img)
                d = keep(G.detect(base, m_cheap, THR))
                d["mono"] = lift(d["xyxy"], d["coarse"], hist.get("c"))
                return d

            # ---- cheap / u960 (fast) and u960 (CPU letterbox, earlier path)
            sync(); t0 = time.perf_counter()
            dc = cheap_fast()
            a_c_mono = dc["mono"]
            sync(); rec["ms_cheap"] = 1e3 * (time.perf_counter() - t0)
            rec["J_cheap"] = J(a_c_mono)
            sync(); t0 = time.perf_counter()
            G.frame(img)
            d9 = keep(G.detect(crop, m_960, THR))
            tt9 = lift(d9["xyxy"], d9["coarse"], hist.get("u960"))
            sync(); rec["ms_u960"] = 1e3 * (time.perf_counter() - t0)
            rec["J_u960"] = J(tt9)
            for nm, det_, md in (("u640", base, MODES["full_640"]), ("u1248", crop, m_1248)):
                if md is None:
                    continue
                sync(); t0 = time.perf_counter()
                G.frame(img)
                du = keep(G.detect(det_, md, THR))
                ttu = lift(du["xyxy"], du["coarse"], hist.get(nm))
                sync(); rec[f"ms_{nm}"] = 1e3 * (time.perf_counter() - t0)
                rec[f"J_{nm}"] = J(ttu)
                hist[nm] = du
            sync(); t0 = time.perf_counter()
            d9c = keep(crop.detect(img, m_960)[0])
            tt9c = lift(d9c["xyxy"], d9c["coarse"], hist.get("u960_cpu"))
            sync(); rec["ms_u960_cpu"] = 1e3 * (time.perf_counter() - t0)
            rec["J_u960_cpu"] = J(tt9c)

            # ---- fast speculative pipelines
            def many(crs):
                return G.crops(crop, list(crs), m_canvas, engines, THR)

            def one(cr):
                return many([cr])[0]
            planb = SG.PlannerB(pbp, pbc)
            planb.prev = prev_plan
            for name, vc, kind in (("v3", 10.0, "brake"), ("v4lite", 0.0, "brake"), ("kitti", 0.0, "union")):
                sync(); t0 = time.perf_counter()
                ch = cheap_fast()
                lf = (lambda full, nm=name: lift(full["xyxy"], full["coarse"], hist.get(nm)))
                if kind == "brake":
                    scene, crops, rounds = SP.run(ch, v, cam, W, H, one, lf, pp, v_close=vc, max_rounds=3, guard=True,
                                                  detect_many=many, batch_round=True)
                    sup = SP.support(scene["mono"], v, pp)
                else:
                    cons = _Union(SG.Brake(pp, vc), planb)
                    scene, crops, rounds = SG.run(ch, v, cam, W, H, one, lf, cons, max_rounds=3, guard=True,
                                                  detect_many=many, batch_round=True)
                    sup = cons.support(scene["mono"], v)
                sync(); t1 = time.perf_counter()
                tt_ref = refine(scene, sup)
                sync(); t2 = time.perf_counter()
                rec.update({f"ms_{name}": 1e3 * (t1 - t0), f"ms_{name}_ref": 1e3 * (t2 - t0), f"n_crops_{name}": len(crops),
                            f"rounds_{name}": rounds, f"J_{name}": J(scene["mono"]), f"J_{name}_ref": J(tt_ref)})
                hist[name] = {"xyxy": scene["xyxy"]}

            # ---- the earlier deployed v3 path (CPU letterbox, one call per crop, guard grown crops in b4 batches)
            def run_crop_cpu(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                dk = crop.detect(img[y1:y2, x1:x2], m_canvas)[0]
                kk = dk["conf"] >= THR
                return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]}

            def many_cpu(crs):
                res = []
                for k0 in range(0, len(crs), 4):
                    chunk, xs, meta = crs[k0:k0 + 4], [], []
                    for cr in chunk:
                        x1, y1 = int(cr[0]), int(cr[1])
                        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                        sub = img[y1:y2, x1:x2]
                        x, r, pad, _ = crop.preprocess(sub, m_canvas)
                        xs.append(x); meta.append((r, pad, sub.shape, x1, y1))
                    raw = engines[4](torch.cat(xs + [torch.zeros_like(xs[0])] * (4 - len(xs))))
                    for i_, (r, pad, shp, x1, y1) in enumerate(meta):
                        dk = crop.postprocess(raw[i_:i_ + 1], m_canvas, r, pad, shp)
                        kk = dk["conf"] >= THR
                        res.append({"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                                    "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]})
                return res
            sync(); t0 = time.perf_counter()
            d = keep(base.detect(img, m_cheap)[0])
            d["mono"] = lift(d["xyxy"], d["coarse"], hist.get("c"))
            scene, crops, _ = SP.run(d, v, cam, W, H, run_crop_cpu, lambda full: lift(full["xyxy"], full["coarse"], hist.get("v3_seq")),
                                     pp, v_close=10.0, max_rounds=3, guard=True, detect_many=many_cpu if 4 in engines else None)
            sync(); t1 = time.perf_counter()
            tt_ref = refine(scene, SP.support(scene["mono"], v, pp))
            sync(); t2 = time.perf_counter()
            rec.update({"ms_v3_seq": 1e3 * (t1 - t0), "ms_v3_seq_ref": 1e3 * (t2 - t0), "J_v3_seq": J(scene["mono"]),
                        "J_v3_seq_ref": J(tt_ref)})
            hist["v3_seq"] = {"xyxy": scene["xyxy"]}

            hist["c"] = {"xyxy": dc["xyxy"], "coarse": dc["coarse"]}
            hist["u960"] = d9
            hist["u960_cpu"] = d9c
            prev_act = P.discrete_action(P.required_decel(*a_c_mono, v, pp)[0], pp)
            prev_plan = PB.plan(*a_c_mono, v, prev_plan, pbp, pbc)
            rec["warm"] = n < args.warmup
            rows.append(rec)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        print(f"  {s} done ({n} frames)", flush=True)
        if args.max_frames and n >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "fast_latency.pkl")
    L = d.J_cheap.to_numpy()
    out = []
    for nm in ("cheap", "u640", "u960", "u1248", "u960_cpu", "v3_seq", "v3_seq_ref", "v3", "v3_ref", "v4lite", "v4lite_ref", "kitti", "kitti_ref"):
        if f"ms_{nm}" not in d:
            continue
        ms = d.loc[~d.warm, f"ms_{nm}"]
        out.append({"pipeline": nm, "G": float((L - d[f"J_{nm}"]).sum() / L.sum()) if nm != "cheap" else 0.0,
                    "mean_ms": ms.mean(), "median_ms": ms.median(), "p95_ms": ms.quantile(0.95)})
    res = pd.DataFrame(out)
    res.to_csv(run / "fast_latency_summary.csv", index=False)
    print(f"frames {len(d)}; crops/frame v3 {d.n_crops_v3.mean():.2f}, v4lite {d.n_crops_v4lite.mean():.2f}, kitti {d.n_crops_kitti.mean():.2f}")
    print(res.round(3).to_string(index=False))


class _Union:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def field(self, t, v, rg, yg):
        return self.a.field(t, v, rg, yg) | self.b.field(t, v, rg, yg)

    def support(self, t, v):
        return np.union1d(self.a.support(t, v), self.b.support(t, v)).astype(int)


if __name__ == "__main__":
    main()
