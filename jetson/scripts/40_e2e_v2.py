#!/usr/bin/env python
"""v2 (the frozen method: proved loop with v_close = 10 m/s, support + field crops, image refiner).

B2 step 2: end-to-end wall-clock of the deployed pipelines on the Jetson (TensorRT FP16), and the decision value of
the deployed speculative pipeline (braking, KITTI mono).

Per frame (image decode excluded, identical for all pipelines), each pipeline = preprocessing + TensorRT inference +
NMS + monocular lift (+ the speculative extras):
  cheap        YOLOv8s 96x320
  full640      YOLOv8s 192x640 alone
  full960      YOLOv8s 288x960 alone
  cascade640   cheap, then full640 on every frame (DEEP's escalation, uniform)
  speculative  cheap -> closed-form DSF + pivots (vtf.dsf) -> each crop letterboxed into a 96x288 engine at up to the
               960 scale -> lift -> fuse -> decision
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap import planner_b as PB
from rap.cache import Camera
from rap.decision import KittiAdapter, _apply_range_source
from rap.detect import MODES, Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.paths import KITTI_IMAGES, MODELS
from vtf import dsf, runmeta
from vtf import speculative as SP
from vtf.fastpost import FastDetector

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX, THR = 1.8, 1.6, 2.0, 0.25
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def sync():
    torch.cuda.synchronize()


class TRTRefiner:
    """The ResNet-18 range refiner as a TensorRT engine with two inputs (crop 1x3x64x64, features 1x13)."""

    def __init__(self, engine, norm):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine(Path(engine).read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.buf = [torch.empty(tuple(self.eng.get_binding_shape(i)), dtype=torch.float32, device="cuda")
                    for i in range(self.eng.num_bindings)]
        self.names = [self.eng.get_binding_name(i) for i in range(self.eng.num_bindings)]
        z = np.load(norm)
        self.fmu, self.fsd = z["fmu"].astype(np.float32), z["fsd"].astype(np.float32)

    def __call__(self, img, box, conf, coarse, cam, R, t, W, H):
        from nusc_crop import crop64
        g = predicted_geometry({"xyxy": box[None].astype(float), "coarse": np.array([coarse])}, None, cam.calib,
                               cam.cam_h, cam.dt, cam_to_ego=(R, t))
        x1, y1, x2, y2 = box
        z, zg, zh = float(g["z"][0]), float(g["z_ground"][0]), float(g["z_height"][0])
        f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf),
                       float({"vehicle": 0, "person": 1, "cyclist": 2}[str(coarse)]), z, zg, zh,
                       float(np.log(max(zg, .5) / max(zh, .5))), 0.0]], np.float32)
        f = (f - self.fmu) / self.fsd
        self.buf[self.names.index("crop")].copy_(torch.from_numpy(crop64(img, box)).permute(2, 0, 1)[None].float() / 255.0)
        self.buf[self.names.index("feats")].copy_(torch.from_numpy(f.astype(np.float32)))
        self.ctx.execute_v2([int(b.data_ptr()) for b in self.buf])
        return float(np.exp(self.buf[self.names.index("log_ratio")].item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--guard", nargs="?", const="all", default=None, choices=["all", "corridor"],
                    help="truncation guard (locality): every truncated box, or only those that can matter")
    ap.add_argument("--obj_h", type=float, default=SP.OBJ_H, help="probe object height for the field crop (m)")
    ap.add_argument("--hist", default="cheap", choices=["cheap", "crop", "scene"],
                    help="ttc history of crop detections: previous cheap boxes, or the same crops on the previous image")
    ap.add_argument("--only", default=None, choices=["cheap", "full640", "full960", "spec", "spec_ref"],
                    help="run one pipeline alone (dedicated energy pass); decision values are then not comparable")
    ap.add_argument("--cells", action="store_true",
                    help="also score every pipeline on DEEP's other KITTI cells: braking oracle, Planner B mono/oracle")
    ap.add_argument("--energy", action="store_true", help="sample the power rails (50 Hz) and attribute them to pipelines")
    ap.add_argument("--v_close", type=float, default=10.0)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--batch_crops", type=int, default=0,
                    help="run the guard's grown crops of one step in batches of this size (needs <verifier>_crop_b<N>_96x288.engine)")
    ap.add_argument("--verifier", default="yolov8s", help="full model for uniform baselines and crops (cheap draft stays YOLOv8s)")
    args = ap.parse_args()
    run = runmeta.new_run("e2e_v2_KITTI" + ("" if args.verifier == "yolov8s" else f"_{args.verifier}") +
                          ("" if not args.guard else "_guard" if args.guard == "all" else f"_guard{args.guard}") + ("" if args.obj_h == SP.OBJ_H else f"_h{args.obj_h:g}") + ("" if args.hist == "cheap" else f"_hist{args.hist}") +
                          (f"_b{args.batch_crops}" if args.batch_crops else "") +
                          ("" if args.v_close == 10.0 else f"_vc{args.v_close:g}") + ("" if args.max_rounds == 3 else f"_r{args.max_rounds}"), vars(args))
    ref = TRTRefiner(runmeta.ROOT / "engines" / "image_refiner.engine", runmeta.ROOT / "engines" / "norm.npz")
    w = str(Path(MODELS) / "yolov8s.pt")
    base = FastDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem=f"{args.verifier}_crop")
    full = base if args.verifier == "yolov8s" else FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"),
                                                                engine_stem=args.verifier)
    m_cheap, m_full = MODES["cheap_320"], MODES["full_640"]
    batch_eng = None
    if args.batch_crops:
        from rap.trt import TRTModule
        batch_eng = TRTModule(runmeta.ROOT / "engines" / f"{args.verifier}_crop_b{args.batch_crops}_96x288.engine")
    m_960, m_canvas = Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    seqs = sorted(set(dv[dv.split == args.split].seq)) if args.split != "all" else sorted(set(dv.seq))

    def detect(det, img, mode):
        d, _ = det.detect(img, mode)
        sync()
        return d

    def lift(d, prev, cam, R, t, keep_thr=THR):
        k = d["conf"] >= keep_thr
        dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
        return dd, predicted_geometry(dd, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))

    samples, stop_s = [], None
    if args.energy:
        import threading
        from rap import power

        def sampler(stop, out):
            while not stop.is_set():
                pw = power.read_power_mw()
                out.append((time.perf_counter(), pw.get("CPU", 0.0) + pw.get("GPU", 0.0) + pw.get("SOC", 0.0)))
                time.sleep(0.02)
        idle, stop_i = [], threading.Event()
        th = threading.Thread(target=sampler, args=(stop_i, idle), daemon=True); th.start()
        time.sleep(15.0); stop_i.set(); th.join()
        idle_mw = float(np.mean([p_ for _, p_ in idle]))
        stop_s = threading.Event()
        th_s = threading.Thread(target=sampler, args=(stop_s, samples), daemon=True); th_s.start()
    rows, n = [], 0
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        prev = {"cheap": None, "full640": None, "full960": None}
        prev_act, prev_img, prev_scene = None, None, None
        prev_plan, prev_act_o, prev_plan_o = None, None, None
        pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
        frames = sorted(int(f) for f in dv[dv.seq == s].frame)
        for fr in frames:
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            H, W = img.shape[:2]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            rec = {"seq": s, "frame": fr}
            gt_t = tuple(np.asarray(g[k_], float) for k_ in ("long_near", "lat_min", "lat_max", "ttc"))

            def cells(name, xyxy, tt):
                """DEEP's other KITTI cells for one pipeline's final scene (cheap histories, DEEP convention)."""
                if not args.cells:
                    return
                geo = dict(zip(("z", "lat_min", "lat_max", "ttc"), (np.asarray(a, float) for a in tt)))
                to = tuple(np.asarray(a, float) for a in _apply_range_source(np.asarray(xyxy, float).reshape(-1, 4), geo, g,
                                                                             "gt", rng, 0.0))
                rec[f"Jbo_{name}"] = P.decision_cost(P.discrete_action(P.required_decel(*to, v, pp)[0], pp), a_gt,
                                                     prev_act_o, pp, cp)["J"]
                for tag, t_, pv in (("Jtm", tt, prev_plan), ("Jto", to, prev_plan_o)):
                    plan = PB.plan(*t_, v, pv, pbp, pbc)
                    rec[f"{tag}_{name}"] = PB.executed_cost(plan, *gt_t, v, pv, pbp, pbc)["J"]
            # ---- baselines
            for name, det, mode in (("full640", full, m_full), ("full960", crop, m_960)):
                if args.only and args.only != name:
                    rec[f"ms_{name}"], rec[f"J_{name}_trt"], rec[f"seg_{name}"] = np.nan, np.nan, (0.0, 0.0)
                    continue
                t0 = time.perf_counter()
                d = detect(det, img, mode)
                dd, gg = lift(d, prev[name], cam, R, t)
                rec[f"ms_{name}"] = 1e3 * (time.perf_counter() - t0)
                rec[f"seg_{name}"] = (t0, time.perf_counter())
                prev[name] = dd
                tt_ = tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                rec[f"J_{name}_trt"] = P.decision_cost(P.discrete_action(P.required_decel(*tt_, v, pp)[0], pp),
                                                       a_gt, prev_act, pp, cp)["J"]
                cells(name, dd["xyxy"], tt_)
            # ---- speculative v2 (its first stage is the cheap pipeline)
            if args.only in ("full640", "full960"):
                rows.append(rec)
                n += 1
                if args.max_frames and n >= args.max_frames:
                    break
                continue
            t0 = time.perf_counter()
            d = detect(base, img, m_cheap)
            dc, gc = lift(d, prev["cheap"], cam, R, t)
            t_cheap = time.perf_counter()
            k_ = d["conf"] >= THR
            cheap_d = {"xyxy": dc["xyxy"], "conf": d["conf"][k_].astype(float), "coarse": dc["coarse"],
                       "mono": tuple(np.asarray(gc[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))}
            tc = {"crops": 0.0}
            prevc = {"xyxy": np.zeros((0, 4)), "coarse": np.zeros(0, "<U8")}
            if args.only == "cheap":
                rec["ms_cheap"] = 1e3 * (t_cheap - t0)
                prev["cheap"] = dc
                rows.append(rec)
                n += 1
                if args.max_frames and n >= args.max_frames:
                    break
                continue

            def run_crop(im, cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                dk = detect(crop, im[y1:y2, x1:x2], m_canvas)
                kk = dk["conf"] >= THR
                return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]}

            def detect_crop(cr):
                t1 = time.perf_counter()
                out = run_crop(img, cr)
                if args.hist == "crop" and prev_img is not None:
                    pc = run_crop(prev_img, cr)
                    for k_ in prevc:
                        prevc[k_] = np.concatenate([prevc[k_], pc[k_]])
                tc["crops"] += time.perf_counter() - t1
                return out

            def detect_many(crs):
                """The grown crops of one guard step in batched engine calls (same pre/post-processing per crop)."""
                t1 = time.perf_counter()
                res, nb = [], args.batch_crops
                for k0 in range(0, len(crs), nb):
                    chunk, xs, meta = crs[k0:k0 + nb], [], []
                    for cr in chunk:
                        x1, y1 = int(cr[0]), int(cr[1])
                        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                        sub = img[y1:y2, x1:x2]
                        x, r, pad, _ = crop.preprocess(sub, m_canvas)
                        xs.append(x)
                        meta.append((r, pad, sub.shape, x1, y1))
                    xb = torch.cat(xs + [torch.zeros_like(xs[0])] * (nb - len(xs)))
                    raw = batch_eng(xb)
                    for i_, (r, pad, shp, x1, y1) in enumerate(meta):
                        dk = crop.postprocess(raw[i_:i_ + 1], m_canvas, r, pad, shp)
                        kk = dk["conf"] >= THR
                        res.append({"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                                    "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]})
                sync()
                tc["crops"] += time.perf_counter() - t1
                return res

            def lift_full(full):
                hist = {"cheap": prev["cheap"], "scene": prev_scene,
                        "crop": prevc if prev_img is not None else None}[args.hist]
                gk = predicted_geometry({"xyxy": full["xyxy"], "coarse": full["coarse"]}, hist, cam.calib,
                                        cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            scene, crops, rounds = SP.run(cheap_d, v, cam, W, H, detect_crop, lift_full, pp, v_close=args.v_close, max_rounds=args.max_rounds,
                                           guard={None: False, "all": True}.get(args.guard, args.guard), obj_h=args.obj_h,
                                           detect_many=detect_many if batch_eng is not None else None)
            a_spec = P.discrete_action(P.required_decel(*scene["mono"], v, pp)[0], pp)
            t_loop = time.perf_counter()
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            n_ref = 0
            for p_ in (SP.support(scene["mono"], v, pp) if args.only != "spec" else []):
                if not scene["from_full"][p_]:
                    continue
                corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], cam, R, t, W, H)
                for a in refined[:3]:
                    a[p_] *= corr
                n_ref += 1
            a_ref = P.discrete_action(P.required_decel(*refined, v, pp)[0], pp)
            t_end = time.perf_counter()
            a_c = P.discrete_action(P.required_decel(*cheap_d["mono"], v, pp)[0], pp)
            rec.update({"seg_cheap": (t0, t_cheap), "seg_speculative": (t0, t_loop), "seg_speculative_ref": (t0, t_end)})
            rec.update({"ms_cheap": 1e3 * (t_cheap - t0), "ms_crops": 1e3 * tc["crops"],
                        "ms_loop_other": 1e3 * (t_loop - t_cheap - tc["crops"]), "ms_refiner": 1e3 * (t_end - t_loop),
                        "ms_speculative": 1e3 * (t_loop - t0), "ms_speculative_ref": 1e3 * (t_end - t0),
                        "n_crops": len(crops), "rounds": rounds, "n_refined": n_ref,
                        "J_c_trt": P.decision_cost(a_c, a_gt, prev_act, pp, cp)["J"],
                        "J_spec_trt": P.decision_cost(a_spec, a_gt, prev_act, pp, cp)["J"],
                        "J_spec_ref_trt": P.decision_cost(a_ref, a_gt, prev_act, pp, cp)["J"]})
            if args.cells:
                cells("c", dc["xyxy"], cheap_d["mono"])
                cells("spec", scene["xyxy"], scene["mono"])
                cells("spec_ref", scene["xyxy"], tuple(refined))
                geo_c = dict(zip(("z", "lat_min", "lat_max", "ttc"), cheap_d["mono"]))
                to_c = tuple(np.asarray(a, float) for a in _apply_range_source(np.asarray(dc["xyxy"], float).reshape(-1, 4),
                                                                               geo_c, g, "gt", rng, 0.0))
                prev_act_o = P.discrete_action(P.required_decel(*to_c, v, pp)[0], pp)
                prev_plan = PB.plan(*cheap_d["mono"], v, prev_plan, pbp, pbc)
                prev_plan_o = PB.plan(*to_c, v, prev_plan_o, pbp, pbc)
            prev["cheap"], prev_act, prev_img, prev_scene = dc, a_c, img, {"xyxy": scene["xyxy"]}
            rows.append(rec)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        if args.max_frames and n >= args.max_frames:
            break
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    if args.only:                                   # dedicated energy pass: one pipeline, whole-run power
        stop_s.set(); th_s.join()
        col = {"cheap": "ms_cheap", "full640": "ms_full640", "full960": "ms_full960", "spec": "ms_speculative",
               "spec_ref": "ms_speculative_ref"}[args.only]
        t_run = samples[-1][0] - samples[0][0]
        exc = float(np.mean([p_ for _, p_ in samples]) - idle_mw)
        ms = float(d[col].iloc[20:].mean())
        per = exc * t_run / len(d)                        # mW x s / frames = mJ per frame (whole pass)
        out = {"pipeline": args.only, "frames": len(d), "idle_mw": idle_mw, "excess_mw": exc, "mean_ms": ms,
               "mj_per_frame_pass": per, "mj_per_frame_timed": exc * ms / 1e3}
        pd.DataFrame([out]).to_csv(run / "energy_only.csv", index=False)
        print(out)
        return
    d.drop(columns=[c for c in d.columns if c.startswith("seg_")]).to_pickle(run / "e2e.pkl")
    t = d.iloc[20:]                                   # warm-up excluded
    lat = {k: (float(t[k].median()), float(t[k].mean()), float(t[k].quantile(0.95)))
           for k in ("ms_cheap", "ms_full640", "ms_full960", "ms_speculative", "ms_speculative_ref", "ms_crops", "ms_loop_other", "ms_refiner")}
    lat["ms_cascade640 (cheap+full640)"] = (float((t.ms_cheap + t.ms_full640).median()),
                                            float((t.ms_cheap + t.ms_full640).mean()),
                                            float((t.ms_cheap + t.ms_full640).quantile(0.95)))
    res = pd.DataFrame(lat, index=["median", "mean", "p95"]).T
    Lc = float(d.J_c_trt.sum())
    print(res.round(2).to_string())
    print(f"crops per frame {d.n_crops.mean():.2f}; decision value, all on the TensorRT branch (braking, KITTI mono, "
          f"{args.split}): uniform 640 {(d.J_c_trt - d.J_full640_trt).sum() / Lc:.4f}, uniform 960 "
          f"{(d.J_c_trt - d.J_full960_trt).sum() / Lc:.4f}, speculative {(d.J_c_trt - d.J_spec_trt).sum() / Lc:.4f}, "
          f"speculative + refiner {(d.J_c_trt - d.J_spec_ref_trt).sum() / Lc:.4f}; rounds {d.rounds.mean():.2f}, refined/frame {d.n_refined.mean():.2f}")
    res.to_csv(run / "latency.csv")
    if args.cells:
        crow = []
        for cell, pre in (("brake oracle", "Jbo"), ("traj mono", "Jtm"), ("traj oracle", "Jto")):
            Lc_ = d[f"{pre}_c"].sum()
            r = {"cell": cell}
            for nm in ("full640", "full960", "spec", "spec_ref"):
                r[nm] = (d[f"{pre}_c"] - d[f"{pre}_{nm}"]).sum() / Lc_
            crow.append(r)
        cr = pd.DataFrame(crow)
        cr.to_csv(run / "deep_cells.csv", index=False)
        print(cr.round(4).to_string(index=False))
    if args.energy:
        stop_s.set(); th_s.join()
        ts = np.array([a for a, _ in samples]); pw = np.array([b for _, b in samples])
        erows = []
        for name in ("cheap", "full640", "full960", "speculative", "speculative_ref"):
            segs = t[f"seg_{name}"].tolist()
            inside = []
            for a, b in segs:
                i0, i1 = np.searchsorted(ts, a), np.searchsorted(ts, b)
                inside.append(pw[i0:i1])
            inside = np.concatenate(inside)
            ms = float(t[f"ms_{name}"].mean())
            exc = float(inside.mean() - idle_mw) if len(inside) else float("nan")
            erows.append({"pipeline": name, "samples": len(inside), "mean_ms": ms, "excess_mw_cpu_gpu_soc": exc,
                          "mj_per_frame": exc * ms / 1e3})
        er = pd.DataFrame(erows)
        er.to_csv(run / "energy.csv", index=False)
        print(f"idle CPU+GPU+SOC {idle_mw / 1e3:.2f} W")
        print(er.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
