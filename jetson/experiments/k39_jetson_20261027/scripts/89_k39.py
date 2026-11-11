#!/usr/bin/env python
"""K34-3 — B2K-A timing, derived from frozen K29 — fast C1 / v3a on the Jetson (docs/paper/handoff/K29/STATUS.md), decision parity and re-timing.
= 61_k28_v6_latency.py (K28's Jetson reference path, imported unchanged: detectors, loop, lift, C1 head, refiner) with every
pipeline that uses C1 or a refiner run in several PATHS on the same frames:
  ref     K28's path: C1 = TempHead torch FP32 on crop64 (61.lift_c1, verbatim), refiner = 58.RefinerBatch (dynamic FP16)
  <enc>   the fast path (vtf.c1fast) with C1 encoder <enc> in trt16 | trt32 | t32 | t16g; suffix "np" = without pruning
          (e.g. trt16np): crop64_fast, per-pipeline crop / feature bank, bucketed batches, refiner RefFast
Pipeline names: the K28 names for ref (u960+comp, u960+c1comp, ours, ours_noref, ours_b2, ...), "<name>@<path>" otherwise.
Datasets (--dataset):
  nusc        K28's 20 old DEEP scenes (first 25 scenes with images, first --warmup frames not reported)
  nusc_test20 the first 20 nuScenes new-test scene tokens in sorted order (~/datasets/k29); a warm-up pass over the first
              --warmup frames runs first and is discarded, then all 20 scenes are timed
  av2         K28's 10 AV2 val logs (first --warmup frames not reported = the 8 timed logs), ego mask --mask_y0 1830
  nusc_val    nuScenes new val, 152 scenes (dev; ~/datasets/k29)
  av2_dev     AV2 CFG-DEV, 128 train logs (dev; ~/datasets/k29), ego mask
Modes:
  default     timing (every pipeline timed from its own frame upload to its decision; each re-detects)
  --parity    no timing: the whole-frame detections are computed once per side and the crop detections are cached per frame
              (key: crop, canvas, detector), so every path of a pipeline sees the same detections and only C1 / the refiner
              differ; the reference C1 uses crop64_fast (bit-identical; checked on the crops of the first --crop_check frames)
  --breakdown per-component times of ours (draft, field, crops per round, guard, lift, C1, refiner, rest = merge) with
              CUDA syncs at the stage boundaries (a separate repetition)
--heur: the dev-best S heuristics + C1 (YOLOv8s tsp2@S + C1 + v3a; YOLO11m tsp8@S + C1 + own refiner; E construction of
k28_test.py, budget = v5.1-S crop px of the frame), in the --config v51S process.
Output (run folder k29_<mode>_<dataset>_<config>[_tag]): frames.pkl, latency.csv, parity.csv, c1stats.csv, config.json.
usage: 64_k29_latency.py --dataset nusc_test20 --config v6S --paths ref,trt16 [--wf u448,u512] [--bounds 2] [--heur]
                         [--parity] [--breakdown] [--mask_y0 1830] [--max_units n] [--tag t]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0,'/home/kongwoang/research/CVPR/scripts')
sys.path.insert(0,'/home/kongwoang/research/CVPR/experiments/k38_deadline_20261026')
import cv2
from helpers import Retro
from fast import FastRetro
from runtime_fast import MissingRuntime,assoc
from recovery_gpu import TracedCrops,Historical
from bounded_seed import bounded_keep
import bridge
from compact import CompactRuntime
from staged import StagedCrops, StagedOneCall
sys.path.insert(0, "/home/kongwoang/research/CVPR/experiments/k39_jetson_20261027")
import graphs as GR
import deadline as DL
import numpy as np
import pandas as pd
import torch

ROOT = Path('/home/kongwoang/research/CVPR')
_s = importlib.util.spec_from_file_location("k61", ROOT / "scripts" / "61_k28_v6_latency.py")
K61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(K61)
K, KF, KB = K61.K, K61.KF, K61.KB
from nusc_crop import crop64                                                     # noqa: E402
from vtf import c1fast as CF                                                     # noqa: E402
from vtf import gpupost as GP                                                    # noqa: E402
from vtf.gpuprep import letterbox                                                # noqa: E402
from vtf.trtdyn import DynTRT                                                    # noqa: E402

P, SP, THR, Mode, ENG = K.P, K.SP, K.THR, K.Mode, K.ENG
STATS = K61.STATS
CFG, chains = K61.CFG, K61.chains
K29D = Path.home() / "datasets" / "k29"
TUBE = [(near, far, w) for near in (5.0, 10.0) for far in (40.0, 70.0, 100.0) for w in ("fill", "lane")]
HEUR = {"tsp8S+C1": ("yolov8s", 8, "v3a_vuw"), "tsp2S+C1": ("yolov8s", 2, "v3a_vuw"), "tsp8S-yolo11m+C1": ("yolo11m", 8, "own_yolo11m_vuw")}
NWS = {"nusc": 1600, "av2": 2048}


def sync():
    torch.cuda.synchronize()


def frames_k29_nusc(args, which):
    from rap.cache import Camera
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    names = [ln.split()[0] for ln in open(K29D / f"k29_nusc_{which}.txt") if ln.strip()]
    for n_sc, s in enumerate(names):
        if args.max_units and n_sc >= args.max_units:
            break
        cam = Camera("nuScenes", s)
        toks = db.samples(by[s])
        geom, speeds = ad.geometry(s), ad.speeds(s)
        R, t = cam.transform("ego")
        for fr, tk in enumerate(toks):
            g = geom[geom["frame"] == fr]
            p = K29D / "nuscenes" / Path(db.image_path(tk)).relative_to(NUSCENES_TRAINVAL)
            if not p.exists():
                MISSING.append(str(p)); continue
            yield s, fr, str(p), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), g[(g["y2"] - g["y1"]) >= 10], cam, (R, t)


def frames_pkl(args, which):
    """the nuScenes frames written by k29_frames.py (no NuScenesDB in the timed process)"""
    import pickle
    from rap.cache import Camera
    cams = {}
    for s, fr, p, v, g in pickle.load(open(K29D / f"frames_nusc_{which}.pkl", "rb")):
        if s not in cams:
            if args.max_units and len(cams) >= args.max_units:
                break
            cams[s] = Camera("nuScenes", s)
        if not Path(p).exists():
            MISSING.append(p); continue
        yield s, fr, p, v, g, cams[s], cams[s].transform("ego")


def frames_k29_av2(args):
    from vtf import av2 as AV
    root = K29D / "av2" / "sensor" / "train"
    logs = json.loads(Path('/home/kongwoang/research/CVPR/experiments/k38_deadline_20261026/verified_log_manifest.json').read_text())["logs"]
    for n_l, log in enumerate(logs):
        if args.max_units and n_l >= args.max_units:
            break
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        for fr in range(len(L.frame_ts)):
            if not Path(L.image_path(fr)).exists():
                MISSING.append(str(L.image_path(fr))); continue
            g = geom[geom["frame"] == fr] if len(geom) else geom
            yield log, fr, str(L.image_path(fr)), float(speeds[fr]), (g[(g["y2"] - g["y1"]) >= 10] if len(g) else g), L, (R, t)


def frames_t40(args):
    """K38: the first 40 sorted AV2 test (val) logs, all annotated frames in order (as K29's AV2 set)"""
    from vtf import av2 as AV
    root = Path.home() / "datasets/k38/av2/sensor/val"
    logs = [l for l in Path("/home/kongwoang/research/CVPR/experiments/k38_deadline_20261026/av2_test40_logs.txt").read_text().split() if l]
    assert len(logs) == 40
    for log in logs:
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        for fr in range(len(L.frame_ts)):
            path = L.image_path(fr)
            assert Path(path).exists(), ("missing frame", path)
            g = geom[geom["frame"] == fr] if len(geom) else geom
            yield log, fr, str(path), float(speeds[fr]), (g[(g["y2"] - g["y1"]) >= 10] if len(g) else g), L, (R, t)


def frames_k34_av2(args):
    """Replay K29's exact chronological frames, even if shared datasets gained more logs."""
    from vtf import av2 as AV
    manifest = json.loads((ROOT / "docs/paper/handoff/K34/reference/av2_frames.json").read_text())
    root = Path.home() / "datasets/av2/sensor/val"
    current = None
    for row in manifest:
        log, fr = row["unit"], row["frame"]
        if log != current:
            L = AV.AV2Log(root, log)
            geom, speeds = L.geometry(), L.speeds()
            R, t = L.transform("ego")
            current = log
        path = L.image_path(fr)
        assert Path(path).exists(), ("missing frozen frame", path)
        g = geom[geom["frame"] == fr] if len(geom) else geom
        yield log, fr, str(path), float(speeds[fr]), (g[(g["y2"] - g["y1"]) >= 10] if len(g) else g), L, (R, t)


MISSING = []


def with_warm_pass(make, n):
    first = []
    for x in make():
        if len(first) >= n:
            break
        first.append(x)
    for x in first:
        yield ("warmup",) + tuple(x[1:])
    yield from make()


class CachedCrops:
    """parity mode: detect_many with a per-frame cache of crop detections (key: crop pixel box, canvas, detector)"""

    def __init__(self, inner, key):
        self.inner, self.key, self.cache = inner, key, {}

    def new_frame(self):
        self.cache = {}

    def _k(self, cr):
        x1, y1 = int(cr[0]), int(cr[1])
        return (self.key, x1, y1, max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8))

    def __call__(self, crops):
        ks = [self._k(c) for c in crops]
        todo = [i for i, k in enumerate(ks) if k not in self.cache]
        if todo:
            res = self.inner([crops[i] for i in todo])
            for i, r in zip(todo, res):
                self.cache[ks[i]] = r
        return [{k: x.copy() for k, x in self.cache[k_].items()} for k_ in ks]


class SpecMany:
    """detect_many that calls hook(results) after the first call of each round (the guard's calls follow it)"""

    def __init__(self, inner, hook):
        self.inner, self.hook, self.stage = inner, hook, "round"

    def __call__(self, crops):
        r = self.inner(crops)
        if self.stage == "round":
            self.hook(r)
        return r


class CachedOne(CachedCrops):
    """the bounded loop's one-call detector: every crop of a call shares the call's canvas, so the call is the key"""

    def __call__(self, crops):
        k = (self.key,) + tuple(self._k(c) for c in crops)
        if k not in self.cache:
            self.cache[k] = self.inner(crops)
        return [{q: x.copy() for q, x in d.items()} for d in self.cache[k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper",choices=["original","fast"],required=True)
    ap.add_argument("--adapter_encoder",choices=["trt32","t32"],default="trt32")
    ap.add_argument("--recovery_disabled",action="store_true")
    ap.add_argument("--runtime", choices=["reference", "staged"], required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--check_bounded", action="store_true", help="validate unchanged bounded path on shared detections")
    ap.add_argument("--config", default="v6S", choices=list(CFG))
    ap.add_argument("--head", default=str(ENG / "k28_temporal_head.pt"))
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc", "nusc_test20", "nusc_val", "av2_dev", "av2t40"])
    ap.add_argument("--paths", default="trt32")
    ap.add_argument("--wf", default="")
    ap.add_argument("--bounds", default="2")
    ap.add_argument("--heur", action="store_true")
    ap.add_argument("--heur_only", default="", help="comma list of HEUR names (one heuristic per process); default all")
    ap.add_argument("--ours", type=int, default=1, help="0: skip ours (second whole-frame process of a dataset)")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--crop_check", type=int, default=30)
    ap.add_argument("--v_close", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mask_y0", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--deadlines", default="", help="comma list of D in ms")
    ap.add_argument("--medians", default="", help="json of the dev round medians")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--wf_c1only", action="store_true")
    ap.add_argument("--wf_dyn", default="", help="comma list of u<side> on the dynamic engine")
    ap.add_argument("--graphs", default="none", choices=["none", "static", "all"])
    args = ap.parse_args()
    GR.install(args.graphs)
    assert args.config == "v6S" and args.paths in ("trt32", "ref,trt32")
    if args.heur_only:                                             # in place: HEUR is read below as a module global
        for k in [k for k in HEUR if k not in args.heur_only.split(",")]:
            HEUR.pop(k)
        assert HEUR, ("unknown --heur_only", args.heur_only)
    DS = "av2" if args.dataset.startswith("av2") else "nusc"
    if args.dataset in ("nusc", "av2") and not args.max_units:
        args.max_units = 25                                        # K28's frame sets
    if args.parity:
        args.warmup = 0
    args.verifier, side, LIFT, args.refiner = CFG[args.config]
    PATHS = [p for p in args.paths.split(",") if p]
    assert PATHS[0] == "ref" or not args.parity, "parity mode needs the reference path first"
    mode = "parity" if args.parity else ("breakdown" if args.breakdown else "timing")
    run = K.runmeta.new_run(f"k39_{args.graphs}_{args.runtime}_{args.helper}_{mode}_{args.dataset}_{args.config}" + (f"_{args.tag}" if args.tag else ""), vars(args))
    ckh = torch.load(args.head, map_location="cuda", weights_only=False)
    head = K61.TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.cuda().eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    if args.mask_y0:                                               # = 61 (ego mask, K10 / K13)
        _pg = K.predicted_geometry

        def masked_pg(dets, *a, **k):
            g = _pg(dets, *a, **k)
            b = np.asarray(dets["xyxy"], float).reshape(-1, 4)
            if not len(b):
                return g
            m = (b[:, 3] - np.maximum(b[:, 1], args.mask_y0)).clip(0, None) / np.maximum(b[:, 3] - b[:, 1], 1e-6) >= 0.5
            if m.any():
                g = dict(g)
                for key in ("z", "z_ground", "z_height", "ttc"):
                    if key in g:
                        arr = np.asarray(g[key], float).copy(); arr[m] = np.inf; g[key] = arr
            return g
        K.predicted_geometry = masked_pg
    det_s = K.FastDetector(str(Path(K.MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    if DS == "av2":
        WF = {"u448": (448, 352), "u512": (512, 416), "u640": (640, 512), "u960": (960, 768), "u1280": (1280, 992),
              "u1536": (1536, 1184), "native": (2048, 1568)}
        MS = {"u448": 448 / 2048, "u512": 512 / 2048, "u640": 640 / 2048, "u960": 960 / 2048, "u1280": 1280 / 2048,
              "u1536": 1536 / 2048, "native": 1.0}
        CH, NATIVE = Mode("320x256", 320, 256), 1550 * 2048
        if args.dataset == "av2":
            gen = frames_k34_av2(args)
        elif args.dataset == "av2t40":
            gen = frames_t40(args)
        else:
            gen = frames_k29_av2(args)
    else:
        WF = {"u448": (256, 448), "u512": (288, 512), "u640": (384, 640), "u768": (448, 768), "u960": (576, 960),
              "u1280": (736, 1280), "u1536": (864, 1536), "native": (928, 1600)}
        MS = {"u448": 0.28, "u512": 0.32, "u640": 0.4, "u768": 0.48, "u960": 0.6, "u1280": 0.8, "u1536": 0.96, "native": 1.0}
        CH, NATIVE = Mode("192x320", 192, 320), 1600 * 900
        if args.dataset == "nusc":
            gen = frames_pkl(args, "old")                          # = K.frames_nusc(args), max_units 25
        elif args.dataset == "nusc_test20":
            gen = with_warm_pass(lambda: frames_pkl(args, "test20"), args.warmup)
        else:
            gen = frames_pkl(args, "val")
    scale = side / NWS[DS]
    keep_wf = [x for x in args.wf.split(",") if x and x != "none"]
    M = {nm: Mode(f"{WF[nm][0]}x{WF[nm][1]}", *WF[nm]) for nm in keep_wf}
    missing = [m.name for m in M.values() if not (ENG / f"yolov8s_crop_{m.name}.engine").exists()]
    assert not missing, ("missing whole-frame engines", missing)
    G = K.GPUDetect()
    many = (KF.Crops(G, scale) if args.runtime == "reference" else
            StagedCrops(G, scale, ENG / "yolov8s_dyn.engine", Mode, THR, STATS))
    if args.verifier != "yolov8s":
        many.eng = DynTRT(ENG / f"{args.verifier}_dyn.engine")
    one = KB.OneCall(G, scale, many.eng) if args.runtime == "reference" else StagedOneCall(many)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400], [0, 0, 1000, 1000]])
    hmany = {}
    if args.heur:
        pass                                                       # K39: budget = v6-S crop px incl. guard
        hscale = 960 / NWS[DS]
        for hn, (hdet, _, _) in HEUR.items():
            if hdet not in hmany and hdet == "yolov8s" and abs(hscale - scale) < 1e-12:
                hmany[hdet] = many                                 # same engine and scale as v5.1-S's crops
            elif hdet not in hmany:
                c = KF.Crops.__new__(KF.Crops)                    # = KF.Crops(G, hscale) without loading yolov8s_dyn
                c.G, c.s, c.eng = G, hscale, DynTRT(ENG / f"{hdet}_dyn.engine")
                c([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400]])
                hmany[hdet] = c
    if args.parity:
        many_p, one_p = CachedCrops(many, "v"), CachedOne(one, "o")
        hmany_p = {k: (many_p if c is many else CachedCrops(c, f"h{k}")) for k, c in hmany.items()}
    else:
        many_p, one_p, hmany_p = many, one, hmany
    # refiners: reference (58.RefinerBatch, shared by the reference pipelines, as in 61) and fast (one per path)
    tags = {args.refiner, "v3a_vuw"} | ({t for _, (_, _, t) in HEUR.items()} if args.heur else set())
    RB = {tg: KF.RefinerBatch(tg) for tg in tags}

    class _RBGuard:
        """58.RefinerBatch sends every support object in one call without checking set_binding_shape: above the engine
        profile's maximum batch, TensorRT keeps the previous shape and the extra outputs are uninitialised. Fail loudly."""

        def __init__(self, rb):
            self.rb, self.mx = rb, int(rb.eng.get_profile_shape(0, rb.names.index("crop"))[2][0])

        def __call__(self, img, items, *a):
            if len(items) > self.mx:
                raise ValueError(f"reference refiner: {len(items)} support objects > engine maximum batch {self.mx}")
            return self.rb(img, items, *a)
    RB = {tg: _RBGuard(rb) for tg, rb in RB.items()}
    # the reference RefinerBatch reads 58's own (unmasked) predicted_geometry; RefFast uses the same function
    # one RefFast per refiner, shared by the fast paths (identical in every fast path; used one pipeline at a time)
    _rf = {tg: CF.RefFast(tg, ENG, KF.K.predicted_geometry) for tg in tags} if any(p != "ref" for p in PATHS) else {}
    RFF = {(p, tg): _rf[tg] for p in PATHS if p != "ref" for tg in tags}
    def parse(p):
        """<encoder>[np][ov]: np = without pruning, ov = speculative overlap of the C1 encode with the guard calls"""
        ov = p.endswith("ov")
        q = p[:-2] if ov else p
        npr = q.endswith("np")
        return (q[:-2] if npr else q), not npr, ov
    ENC, OV = {}, {}
    for p in PATHS:
        if p != "ref":
            ek, _, OV[p] = parse(p)
            if ek not in ENC:
                ENC[ek] = CF.Encoder(ek, ENG, ckh)
    HEAD = CF.Head(ckh)
    ADAPTER_ENCODER=ENC['trt32'] if args.adapter_encoder=='trt32' else CF.Encoder('t32',ENG,ckh)
    adapters={}
    def adapter(name):
        if name not in adapters:adapters[name]=CompactRuntime(DS,head,CF.crop64_fast,assoc) if '_compact' in name else MissingRuntime(DS,head,ADAPTER_ENCODER)
        return adapters[name]
    print(f"  setup done (engines loaded) {time.strftime('%T')}", flush=True)
    C1F = {p: CF.C1Fast(ENC[parse(p)[0]], HEAD, hmu, hsd, P.PlannerParams(), prune=parse(p)[1]) for p in PATHS if p != "ref"}
    bounds = [int(b) for b in args.bounds.split(",") if b]
    pp, cp = P.PlannerParams(), P.CostParams()
    BD = {}                                                       # breakdown clock of the current pipeline
    if args.breakdown:
        _fc = SP.field_crop

        def fc_timed(*a, **k):
            sync(); t_ = time.perf_counter()
            r = _fc(*a, **k)
            sync(); BD["field"] = BD.get("field", 0.0) + 1e3 * (time.perf_counter() - t_)
            return r
        SP.field_crop = fc_timed

        class TimedMany:
            def __init__(self, f):
                self.f, self.stage = f, "round"

            def __call__(self, crs):
                sync(); t_ = time.perf_counter()
                r = self.f(crs)
                sync(); d_ = 1e3 * (time.perf_counter() - t_)
                if self.stage == "round":
                    BD["_round"] = BD.get("_round", 0) + 1
                    k_ = f"crops_r{BD['_round']}"
                else:
                    k_ = "guard"
                BD[k_] = BD.get(k_, 0.0) + d_
                return r
        many_p = TimedMany(many_p)

    def gdetect(dt_, m):
        xb, r, pad = letterbox(G.x, m.net_h, m.net_w)
        raw = dt_.forward(xb[None], m)
        return GP.post_lean(GP.host_columns(raw, m, THR)[0], m, r, pad, G.shape, min_conf=THR)

    def canvas(cr, W, H, sc):
        """k28_test.ckey canvas (the heuristics' budget unit)"""
        x1, y1 = int(cr[0]), int(cr[1])
        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
        xe, ye = min(x2, W), min(y2, H)
        return max(32, int(math.ceil((ye - y1) * sc / 32)) * 32) * max(32, int(math.ceil((xe - x1) * sc / 32)) * 32)

    rows, hist, banks, cur, prev_act, imgs = [], {}, {}, None, None, []
    EX0, DYN = {}, {}

    def dyn_runner(sc):
        """StagedCrops on the shared dynamic engine at scale sc (whole image as one crop)"""
        r = StagedCrops.__new__(StagedCrops)
        r.G, r.s, r.eng, r.Mode, r.threshold, r.stats = G, sc, many.eng, Mode, THR, STATS
        return r

    def getbank(name):
        if name not in banks:
            banks[name] = CF.Bank()
        return banks[name]
    n_crop_checked, n_crop_bad = 0, 0
    fid = 0
    gpu_history={}
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if unit != cur:
            hist, cur, prev_act, imgs = {}, unit, None, []
            gpu_history.clear()
            for ar in adapters.values():ar.bank={}
            for b_ in banks.values():
                b_.reset()
        fid += 1
        img = cv2.imread(path)
        H, W = img.shape[:2]
        dt = float(cam.dt)
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr, "v": v, "a_gt":float(a_gt),"dt":dt,"act_gt": P.discrete_action(a_gt, pp)}
        rec["image_filename"]=Path(path).name
        rec["annotation_timestamp_ns"]=int(cam.frame_ts[fr]) if hasattr(cam,"frame_ts") else -1
        RETROS={}
        for ar in adapters.values():ar.frame(fid)
        if args.parity:
            for c_ in [many_p, one_p] + list(hmany_p.values()):
                c_.new_frame()
        crop_check = args.parity and len(rows) < args.crop_check

        def lift(xyxy, coarse, pv):
            gk = K.predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def lift_t4(name, sc):
            h = hist.get(name, {})

            def f(full):
                z, lo, hi, _ = lift(full["xyxy"], full["coarse"], None)
                tt, _ = K61.ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                return z, lo, hi, tt
            return f

        def c1_feats(X, full, z, chn, h, zz, sel):
            fe = {}
            for n in sel:
                i, j1, j2 = chn[n]
                b1, b2 = h["s1"][j1], h["s2"][j2]
                x1, y1, x2, y2 = X[i]
                zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                fe[n] = [x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                         float(K.CLS[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                         float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))), dt, 1.0]
            return fe

        def lift_c1(name):
            """61.lift_c1 (k28_test.lift_c, mode C1), verbatim; parity mode cuts the crops with crop64_fast (bit-identical)"""
            h = hist.get(name, {})
            cut = CF.crop64_fast if args.parity else crop64

            def f(full):
                X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                z, lo, hi, _ = lift(X, full["coarse"], None)
                ttc = np.full(len(X), np.inf)
                chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                if not chn:
                    return z, lo, hi, ttc
                if args.breakdown:
                    sync(); t_ = time.perf_counter()
                idx = np.array([c[0] for c in chn])
                zz = K.predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                          cam_to_ego=(R, t))
                fe = c1_feats(X, full, z, chn, h, zz, range(len(chn)))
                cr = []
                for n, (i, j1, j2) in enumerate(chn):
                    trip = [(img, X[i]), (imgs[-1], h["s1"][j1]), (imgs[-2], h["s2"][j2])]
                    cr.append(np.stack([cut(im_, b_) for im_, b_ in trip]))
                    if crop_check:
                        nonlocal_bad[0] += CF.check_crop64(img, [X[i]]) + CF.check_crop64(imgs[-1], [h["s1"][j1]]) + CF.check_crop64(imgs[-2], [h["s2"][j2]])
                        nonlocal_bad[1] += 3
                xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().cuda() / 255.0
                fin = torch.from_numpy(((np.array([fe[n] for n in range(len(chn))], np.float32) - hmu) / hsd).astype(np.float32)).cuda()
                with torch.no_grad():
                    rh = np.clip(head(xin, fin).float().cpu().numpy(), 0, 2)
                STATS["calls"] += 1
                C1REF["chains"] += len(chn)
                for n, (i, _, _) in enumerate(chn):
                    r = rh[n]
                    ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                if args.breakdown:
                    sync(); BD["c1"] = BD.get("c1", 0.0) + 1e3 * (time.perf_counter() - t_)
                return z, lo, hi, ttc
            return f

        def lift_c1_fast(name, p):
            original_h = hist.get(name, {})
            bank = getbank(name)
            c1 = C1F[p]
            def apply_missing(full,h,chn,mono):
                if '_missing' not in name:return mono
                missing=set(range(len(full['xyxy'])))-{i for i,_,_ in chn}
                if not missing:return mono
                ar=adapter(name);ar.fr=fid;ar.shared_bank=bank
                zz=K.predicted_geometry({'xyxy':full['xyxy'],'coarse':full['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                pred,masks=ar.infer('geometry',full,mono,zz,h,chn,hist.get('cheap_history',{}),imgs,img,W,H,v,dt,only_indices=missing)
                out=[np.asarray(q).copy() for q in mono]
                corroborated=set()
                if '_corroborated' in name:
                    links=bridge.match(full['xyxy'],full['coarse'],dd['xyxy'],dd['coarse'])
                    cheap_ttc=np.array([dd['mono'][3][j] if j>=0 else np.inf for j in links])
                    dc=P.required_decel(*mono[:3],cheap_ttc,v,pp)[1]
                    corroborated={i for i,j in enumerate(links) if j>=0 and np.isfinite(dd['mono'][3][j]) and dd['mono'][3][j]>0 and dc[i]>=pp.thr_decel}
                for i in missing:
                    if np.isfinite(pred[i,0]):
                        rate=float(pred[i,4] if i in corroborated and masks[i].sum()>=2 else pred[i,0]);out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf
                return tuple(out)

            def f(full):
                h=dict(original_h)
                X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                z, lo, hi, _ = lift(X, full["coarse"], None)
                ttc = np.full(len(X), np.inf)
                chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                if name in RETROS:
                    h,chn=(RETROS[name].augment(full,(z,lo,hi,ttc),h,known_chains=chn)
                           if args.helper=='fast' else RETROS[name].augment(full,(z,lo,hi,ttc),h))
                if not chn:
                    return apply_missing(full,h,chn,(z,lo,hi,ttc))
                if args.breakdown:
                    sync(); t_ = time.perf_counter()
                kp = c1.keep(chn, z, lo, hi)
                if kp.any():
                    idx = np.array([c[0] for c in chn])
                    zz = K.predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                              cam_to_ego=(R, t))
                    fe = c1_feats(X, full, z, chn, h, zz, np.flatnonzero(kp))
                    rh = c1(bank, fid, img, imgs[-1], imgs[-2], X, chn, h["s1"], h["s2"], fe, kp)
                    for n, r in rh.items():
                        i = chn[n][0]
                        ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                else:
                    c1.stats["chains"] += len(chn); c1.stats["pruned"] += len(chn)
                if args.breakdown:
                    sync(); BD["c1"] = BD.get("c1", 0.0) + 1e3 * (time.perf_counter() - t_)
                return apply_missing(full,h,chn,(z,lo,hi,ttc))
            return f

        def lift_c(name, p):
            return lift_c1(name) if p == "ref" else lift_c1_fast(name, p)

        def spec_hook(name, p):
            """overlap: after a round's first detection call, encode the current-frame crops of the returned boxes that can
            be in a chain (IoU >= 0.3 with a box of the previous frame) and pass the pruning test, on a second stream"""
            def hook(res):
                h = hist.get(name, {})
                if len(imgs) < 2 or h.get("s1") is None or h.get("s2") is None or not len(h["s1"]) or not len(h["s2"]):
                    return
                rs = [r_ for r_ in res if len(r_["conf"])]
                if not rs:
                    return
                bx = np.concatenate([np.asarray(r_["xyxy"], float).reshape(-1, 4) for r_ in rs])
                z, lo, hi, _ = lift(bx, np.concatenate([np.asarray(r_["coarse"]) for r_ in rs]), None)
                ok = P.in_corridor(lo, hi, pp.corridor_half_w) & (z <= pp.max_range)
                if ok.any():
                    ok &= K61.box_iou(bx, h["s1"]).max(1) >= 0.3
                if ok.any():
                    C1F[p].spec(getbank(name), fid, img, bx[ok])
            return hook

        def push(name, xyxy):
            h = hist.setdefault(name, {})
            h["s2"] = h.get("s1")
            h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)

        def act(tt):
            return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

        def J(a):
            return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

        def refiner(p, tag, name):
            return RB[tag] if p == "ref" else RFF[(p, tag)].bind(getbank(name), fid)

        def refine(scene, rb):
            if args.breakdown:
                sync(); t_ = time.perf_counter()
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
            corr = rb(img, [(scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_]) for p_ in sup], cam, R, t, W, H)
            for p_, c in zip(sup, corr):
                for a in refined[:3]:
                    a[p_] *= c
            if args.breakdown:
                sync(); BD["refiner"] = BD.get("refiner", 0.0) + 1e3 * (time.perf_counter() - t_)
            return tuple(refined), len(sup)

        def sfx(p):
            return "" if p == "ref" else f"@{p}"

        def start(name):
            EX0[name] = GR.EXEC[0]
            if name in banks:
                banks[name].new_frame(fid)
            BD.clear()
            sync()
            return time.perf_counter()

        def stop(name, t0):
            sync()
            ms = 1e3 * (time.perf_counter() - t0)
            rec[f"ms_{name}"] = ms
            rec[f"execs_{name}"] = GR.EXEC[0] - EX0.pop(name, GR.EXEC[0])
            if args.breakdown:
                for k_, v_ in BD.items():
                    if not k_.startswith("_"):
                        rec[f"bd_{k_}_{name}"] = v_
            return ms
        nonlocal_bad = [0, 0]
        # ---- cheap draft
        t0 = start("cheap")
        G.frame(img)
        d = keep(gdetect(det_s, CH))
        tc = lift(d["xyxy"], d["coarse"], hist.get("cheap_prev"))
        a_c = act(tc)
        stop("cheap", t0)
        rec.update({"a_cheap": a_c, "J_cheap": J(a_c), "px_cheap": CH.net_h * CH.net_w / NATIVE})
        # ---- whole frame: plain; + T4 + refiner and + C1 + refiner in every path
        for nm, m in M.items():
            if args.wf_c1only and not args.parity:
                pass
            elif args.parity:
                G.frame(img)
                dwp = keep(gdetect(det_s, m))
            else:
                t0 = start(nm)
                G.frame(img)
                dw = keep(gdetect(det_s, m))
                tt = lift(dw["xyxy"], dw["coarse"], hist.get(f"{nm}_prev"))
                a = act(tt)
                stop(nm, t0)
                rec[f"a_{nm}"], rec[f"J_{nm}"], rec[f"px_{nm}"] = a, J(a), m.net_h * m.net_w / NATIVE
                hist[f"{nm}_prev"] = dw
            for p in PATHS:
                for kind in (("c1comp",) if args.wf_c1only else ("comp", "c1comp")):
                    name = f"{nm}+{kind}{sfx(p)}"
                    t0 = start(name)
                    if args.parity:
                        dw2 = dwp
                    else:
                        G.frame(img)
                        dw2 = keep(gdetect(det_s, m))
                    lf = lift_t4(name, MS[nm]) if kind == "comp" else lift_c(name, p)
                    sc_ = {"xyxy": dw2["xyxy"], "conf": dw2["conf"], "coarse": dw2["coarse"], "mono": lf(dw2),
                           "from_full": np.ones(len(dw2["conf"]), bool)}
                    tt2, _ = refine(sc_, refiner(p, "v3a_vuw", name))
                    a2 = act(tt2)
                    stop(name, t0)
                    rec[f"a_{name}"], rec[f"J_{name}"], rec[f"px_{name}"] = a2, J(a2), m.net_h * m.net_w / NATIVE
                    push(name, dw2["xyxy"])
        # ---- K39: whole frame + C1 + v3a on the dynamic engine (u800; u960 control)
        for nm_d in [x for x in args.wf_dyn.split(",") if x]:
            run_d = DYN.setdefault(nm_d, dyn_runner(int(nm_d[1:]) / NWS[DS]))
            it_d = run_d.items([[0, 0, W, H]])[0]
            for p in PATHS:
                name = f"{nm_d}d+c1comp{sfx(p)}"
                t0 = start(name)
                G.frame(img)
                dw2 = keep(run_d([[0, 0, W, H]])[0])
                sc_ = {"xyxy": dw2["xyxy"], "conf": dw2["conf"], "coarse": dw2["coarse"], "mono": lift_c(name, p)(dw2),
                       "from_full": np.ones(len(dw2["conf"]), bool)}
                tt2, _ = refine(sc_, refiner(p, "v3a_vuw", name))
                a2 = act(tt2)
                stop(name, t0)
                rec[f"a_{name}"], rec[f"J_{name}"], rec[f"px_{name}"] = a2, J(a2), it_d[2] * it_d[3] / NATIVE
                push(name, dw2["xyxy"])
        # ---- ours (the configuration), <= 3 rounds and bounded, in every path
        crop51 = None
        DLS = [float(x) for x in args.deadlines.split(",") if x]
        MED = json.loads(Path(args.medians).read_text())["nusc" if DS == "nusc" else "av2"] if args.medians else None

        def hooks(nm0, t0):
            """(stop_fn, mark, log) of pipeline nm0: deadline rows stop by the dev medians; calibrate records times"""
            log = {"top": [], "end": [], "stopped": False}
            def mark(ev, r):
                sync(); log[ev].append(1e3 * (time.perf_counter() - t0))
            if "_dl" in nm0:
                base, D_ = nm0.split("_dl")[0], float(nm0.split("_dl")[1])
                med = MED[base]
                def stop_fn(r):
                    sync()
                    el = 1e3 * (time.perf_counter() - t0)
                    nxt = med["round"][min(r, len(med["round"]) - 1)]
                    if el + nxt + med["finish"] > D_:
                        log["stopped"] = True
                        return True
                    return False
                return stop_fn, None, log
            return None, (mark if args.calibrate else None), log
        for p in (PATHS if args.ours else []):
            variants = [("ours",None),("ours_b2A",2),("ours_b1A",1)] + [(f"{b}_dl{D_:g}", n_) for D_ in DLS for b, n_ in (("ours",None),("ours_b2A",2),("ours_b1A",1))]
            rotation = (args.rep - 1) % len(variants)
            variants = variants[rotation:] + variants[:rotation]
            for nm0, N in variants:
                nm = nm0 + sfx(p)
                STATS["calls"], STATS["tpx"], STATS["exec"], STATS["tpx_b"] = 0, 0, 0, 0
                t0 = start(nm)
                G.frame(img)
                dd = keep(gdetect(det_s, CH)); STATS["calls"] += 1
                dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
                if args.breakdown:
                    sync(); BD["draft"] = 1e3 * (time.perf_counter() - t0)
                assert p=='trt32' and not args.parity
                is_recovery=nm0.endswith('_retro')
                trace={}
                traced_many=TracedCrops(many_p,G,scale,trace)
                traced_one=TracedCrops(one_p,G,scale,trace,common=True)
                eligible_frame=is_recovery and not args.recovery_disabled and (args.helper=='original' or np.any(np.asarray(P.required_decel(*dd['mono'],v,pp)[1])>=pp.thr_decel))
                if eligible_frame:
                    history=gpu_history.get(nm,[])
                    assert len(history)==len(imgs)
                    historical=Historical(imgs,history,many.eng,Mode,THR,STATS)
                    cls=FastRetro if args.helper=='fast' else Retro
                    RETROS[nm]=cls(dd,imgs,trace,historical,chains,v,pp)
                lf = lift_c(nm, p) if LIFT == "C1" else lift_t4(nm, scale)
                mu=traced_many if is_recovery and (args.helper=='original' or eligible_frame) else many_p
                stop_fn, mark, hlog = hooks(nm0, t0)
                if N is None:
                    scene, crops, rounds = DL.run(dd, v, cam, W, H, lambda cr: mu([cr])[0], lf, pp,
                                                  v_close=args.v_close, max_rounds=3, guard=True, detect_many=mu, batch_round=True,
                                                  stop_fn=stop_fn, mark=mark)
                    info = {"n_crops": len(crops), "rounds": rounds}
                    rec[f"crops_{nm}"] = [np.asarray(c, float) for c in crops]
                    if nm0 == "ours" and crop51 is None:
                        crop51 = STATS["tpx"]                      # K39: v6-S crop-detector px incl. guard
                else:
                    if nm0.startswith(("ours_b2A", "ours_b1A")):
                        seed=bridge.extent_seed(hist.setdefault(nm,{}),dd,W,H,lambda box,w,h,ctx:SP.box_crop(box,w,h)) if '_extent' in nm else []
                        scene, calls, hit = DL.bounded_keep(dd,v,cam,W,H,traced_one if is_recovery and (args.helper=='original' or eligible_frame) else one_p,lf,pp,args.v_close,N,seed=seed,stop_fn=stop_fn,mark=mark)
                    else:
                        scene, calls, hit = KB.bounded_run(dd, v, cam, W, H, one_p, lf, pp, args.v_close, N)
                        if args.check_bounded:
                            assert args.parity, "check mode must share detections"
                            check, cc, hh = bounded_keep(dd, v, cam, W, H, one_p, lf, pp, args.v_close, N, keep_pending=False)
                            assert (cc, hh) == (calls, hit)
                            for key in scene:
                                a, b = scene[key], check[key]
                                if key == "mono":
                                    assert all(np.array_equal(x,y) for x,y in zip(a,b)), key
                                else:
                                    assert np.array_equal(a,b), key
                    info = {"calls_loop": calls, "bound_hit": hit}
                ar=adapters.get(nm)
                rec['adapter_encoded_'+nm]=ar.encoded if ar else 0
                rec['adapter_items_'+nm]=ar.head_items if ar else 0
                recovery=RETROS.get(nm)
                assert recovery is None or recovery.calls<=2
                if recovery is not None:
                    assert historical.provenance in ([],[1],[1,0])
                    assert len(historical.provenance)==recovery.calls
                a_noref = act(scene["mono"])
                t_loop = time.perf_counter()
                tt, n_ref = refine(scene, refiner(p, args.refiner, nm))
                a_o = act(tt)
                if "_extent" in nm:bridge.extent_update(hist.setdefault(nm,{}),scene,tt,v,pp)
                ms = stop(nm, t0)
                rec[f"fin_{nm}"] = ms - 1e3 * (t_loop - t0)
                rec[f"dl_stopped_{nm}"] = hlog["stopped"]
                if mark is not None:
                    rec[f"rt_top_{nm}"], rec[f"rt_end_{nm}"] = list(hlog["top"]), list(hlog["end"])
                px = (CH.net_h*CH.net_w+(STATS["tpx"] if N is None else STATS["tpx_b"])+(recovery.pixels if recovery else 0))/NATIVE
                rec.update({f'retro_calls_{nm}':recovery.calls if recovery else 0,
                    f'retro_attempted_{nm}':recovery.attempted if recovery else 0,
                    f'retro_success_{nm}':recovery.success if recovery else 0,
                    f'retro_pixels_{nm}':recovery.pixels if recovery else 0})
                if is_recovery:
                    gpu_history[nm]=(gpu_history.get(nm,[])+[G.x])[-2:]
                rec['gpu_history_bytes']=sum(x.numel()*x.element_size() for hh in gpu_history.values() for x in hh)
                rec.update({f"ms_{nm}_noref": 1e3 * (t_loop - t0), f"a_{nm}_noref": a_noref, f"a_{nm}": a_o,
                            f"J_{nm}_noref": J(a_noref), f"J_{nm}": J(a_o), f"px_{nm}": px, f"px_{nm}_noref": px,
                            f"calls_{nm}": STATS["calls"] + STATS["exec"], f"crop_engine_exec_{nm}": STATS["exec"], f"draft_calls_{nm}": 1, f"n_refined_{nm}": n_ref,
                            **{f"{k}_{nm}": v_ for k, v_ in info.items()}})
                ff = np.asarray(scene["from_full"], bool)
                push(nm, np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff])
        # ---- the dev-best S heuristics + C1 (budget = v5.1-S's crop px of this frame)
        if args.heur:
            P2 = cam.calib.P2
            fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
            hs = 960 / NWS[DS]

            def fill_px(cands, bud):
                sel, used_ = [], 0
                for c in cands:
                    cv_ = canvas(c, W, H, hs)
                    if used_ + cv_ <= bud:
                        sel.append(c); used_ += cv_
                return sel, used_

            def tube(bud, near, far, wmode):
                vb_n = min(cy + fy * cam.cam_h / near, H)
                top = max(cy + fy * cam.cam_h / far - fy * 1.6 / far, 0)
                best = []
                for half_m in np.arange(0.5, 12.01, 0.25):
                    if wmode == "lane" and half_m > 2.5:
                        break
                    half = fx * half_m / near
                    c = np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])
                    if canvas(c, W, H, hs) > bud:
                        break
                    best = [c]
                return best
            rec["budget_heur"] = crop51
            for hn, (hdet, ti, rtag) in HEUR.items():
                near, far, wm = TUBE[ti]
                for p in PATHS:
                    nm = hn + sfx(p)
                    STATS["tpx"] = 0
                    t0 = start(nm)
                    G.frame(img)
                    dd = keep(gdetect(det_s, CH))
                    base = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"],
                            "mono": lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))}
                    sup_d = SP.support(base["mono"], v, pp)
                    bud = crop51
                    sp_, used_ = fill_px([SP.box_crop(base["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)
                    sel = sp_ + (tube(bud - used_, near, far, wm) if bud - used_ > 0 else [])
                    # IE.merge_once with SP.guarded, the crops in shared engine calls (guarded_batch: same per-crop result)
                    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                    grown = []
                    if sel:
                        for d_, ex in SP.guarded_batch(hmany_p[hdet], W, H)(sel):
                            grown += ex
                            full = {k: np.concatenate([full[k], np.asarray(d_[k])]) for k in full}
                    if len(full["conf"]) > 1:
                        o = np.argsort(-full["conf"], kind="stable"); iou = K61.box_iou(full["xyxy"], full["xyxy"]); kk = []
                        for i in o:
                            if not kk or iou[i, kk].max() < 0.5:
                                kk.append(i)
                        full = {k: full[k][np.sort(np.array(kk))] for k in full}
                    out = SP.centres_in(base["xyxy"], sel) if sel else np.zeros(len(base["xyxy"]), bool)
                    if grown and len(full["conf"]) and len(base["xyxy"]):
                        out |= K61.box_iou(np.asarray(base["xyxy"], float), full["xyxy"]).max(1) >= 0.5
                    fz = lift_c(nm, p)(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
                    sh = {"xyxy": np.r_[base["xyxy"][~out], full["xyxy"]], "conf": np.r_[base["conf"][~out], full["conf"]],
                          "coarse": np.r_[base["coarse"][~out], full["coarse"]],
                          "mono": tuple(np.r_[a[~out], b] for a, b in zip(base["mono"], fz)),
                          "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
                    a_noref = act(sh["mono"])
                    tt, _ = refine(sh, refiner(p, rtag, nm))
                    a_h = act(tt)
                    stop(nm, t0)
                    rec.update({f"a_{nm}": a_h, f"J_{nm}": J(a_h), f"a_{nm}_noref": a_noref, f"J_{nm}_noref": J(a_noref),
                                f"px_{nm}": (CH.net_h * CH.net_w + STATS["tpx"]) / NATIVE, f"ncrops_{nm}": len(sel)})
                    push(nm, np.asarray(sh["xyxy"], float).reshape(-1, 4)[np.asarray(sh["from_full"], bool)])
        chist=hist.setdefault('cheap_history',{})
        chist['s2']=chist.get('s1');chist['c2']=chist.get('c1')
        chist['s1']=np.asarray(d['xyxy']).copy();chist['c1']=np.asarray(d['coarse']).copy()
        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        prev_act = a_c
        imgs = (imgs + [img])[-2:]
        rec["warm"] = (unit == "warmup") if args.dataset == "nusc_test20" else (len(rows) < args.warmup)
        n_crop_checked += nonlocal_bad[1]; n_crop_bad += nonlocal_bad[0]
        rows.append(rec)
        if args.parity and len(rows) % 100 == 0:
            torch.cuda.empty_cache()                               # memory only (unified memory on the Jetson)
        if len(rows) % 100 == 0 or (args.heur_only and len(rows) in (1, 5, 20, 50, 100)):   # early markers in the heuristic runs
            import resource
            print(f"  {len(rows)} frames  {time.strftime('%T')}  maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "frames.pkl")
    c1s = [{"path": p, **c.stats} for p, c in C1F.items()] + [{"path": "ref", **C1REF}]
    pd.DataFrame(c1s).to_csv(run / "c1stats.csv", index=False)
    meta = {"frames": len(d), "missing_images": len(MISSING), "crop64_checked": n_crop_checked, "crop64_mismatch": n_crop_bad,
            "engines_md5": md5s(), "paths": PATHS}
    meta["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
    meta["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()
    (run / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v_ for k, v_ in meta.items() if k != "engines_md5"}))
    if not args.max_frames and args.dataset in ("nusc_test20", "av2") and (not args.parity):
        expected = json.loads((ROOT / "docs/paper/handoff/K34/reference" / (args.dataset + "_frames.json")).read_text())
        actual = [{"unit":str(r.unit),"frame":int(r.frame),"warm":bool(r.warm)} for r in d.itertuples()]
        assert actual == expected, "K29 frame manifest mismatch"
    assert not MISSING, "missing input images"
    summarise(d, run, args, PATHS)
    (run / "DONE.json").write_text(json.dumps({"frames":len(d), "timed_frames":int((~d.warm).sum()), "rep":args.rep}))


C1REF = {"chains": 0}


def md5s():
    import hashlib
    out = {}
    for f in sorted(list(ENG.glob("c1enc_*.engine")) + list(ENG.glob("refiner_*_dyn.engine")) + [ENG / "k28_temporal_head.pt"]):
        out[f.name] = hashlib.md5(f.read_bytes()).hexdigest()[:8]
    return out


def summarise(d, run, args, PATHS):
    dd = d[~d.warm].reset_index(drop=True)
    L = dd.J_cheap.to_numpy()
    lat, par = [], []
    names = [c[3:] for c in dd.columns if c.startswith("ms_")]
    for nm in names:
        ms = dd[f"ms_{nm}"]
        jn = f"J_{nm}"
        r = {"pipeline": nm, "frames": len(dd), "G": (L.sum() - dd[jn].sum()) / L.sum() if jn in dd and nm != "cheap" else (0.0 if nm == "cheap" else np.nan),
             "mean_ms": ms.mean(), "median_ms": ms.median(), "p95_ms": ms.quantile(.95), "p99_ms": ms.quantile(.99), "max_ms": ms.max(),
             "tensor_px_mean": dd[f"px_{nm}"].mean() if f"px_{nm}" in dd else np.nan}
        lat.append(r)
    pd.DataFrame(lat).to_csv(run / "latency.csv", index=False)
    for c in dd.columns:
        if not c.startswith("a_") or "@" not in c:
            continue
        base, p = c[2:].split("@")
        p, suf = (p.split("_", 1) + [""])[:2]
        ref = f"a_{base}" + (f"_{suf}" if suf else "")
        if ref not in dd:
            continue
        nm_ref, nm_f = ref[2:], c[2:]
        same = dd[c].to_numpy() == dd[ref].to_numpy()
        par.append({"pipeline": nm_ref, "path": p, "frames": len(dd), "identical_action": same.mean(), "differing_frames": int((~same).sum()),
                    "G_ref": (L.sum() - dd[f"J_{nm_ref}"].sum()) / L.sum(), "G_fast": (L.sum() - dd[f"J_{nm_f}"].sum()) / L.sum(),
                    "differing_units": int(dd.loc[~same, "unit"].nunique())})
    pr = pd.DataFrame(par)
    pr.to_csv(run / "parity.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(pd.DataFrame(lat).round(3).to_string(index=False))
    if len(pr):
        print(pr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
