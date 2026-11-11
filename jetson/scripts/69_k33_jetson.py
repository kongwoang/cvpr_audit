#!/usr/bin/env python
"""K33 G on the Jetson (docs/paper/handoff/K33/table1_design.md §2 G; post-hoc): 67_k30_jetson.py (K30 J) with
  --k33 de        the D (AnA-style, "ana*") and E (counterfactual importance, "cf*") rows of vtf.k33sel, one pass ("o") and
                  iterated ("i", <= 3 rounds; crop detections of earlier rounds are reused within the frame), + C1 (fast path)
                  + refiner v3a, YOLOv8s on the crops at 960 / native with the guard (guarded_batch) and K28's merge
  --k33 pub       A / B rows within the dynamic engine's 1024 px limit (vtf.k33pub): AdaScale top 960 (scale net of K24 /
                  av2_sota, the frame on the GPU at the predicted long side), SAHI (12; slices batched, GREEDYNMM), FOVEA
                  KDE_I / KDE_S canvases 640 / 960 (official grid generator on the GPU frame), UFP mosaics 512 / 768 / 1024
                  (official packing on the CPU, the guard as k33_eval.py); every row + C1 (fast path) + refiner v3a
  --k33_rows a,b  only these rows; --k33_cap also "<row>@cap" (crops within v6-S's crop px of the frame; runs ours)
                  named C rows (bar1280_0.5, dvr1280_0.5, ch128B_csc; with --k33 pub): k33_rc.py's routers / Chanakya on the
                  device (estimator, cond_V from its JSON, the MLP), the routing score / action stored for the parity check
Candidate timing happens before the dev selection on VUW. These K33 rows store the action (for the VUW parity check) and
the latency, but no cost J: they are on test frames, and the selection never reads Jetson files.
Everything else is 67 unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k61", ROOT / "scripts" / "61_k28_v6_latency.py")
K61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(K61)
K, KF, KB = K61.K, K61.KF, K61.KB
from nusc_crop import crop64                                                     # noqa: E402
from vtf import c1fast as CF                                                     # noqa: E402
from vtf import gpupost as GP                                                    # noqa: E402
from vtf.gpuprep import letterbox                                                # noqa: E402
from vtf.trtdyn import DynTRT                                                    # noqa: E402
from vtf import k33sel as KS                                                      # noqa: E402
from vtf import k33pub as KP                                                      # noqa: E402

P, SP, THR, Mode, ENG = K.P, K.SP, K.THR, K.Mode, K.ENG
STATS = K61.STATS
CFG, chains = K61.CFG, K61.chains
K29D = Path.home() / "datasets" / "k29"
TUBE = [(near, far, w) for near in (5.0, 10.0) for far in (40.0, 70.0, 100.0) for w in ("fill", "lane")]
HEUR = {"tsp2S+C1": ("yolov8s", 2, "v3a_vuw", "tsp"), "tsp8S-yolo11m+C1": ("yolo11m", 8, "own_yolo11m_vuw", "tsp"),
        "tsp8S+C1": ("yolov8s", 8, "v3a_vuw", "tsp"), "cor0S+C1": ("yolov8s", 0, "v3a_vuw", "cor"),
        "cor8S+C1": ("yolov8s", 8, "v3a_vuw", "cor")}
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}


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
    logs = [ln.strip() for ln in open(K29D / "k29_av2dev_logs.txt") if ln.strip()]
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
    ap.add_argument("--config", default="v6S", choices=list(CFG))
    ap.add_argument("--head", default=str(ENG / "k28_temporal_head.pt"))
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc", "nusc_test20", "nusc_val", "av2_dev", "ahd"])
    ap.add_argument("--only", default="", help="time only these pipelines (base names, e.g. cheap,u960+c1comp,ours,ours_b2,tsp8S+C1)")
    ap.add_argument("--paths", default="ref,trt16")
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
    ap.add_argument("--k33", default="", help="comma list of K33 families: de")
    ap.add_argument("--k33_rows", default="", help="only these K33 rows")
    ap.add_argument("--k33_cap", action="store_true")
    args = ap.parse_args()
    K33 = {x for x in args.k33.split(",") if x}
    if args.heur_only:                                             # in place: HEUR is read below as a module global
        for k in [k for k in HEUR if k not in args.heur_only.split(",")]:
            HEUR.pop(k)
        assert HEUR, ("unknown --heur_only", args.heur_only)
    DS = "av2" if args.dataset.startswith("av2") else ("ahd" if args.dataset == "ahd" else "nusc")
    ONLY = {x for x in args.only.split(",") if x}

    def want(name):
        return not ONLY or name.split("@")[0] in ONLY
    if args.dataset in ("nusc", "av2") and not args.max_units:
        args.max_units = 25                                        # K28's frame sets
    if args.parity:
        args.warmup = 0
    args.verifier, side, LIFT, args.refiner = CFG[args.config]
    PATHS = [p for p in args.paths.split(",") if p]
    assert PATHS[0] == "ref" or not args.parity, "parity mode needs the reference path first"
    mode = "parity" if args.parity else ("breakdown" if args.breakdown else "timing")
    run = K.runmeta.new_run(f"k33j_{mode}_{args.dataset}_{args.config}" + (f"_{args.tag}" if args.tag else ""), vars(args))
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
    if DS == "ahd":
        WF = {"u640": (416, 640), "u960": (608, 960), "u1280": (800, 1280), "native": (1216, 1920)}
        MS = {"u640": 640 / 1920, "u960": 960 / 1920, "u1280": 1280 / 1920, "native": 1.0}
        CH, NATIVE = Mode("224x320", 224, 320), 1920 * 1200
        gen = K61.frames_ahd(args)
    elif DS == "av2":
        WF = {"u448": (448, 352), "u512": (512, 416), "u640": (640, 512), "u960": (960, 768), "u1280": (1280, 992),
              "u1536": (1536, 1184), "native": (2048, 1568)}
        MS = {"u448": 448 / 2048, "u512": 512 / 2048, "u640": 640 / 2048, "u960": 960 / 2048, "u1280": 1280 / 2048,
              "u1536": 1536 / 2048, "native": 1.0}
        CH, NATIVE = Mode("320x256", 320, 256), 1550 * 2048
        if args.dataset == "av2":
            gen = K.frames_av2(args)
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
    many = KF.Crops(G, scale)
    if args.verifier != "yolov8s":
        many.eng = DynTRT(ENG / f"{args.verifier}_dyn.engine")
    one = KB.OneCall(G, scale, many.eng)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400], [0, 0, 1000, 1000]])
    hmany = {}
    if "pub" in K33:
        import torchvision
        ada_net = torchvision.models.resnet18(weights=None); ada_net.fc = torch.nn.Linear(512, 5)

        class _Ada(torch.nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
                self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x):
                return self.m((x - self.mu) / self.sd)
        ADA = _Ada(ada_net)
        ADA.load_state_dict(torch.load(ENG / "k33" / ("adascale_nusc.pt" if DS == "nusc" else "adascale_av2.pt"), map_location="cpu"))
        ADA = ADA.cuda().eval()
        FGG = KP.fovea_gg(ROOT / "third_party" / "fovea")
        FSAL = {(bw, am): torch.load(ENG / "k33" / "fovsal" / f"{'nusc' if DS == 'nusc' else 'av2'}_bw{bw}_amp{am}.pt").cuda()
                for bw in (32, 64) for am in (1, 2)}
        FOVG = {}
        sys.path.insert(0, str(ROOT / "third_party" / "UFPMP-Det-Tools"))
        from ufp import UnifiedForegroundPacking as UFPACK                                 # noqa: F401
        RTR = None
        if any(r_.startswith(("bar", "dvr", "ch")) for r_ in args.k33_rows.split(",")):
            # C rows (k33_rc.py): BAR = bgt-ada's MobileNetV2-lite skip_offbin estimator (its transform: PIL resize to
            # 128 x 128, RGB, ImageNet normalisation; a cv2 resize moved the scores by up to 28 on nuScenes), DVR = cond_V (XGBoost, evaluated from its JSON), Chanakya = the selected MLP; Argoverse-HD
            # takes AV2's estimator, thresholds and policy
            src_ = "nusc" if DS == "nusc" else "av2"
            mb_ = torchvision.models.mobilenet_v2(weights=None)
            BARN = torch.nn.Sequential(mb_.features[:14], torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(96, 1))
            BARN.load_state_dict(torch.load(ENG / "k33" / f"skip_offbin_{src_}.pt", map_location="cpu"))
            BARN = BARN.cuda().eval()
            # bgt-ada's features.py uses PEP 585 annotations (tuple[...]); the Jetson's Python 3.8 needs them postponed
            fpy = ROOT / "third_party" / "bgt-ada" / "src" / "features.py"
            BGF = type(sys)("bgt_features")
            exec(compile("from __future__ import annotations\n" + fpy.read_text(), str(fpy), "exec"), BGF.__dict__)
            ckc = torch.load(ENG / "k33" / "router" / f"chanakya_ch128B_csc_{src_}_t.pt")
            hc_ = ckc["hidden"]
            CHN = torch.nn.Sequential(torch.nn.Linear(len(ckc["mu"]), hc_), torch.nn.ReLU(), torch.nn.Linear(hc_, hc_), torch.nn.ReLU(),
                                      torch.nn.Linear(hc_, 4))
            CHN.load_state_dict(ckc["state"])
            RTR = {"src": src_, "xgb": KP.xgb_json(ENG / "k33" / "router" / f"condV_{src_}.json"), "chn": CHN.eval(),
                   "mu": ckc["mu"].numpy(), "sd": ckc["sd"].numpy(), "fs": ckc["fs"],
                   "thr": json.load(open(ENG / "k33" / "router" / "router_thresholds.json")),
                   "tf": torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
            from PIL import Image as PILImage
    if K33:
        assert args.config == "v6S" and abs(960 / NWS[DS] - scale) < 1e-12
        hmany["yolov8s"] = many                                    # the crops of D / E: v6-S's detector and scale
    if args.heur:
        assert args.config in ("v51S", "v6S"), "budget = the configuration's crop px of the frame (v51S: S, v6S: x1.0)"
        hscale = 960 / NWS[DS]
        for hn, (hdet, _, _, _) in HEUR.items():
            if hdet not in hmany and hdet == "yolov8s" and abs(hscale - scale) < 1e-12:
                hmany[hdet] = many                                 # same engine and scale as v5.1-S's crops
            elif hdet not in hmany:
                c = KF.Crops.__new__(KF.Crops)                    # = KF.Crops(G, hscale) without loading yolov8s_dyn
                c.G, c.s, c.eng = G, hscale, DynTRT(ENG / f"{hdet}_dyn.engine")
                c([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400]])
                hmany[hdet] = c
    if "pub" in K33:
        def _crops_at(sc):
            c = KF.Crops.__new__(KF.Crops)                                       # KF.Crops at another scale, same engine
            c.G, c.s, c.eng = G, sc, many.eng
            return c
        SAHI_MANY = {sz / side: _crops_at(sz / side) for side in (480, 640, 800) for sz in
                     {max(32, int(round(side * sc_ / 32)) * 32) for sc_ in (0.6, 1.0)}}
        CANVAS_MANY = _crops_at(1.0)

        def KF_whole(sd):
            """the whole frame at long side sd (one crop = the frame, scale sd / long side; KF.Crops caps at 1024)"""
            Hh, Ww = G.shape[:2]
            return _crops_at(sd / max(Hh, Ww))([[0, 0, Ww, Hh]])[0]
        UFP_MANY = {S_: (lambda mw, mh, S_=S_: _crops_at(S_ / max(mh, mw))) for S_ in (512, 768, 1024)}
    if args.parity:
        many_p, one_p = CachedCrops(many, "v"), CachedOne(one, "o")
        hmany_p = {k: (many_p if c is many else CachedCrops(c, f"h{k}")) for k, c in hmany.items()}
    else:
        many_p, one_p, hmany_p = many, one, hmany
    # refiners: reference (58.RefinerBatch, shared by the reference pipelines, as in 61) and fast (one per path)
    tags = {args.refiner, "v3a_vuw"} | ({t for _, (_, _, t, _) in HEUR.items()} if args.heur else set())
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

    def getbank(name):
        if name not in banks:
            banks[name] = CF.Bank()
        return banks[name]
    n_crop_checked, n_crop_bad = 0, 0
    fid = 0
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if unit != cur:
            hist, cur, prev_act, imgs = {}, unit, None, []
            for b_ in banks.values():
                b_.reset()
        fid += 1
        img = cv2.imread(path)
        H, W = img.shape[:2]
        dt = float(cam.dt)
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr, "v": v, "act_gt": P.discrete_action(a_gt, pp), "t_wall": time.time()}
        if DS == "ahd":
            rec["image_id"], rec["gt"] = K61.AHD_META[(unit, fr)]
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
            h = hist.get(name, {})
            bank = getbank(name)
            c1 = C1F[p]

            def f(full):
                X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                z, lo, hi, _ = lift(X, full["coarse"], None)
                ttc = np.full(len(X), np.inf)
                chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                if not chn:
                    return z, lo, hi, ttc
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
                return z, lo, hi, ttc
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
            if name in banks:
                banks[name].new_frame(fid)
            BD.clear()
            sync()
            return time.perf_counter()

        def stop(name, t0):
            sync()
            ms = 1e3 * (time.perf_counter() - t0)
            rec[f"ms_{name}"] = ms
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
            if not (want(nm) or want(f"{nm}+comp") or want(f"{nm}+c1comp")):
                continue
            if args.parity:
                G.frame(img)
                dwp = keep(gdetect(det_s, m))
            elif want(nm):
                t0 = start(nm)
                G.frame(img)
                dw = keep(gdetect(det_s, m))
                tt = lift(dw["xyxy"], dw["coarse"], hist.get(f"{nm}_prev"))
                a = act(tt)
                stop(nm, t0)
                rec[f"a_{nm}"], rec[f"J_{nm}"], rec[f"px_{nm}"] = a, J(a), m.net_h * m.net_w / NATIVE
                hist[f"{nm}_prev"] = dw
            for p in PATHS:
                for kind in ("comp", "c1comp"):
                    name = f"{nm}+{kind}{sfx(p)}"
                    if not want(name):
                        continue
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
        # ---- ours (the configuration), <= 3 rounds and bounded, in every path
        crop51 = None
        for p in (PATHS if args.ours else []):
            for nm0, N in [("ours", None)] + [(f"ours_b{b}", b) for b in bounds]:
                nm = nm0 + sfx(p)
                if not (want(nm) or (nm0 == "ours" and (args.heur or args.k33_cap))):   # ours also gives the budget
                    continue
                STATS["calls"], STATS["tpx"], STATS["exec"], STATS["tpx_b"] = 0, 0, 0, 0
                t0 = start(nm)
                G.frame(img)
                dd = keep(gdetect(det_s, CH)); STATS["calls"] += 1
                dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
                if args.breakdown:
                    sync(); BD["draft"] = 1e3 * (time.perf_counter() - t0)
                lf = lift_c(nm, p) if LIFT == "C1" else lift_t4(nm, scale)
                mu = SpecMany(many_p, spec_hook(nm, p)) if (N is None and LIFT == "C1" and OV.get(p)) else many_p
                if N is None:
                    scene, crops, rounds = SP.run(dd, v, cam, W, H, lambda cr: mu([cr])[0], lf, pp,
                                                  v_close=args.v_close, max_rounds=3, guard=True, detect_many=mu, batch_round=True)
                    info = {"n_crops": len(crops), "rounds": rounds}
                    if nm0 == "ours" and crop51 is None:
                        crop51 = sum(canvas(c, W, H, 960 / NWS[DS]) for c in crops)
                else:
                    scene, calls, hit = KB.bounded_run(dd, v, cam, W, H, one_p, lf, pp, args.v_close, N)
                    info = {"calls_loop": calls, "bound_hit": hit}
                a_noref = act(scene["mono"])
                t_loop = time.perf_counter()
                tt, n_ref = refine(scene, refiner(p, args.refiner, nm))
                a_o = act(tt)
                ms = stop(nm, t0)
                px = (CH.net_h * CH.net_w + (STATS["tpx"] if N is None else STATS["tpx_b"])) / NATIVE
                rec.update({f"ms_{nm}_noref": 1e3 * (t_loop - t0), f"a_{nm}_noref": a_noref, f"a_{nm}": a_o,
                            f"J_{nm}_noref": J(a_noref), f"J_{nm}": J(a_o), f"px_{nm}": px, f"px_{nm}_noref": px,
                            f"calls_{nm}": STATS["calls"] + STATS["exec"], f"n_refined_{nm}": n_ref,
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
            for hn, (hdet, ti, rtag, fam) in HEUR.items():
                near, far, wm = TUBE[ti]
                for p in PATHS:
                    nm = hn + sfx(p)
                    if not want(nm):
                        continue
                    STATS["tpx"] = 0
                    t0 = start(nm)
                    G.frame(img)
                    dd = keep(gdetect(det_s, CH))
                    base = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"],
                            "mono": lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))}
                    sup_d = SP.support(base["mono"], v, pp)
                    bud = crop51
                    sp_, used_ = (fill_px([SP.box_crop(base["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)) \
                        if fam == "tsp" else ([], 0)                                  # cor: the tube only (k28_test)
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
        # ---- K33 D / E rows (vtf.k33sel; the same selection as k33_eval.py part de)
        if "de" in K33:
            hs = 960 / NWS[DS]
            lo_keep = None

            def merge_rounds(base, crops, res, nm, p):
                """K28 merge of all crop results so far (NMS 0.5 over crop detections, draft boxes inside crops or
                overlapping grown ones replaced), C1 lift of the crop detections"""
                full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                grown = []
                for d_, ex in res:
                    grown += ex
                    full = {k: np.concatenate([full[k], np.asarray(d_[k])]) for k in full}
                if len(full["conf"]) > 1:
                    o = np.argsort(-full["conf"], kind="stable"); iou = K61.box_iou(full["xyxy"], full["xyxy"]); kk = []
                    for i in o:
                        if not kk or iou[i, kk].max() < 0.5:
                            kk.append(i)
                    full = {k: full[k][np.sort(np.array(kk))] for k in full}
                out = SP.centres_in(base["xyxy"], crops) if crops else np.zeros(len(base["xyxy"]), bool)
                if grown and len(full["conf"]) and len(base["xyxy"]):
                    out |= K61.box_iou(np.asarray(base["xyxy"], float), full["xyxy"]).max(1) >= 0.5
                fz = lift_c(nm, p)(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
                return {"xyxy": np.r_[base["xyxy"][~out], full["xyxy"]], "conf": np.r_[base["conf"][~out], full["conf"]],
                        "coarse": np.r_[base["coarse"][~out], full["coarse"]],
                        "mono": tuple(np.r_[a[~out], b] for a, b in zip(base["mono"], fz)),
                        "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}, grown
            rows_k33 = [(f"ana{int(round(t_ * 100)):02d}h{'inf' if not np.isfinite(th_) else int(th_)}{o_}", "ana", (t_, th_), it_)
                        for t_, th_ in KS.ana_cands() for o_, it_ in (("o", 1), ("i", 3))]
            rows_k33 += [(f"cf{int(round(p_ * 100)):02d}q{int(round(q_ * 10)):02d}{o_}", "cf", (p_, q_), it_)
                         for p_, q_ in KS.cf_cands() for o_, it_ in (("o", 1), ("i", 3))]
            sel_rows = {x for x in args.k33_rows.split(",") if x}
            for nm0, fam, par, iters in rows_k33:
                if sel_rows and nm0 not in sel_rows:
                    continue
                for capped in ((False, True) if args.k33_cap else (False,)):
                    for p in PATHS:
                        nm = nm0 + ("@cap" if capped else "") + sfx(p)
                        STATS["tpx"] = 0
                        t0 = start(nm)
                        G.frame(img)
                        xb, r_, pad = letterbox(G.x, CH.net_h, CH.net_w)
                        raw_ = det_s.forward(xb[None], CH)
                        lo_d = GP.post_lean(GP.host_columns(raw_, CH, 0.05)[0], CH, r_, pad, G.shape, min_conf=0.05)
                        lo_d = {"xyxy": np.asarray(lo_d["xyxy"], float).reshape(-1, 4), "conf": np.asarray(lo_d["conf"], float),
                                "coarse": np.asarray(lo_d["coarse"])}
                        dd = keep(lo_d)
                        base = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"],
                                "mono": lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev")),
                                "from_full": np.zeros(len(dd["conf"]), bool)}
                        sup0 = SP.support(base["mono"], v, pp)
                        if fam == "ana":
                            lo_mono = lift(lo_d["xyxy"], lo_d["coarse"], hist.get("lo_prev"))
                            lo_keep = lo_d
                        crops, res, scene, grown, rd = [], [], base, [], 0
                        cap = crop51 if capped else None
                        for rd in range(iters):
                            if fam == "ana":
                                if rd == 0:
                                    xt, idx, xs, sup = lo_d["xyxy"], KS.ana_targets(lo_d["conf"], lo_mono, v, pp, *par), base["xyxy"], sup0
                                else:
                                    xt, idx, xs, sup = (scene["xyxy"], KS.ana_targets(scene["conf"], scene["mono"], v, pp, *par),
                                                        scene["xyxy"], SP.support(scene["mono"], v, pp))
                            else:
                                xt, idx, xs, sup = (scene["xyxy"], KS.cf_targets(scene["mono"], v, pp, *par), scene["xyxy"],
                                                    SP.support(scene["mono"], v, pp))
                            new = KS.crops_for(xt, idx, [], W, H, crops)
                            new += KS.crops_for(xs, [], sup, W, H, crops + new)
                            if cap is not None:
                                new = KS.fill_cap(new, lambda c: canvas(c, W, H, hs), cap - sum(canvas(c, W, H, hs) for c in crops))
                            if not new:
                                break
                            crops = crops + new
                            res = res + list(SP.guarded_batch(hmany_p["yolov8s"], W, H)(new))
                            scene, grown = merge_rounds(base, crops, res, nm, p)
                        else:
                            rd = iters
                        tt, _ = refine(scene, refiner(p, "v3a_vuw", nm))
                        a_k = act(tt)
                        stop(nm, t0)
                        rec.update({f"a_{nm}": a_k, f"px_{nm}": (CH.net_h * CH.net_w + sum(canvas(c, W, H, hs) for c in crops + grown)) / NATIVE,
                                    f"ncrops_{nm}": len(crops), f"rounds_{nm}": rd})
                        push(nm, np.asarray(scene["xyxy"], float).reshape(-1, 4)[np.asarray(scene["from_full"], bool)])
            if lo_keep is not None:
                hist["lo_prev"] = {"xyxy": lo_keep["xyxy"], "coarse": lo_keep["coarse"]}
        # ---- K33 A / B rows (vtf.k33pub; the same candidates as k33_eval.py part pub, within 1024 px)
        if "pub" in K33:
            sel_rows = {x for x in args.k33_rows.split(",") if x}
            hs = 960 / NWS[DS]

            def want33(nm0):
                return not sel_rows or nm0 in sel_rows

            def draft():
                dd = keep(gdetect(det_s, CH))
                dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
                return dd

            def finish(nm, dd, px, p, t0):
                dd = {"xyxy": np.asarray(dd["xyxy"], float).reshape(-1, 4), "conf": np.asarray(dd["conf"], float),
                      "coarse": np.asarray(dd["coarse"])}
                mono = lift_c(nm, p)(dd) if len(dd["conf"]) else tuple(np.zeros(0) for _ in range(4))
                sc_ = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                       "from_full": np.ones(len(dd["conf"]), bool)}
                tt, _ = refine(sc_, refiner(p, "v3a_vuw", nm))
                a_k = act(tt)
                stop(nm, t0)
                rec.update({f"a_{nm}": a_k, f"px_{nm}": px / NATIVE})
                push(nm, dd["xyxy"])
            land = W >= H

            def canv(sd, ada=False):
                """k33_eval.py part pub's modes: AdaScale's official ones up to 960, else K28's wf_mode"""
                if ada and DS != "ahd" and sd <= 960:
                    return (int(round(sd * 9 / 16 / 32)) * 32, sd) if DS == "nusc" else (sd, int(round(0.8 * sd / 32)) * 32)
                if DS == "ahd":
                    return (int(math.ceil(sd * 1200 / 1920 / 32)) * 32, sd)
                if DS == "nusc":
                    return {640: (384, 640), 960: (576, 960), 1280: (736, 1280)}.get(sd, (int(math.ceil(sd * 900 / 1600 / 32)) * 32, sd))
                return {640: (640, 512), 960: (960, 768), 1280: (1280, 992)}.get(sd, (sd, int(math.ceil(sd * 1550 / 2048 / 32)) * 32))
            for p in PATHS:
                # AdaScale (top of the ladder 960)
                SC0 = (320, 480, 640, 800, 960)
                for top, k_ in [(960, k) for k in (-1, 0, 1, 2)] + [(1280, k) for k in (-1, 0, 1, 2)]:
                    nm0 = f"ada{k_:+d}t{top}"
                    if not want33(nm0) or (top == 1280 and not sel_rows):     # top 1280 only when asked for by name
                        continue
                    LAD = [320, 480, 640, 800, 960, 1120, 1280]
                    nm = nm0 + sfx(p)
                    t0 = start(nm)
                    G.frame(img)
                    dd = draft()
                    with torch.no_grad():
                        xi = torch.from_numpy(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255.0
                        i0 = int(ADA(xi).argmax(1).item())
                    sd = LAD[int(np.clip(i0 + k_, 0, LAD.index(top)))]
                    if sd == 320:
                        finish(nm, dd, CH.net_h * CH.net_w, p, t0)
                    else:
                        h_, w_ = canv(sd, ada=True)
                        # above the dynamic engine's 1024 px: the static engine of exactly that shape (K28 modes)
                        dw = keep(gdetect(det_s, Mode(f"{h_}x{w_}", h_, w_))) if max(h_, w_) > 1024 else KF_whole(sd)
                        finish(nm, dw, CH.net_h * CH.net_w + h_ * w_, p, t0)
                # SAHI
                for side in (480, 640, 800):
                    for ov in (0.1, 0.2):
                        for scl in (0.6, 1.0):
                            nm0 = f"sahi{side}o{int(ov * 10)}s{int(scl * 10):02d}"
                            if not want33(nm0):
                                continue
                            nm = nm0 + sfx(p)
                            sz = max(32, int(round(side * scl / 32)) * 32)
                            t0 = start(nm)
                            G.frame(img)
                            sl = KP.sahi_slices(H, W, side, ov)
                            res_ = SAHI_MANY[sz / side](sl)
                            full_ = KF_whole(sz)
                            X = np.concatenate([r_["xyxy"] for r_ in res_] + [full_["xyxy"]])
                            C_ = np.concatenate([r_["conf"] for r_ in res_] + [full_["conf"]])
                            K_ = np.concatenate([r_["coarse"] for r_ in res_] + [full_["coarse"]])
                            mx, mc, mk = KP.greedy_nmm(X, C_, K_)
                            short = int(math.ceil(sz * min(W, H) / max(W, H) / 32)) * 32
                            finish(nm, {"xyxy": mx, "conf": mc, "coarse": mk}, len(sl) * sz * sz + sz * short, p, t0)
                # FOVEA KDE_I / KDE_S, canvases 640 / 960
                for cl in (640, 960, 1280):
                    if cl == 1280 and not sel_rows:                      # 1280 canvases only when asked for by name
                        continue
                    h_, w_ = canv(cl)
                    for bw in (32, 64):
                        for am in (1, 2):
                            for var in ("i", "s"):
                                nm0 = f"fov{var}{cl}b{bw}a{am}"
                                if not want33(nm0):
                                    continue
                                nm = nm0 + sfx(p)
                                gk = (h_, w_, bw, am)
                                if gk not in FOVG:
                                    FOVG[gk] = FGG.PlainKDEGrid(output_shape=(h_, w_), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                                                anti_crop=True, bandwidth_scale=bw, amplitude_scale=am)
                                gen = FOVG[gk]
                                t0 = start(nm)
                                G.frame(img)
                                meta = [{"pad_shape": (H, W, 3)}]
                                xt = G.x[None] * 255.0
                                with torch.no_grad():
                                    if var == "i":
                                        dd = draft()
                                        grid = gen(xt, meta, [torch.tensor(dd["xyxy"], dtype=torch.float32, device="cuda")])
                                    else:
                                        sal = FSAL[(bw, am)]
                                        grid = gen.separable_saliency_to_grid(xt, meta, sal.sum(dim=2), sal.sum(dim=3), xt.device)
                                    warped = torch.nn.functional.grid_sample(G.x[None], grid, align_corners=True)[0]
                                G.x, G.shape = warped.contiguous(), (h_, w_, 3)
                                dw = keep(gdetect(det_s, Mode(f"{h_}x{w_}", h_, w_))) if max(h_, w_) > 1024 else CANVAS_MANY([[0, 0, w_, h_]])[0]
                                if len(dw["conf"]):
                                    bb = FGG.unwarp_bboxes(torch.tensor(dw["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W))
                                    dw = dict(dw, xyxy=bb.cpu().numpy().astype(float))
                                finish(nm, dw, (CH.net_h * CH.net_w if var == "i" else 0) + h_ * w_, p, t0)
                # UFP mosaics 512 / 768 / 1024
                for msc in (1.2, 1.5, 2.0):
                    for S_ in (512, 768, 1024):
                        nm0 = f"ufp{int(msc * 10)}s{S_}"
                        if not want33(nm0):
                            continue
                        nm = nm0 + sfx(p)
                        t0 = start(nm)
                        G.frame(img)
                        xb_, r_, pad_ = letterbox(G.x, CH.net_h, CH.net_w)
                        raw_ = det_s.forward(xb_[None], CH)
                        rc = GP.post_lean(GP.host_columns(raw_, CH, 0.10)[0], CH, r_, pad_, G.shape, min_conf=0.10)
                        dd = keep(rc)
                        dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
                        cxy = np.asarray(rc["xyxy"], float).reshape(-1, 4)
                        recu, mw_, mh_ = UFPACK(cxy.copy(), msc, input_shape=[W, H]) if len(cxy) else ([], 0, 0)
                        full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                        srcs, mpx, gpx = [], 0, 0
                        if recu and mw_ >= 1 and mh_ >= 1:
                            mos = KP.ufp_mosaic(recu, img, mw_, mh_).astype(np.uint8)
                            chips = [[math.floor(_) for _ in c_] for c_ in recu]
                            srcs = [np.array([c_[0], c_[1], c_[0] + c_[2], c_[1] + c_[3]], float) for c_ in chips]
                            mh, mw = mos.shape[:2]
                            G.frame(mos)
                            sc_m = S_ / max(mh, mw)
                            mpx = int(math.ceil(mh * sc_m / 32) * 32) * int(math.ceil(mw * sc_m / 32) * 32)
                            dm = UFP_MANY[S_](mw, mh)([[0, 0, mw, mh]])[0]
                            G.frame(img)
                            out_ = {}
                            for (x1, y1, x2, y2), scr, cl in zip(dm["xyxy"], dm["conf"], dm["coarse"]):
                                for o_x1, o_y1, cw, chh, n_x, n_y, sf in chips:
                                    if KP.ufp_iof([x1, y1, x2, y2], [n_x, n_y, n_x + cw * sf, n_y + chh * sf]) > 0.9:
                                        nx, ny = (x1 - n_x) / sf + o_x1, (y1 - n_y) / sf + o_y1
                                        out_.setdefault(str(cl), []).append([nx, ny, nx + (x2 - x1) / sf, ny + (y2 - y1) / sf, float(scr),
                                                                            o_x1, o_y1, o_x1 + cw, o_y1 + chh])
                                        break
                            bx_, cf_, cl_ = [], [], []
                            for cl, lst in out_.items():
                                arr = np.array(lst)
                                for i_ in KP.ufp_nms(arr[:, :5], 0.6):
                                    b_, src_ = arr[i_, :4], arr[i_, 5:9]
                                    if any(abs(b_[k2] - src_[k2]) < 2 and abs(src_[k2] - (0, 0, W, H)[k2]) > 2 for k2 in range(4)):
                                        gc = SP.box_crop(b_, W, H)
                                        gpx += canvas(gc, W, H, hs)
                                        gd = many_p([gc])[0]
                                        if len(gd["conf"]):
                                            io_ = K61.box_iou(b_[None], gd["xyxy"])[0]
                                            j_ = int(np.argmax(io_))
                                            if io_[j_] >= 0.3:
                                                bx_.append(gd["xyxy"][j_]); cf_.append(float(gd["conf"][j_])); cl_.append(str(gd["coarse"][j_]))
                                        continue
                                    bx_.append(b_); cf_.append(float(arr[i_, 4])); cl_.append(cl)
                            full = {"xyxy": np.array(bx_, float).reshape(-1, 4), "conf": np.array(cf_, float),
                                    "coarse": np.array(cl_) if cl_ else np.zeros(0, "<U8")}
                        out = SP.centres_in(dd["xyxy"], srcs) if srcs else np.zeros(len(dd["xyxy"]), bool)
                        fz = lift_c(nm, p)(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
                        sh = {"xyxy": np.r_[dd["xyxy"][~out], full["xyxy"]], "conf": np.r_[dd["conf"][~out], full["conf"]],
                              "coarse": np.r_[dd["coarse"][~out], full["coarse"]],
                              "mono": tuple(np.r_[a_[~out], b_] for a_, b_ in zip(dd["mono"], fz)),
                              "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
                        tt, _ = refine(sh, refiner(p, "v3a_vuw", nm))
                        a_k = act(tt)
                        stop(nm, t0)
                        rec.update({f"a_{nm}": a_k, f"px_{nm}": (CH.net_h * CH.net_w + mpx + gpx) / NATIVE})
                        push(nm, np.asarray(sh["xyxy"], float).reshape(-1, 4)[np.asarray(sh["from_full"], bool)])
                # C: routers (BAR, DVR; strong arm 1280) and Chanakya (ch128B_csc), K28's router convention: the cheap
                # draft always; a routed frame also gets the whole frame at s + C1 + refiner v3a, which replaces the draft
                if RTR is not None:
                    for nm0 in ("bar1280_0.5", "dvr1280_0.5", "ch128B_csc"):
                        if not want33(nm0):
                            continue
                        nm = nm0 + sfx(p)
                        t0 = start(nm)
                        G.frame(img)
                        dd = draft()
                        if nm0.startswith("bar"):
                            with torch.no_grad():                # the estimator's own transform (PIL resize, ImageNet norm.)
                                xi = RTR["tf"](PILImage.fromarray(np.ascontiguousarray(img[:, :, ::-1])))[None].cuda()
                                sc = float(BARN(xi).item())
                            sd = 1280 if sc >= RTR["thr"][f"{nm0}|{RTR['src']}"] else 0
                        else:
                            bx_ = np.asarray(dd["xyxy"], float).reshape(-1, 4)
                            e_ = BGF.extract_detection_features([(float(b_[0]), float(b_[1]), float(b_[2] - b_[0]), float(b_[3] - b_[1]), float(c_), str(k_))
                                                                 for b_, c_, k_ in zip(bx_, dd["conf"], dd["coarse"])])
                            f_ = [float(e_[k_]) for k_ in BGF.BASE_FEATURES]
                            if nm0.startswith("dvr"):
                                sc = RTR["xgb"](f_ + [v])
                                sd = 1280 if sc >= RTR["thr"][f"{nm0}|{RTR['src']}"] else 0
                            else:
                                xf = np.array(f_ + [v, float(P.required_decel(*dd["mono"], v, pp)[0]), float(act(dd["mono"])),
                                                    float(-1 if prev_act is None else prev_act)], np.float32)
                                with torch.no_grad():
                                    sc = int(RTR["chn"](torch.from_numpy((xf - RTR["mu"]) / RTR["sd"])[None]).argmax(1).item())
                                sd = (0, 640, 960, 1280)[sc]
                        if sd == 0:
                            a_k = act(dd["mono"])
                            stop(nm, t0)
                            rec.update({f"a_{nm}": a_k, f"px_{nm}": CH.net_h * CH.net_w / NATIVE})
                        else:
                            h_, w_ = canv(sd)
                            dw = keep(gdetect(det_s, Mode(f"{h_}x{w_}", h_, w_))) if max(h_, w_) > 1024 else KF_whole(sd)
                            finish(nm, dw, CH.net_h * CH.net_w + h_ * w_, p, t0)
                        rec.update({f"score_{nm}": sc, f"size_{nm}": sd})
        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        prev_act = a_c
        imgs = (imgs + [img])[-2:]
        rec["warm"] = (unit == "warmup") if args.dataset in ("nusc_test20", "ahd") else (len(rows) < args.warmup)
        n_crop_checked += nonlocal_bad[1]; n_crop_bad += nonlocal_bad[0]
        rows.append(rec)
        if args.parity and len(rows) % 100 == 0:
            torch.cuda.empty_cache()                               # memory only (unified memory on the Jetson)
        if len(rows) % 200 == 0 or (args.heur_only and len(rows) in (1, 5, 20, 50, 100)):   # early markers in the heuristic runs
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
    (run / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v_ for k, v_ in meta.items() if k != "engines_md5"}))
    summarise(d, run, args, PATHS)


C1REF = {"chains": 0}


def md5s():
    import hashlib
    out = {}
    for f in sorted(list(ENG.glob("c1enc_*.engine")) + list(ENG.glob("refiner_*_dyn.engine")) + [ENG / "k28_temporal_head.pt"]):
        out[f.name] = hashlib.md5(f.read_bytes()).hexdigest()[:8]
    return out


def summarise(d, run, args, PATHS):
    dd = d[~d.warm].reset_index(drop=True)
    if "gt" in dd:                                          # Argoverse-HD: G on the frames with 3D labels (as 61); all frames timed
        dd = dd[dd["gt"].astype(bool)].reset_index(drop=True)
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
        base, p = c[2:].rsplit("@", 1)
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
