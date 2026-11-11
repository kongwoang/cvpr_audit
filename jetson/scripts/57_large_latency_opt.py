#!/usr/bin/env python
"""K5 follow-up — the large-frame latency of scripts/55_large_latency.py (imported, file unchanged) with deployment
engineering that leaves every detection identical (parity is checked against a 55 run on the same frames):

  --gpupost   detector post-processing: candidate selection on the GPU before the host copy for canvases with
              >= 5000 anchors (vtf.gpupost.select) and a lean numpy post-processing that computes only the outputs the
              pipelines use (vtf.gpupost.post_lean); applied to EVERY pipeline (whole-frame passes and crops alike)
  --dyncrops  crops on a canvas of exactly ceil32(crop x scale) with the frame letterbox (the VUW evaluations'
              geometry) through one dynamic-shape engine (engines/yolov8s_dyn.engine, vtf.trtdyn), crops of equal canvas
              batched (<= 8); detections then differ from 55's static canvases (closer to VUW): G is re-reported
  --verifier yolov8l  heavy-verifier operating point: every whole-frame pass and every crop use YOLOv8l (engines
              yolov8l_crop_<HxW>.engine, crops through yolov8l_dyn.engine; needs --dyncrops); the cheap draft stays
              YOLOv8s at 320 (as in the KITTI YOLOv8l measurement); G is re-reported
  --nosync    no CUDA synchronisations inserted only for the per-stage breakdown (55 syncs before / after every engine
              call and at each stage boundary); every pipeline still ends on host values (detections copied to the
              host, refiner output read back), so each pipeline's time still covers all of its GPU work. The stage
              breakdown is then not meaningful (it is reported by the synced 55 run).

Writes the same outputs as 55 (latency.csv, breakdown.csv, large_latency.pkl) in a run folder named large_latency_opt_*.
usage: 57_large_latency_opt.py --dataset av2|nusc [--gpupost] [--nosync] [--max_frames N] [--warmup W]
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k5", ROOT / "scripts" / "55_large_latency.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
from vtf import gpupost as GP                                                    # noqa: E402

FLAGS = {f for f in ("--gpupost", "--nosync", "--dyncrops") if f in sys.argv}
VERIFIER = "yolov8s"
if "--verifier" in sys.argv:
    i = sys.argv.index("--verifier"); VERIFIER = sys.argv[i + 1]; del sys.argv[i:i + 2]
    FLAGS.add(f"verifier={VERIFIER}")
    assert "--dyncrops" in FLAGS, "--verifier needs --dyncrops (no static crop canvases for it)"
sys.argv = [a for a in sys.argv if a not in FLAGS]
CHEAP_MODES = {"320x256", "192x320", "224x320"}

if VERIFIER != "yolov8s":
    _engine_for = K.FastDetector.engine_for

    def engine_for(self, mode):
        """the cheap draft from the YOLOv8s engines, every other whole-frame mode from the verifier's engines"""
        if mode.name in CHEAP_MODES:
            return _engine_for(self, mode)
        key = f"{VERIFIER}:{mode.name}"
        if key not in self._engines:
            self._engines[key] = K.TRTModule(K.ENG / f"{VERIFIER}_crop_{mode.name}.engine")
        return self._engines[key]
    K.FastDetector.engine_for = engine_for

if "--gpupost" in FLAGS:
    from vtf.gpuprep import letterbox                                        # noqa: E402

    def detect(self, det, mode, thr=None):
        """= vtf.gpuprep.GPUDetect.detect with the post-processing of vtf.gpupost (identical outputs)."""
        xb, r, pad = letterbox(self.x, mode.net_h, mode.net_w)
        raw = det.forward(xb[None], mode)
        return GP.post_lean(GP.host_columns(raw, mode, thr)[0], mode, r, pad, self.shape, min_conf=thr)
    K.GPUDetect.detect = detect

    def crops_call(self, crops):
        """= 55's ScaledCrops.__call__ with the host copy / post-processing of vtf.gpupost (identical outputs)."""
        K.sync(); t0 = time.perf_counter()
        groups = {}
        for i, cr in enumerate(crops):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = self.G.x[:, y1:y2, x1:x2]
            c = self.canvas(y2 - y1, x2 - x1)
            s = min(self.s, c[0] / (y2 - y1), c[1] / (x2 - x1))
            xb, r, pad = K.fixed_scale(sub, s, *c)
            groups.setdefault(c, []).append((i, xb, r, pad, (y2 - y1, x2 - x1, 3), x1, y1))
        K.sync()
        st = K.STATS.setdefault("crop_prep", [0, 0.0]); st[0] += 1; st[1] += time.perf_counter() - t0
        out = [None] * len(crops)
        for c, items in groups.items():
            mode = self.modes[c]
            engines = self.batch if c == (96, 288) else {1: None}
            sizes = sorted(engines)
            k0 = 0
            while k0 < len(items):
                b = next((q for q in sizes if q >= len(items) - k0), sizes[-1])
                chunk = items[k0:k0 + b]
                xb = torch.stack([it[1] for it in chunk] + [torch.full_like(chunk[0][1], K.PAD)] * (b - len(chunk)))
                call = K.Counted(self.det.raw_forward if b == 1 else engines[b], f"det_{self.stage}")
                raw = (call(xb, mode) if b == 1 else call(xb))[: len(chunk)]
                cols = GP.host_columns(raw, mode, K.THR)
                px = K.STATS.setdefault(f"tpx_{self.stage}", [0, 0.0]); px[1] += b * c[0] * c[1]
                for k, (i, _, r, pad, shp, x1, y1) in enumerate(chunk):
                    d = GP.post_lean(cols[k], mode, r, pad, shp, min_conf=K.THR)
                    kk = d["conf"] >= K.THR
                    out[i] = {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                              "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
                k0 += len(chunk)
        return out
    K.ScaledCrops.__call__ = crops_call

if "--dyncrops" in FLAGS:
    from vtf.gpuprep import letterbox as _lbox                                  # noqa: E402
    from vtf.trtdyn import DynTRT                                               # noqa: E402
    _DYN = {}

    def dyn_call(self, crops):
        if "e" not in _DYN:
            _DYN["e"] = DynTRT(K.ENG / f"{VERIFIER}_dyn.engine")
        eng = _DYN["e"]
        K.sync(); t0 = time.perf_counter()
        groups = {}
        for i, cr in enumerate(crops):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = self.G.x[:, y1:y2, x1:x2]
            nh = min(1024, max(32, int(math.ceil(sub.shape[1] * self.s / 32)) * 32))
            nw = min(1024, max(32, int(math.ceil(sub.shape[2] * self.s / 32)) * 32))
            xb, r, pad = _lbox(sub, nh, nw)
            groups.setdefault((nh, nw), []).append((i, xb, r, pad, (y2 - y1, x2 - x1, 3), x1, y1))
        K.sync()
        st = K.STATS.setdefault("crop_prep", [0, 0.0]); st[0] += 1; st[1] += time.perf_counter() - t0
        out = [None] * len(crops)
        for (nh, nw), items in groups.items():
            mode = K.Mode(f"c{nh}x{nw}", nh, nw)
            for k0 in range(0, len(items), eng.max_batch):
                chunk = items[k0:k0 + eng.max_batch]
                raw = K.Counted(eng, f"det_{self.stage}")(torch.stack([it[1] for it in chunk]))
                px = K.STATS.setdefault(f"tpx_{self.stage}", [0, 0.0]); px[1] += len(chunk) * nh * nw
                cols = GP.host_columns(raw, mode, K.THR)
                for k, (i, _, r, pad, shp, x1, y1) in enumerate(chunk):
                    d = GP.post_lean(cols[k], mode, r, pad, shp, min_conf=K.THR)
                    kk = d["conf"] >= K.THR
                    out[i] = {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                              "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
        return out
    K.ScaledCrops.__call__ = dyn_call

if "--nosync" in FLAGS:
    K.sync = lambda: None

_new_run = K.runmeta.new_run
K.runmeta.new_run = lambda name, cfg: _new_run(name.replace("large_latency", "large_latency_opt"),
                                               {**cfg, "flags": sorted(FLAGS)})

if __name__ == "__main__":
    K.main()
