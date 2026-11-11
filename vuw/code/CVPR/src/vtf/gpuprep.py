"""GPU pre-processing for the deployed Jetson pipelines: the frame is uploaded once, every letterbox (whole frame or crop)
is a GPU resize + pad, and the crops of one call share one batched TensorRT engine call and one device -> host copy.

Same geometry as rap.detect.letterbox (scale r = min(net_h / h, net_w / w), centred padding with value 114, same
rounding of the pad), bilinear resize (cv2.INTER_LINEAR's sampling, no antialiasing), so detections match the CPU path
up to resize rounding / FP16 differences; parity is checked by scripts/54_fast_latency.py (decision values per pipeline).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

PAD = 114.0 / 255.0


def upload(img_bgr, device):
    """HxWx3 uint8 BGR -> 3xHxW float RGB in [0, 1] on the device (one host -> device copy per frame)."""
    t = torch.from_numpy(img_bgr).to(device, non_blocking=True)
    return t.permute(2, 0, 1).flip(0).float().div_(255.0)


def letterbox(x, net_h, net_w):
    """x: 3xhxw on the device -> (3 x net_h x net_w, r, (left, top)), as rap.detect.letterbox."""
    h, w = x.shape[1:]
    r = min(net_h / h, net_w / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    padx, pady = (net_w - nw) / 2, (net_h - nh) / 2
    top, left = int(round(pady - 0.1)), int(round(padx - 0.1))
    out = torch.full((3, net_h, net_w), PAD, device=x.device, dtype=x.dtype)
    out[:, top:top + nh, left:left + nw] = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    return out, r, (left, top)


class GPUDetect:
    """detect(det, mode) on the uploaded frame; crops(det, crops, mode, batch_engines) for many crops at once.
    `det` is a vtf.fastpost.FastDetector with TensorRT engines (forward(x, mode) / postprocess_np)."""

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.x = None
        self.shape = None

    def frame(self, img_bgr):
        self.x = upload(img_bgr, self.device)
        self.shape = img_bgr.shape

    def detect(self, det, mode, thr=None):
        xb, r, pad = letterbox(self.x, mode.net_h, mode.net_w)
        raw = det.forward(xb[None], mode)
        return det.postprocess_np(raw[0].float().cpu().numpy(), mode, r, pad, self.shape, min_conf=thr)

    def warm(self, det, mode, engines):
        """first call of every engine (TensorRT lazy initialisation) outside the timed frames"""
        x = torch.full((3, 64, 64), PAD, device=self.device)
        self.x, self.shape = x, (64, 64, 3)
        self.crops(det, [[0, 0, 32, 32]] * max(engines), mode, engines, 0.25)
        for b in engines:
            self.crops(det, [[0, 0, 32, 32]] * b, mode, engines, 0.25)

    def crops(self, det, crops, mode, engines, thr):
        """crops: [x1, y1, x2, y2] in frame pixels -> [dict(xyxy, conf, coarse)] (conf >= thr), frame coordinates.
        engines: {batch size: TRTModule} for the crop canvas (1 = det's own engine for `mode`)."""
        if not crops:
            return []
        xs, meta = [], []
        for cr in crops:
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = self.x[:, y1:y2, x1:x2]
            xb, r, pad = letterbox(sub, mode.net_h, mode.net_w)
            xs.append(xb)
            meta.append((r, pad, (sub.shape[1], sub.shape[2], 3), x1, y1))
        sizes = sorted(engines)
        out, i0 = [], 0
        while i0 < len(xs):
            n = len(xs) - i0
            b = next((s for s in sizes if s >= n), sizes[-1])
            chunk = xs[i0:i0 + b]
            xb = torch.stack(chunk + [torch.full_like(chunk[0], PAD)] * (b - len(chunk)))
            raw = (det.forward(xb, mode) if b == 1 else engines[b](xb))[: len(chunk)].float().cpu().numpy()
            for k, (r, pad, shp, x1, y1) in enumerate(meta[i0:i0 + len(chunk)]):
                d = det.postprocess_np(raw[k], mode, r, pad, shp, min_conf=thr)
                kk = d["conf"] >= thr
                out.append({"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                            "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]})
            i0 += len(chunk)
        return out
