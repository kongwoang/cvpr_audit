"""Candidate selection on the GPU before the device -> host copy (deployment engineering; identical outputs).

vtf.fastpost.FastDetector.postprocess_np receives the whole raw head output (4 + nc, N) on the host: for a large canvas
that is a multi-MB copy plus numpy work over every anchor, ~3 ms per call on the Jetson. Its first step keeps only the
anchors whose best kept-class score passes the thresholds (best > mode.conf, and best >= min_conf when given). Here that
mask is computed on the GPU from the same float32 values (max is tie-free, so the mask is bit-identical) and only the
selected columns, in their original order, are copied; postprocess_np then runs unchanged on them, so every returned
array is identical. n_cand_raw (a diagnostic counted over all anchors) is computed on the GPU as well and restored.
"""
from __future__ import annotations

import numpy as np
import torch

from .fastpost import KEEP

_KEEP_T = {}


def _keep_idx(device):
    if device not in _KEEP_T:
        _KEEP_T[device] = torch.as_tensor(KEEP, device=device, dtype=torch.long)
    return _KEEP_T[device]


def select(raw, conf, min_conf=None):
    """raw: (4 + nc, N) or (B, 4 + nc, N) tensor on the device. Returns a list of (columns as host numpy, n_cand_raw)."""
    t = raw.float()
    if t.dim() == 2:
        t = t[None]
    cls = t[:, 4:]
    best = cls.index_select(1, _keep_idx(t.device)).max(1).values            # (B, N)
    keep = best > conf
    if min_conf is not None:
        keep &= best >= min_conf
    n_raw = (cls.max(1).values > conf).sum(1)                                 # (B,)
    counts = keep.sum(1)
    sel = t.permute(0, 2, 1)[keep]                                            # (sum n, 4 + nc), batch-major, ordered
    host = sel.cpu().numpy()                                                  # the only device -> host copy (+ counts)
    cnt = torch.stack([counts, n_raw], 1).cpu().numpy()
    out, i0 = [], 0
    for n, nr in cnt:
        out.append((np.ascontiguousarray(host[i0:i0 + n].T), int(nr)))
        i0 += n
    return out


def postprocess(det, raw, mode, r, pad, img_shape, min_conf=None):
    """= det.postprocess_np(raw.float().cpu().numpy(), ...) for one image's raw output on the device."""
    p, n_raw = select(raw, mode.conf, min_conf)[0]
    if p.shape[1] == 0:
        p = np.zeros((raw.shape[-2], 0), np.float32)
    d = det.postprocess_np(p, mode, r, pad, img_shape, min_conf=min_conf)
    d["n_cand_raw"] = n_raw
    return d


GPU_SELECT_MIN_ANCHORS = 5000          # below this the extra kernel launches cost more than the copy they save (Jetson)


def post_lean(p, mode, r, pad, img_shape, min_conf=None):
    """postprocess_np restricted to the outputs the pipelines use (xyxy, conf, coarse): same operations in the same
    order on those arrays, the uncertainty features (entropy / margin / binent) and diagnostics are skipped."""
    from .fastpost import COARSE, nms, nms_matrix
    cls_scores = p[4:]
    traffic = cls_scores[KEEP]
    best_i = traffic.argmax(0)
    best = traffic[best_i, np.arange(traffic.shape[1])]
    keep = best > mode.conf
    if min_conf is not None:
        keep &= best >= min_conf
    if not keep.any():
        return {"xyxy": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32), "coarse": np.zeros(0, "<U8")}
    bx = p[:4, keep].T
    best, best_i = best[keep], best_i[keep]
    xyxy = np.concatenate([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1)
    off = KEEP[best_i].astype(np.float32)[:, None] * (xyxy.max() + 1)
    sel = (nms(xyxy + off, best, mode.iou) if min_conf is None else nms_matrix(xyxy + off, best, mode.iou))[: mode.max_det]
    xyxy = xyxy[sel].copy()
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= r
    h0, w0 = img_shape[:2]
    xyxy[:, [0, 2]] = np.minimum(np.maximum(xyxy[:, [0, 2]], 0), w0 - 1)
    xyxy[:, [1, 3]] = np.minimum(np.maximum(xyxy[:, [1, 3]], 0), h0 - 1)
    return {"xyxy": xyxy.astype(np.float32), "conf": best[sel].astype(np.float32), "coarse": COARSE[best_i[sel]]}


def host_columns(raw, mode, min_conf=None):
    """raw (B, 4 + nc, N) on the device -> list of host arrays (4 + nc, n_b): GPU candidate selection for large N,
    a plain copy otherwise (post_lean / postprocess_np give identical outputs on either)."""
    if raw.shape[-1] >= GPU_SELECT_MIN_ANCHORS:
        return [c for c, _ in select(raw, mode.conf, min_conf)]
    h = raw.float().cpu().numpy()
    return [h[i] for i in range(h.shape[0])] if h.ndim == 3 else [h]
