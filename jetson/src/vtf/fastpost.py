"""CPU post-processing for DEEP's YOLO detector (same outputs as rap.detect.TwoFidelityDetector.postprocess).

On the Jetson the original runs ~20 small GPU kernels with host syncs per call (~6 ms at low GPU clocks); here the raw
head output is copied to the host once and thresholding, class-aware NMS and the uncertainty features run in numpy.
Used for every pipeline of the deployed comparison alike (scripts/40_e2e_v2.py), after a parity check.
"""
from __future__ import annotations

import numpy as np

from rap.detect import COCO_KEEP, COCO_TRAFFIC, TwoFidelityDetector, _empty_dets

KEEP = np.array(COCO_KEEP)
COARSE = np.array([COCO_TRAFFIC[c] for c in COCO_KEEP])


def nms_matrix(boxes, scores, iou_thr):
    """Same result as nms() (greedy, suppress IoU > iou_thr, descending score, stable), one IoU matrix instead of a
    numpy call per kept box: faster for the few-dozen candidates of a crop."""
    order = np.argsort(-scores, kind="stable")
    b = boxes[order]
    x1, y1, x2, y2 = b.T
    area = (x2 - x1) * (y2 - y1)
    w = np.maximum(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0)
    h = np.maximum(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0)
    inter = w * h
    iou = inter / np.maximum(area[:, None] + area[None] - inter, 1e-9)
    sup = np.zeros(len(b), bool)
    keep = []
    for i in range(len(b)):
        if sup[i]:
            continue
        keep.append(i)
        sup |= iou[i] > iou_thr
    return order[np.array(keep, int)]


def nms(boxes, scores, iou_thr):
    """Greedy NMS; suppresses IoU > iou_thr (torchvision semantics).  Returns kept indices by descending score."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, int)


class FastDetector(TwoFidelityDetector):
    def postprocess(self, raw, mode, r, pad, img_shape):
        return self.postprocess_np(raw[0].float().cpu().numpy(), mode, r, pad, img_shape)   # the only device -> host copy

    def postprocess_np(self, p, mode, r, pad, img_shape, min_conf=None):
        """p: one image's raw head output (4+nc, N) already on the host (batched callers copy the batch once).
        min_conf: callers that keep only conf >= min_conf may drop lower candidates BEFORE NMS — greedy NMS decides a
        box only from higher-scored boxes, so every returned box with conf >= min_conf is identical (only the low-
        confidence tail and the n_cand diagnostics change); the IoU-matrix NMS is used then (same greedy result)."""
        boxes_xywh, cls_scores = p[:4], p[4:]
        traffic = cls_scores[KEEP]
        best_i = traffic.argmax(0)
        best = traffic[best_i, np.arange(traffic.shape[1])]
        n_cand_raw = int((cls_scores.max(0) > mode.conf).sum())
        keep = best > mode.conf
        if min_conf is not None:
            keep &= best >= min_conf
        n_cand = int(keep.sum())
        if n_cand == 0:
            return _empty_dets(n_cand_raw, 0)
        bx = boxes_xywh[:, keep].T
        traffic, full_scores = traffic[:, keep].T, cls_scores[:, keep].T
        best, best_i = best[keep], best_i[keep]
        xyxy = np.concatenate([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1)
        off = KEEP[best_i].astype(np.float32)[:, None] * (xyxy.max() + 1)   # class-aware NMS by coordinate offset
        sel = (nms(xyxy + off, best, mode.iou) if min_conf is None else nms_matrix(xyxy + off, best, mode.iou))[: mode.max_det]
        xyxy = xyxy[sel].copy()
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= r
        h0, w0 = img_shape[:2]
        xyxy[:, [0, 2]] = np.minimum(np.maximum(xyxy[:, [0, 2]], 0), w0 - 1)        # np.clip is slow on this numpy
        xyxy[:, [1, 3]] = np.minimum(np.maximum(xyxy[:, [1, 3]], 0), h0 - 1)
        tr = traffic[sel]
        top2 = -np.sort(-tr, 1)[:, :2]
        margin = top2[:, 0] - top2[:, 1]
        fs = np.maximum(full_scores[sel], 1e-9)
        pn = fs / fs.sum(1, keepdims=True)
        conf = np.minimum(np.maximum(best[sel], 1e-6), 1 - 1e-6)
        return {"xyxy": xyxy.astype(np.float32), "conf": best[sel].astype(np.float32),
                "coco_id": KEEP[best_i[sel]].astype(np.int16), "coarse": COARSE[best_i[sel]],
                "entropy": (-(pn * np.log(pn)).sum(1)).astype(np.float32), "margin": margin.astype(np.float32),
                "binent": (-(conf * np.log(conf) + (1 - conf) * np.log(1 - conf))).astype(np.float32),
                "n_cand_raw": n_cand_raw, "n_cand": n_cand, "n_post": int(len(sel))}
