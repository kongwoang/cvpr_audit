"""Speculative perception loop for the braking controller, as proved in docs/paper_theory.md.

Draft with the cheap detections; repeatedly crop (a) the image footprint of the insertion-sensitive ground cells of the
current scene and (b) the boxes of the objects that support the current action; run the full model on the new crops;
merge (full detections replace everything whose centre lies in any crop); stop when the action is stable and no new
region is needed.  Detection and lifting are callbacks, so the same loop serves the Jetson (TensorRT) and the GPU
server (PyTorch), KITTI and nuScenes.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P
from rap.mono import box_iou

from . import dsf

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX = 1.8, 1.6, 2.0


def support(t, v, pp) -> np.ndarray:
    """Objects whose demand alone reaches the threshold of the current action (empty for KEEP)."""
    a_max, a_obj = P.required_decel(*t, v, pp)
    act = P.discrete_action(a_max, pp)
    if act == P.KEEP or not len(a_obj):
        return np.zeros(0, int)
    thr = pp.thr_decel if act == P.DECELERATE else pp.thr_hard
    return np.flatnonzero(np.asarray(a_obj) >= thr)


def field_crop(mono_t, v, cam, img_w, img_h, pp, v_close=0.0, r_grid=None, obj_h=OBJ_H):
    r_grid = R_GRID if r_grid is None else r_grid
    sens = dsf.insertion_grid(mono_t, v, r_grid, Y_GRID, OBJ_W, pp, v_close)
    if not sens.any():
        return None
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(r_grid, Y_GRID, indexing="ij")
    rr, yy = rr[sens], yy[sens]
    vb = cy + fy * cam.cam_h / rr
    uc, hw = cx - fx * yy / rr, fx * OBJ_W / (2 * rr)
    return np.array([max((uc - hw).min(), 0), max((vb - fy * obj_h / rr).min(), 0), min((uc + hw).max(), img_w),
                     min(vb.max(), img_h)])


def box_crop(b, img_w, img_h):
    x1, y1, x2, y2 = b
    cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
    return np.array([max((x1 + x2) / 2 - cw / 2, 0), max((y1 + y2) / 2 - ch / 2, 0), min((x1 + x2) / 2 + cw / 2, img_w),
                     min((y1 + y2) / 2 + ch / 2, img_h)])


def centres_in(xyxy, crops):
    if not len(xyxy) or not crops:
        return np.zeros(len(xyxy), bool)
    u, w = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    m = np.zeros(len(xyxy), bool)
    for c in crops:
        m |= (u >= c[0]) & (u <= c[2]) & (w >= c[1]) & (w <= c[3])
    return m


def covered(crop, crops, tol=0.95):
    """Is `crop` (almost entirely) inside the union of earlier crops? (area fraction on an 4-px grid)"""
    if not crops:
        return False
    x1, y1, x2, y2 = crop
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    inter = max(max(0.0, min(x2, c[2]) - max(x1, c[0])) * max(0.0, min(y2, c[3]) - max(y1, c[1])) for c in crops)
    return inter / area >= tol


def truncated(xyxy, crop, img_w, img_h, tol=2.0, top_only=False):
    """Boxes touching an edge of `crop` that is not an image edge (the object may extend beyond the crop).
    top_only=True returns (top-edge truncation, truncation at any other edge) separately."""
    x1, y1, x2, y2 = crop
    b = np.asarray(xyxy, float).reshape(-1, 4)
    top = (b[:, 1] - y1 < tol) & (y1 > tol)
    other = (((b[:, 0] - x1 < tol) & (x1 > tol)) | ((x2 - b[:, 2] < tol + 1) & (x2 < img_w - tol - 1)) |
             ((y2 - b[:, 3] < tol + 1) & (y2 < img_h - tol - 1)))
    return (top, other) if top_only else (top | other)


def corridor_filter(cam, pp, margin=1.0):
    """For a box cut only at its top, the bottom edge and both sides are the object's own, so its ground-contact range
    and lateral extent are exact: an object whose lateral extent misses the corridor (+ margin) or that lies beyond the
    planner's range has zero demand whatever its height, and need not be re-detected."""
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

    def relevant(b):
        b = np.asarray(b, float).reshape(-1, 4)
        dy = b[:, 3] - cy
        z = np.where(dy > 1.0, fy * cam.cam_h / np.maximum(dy, 1.0), np.inf)
        la, lb = -(b[:, 2] - cx) * z / fx, -(b[:, 0] - cx) * z / fx
        hw = pp.corridor_half_w + margin
        return ((la < hw) & (lb > -hw) & (z <= pp.max_range + margin)) | ~np.isfinite(z)
    return relevant


def grow(box, crop, img_w, img_h):
    """A crop around a truncated box, extended by the box's own size on each truncated side, with context."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    tol = 2.0
    ex1 = x1 - w if (x1 - crop[0] < tol and crop[0] > tol) else x1
    ex2 = x2 + w if (crop[2] - x2 < tol + 1) else x2
    ey1 = y1 - h if (y1 - crop[1] < tol and crop[1] > tol) else y1
    ey2 = y2 + h if (crop[3] - y2 < tol + 1) else y2
    return box_crop(np.array([ex1, ey1, ex2, ey2]), img_w, img_h)


def guarded(detect_crop, img_w, img_h, max_grow=2, relevant=None, detect_many=None):
    """Enforce locality (L): a detection truncated by its crop is replaced by the detection of the same object in a
    crop grown around it (repeated up to `max_grow` times; dropped if the grown crop shows no such object).
    The grown crops of one growth step are independent, so they go to `detect_many(crops) -> [dets]` together (one
    batched engine call on the Jetson); default: one call each.  Returns det(crop) -> (dets, [grown crops run])."""
    many = detect_many or (lambda cs: [detect_crop(c) for c in cs])

    def det(crop):
        d = {k: np.asarray(x) for k, x in detect_crop(crop).items()}
        if not len(d["conf"]):
            return d, []
        top, other = truncated(d["xyxy"], crop, img_w, img_h, top_only=True)
        tr = other | (top & (relevant(d["xyxy"]) if relevant is not None else True))
        out = {k: x[~tr] for k, x in d.items()}
        extra = []
        live = [(d["xyxy"][i], crop) for i in np.flatnonzero(tr)]          # (truncated box, the crop that cut it)
        for step in range(max_grow):
            if not live:
                break
            gs = [grow(b, c, img_w, img_h) for b, c in live]
            extra += gs
            nxt = []
            for (b, _), g, dg in zip(live, gs, many(gs)):
                dg = {k: np.asarray(x) for k, x in dg.items()}
                if not len(dg["conf"]):
                    continue
                # the grown crop's detection covering the truncated box most (intersection over the truncated area)
                ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                j = int(np.argmax(io))
                if io[j] < 0.5:
                    continue
                if not truncated(dg["xyxy"][j:j + 1], g, img_w, img_h)[0] or step == max_grow - 1:
                    out = {k: np.concatenate([out[k], dg[k][j:j + 1]]) for k in out}
                else:
                    nxt.append((dg["xyxy"][j], g))
            live = nxt
        return out, extra
    return det


def guarded_batch(detect_many, img_w, img_h, max_grow=2, relevant=None):
    """`guarded` for all crops of one round at once: the crops go to detect_many together, then the grown crops of
    every crop at each growth step together. Per crop the result is identical to guarded(...)(crop) (crops are
    processed independently; only the engine calls are shared). Returns det_all(crops) -> [(dets, grown crops)]."""
    def det_all(crops):
        res = []
        live = []                                                   # (crop index, truncated box, crop that cut it)
        for ci, (crop, d) in enumerate(zip(crops, detect_many(list(crops)))):
            d = {k: np.asarray(x) for k, x in d.items()}
            if not len(d["conf"]):
                res.append([d, []])
                continue
            top, other = truncated(d["xyxy"], crop, img_w, img_h, top_only=True)
            tr = other | (top & (relevant(d["xyxy"]) if relevant is not None else True))
            res.append([{k: x[~tr] for k, x in d.items()}, []])
            live += [(ci, d["xyxy"][i], crop) for i in np.flatnonzero(tr)]
        for step in range(max_grow):
            if not live:
                break
            gs = [grow(b, c, img_w, img_h) for _, b, c in live]
            for (ci, _, _), g in zip(live, gs):
                res[ci][1].append(g)
            nxt = []
            for (ci, b, _), g, dg in zip(live, gs, detect_many(gs)):
                dg = {k: np.asarray(x) for k, x in dg.items()}
                if not len(dg["conf"]):
                    continue
                ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                j = int(np.argmax(io))
                if io[j] < 0.5:
                    continue
                if not truncated(dg["xyxy"][j:j + 1], g, img_w, img_h)[0] or step == max_grow - 1:
                    res[ci][0] = {k: np.concatenate([res[ci][0][k], dg[k][j:j + 1]]) for k in res[ci][0]}
                else:
                    nxt.append((ci, dg["xyxy"][j], g))
            live = nxt
        return [tuple(r) for r in res]
    return det_all


def run(cheap, v, cam, img_w, img_h, detect_crop, lift, pp, max_rounds=3, v_close=0.0, r_grid=None, guard=False,
        obj_h=OBJ_H, detect_many=None, batch_round=False):
    """cheap: dict(xyxy, conf, coarse, mono=(z, lat_min, lat_max, ttc)).  detect_crop(crop) -> dict(xyxy, conf, coarse)
    in image coordinates.  lift(dets) -> (z, lat_min, lat_max, ttc).  Returns (scene dict, crops, rounds).
    guard enforces locality (detections truncated by a crop edge are re-detected in a grown crop): False, True (every
    truncated box) or "corridor" (skip top-truncated boxes whose exact ground-contact range and lateral extent already
    give zero demand)."""
    gdet = (guarded(detect_crop, img_w, img_h, relevant=corridor_filter(cam, pp) if guard == "corridor" else None,
                    detect_many=detect_many) if guard else None)
    # batch_round: all crops of a round (and the guard's grown crops of each step) in shared detect_many calls;
    # same per-crop results and order as the sequential loop below
    gall = (guarded_batch(detect_many, img_w, img_h, relevant=corridor_filter(cam, pp) if guard == "corridor" else None)
            if (batch_round and guard and detect_many is not None) else None)
    crops, grown, full = [], [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene = dict(cheap)
    rounds = 0
    for _ in range(max_rounds):
        new = []
        fc = field_crop(scene["mono"], v, cam, img_w, img_h, pp, v_close, r_grid, obj_h)
        if fc is not None and not covered(fc, crops):
            new.append(fc)
        for k in support(scene["mono"], v, pp):
            bc = box_crop(scene["xyxy"][k], img_w, img_h)
            if not covered(bc, crops + grown + new):
                new.append(bc)
        if not new:
            break
        rounds += 1
        if gall is not None:
            per = gall(new)
        elif batch_round and gdet is None and detect_many is not None:
            per = [(d, []) for d in detect_many(new)]
        else:
            per = None
        for i_c, c in enumerate(new):
            if per is not None:
                d, ex = per[i_c]
                grown += ex
            elif gdet is None:
                d = detect_crop(c)
            else:
                d, ex = gdet(c)
                grown += ex
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops += new
        # de-duplicate detections of overlapping crops: keep the more confident of any pair with IoU >= 0.5
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable")
            iou = box_iou(full["xyxy"], full["xyxy"])
            keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            keep = np.sort(np.array(keep))
            full = {k: full[k][keep] for k in full}
        out = centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            # an object re-detected in a grown crop may have its centre outside the original crops: drop its cheap twin
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                 "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                 "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops + grown, rounds
