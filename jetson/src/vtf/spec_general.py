"""The speculative loop for any consumer that exposes a field (where an insertion changes its decision) and a support
(what its decision rests on).  Same loop, guard and merge as `vtf.speculative.run` (braking), which stays the frozen
reference; this module adds Planner B (argmin over candidate trajectories, decomposable costs, exact fields in
`vtf.dsf.planner_b_*`), whose support is taken as its exact leave-one-out pivots.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P
from rap import planner_b as PB
from rap.mono import box_iou

from . import dsf
from . import speculative as SP


class Brake:
    def __init__(self, pp, v_close=10.0):
        self.pp, self.v_close = pp, v_close

    def field(self, t, v, r_grid, y_grid):
        return dsf.insertion_grid(t, v, r_grid, y_grid, SP.OBJ_W, self.pp, self.v_close)

    def support(self, t, v):
        return SP.support(t, v, self.pp)

    def decide(self, t, v):
        return P.discrete_action(P.required_decel(*t, v, self.pp)[0], self.pp)


class PlannerB:
    def __init__(self, pb=None, cb=None):
        self.pb = pb or PB.PARAMS_B["static_obstacles"]
        self.cb = cb or PB.COSTS_B["default"]
        self.prev = None                                 # the plan history the consumer is run with (set per frame)

    def field(self, t, v, r_grid, y_grid):
        return dsf.planner_b_insertion_grid(t, v, self.prev, r_grid, y_grid, SP.OBJ_W, self.pb, self.cb)

    def support(self, t, v):
        return np.asarray(dsf.planner_b_pivots(t, v, self.prev, self.pb, self.cb), int).reshape(-1)

    def decide(self, t, v):
        return PB.plan(*t, v, self.prev, self.pb, self.cb)


def footprint(sens, cam, img_w, img_h, r_grid, y_grid, obj_h=SP.OBJ_H):
    if not sens.any():
        return None
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(r_grid, y_grid, indexing="ij")
    rr, yy = rr[sens], yy[sens]
    vb = cy + fy * cam.cam_h / rr
    uc, hw = cx - fx * yy / rr, fx * SP.OBJ_W / (2 * rr)
    return np.array([max((uc - hw).min(), 0), max((vb - fy * obj_h / rr).min(), 0), min((uc + hw).max(), img_w),
                     min(vb.max(), img_h)])


def run(cheap, v, cam, img_w, img_h, detect_crop, lift, consumer, max_rounds=3, guard=True, detect_many=None,
        batch_round=False):
    gdet = SP.guarded(detect_crop, img_w, img_h, detect_many=detect_many) if guard else None
    gall = SP.guarded_batch(detect_many, img_w, img_h) if (batch_round and guard and detect_many is not None) else None
    crops, grown = [], []
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds = dict(cheap), 0
    for _ in range(max_rounds):
        new = []
        fc = footprint(consumer.field(scene["mono"], v, SP.R_GRID, SP.Y_GRID), cam, img_w, img_h, SP.R_GRID, SP.Y_GRID)
        if fc is not None and not SP.covered(fc, crops):
            new.append(fc)
        for k in consumer.support(scene["mono"], v):
            bc = SP.box_crop(scene["xyxy"][k], img_w, img_h)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        if not new:
            break
        rounds += 1
        per = gall(new) if gall is not None else None              # one batched call per round (same results)
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
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable")
            iou = box_iou(full["xyxy"], full["xyxy"])
            keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            full = {k: full[k][np.sort(np.array(keep))] for k in full}
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                 "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                 "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops + grown, rounds
