"""Exploratory headroom (post hoc, not pre-registered): spend perception compute on the decision's pivot, not the frame.

For a perceived set T and a consumer, the *pivot* of the decision is the object whose removal alone changes the
decision (the nearest if several; if none, the object the braking rule reacts to most, if any).  An oracle verifier
checks the pivot against the reference scene: a detection that matches a reference box at IoU >= 0.5 gets that box's
geometry, one that matches none is removed.  Up to K pivots are verified in turn, each time on the updated set.

A noisy verifier makes the existence call wrong with probability `eps` and refines range with multiplicative
Gaussian error `sigma`; it measures how good a learned verifier/refiner would have to be.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P
from rap.mono import box_iou

from .flips import MATCH_IOU as _M

MATCH_IOU = _M          # the verifier's existence criterion; 03_headroom.py can lower it


class Obj:
    """A mutable perceived set: image boxes plus consumer geometry, with a per-object `verified` flag."""

    def __init__(self, xyxy, cons):
        self.xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        self.t = [np.asarray(a, float).copy() for a in cons]
        self.verified = np.zeros(len(self.xyxy), bool)

    def cons(self):
        return tuple(self.t)

    def drop(self, i):
        self.xyxy = np.delete(self.xyxy, i, 0)
        self.t = [np.delete(a, i) for a in self.t]
        self.verified = np.delete(self.verified, i)

    def __len__(self):
        return len(self.xyxy)


def pivot_of(S: Obj, decide, v, pp):
    """Index of the unverified object the current decision hinges on, or None."""
    n = len(S)
    if n == 0:
        return None
    base = decide(S.cons())
    cand = [i for i in range(n) if not S.verified[i]]
    if not cand:
        return None
    nec = [i for i in cand if decide(tuple(np.delete(a, i) for a in S.t)) != base]
    if nec:
        return min(nec, key=lambda i: S.t[0][i])
    a_obj = [P.required_decel(*(a[i:i + 1] for a in S.t), v, pp)[0] for i in cand]
    k = int(np.argmax(a_obj))
    return cand[k] if a_obj[k] > 0 else None


def verify(S: Obj, i: int, g, rng=None, eps: float = 0.0, sigma: float = 0.0) -> None:
    """Replace object i by the reference geometry it matches, or remove it; with optional verifier errors."""
    gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4)))
    iou = box_iou(S.xyxy[i:i + 1], gt)[0] if len(gt) else np.zeros(0)
    real = bool(len(iou) and iou.max() >= MATCH_IOU)
    if rng is not None and eps > 0 and rng.random() < eps:
        real = not real                                              # a wrong existence call
    if not real:
        S.drop(i)
        return
    if len(iou) and iou.max() >= MATCH_IOU:
        b = int(np.argmax(iou))
        s = float(np.clip(1.0 + rng.normal(0.0, sigma), 0.3, 3.0)) if (rng is not None and sigma > 0) else 1.0
        S.t[0][i] = float(g["long_near"][b]) * s
        S.t[1][i] = float(g["lat_min"][b]) * s
        S.t[2][i] = float(g["lat_max"][b]) * s
        S.t[3][i] = float(g["ttc"][b])
    # a phantom wrongly kept stays with its own geometry
    S.verified[i] = True


def refine(xyxy, cons, g, decide, v, pp, K: int, rng=None, eps=0.0, sigma=0.0):
    """The set after verifying up to K successive pivots; returns (set, number of verifications spent)."""
    S = Obj(xyxy, cons)
    spent = 0
    for _ in range(K):
        i = pivot_of(S, decide, v, pp)
        if i is None:
            break
        verify(S, i, g, rng, eps, sigma)
        spent += 1
    return S, spent


def verify_all(xyxy, cons, g) -> Obj:
    """Upper bound of verification alone: every detection checked (misses stay missed)."""
    S = Obj(xyxy, cons)
    i = 0
    while i < len(S):
        n0 = len(S)
        verify(S, i, g)
        if len(S) == n0:
            i += 1
    return S
