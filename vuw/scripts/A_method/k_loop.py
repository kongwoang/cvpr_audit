"""The frozen loop (vtf.speculative.run with guard=True, sequential) with switches, for the K16 / K17 / K21 variants.
With the defaults (field on, support crops, one closing speed, car probe 1.8 x 1.6 m, frozen grid, context 2, 3 rounds,
no extra crops) it is speculative.run step for step (same crops, same merge, same scene).
  field          the unseen-object field crop on / off (off = "support only")
  v_closes       closing speeds of the probe (footprint = bounding box of the union over speeds and probes)
  probes         probe sizes (width, height) in metres
  r_grid, y_grid ground grid of the field
  ctx            context factor of the support crops (2 = frozen)
  max_rounds     round cap
  fmargin        field crop enlarged by fmargin x its size on each side (0 = frozen)
  seed           crops added to the first round (K23 C3 temporal support: previous support objects missing in the draft)
  max_calls      K23 C5 / K18(B): at most this many detector rounds, the guard's grown crops deferred into the NEXT round
                 (detected together with that round's new crops; at the bound the current scene is returned);
                 None = the frozen loop (guard re-detections inside the round)
  extra          optional hook extra(crops, grown) -> list of crops, run once after the loop has stopped (detected with
                 the guard and merged like a further round; not counted as a round): equal-compute controls
Returns scene (xyxy, conf, coarse, mono, from_full), crops, grown crops, rounds, converged (False = stopped at the cap
while a further round would still add a crop).
"""
from __future__ import annotations

import numpy as np

from rap.mono import box_iou
from vtf import dsf
from vtf import speculative as SP

CAR = (SP.OBJ_W, SP.OBJ_H)
PED, TRUCK = (0.6, 1.8), (2.6, 3.6)


def field_crop_g(mono, v, cam, W, H, pp, v_closes, r_grid, y_grid, probes):
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    lo_u, lo_v, hi_u, hi_v = [], [], [], []
    for vc in v_closes:
        for (w, h) in probes:
            sens = dsf.insertion_grid(mono, v, r_grid, y_grid, w, pp, vc)
            if not sens.any():
                continue
            rr, yy = np.meshgrid(r_grid, y_grid, indexing="ij")
            rr, yy = rr[sens], yy[sens]
            vb = cy + fy * cam.cam_h / rr
            uc, hw = cx - fx * yy / rr, fx * w / (2 * rr)
            lo_u.append((uc - hw).min()); hi_u.append((uc + hw).max())
            lo_v.append((vb - fy * h / rr).min()); hi_v.append(vb.max())
    if not lo_u:
        return None
    return np.array([max(min(lo_u), 0), max(min(lo_v), 0), min(max(hi_u), W), min(max(hi_v), H)])


def box_crop_ctx(b, W, H, ctx=SP.CTX):
    x1, y1, x2, y2 = b
    cw, ch = (x2 - x1) * ctx, (y2 - y1) * ctx
    return np.array([max((x1 + x2) / 2 - cw / 2, 0), max((y1 + y2) / 2 - ch / 2, 0), min((x1 + x2) / 2 + cw / 2, W),
                     min((y1 + y2) / 2 + ch / 2, H)])


def run_g(cheap, v, cam, W, H, detect_crop, lift, pp, max_rounds=3, v_closes=(0.0,), r_grid=SP.R_GRID, y_grid=SP.Y_GRID,
          probes=(CAR,), field=True, ctx=SP.CTX, guard=True, extra=None, extra_gate=None, fmargin=0.0, seed=None,
          max_calls=None):
    if max_calls is not None:
        return run_bounded(cheap, v, cam, W, H, detect_crop, lift, pp, max_calls, v_closes, r_grid, y_grid, probes, field,
                           ctx, fmargin, seed)
    gdet = SP.guarded(detect_crop, W, H) if guard else None
    crops, grown, full = [], [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds = dict(cheap), 0

    first = [True]

    def new_crops(scene):
        new = []
        if field:
            fc = field_crop_g(scene["mono"], v, cam, W, H, pp, v_closes, r_grid, y_grid, probes)
            if fc is not None and fmargin:
                fc = margin(fc, fmargin, W, H)
            if fc is not None and not SP.covered(fc, crops):
                new.append(fc)
        if first[0] and seed:
            new += [c for c in seed if not SP.covered(c, crops + grown + new)]
        for k in SP.support(scene["mono"], v, pp):
            bc = box_crop_ctx(scene["xyxy"][k], W, H, ctx)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        return new

    def detect_merge(new, gate=None):
        nonlocal full
        acc = []
        for c in new:
            if gate is not None and not gate(c):
                continue
            acc.append(c)
            if gdet is None:
                d = detect_crop(c)
            else:
                d, ex = gdet(c)
                grown.extend(ex)
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops.extend(acc)
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable")
            iou = box_iou(full["xyxy"], full["xyxy"])
            keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            keep = np.sort(np.array(keep))
            full = {k: full[k][keep] for k in full}
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        return {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
    converged = False
    for _ in range(max_rounds):
        new = new_crops(scene)
        if not new:
            converged = True
            break
        rounds += 1
        first[0] = False
        scene = detect_merge(new)
    if not converged:
        converged = not new_crops(scene)
    if extra is not None:
        ex = extra(list(crops), list(grown))
        if ex:
            scene = detect_merge(ex, extra_gate)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops, grown, rounds, converged


def margin(c, f, W, H):
    w, h = c[2] - c[0], c[3] - c[1]
    return np.array([max(c[0] - f * w, 0), max(c[1] - f * h, 0), min(c[2] + f * w, W), min(c[3] + f * h, H)])


def run_bounded(cheap, v, cam, W, H, detect_crop, lift, pp, N, v_closes=(0.0,), r_grid=SP.R_GRID, y_grid=SP.Y_GRID,
                probes=(CAR,), field=True, ctx=SP.CTX, fmargin=0.0, seed=None, max_grow=2):
    """at most N detector rounds; a round detects its new crops and the grown crops queued by the guard in the previous
    round (no in-round re-detection); at the bound the current merged scene is returned. Same merge as the frozen loop.
    Returns (scene, crops, grown, rounds, converged) like run_g (converged = stopped because nothing was requested)."""
    crops, grown, pending = [], [], []
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds, converged = dict(cheap), 0, False
    while True:
        new = []
        if field:
            fc = field_crop_g(scene["mono"], v, cam, W, H, pp, v_closes, r_grid, y_grid, probes)
            if fc is not None and fmargin:
                fc = margin(fc, fmargin, W, H)
            if fc is not None and not SP.covered(fc, crops):
                new.append(fc)
        if rounds == 0 and seed:
            new += [c for c in seed if not SP.covered(c, crops + grown + new)]
        for k in SP.support(scene["mono"], v, pp):
            bc = box_crop_ctx(scene["xyxy"][k], W, H, ctx)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        gs = [SP.grow(b, c, W, H) for b, c, _ in pending]
        if not new and not gs:
            converged = True
            break
        if rounds >= N:
            break
        rounds += 1
        nxt = []
        for c in new:
            d = {k: np.asarray(x) for k, x in detect_crop(c).items()}
            if len(d["conf"]):
                top, other = SP.truncated(d["xyxy"], c, W, H, top_only=True)
                tr = other | top
                nxt += [(d["xyxy"][i], c, 0) for i in np.flatnonzero(tr)]
                d = {k: x[~tr] for k, x in d.items()}
            full = {k: np.concatenate([full[k], d[k]]) for k in full}
        for (b, _, step), g in zip(pending, gs):
            dg = {k: np.asarray(x) for k, x in detect_crop(g).items()}
            if not len(dg["conf"]):
                continue
            ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
            iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
            io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
            j = int(np.argmax(io))
            if io[j] < 0.5:
                continue
            if not SP.truncated(dg["xyxy"][j:j + 1], g, W, H)[0] or step == max_grow - 1:
                full = {k: np.concatenate([full[k], dg[k][j:j + 1]]) for k in full}
            else:
                nxt.append((dg["xyxy"][j], g, step + 1))
        crops += new
        grown += gs
        pending = nxt
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
    return scene, crops, grown, rounds, converged
