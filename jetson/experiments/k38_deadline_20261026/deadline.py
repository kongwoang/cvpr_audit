"""K38-1: the frozen v6-S loop (vtf.speculative.run, verbatim) and the K31 B2K-A port (bounded_seed.bounded_keep,
verbatim) with two hooks only: stop_fn(rounds_done) -> True stops before the next round (v6-S: the merged scene of the
last completed round; B2K-A: exactly as reaching its cap, pending truncated detections retained); mark(event, r)
records times. With stop_fn None / never True the outputs are identical to the originals."""
import numpy as np
from rap.mono import box_iou
from vtf import speculative as SP
from vtf.speculative import (guarded, guarded_batch, corridor_filter, field_crop, covered, support, box_crop,
                             centres_in, OBJ_H)


def run(cheap, v, cam, img_w, img_h, detect_crop, lift, pp, max_rounds=3, v_close=0.0, r_grid=None, guard=False,
        obj_h=OBJ_H, detect_many=None, batch_round=False, stop_fn=None, mark=None):
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
        if stop_fn is not None and stop_fn(rounds):                  # K38: deadline check before the next round
            break
        if mark is not None:
            mark("top", rounds)
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
        if mark is not None:
            mark("end", rounds)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops + grown, rounds


def bounded_keep(cheap, v, cam, W, H, call, lift, pp, v_close, N, max_grow=2, keep_pending=True, seed=None, stop_fn=None, mark=None):
    from rap.mono import box_iou
    crops, grown, pending = [], [], []                  # pending: (truncated box, crop that cut it, growth step)
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, calls, hit = dict(cheap), 0, False
    def merge_final(full_):
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full_["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full_["xyxy"]).max(1) >= 0.5
        fz = lift(full_) if len(full_["conf"]) else tuple(np.zeros(0) for _ in range(4))
        return {"xyxy": np.r_[cheap["xyxy"][~out], full_["xyxy"]], "conf": np.r_[cheap["conf"][~out], full_["conf"]],
                "coarse": np.r_[cheap["coarse"][~out], full_["coarse"]],
                "mono": tuple(np.r_[a[~out], b] for a,b in zip(cheap["mono"],fz)),
                "from_full": np.r_[np.zeros(int((~out).sum()),bool), np.ones(len(full_["conf"]),bool)]}

    while True:
        new = []
        fc = SP.field_crop(scene["mono"], v, cam, W, H, pp, v_close)
        if fc is not None and not SP.covered(fc, crops):
            new.append(fc)
        if calls==0 and seed:
            new += [c for c in seed if not SP.covered(c,crops+grown+new)]
        for k in SP.support(scene["mono"], v, pp):
            bc = SP.box_crop(scene["xyxy"][k], W, H)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        gs = [SP.grow(b, c, W, H) for b, c, *_ in pending]
        if not new and not gs:
            break
        stopped = stop_fn is not None and stop_fn(calls)               # K38: deadline = the cap is reached now
        if calls >= N or stopped:
            hit = True
            if keep_pending and pending:
                add = {"xyxy": np.array([p[0] for p in pending],float).reshape(-1,4),
                       "conf": np.array([p[3] for p in pending],float),
                       "coarse": np.array([p[4] for p in pending]).astype("<U8")}
                retained = {k: np.concatenate([full[k],add[k]]) for k in full}
                if len(retained["conf"]) > 1:
                    order = np.argsort(-retained["conf"],kind="stable")
                    iou = box_iou(retained["xyxy"],retained["xyxy"]); keep = []
                    for i in order:
                        if not keep or iou[i,keep].max() < 0.5: keep.append(i)
                    retained = {k:x[np.sort(np.array(keep))] for k,x in retained.items()}
                scene = merge_final(retained)
            break
        if mark is not None:
            mark("top", calls)
        res = call(new + gs)
        calls += 1
        nxt = []
        for c, d in zip(new, res[:len(new)]):
            d = {k: np.asarray(x) for k, x in d.items()}
            if len(d["conf"]):
                top, other = SP.truncated(d["xyxy"], c, W, H, top_only=True)
                tr = other | top
                nxt += [(d["xyxy"][i], c, 0, d["conf"][i], d["coarse"][i]) for i in np.flatnonzero(tr)]
                d = {k: x[~tr] for k, x in d.items()}
            full = {k: np.concatenate([full[k], d[k]]) for k in full}
        for (b, _, step, _, _), g, dg in zip(pending, gs, res[len(new):]):
            dg = {k: np.asarray(x) for k, x in dg.items()}
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
                nxt.append((dg["xyxy"][j], g, step + 1, dg["conf"][j], dg["coarse"][j]))
        crops += new
        grown += gs
        pending = nxt
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable"); iou = box_iou(full["xyxy"], full["xyxy"]); keep = []
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
        if mark is not None:
            mark("end", calls)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, calls, hit
