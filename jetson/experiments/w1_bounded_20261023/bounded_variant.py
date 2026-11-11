"""Verbatim frozen K31 B2K-A function; KL is bound by the sandbox harness."""
import numpy as np
from vtf import speculative as SP
KL = None

def run_bounded_keep(cheap, v, cam, W, H, detect_crop, lift, pp, N, mode, v_closes=(0.0,), r_grid=SP.R_GRID, y_grid=SP.Y_GRID,
                     probes=None, field=True, ctx=SP.CTX, max_grow=2):
    """K31 variant B2K (pre-registered): k_loop.run_bounded (verbatim logic) whose pending edge-truncated crop detections are
    not dropped at the bound: mode "A" adds them to the merged scene as detected, mode "B" restores the draft box with the
    largest IoU (>= 0.3) to each of them. No extra detector call."""
    from rap.mono import box_iou
    probes = probes or (KL.CAR,)
    crops, grown, pending = [], [], []                  # pending: (box, crop, growth step, conf, coarse)
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds, converged = dict(cheap), 0, False

    def merged(full_, restore=None):
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full_["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full_["xyxy"]).max(1) >= 0.5
        if restore is not None:
            out &= ~restore
        fz = lift(full_) if len(full_["conf"]) else tuple(np.zeros(0) for _ in range(4))
        return {"xyxy": np.r_[cheap["xyxy"][~out], full_["xyxy"]], "conf": np.r_[cheap["conf"][~out], full_["conf"]],
                "coarse": np.r_[cheap["coarse"][~out], full_["coarse"]],
                "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full_["conf"]), bool)]}

    def nms(f_):
        if len(f_["conf"]) <= 1:
            return f_
        o = np.argsort(-f_["conf"], kind="stable"); iou = box_iou(f_["xyxy"], f_["xyxy"]); keep = []
        for i in o:
            if not keep or iou[i, keep].max() < 0.5:
                keep.append(i)
        return {k: f_[k][np.sort(np.array(keep))] for k in f_}
    while True:
        new = []
        if field:
            fc = KL.field_crop_g(scene["mono"], v, cam, W, H, pp, v_closes, r_grid, y_grid, probes)
            if fc is not None and not SP.covered(fc, crops):
                new.append(fc)
        for k in SP.support(scene["mono"], v, pp):
            bc = KL.box_crop_ctx(scene["xyxy"][k], W, H, ctx)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        gs = [SP.grow(b, c, W, H) for b, c, *_ in pending]
        if not new and not gs:
            converged = True
            break
        if rounds >= N:
            if pending and mode == "A":
                add = {"xyxy": np.array([p_[0] for p_ in pending], float).reshape(-1, 4), "conf": np.array([p_[3] for p_ in pending], float),
                       "coarse": np.array([p_[4] for p_ in pending]).astype("<U8")}
                scene = merged(nms({k: np.concatenate([full[k], add[k]]) for k in full}))
            elif pending and mode == "B" and len(cheap["xyxy"]):
                M = box_iou(np.asarray(cheap["xyxy"], float), np.array([p_[0] for p_ in pending], float).reshape(-1, 4))
                restore = np.zeros(len(cheap["xyxy"]), bool)
                for j in range(M.shape[1]):
                    i = int(np.argmax(M[:, j]))
                    if M[i, j] >= 0.3:
                        restore[i] = True
                scene = merged(full, restore)
            break
        rounds += 1
        nxt = []
        for c in new:
            d = {k: np.asarray(x) for k, x in detect_crop(c).items()}
            if len(d["conf"]):
                top, other = SP.truncated(d["xyxy"], c, W, H, top_only=True)
                tr = other | top
                nxt += [(d["xyxy"][i], c, 0, d["conf"][i], d["coarse"][i]) for i in np.flatnonzero(tr)]
                d = {k: x[~tr] for k, x in d.items()}
            full = {k: np.concatenate([full[k], d[k]]) for k in full}
        for (b, _, step, _, _), g in zip(pending, gs):
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
                nxt.append((dg["xyxy"][j], g, step + 1, dg["conf"][j], dg["coarse"][j]))
        crops += new
        grown += gs
        pending = nxt
        full = nms(full)
        scene = merged(full)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops, grown, rounds, converged
