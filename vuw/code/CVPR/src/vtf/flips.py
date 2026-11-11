"""Per-frame escalation records: both branches, each consumer's decision, the pivotal element, and features.

The loop mirrors `rap.decision.build` (braking controller) and `65_planner_b_decision.build_b` (Planner B) line for
line, so the recomputed losses can be checked against DEEP's canonical tables (pre-registration gate V1), and adds what
an arbiter needs after escalation:

  * the difference between the two branches' detection sets (added, removed, and matched pairs whose geometry moved),
  * for each consumer, the element of that difference the decision change hinges on (the pivot), found by applying
    one element at a time to the cheap set and reverting one at a time in the full set,
  * features of the frame and of the pivot, computed from detections and the *monocular* lift only.

Anything computed from the reference scene state is written under a `gt_` prefix and is diagnostic only; `features.py`
refuses such columns.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import _apply_range_source
from rap.mono import box_iou
from rap.risk import RiskConfig

MATCH_IOU = 0.5          # full-to-cheap matching, as DEEP's matching of detections
SUPPORT_IOU = 0.3        # a lower-confidence or earlier detection "supports" an element at this overlap
MAX_RANGE = 80.0
TTC_CLIP = 20.0
CLASSES = ("vehicle", "person", "cyclist")
KINDS = ("add", "remove", "move", "multi")
IMG_SCALARS = ("img_bright_mean", "img_bright_std", "img_dark_frac", "img_bright_frac", "img_lap_var",
               "img_edge_density", "img_contrast_p95_p5", "motion_mean", "motion_p95", "motion_first")


class Branch:
    """One fidelity on one frame: every cached detection, and the kept ones with mono and consumer geometry."""

    def __init__(self, d: dict, geo: dict, thr: float, g, source: str, rng):
        k = d["conf"] >= thr
        self.all_xyxy = d["xyxy"].astype(float)
        self.all_conf = d["conf"].astype(float)
        self.xyxy = d["xyxy"][k].astype(float)
        self.conf = d["conf"][k].astype(float)
        self.binent = d["binent"][k].astype(float)
        self.entropy = d["entropy"][k].astype(float)
        self.coarse = d["coarse"][k]
        geo_k = {kk: vv[k] for kk, vv in geo.items()}
        self.mono = {kk: np.asarray(vv, float) for kk, vv in geo_k.items()}
        # exactly the arrays `rap.decision.build` hands to the consumers
        self.cons_raw = _apply_range_source(self.xyxy, geo_k, g, source, rng, 0.0)
        self.cons = tuple(np.asarray(a, float) for a in self.cons_raw)
        self.mono_t = (self.mono["z"], self.mono["lat_min"], self.mono["lat_max"], self.mono["ttc"])

    def __len__(self) -> int:
        return len(self.conf)


def match(full: Branch, cheap: Branch):
    """Greedy, full-confidence-ordered matching of full detections to cheap ones at image IoU >= 0.5."""
    nf, nc = len(full), len(cheap)
    pairs = []
    if nf and nc:
        iou = box_iou(full.xyxy, cheap.xyxy)
        used = np.zeros(nc, bool)
        for j in np.argsort(-full.conf, kind="stable"):
            row = np.where(used, -1.0, iou[j])
            i = int(np.argmax(row))
            if row[i] >= MATCH_IOU:
                used[i] = True
                pairs.append((i, int(j)))
    mf = {j for _, j in pairs}
    mc = {i for i, _ in pairs}
    added = [j for j in range(nf) if j not in mf]
    removed = [i for i in range(nc) if i not in mc]
    return pairs, added, removed


# ---- counterfactual sets over consumer geometry (z, lat_min, lat_max, ttc) -------------------------------------

def _row(t, i):
    return tuple(a[i:i + 1] for a in t)


def _cat(t, u):
    return tuple(np.concatenate([a, b]) for a, b in zip(t, u))


def _drop(t, i):
    return tuple(np.delete(a, i) for a in t)


def _replace(t, i, u, j):
    out = tuple(a.copy() for a in t)
    for a, b in zip(out, u):
        a[i] = b[j]
    return out


def apply_to_cheap(C, F, e):
    kind, i, j = e
    if kind == "add":
        return _cat(C, _row(F, j))
    if kind == "remove":
        return _drop(C, i)
    return _replace(C, i, F, j)


def revert_in_full(C, F, e):
    kind, i, j = e
    if kind == "add":
        return _drop(F, j)
    if kind == "remove":
        return _cat(F, _row(C, i))
    return _replace(F, j, C, i)


def elements(cheap: Branch, full: Branch, pairs, added, removed):
    els = [("add", -1, j) for j in added] + [("remove", i, -1) for i in removed]
    for i, j in pairs:
        a = np.array([x[i] for x in cheap.cons])
        b = np.array([x[j] for x in full.cons])
        if not np.allclose(a, b, rtol=0.0, atol=1e-6, equal_nan=True):
            els.append(("move", i, j))
    return els


def _elem_src(e, cheap, full):
    kind, i, j = e
    return (cheap, i) if kind == "remove" else (full, j)


def salience(e, cheap, full, v, pp) -> float:
    """Required deceleration of the element's object alone under the braking rule and mono geometry."""
    src, k = _elem_src(e, cheap, full)
    a, _ = P.required_decel(*_row(src.mono_t, k), v, pp)
    return float(a)


def find_pivot(cheap, full, els, decide, dec_full, v, pp):
    """(pivot element, kind, n_sufficient, n_necessary) for one consumer's decision change."""
    if not els:
        return None, "multi", 0, 0
    C, F = cheap.cons, full.cons
    suff = np.array([decide(apply_to_cheap(C, F, e)) == dec_full for e in els])
    nec = np.array([decide(revert_in_full(C, F, e)) != dec_full for e in els])
    rank = np.where(suff & nec, 0, np.where(suff, 1, np.where(nec, 2, 3)))
    best = rank.min()
    if best == 3:
        sal = [salience(e, cheap, full, v, pp) for e in els]
        e = els[int(np.argmax(sal))]
        return e, "multi", int(suff.sum()), int(nec.sum())
    cand = [e for e, r in zip(els, rank) if r == best]
    zs = [(_elem_src(e, cheap, full)[0].cons[0][_elem_src(e, cheap, full)[1]]) for e in cand]
    e = cand[int(np.argmin(zs))]
    return e, e[0], int(suff.sum()), int(nec.sum())


def _best_iou(box, xyxy, conf, lo=0.0, hi=np.inf):
    k = (conf >= lo) & (conf < hi)
    if not k.any():
        return 0.0, 0.0
    iou = box_iou(box[None, :], xyxy[k])[0]
    b = int(np.argmax(iou))
    return float(iou[b]), float(conf[k][b])


def elem_features(p: str, e, kind: str, cheap: Branch, full: Branch, prev1, prev2, img_w, img_h, v, pp) -> dict:
    """Features of one element, from its detection and its mono geometry; `p` is the column prefix."""
    out = {f"{p}{k}": float(kind == k) for k in KINDS}
    if e is None:
        out[f"{p}none"] = 1.0
        return out
    out[f"{p}none"] = 0.0
    src, k = _elem_src(e, cheap, full)
    other = full if e[0] == "remove" else cheap
    x1, y1, x2, y2 = src.xyxy[k]
    m = src.mono
    zg, zh = float(m["z_ground"][k]), float(m["z_height"][k])
    lg = (np.log(max(zg, 0.1) / max(zh, 0.1)) if np.isfinite(zg) and np.isfinite(zh) else np.nan)
    ttc = float(m["ttc"][k])
    oi, oc = _best_iou(src.xyxy[k], other.all_xyxy, other.all_conf)
    p1 = _best_iou(src.xyxy[k], *prev1)[0] if prev1 is not None else 0.0
    p2 = _best_iou(src.xyxy[k], *prev2)[0] if prev2 is not None else 0.0
    out.update({
        f"{p}conf": float(src.conf[k]), f"{p}binent": float(src.binent[k]), f"{p}entropy": float(src.entropy[k]),
        **{f"{p}cls_{c}": float(src.coarse[k] == c) for c in CLASSES},
        f"{p}box_h": (y2 - y1) / img_h, f"{p}box_w": (x2 - x1) / img_w, f"{p}bottom": y2 / img_h,
        f"{p}xc": (x1 + x2) / (2 * img_w) - 0.5,
        f"{p}border": float(x1 <= 2 or y1 <= 2 or x2 >= img_w - 3 or y2 >= img_h - 3),
        f"{p}z": float(m["z"][k]), f"{p}z_ground": zg, f"{p}z_height": zh, f"{p}log_zg_zh": lg,
        f"{p}abs_log_zg_zh": abs(lg) if np.isfinite(lg) else np.nan,
        f"{p}lat_c": float((m["lat_min"][k] + m["lat_max"][k]) / 2),
        f"{p}lat_hw": float((m["lat_max"][k] - m["lat_min"][k]) / 2),
        f"{p}ttc": float(np.clip(ttc, 0.0, TTC_CLIP)) if np.isfinite(ttc) else TTC_CLIP,
        f"{p}a_obj": salience(e, cheap, full, v, pp),
        f"{p}other_iou": oi, f"{p}other_conf": oc, f"{p}prev1_iou": p1, f"{p}prev2_iou": p2,
    })
    if e[0] == "move":
        i, j = e[1], e[2]
        zc = float(cheap.mono["z"][i])
        out[f"{p}dz_rel"] = (float(full.mono["z"][j]) - zc) / max(zc, 1.0)
        out[f"{p}pair_iou"] = float(box_iou(full.xyxy[j:j + 1], cheap.xyxy[i:i + 1])[0, 0])
        out[f"{p}dconf"] = float(full.conf[j] - cheap.conf[i])
    else:
        out[f"{p}dz_rel"] = out[f"{p}pair_iou"] = out[f"{p}dconf"] = 0.0
    return out


def elem_correct(e, cheap: Branch, full: Branch, g) -> float:
    """Diagnostic (reference state): is the full branch right about this element?"""
    if e is None:
        return np.nan
    kind, i, j = e
    gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4)))
    if kind == "add":
        return float(len(gt) > 0 and box_iou(full.xyxy[j:j + 1], gt).max() >= MATCH_IOU)
    if kind == "remove":
        return float(not (len(gt) > 0 and box_iou(cheap.xyxy[i:i + 1], gt).max() >= MATCH_IOU))
    if not len(gt):
        return 0.0
    iou = box_iou(full.xyxy[j:j + 1], gt)[0]
    if iou.max() < MATCH_IOU:
        return 0.0
    zt = float(g["long_near"][int(np.argmax(iou))])
    return float(abs(full.mono["z"][j] - zt) < abs(cheap.mono["z"][i] - zt))


def frame_features(cheap: Branch, full: Branch, pairs, added, removed, scal: dict, img_w: float) -> dict:
    half_w = P.PlannerParams().corridor_half_w

    def in_corr(br, idx):
        if not idx:
            return 0.0
        m = br.mono
        idx = np.asarray(idx)
        return float((P.in_corridor(m["lat_min"][idx], m["lat_max"][idx], half_w) & (m["z"][idx] <= MAX_RANGE)).sum())

    sub = []
    for j in added:
        sub.append(_best_iou(full.xyxy[j], cheap.all_xyxy, cheap.all_conf, 0.0, RiskConfig().op_conf)[0] >= SUPPORT_IOU)
    dz = [abs(full.mono["z"][j] - cheap.mono["z"][i]) / max(cheap.mono["z"][i], 1.0) for i, j in pairs]
    out = {
        "fr_n_cheap": float(len(cheap)), "fr_n_full": float(len(full)),
        "fr_n_add": float(len(added)), "fr_n_rem": float(len(removed)), "fr_n_match": float(len(pairs)),
        "fr_add_conf_max": float(full.conf[added].max()) if added else 0.0,
        "fr_add_conf_mean": float(full.conf[added].mean()) if added else 0.0,
        "fr_rem_conf_min": float(cheap.conf[removed].min()) if removed else np.nan,
        "fr_add_in_corr": in_corr(full, added), "fr_rem_in_corr": in_corr(cheap, removed),
        "fr_match_dz_rel_mean": float(np.mean(dz)) if dz else 0.0,
        "fr_add_sub_support": float(np.mean(sub)) if sub else np.nan,
    }
    out.update({f"fr_{k}": float(scal.get(k, np.nan)) for k in IMG_SCALARS})
    return out


def _plan_parts(cand: int):
    a_i, d_i = divmod(int(cand), len(B.D_LAT))
    return float(B.A_LON[a_i]), float(B.D_LAT[d_i])


def build(det_dir, cheap_mode: str, full_mode: str, seqs, adapter, source: str, v_causal=None,
          with_traj: bool = True, pb=None, cb=None) -> list[dict]:
    """One record per frame for one (dataset, geometry): braking and, if asked, Planner B.

    `v_causal(seq)` returns {frame: causal ego speed} for the features (DEEP's signals read it); the consumers read
    `adapter.speeds`, as the canonical tables do.
    """
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb = pb or B.PARAMS_B["static_obstacles"]
    cb = cb or B.COSTS_B["default"]
    rng = np.random.default_rng(0)
    rows = []
    for s in seqs:
        geom = adapter.geometry(s)
        speeds = adapter.speeds(s)
        vc = v_causal(s) if v_causal is not None else None
        c = DetCache(det_dir / cheap_mode / f"{s}.npz")
        f = DetCache(det_dir / full_mode / f"{s}.npz")
        prev_act = None
        prev_plan = None
        hist = []                                     # cheap detections of earlier frames, for temporal support
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            sel = geom["frame"] == fr
            g = geom[sel]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            dc, df = c.det(i), f.det(j)
            cheap = Branch(dc, c.geo(i), cfg.thr("cheap"), g, source, rng)
            full = Branch(df, f.geo(j), cfg.thr("full"), g, source, rng)
            scal = c.scalars(i)
            img_w, img_h = float(scal.get("img_w", 1242.0)), float(scal.get("img_h", 375.0))
            prev1 = hist[-1] if len(hist) >= 1 else None
            prev2 = hist[-2] if len(hist) >= 2 else None
            hist.append((dc["xyxy"].astype(float), dc["conf"].astype(float)))

            pairs, added, removed = match(full, cheap)
            els = elements(cheap, full, pairs, added, removed)
            rec = {"seq": s, "frame": fr, "v_ego": v,
                   "v_causal": float(vc[fr]) if vc is not None else v}
            rec.update(frame_features(cheap, full, pairs, added, removed, scal, img_w))

            # ---- braking controller, exactly as rap.decision.build with history="shared"
            a_c, _ = P.required_decel(*cheap.cons_raw, v, pp)
            a_f, _ = P.required_decel(*full.cons_raw, v, pp)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act_c, act_f = P.discrete_action(a_c, pp), P.discrete_action(a_f, pp)
            Jc = P.decision_cost(act_c, a_gt, prev_act, pp, cp)["J"]
            Jf = P.decision_cost(act_f, a_gt, prev_act, pp, cp)["J"]
            prev_act = act_c
            am_c, _ = P.required_decel(*cheap.mono_t, v, pp)
            am_f, _ = P.required_decel(*full.mono_t, v, pp)
            rec.update({"brake_act_c": act_c, "brake_act_f": act_f, "brake_J_c": Jc, "brake_J_f": Jf,
                        "brake_V": Jc - Jf, "brake_flip": int(act_c != act_f),
                        "brake_a_c_mono": am_c, "brake_a_f_mono": am_f,
                        "brake_dir": float(np.sign(act_f - act_c))})
            if act_c != act_f:
                decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)  # noqa: E731
                e, kind, ns, nn = find_pivot(cheap, full, els, decide, act_f, v, pp)
            else:
                e, kind, ns, nn = None, "none", 0, 0
            rec.update({"brake_n_suff": ns, "brake_n_nec": nn, "brake_n_elems": len(els)})
            rec.update(elem_features("brake_piv_", e, kind, cheap, full, prev1, prev2, img_w, img_h, v, pp))
            rec["gt_brake_piv_correct"] = elem_correct(e, cheap, full, g)

            # ---- Planner B, exactly as 65_planner_b_decision.build_b with history="shared"
            if with_traj:
                truth = (g["long_near"].astype(float), g["lat_min"].astype(float),
                         g["lat_max"].astype(float), g["ttc"].astype(float))
                pc = B.plan(*cheap.cons_raw, v, prev_plan, pb, cb)
                pf = B.plan(*full.cons_raw, v, prev_plan, pb, cb)
                JBc = B.executed_cost(pc, *truth, v, prev_plan, pb, cb)["J"]
                JBf = B.executed_cost(pf, *truth, v, prev_plan, pb, cb)["J"]
                prev_used = prev_plan
                prev_plan = pc
                alc, dlc = _plan_parts(pc)
                alf, dlf = _plan_parts(pf)
                rec.update({"traj_plan_c": pc, "traj_plan_f": pf, "traj_J_c": JBc, "traj_J_f": JBf,
                            "traj_V": JBc - JBf, "traj_flip": int(pc != pf),
                            "traj_alon_c": alc, "traj_alon_f": alf, "traj_dlat_c": dlc, "traj_dlat_f": dlf,
                            "traj_dir": float(np.sign(alc - alf)), "traj_dlat_change": abs(dlf - dlc)})
                if pc != pf:
                    decide_b = lambda t: B.plan(*t, v, prev_used, pb, cb)                    # noqa: E731
                    e, kind, ns, nn = find_pivot(cheap, full, els, decide_b, pf, v, pp)
                else:
                    e, kind, ns, nn = None, "none", 0, 0
                rec.update({"traj_n_suff": ns, "traj_n_nec": nn})
                rec.update(elem_features("traj_piv_", e, kind, cheap, full, prev1, prev2, img_w, img_h, v, pp))
                rec["gt_traj_piv_correct"] = elem_correct(e, cheap, full, g)

            # ---- salient element: the added or removed detection the braking rule would react to most (mono)
            ar = [x for x in els if x[0] in ("add", "remove")]
            if ar:
                sal = np.array([salience(x, cheap, full, v, pp) for x in ar])
                zz = np.array([_elem_src(x, cheap, full)[0].mono["z"][_elem_src(x, cheap, full)[1]] for x in ar])
                order = np.lexsort((zz, -sal))
                es = ar[int(order[0])]
                rec.update(elem_features("sal_", es, es[0], cheap, full, prev1, prev2, img_w, img_h, v, pp))
                rec["gt_sal_correct"] = elem_correct(es, cheap, full, g)
            else:
                rec.update(elem_features("sal_", None, "none", cheap, full, prev1, prev2, img_w, img_h, v, pp))
                rec["gt_sal_correct"] = np.nan
            rows.append(rec)
    return rows
