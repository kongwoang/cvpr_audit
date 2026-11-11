"""Closed-form decision-sensitivity field and pivots for DEEP's braking controller (a max-composable rule).

The braking controller acts on the largest per-object required deceleration, so both questions the speculative loop
asks have closed forms, with no planner calls:

  * insertion sensitivity: an obstacle inserted at (r, y) changes the action iff it lies in the corridor and its own
    requirement reaches the threshold of the next stronger action;
  * pivots: an object is a pivot iff removing it changes the action, which depends only on the two largest demands.

Both mirror `rap.planner.required_decel` term by term (including the closing-speed term with an infinite TTC and the
clip at `a_cap`), so they agree with probing exactly; `tests/test_dsf.py` checks this.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P


def _insert_demand(r: np.ndarray, v: float, p: P.PlannerParams, v_close: float = 0.0) -> np.ndarray:
    """Required deceleration of an obstacle at range r alone, closing at v_close (0: static, the probe object with
    ttc = inf); a closing object's TTC is r / v_close, exactly as the controller turns TTC into a closing speed."""
    if v <= 0.1:
        return np.zeros_like(r, dtype=float)
    gap = np.maximum(r - p.standoff - v * p.reaction_t, 0.05)
    a = v ** 2 / (2.0 * gap)
    if p.use_ttc:
        ttc = np.clip(r / v_close, 0.1, 1e3) if v_close > 0 else np.full_like(r, 1e3, dtype=float)
        v_rel = np.clip(r / ttc, 0.0, 60.0)                      # ttc = inf is clipped to 1e3 in the controller
        a = np.maximum(a, v_rel ** 2 / (2.0 * np.maximum(r - p.standoff, 0.05)))
    a = np.clip(a, 0.0, p.a_cap)
    return np.where(r <= p.max_range, a, 0.0)


def insertion_grid(mono_t, v: float, r_grid, y_grid, obj_w: float, p: P.PlannerParams,
                   v_close: float = 0.0) -> np.ndarray:
    """Boolean (len(r_grid), len(y_grid)): does inserting the probe object there change the braking action?"""
    a_cur, _ = P.required_decel(*mono_t, v, p)
    act = P.discrete_action(a_cur, p)
    if act == P.HARD_BRAKE:
        return np.zeros((len(r_grid), len(y_grid)), bool)
    thr_next = p.thr_decel if act == P.KEEP else p.thr_hard
    a_ins = _insert_demand(np.asarray(r_grid, float), v, p, v_close)           # (R,)
    y = np.asarray(y_grid, float)
    in_corr = (y - obj_w / 2 < p.corridor_half_w) & (y + obj_w / 2 > -p.corridor_half_w)
    return (a_ins[:, None] >= thr_next) & in_corr[None, :]


def brake_pivots(cons, v: float, p: P.PlannerParams) -> np.ndarray:
    """Indices of objects whose removal alone changes the braking action (O(n), no planner calls)."""
    a_max, a_obj = P.required_decel(*cons, v, p)
    n = len(a_obj)
    if n == 0:
        return np.zeros(0, int)
    act = P.discrete_action(a_max, p)
    order = np.argsort(-a_obj, kind="stable")
    second = a_obj[order[1]] if n > 1 else 0.0
    top = order[0]
    # removing any object but the unique maximum leaves the maximum unchanged
    if P.discrete_action(second, p) != act and a_obj[top] > second:
        return np.array([top])
    return np.zeros(0, int)


# ---- Planner B: a decomposable (OR / min over objects) receding-horizon planner -------------------------------------
# Its candidate cost depends on the obstacle set only through  coll = OR_i coll_i  (per step) and  clear = min_i clear_i,
# so the effect of inserting or deleting objects is computed from per-object arrays without re-running the planner.

def _b_per_object(z, lo, hi, ttc, v, p):
    """Per-object collision steps (na, nd, n, N) and clearance (na, nd, N), as rap.planner_b._simulate before its
    reduction over objects (same filtering of far/non-finite objects: they get no collision and infinite clearance)."""
    from rap import planner_b as Bm
    s, _, y, t = Bm._rollout(v, p)
    na, nd, n = len(Bm.A_LON), len(Bm.D_LAT), s.shape[1]
    z = np.asarray(z, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float); ttc = np.asarray(ttc, float)
    N = z.size
    keep = np.isfinite(z) & (z <= p.max_range)
    coll = np.zeros((na, nd, n, N), bool)
    clear = np.full((na, nd, N), np.inf)
    if not keep.any():
        return coll, clear
    zk, lok, hik, tk = z[keep], lo[keep], hi[keep], ttc[keep]
    if p.obstacle_closes:
        rate = np.zeros_like(zk)
        good = np.isfinite(tk) & (tk > 1e-3)
        rate[good] = -zk[good] / tk[good]
        v_obs = np.maximum(v + rate, 0.0)
    else:
        v_obs = np.zeros_like(zk)
    zt = zk[None, :] + v_obs[None, :] * t[:, None]
    gap = zt[None, None, :, :] - s[:, None, :, None]
    reach = gap <= p.front_margin
    lat_gap = np.minimum(hik[None, None, :] - (y[:, :, None] - p.ego_half_w), (y[:, :, None] + p.ego_half_w) - lok[None, None, :])
    hit = reach & (lat_gap > 0.0)[None]
    gaps = np.where(np.broadcast_to(reach, hit.shape), np.broadcast_to(-lat_gap[None], hit.shape), np.inf)
    coll[..., keep] = hit
    clear[..., keep] = gaps.min(axis=2)
    return coll, clear


def _b_cost(coll, clear, v, prev, p, c):
    """rap.planner_b._candidate_costs from aggregated coll (na, nd, n, ...) and clear (na, nd, ...); extra trailing axes
    are vectorised over (e.g. one per probe)."""
    from rap import planner_b as Bm
    n = coll.shape[2]
    disc = c.discount ** np.arange(n)
    J = c.w_collision * (coll * disc.reshape((1, 1, n) + (1,) * (coll.ndim - 3))).max(axis=2)
    cl = np.where(np.isfinite(clear), clear, c.clear_req)
    short = np.clip(c.clear_req - cl, 0.0, 2.0 * c.clear_req)
    J = J + c.w_clearance * short ** 2
    s, _, _, _ = Bm._rollout(v, p)
    v_ref = max(v, p.v_ref_floor)
    ref = v_ref * p.horizon
    base = c.w_progress * ((ref - s[:, -1]) / ref)[:, None] ** 2
    base = base + c.w_accel * (Bm.A_LON ** 2 * p.horizon)[:, None]
    base = base + c.w_lateral * (Bm.D_LAT ** 2)[None, :]
    if prev is not None:
        pa, pd_ = divmod(int(prev), len(Bm.D_LAT))
        base = base + c.w_switch * ((Bm.A_LON[:, None] - Bm.A_LON[pa]) ** 2 / 4.0 + (Bm.D_LAT[None, :] - Bm.D_LAT[pd_]) ** 2)
    return J + base[(slice(None), slice(None)) + (None,) * (J.ndim - 2)]


def _b_plan_from(J):
    na, nd = J.shape[:2]
    return J.reshape(na * nd, -1).argmin(axis=0) if J.ndim > 2 else int(np.argmin(J))


def planner_b_insertion_grid(t, v, prev, r_grid, y_grid, obj_w, p, c) -> np.ndarray:
    """Does inserting the probe object at (r, y) change Planner B's plan? Exact, one vectorised evaluation."""
    coll0, clear0 = _b_per_object(*t, v, p)
    base_coll, base_clear = coll0.any(axis=3), clear0.min(axis=2) if clear0.shape[2] else np.full(coll0.shape[:2], np.inf)
    base_plan = _b_plan_from(_b_cost(base_coll, base_clear, v, prev, p, c))
    rr, yy = np.meshgrid(np.asarray(r_grid, float), np.asarray(y_grid, float), indexing="ij")
    pz, plo, phi = rr.ravel(), (yy - obj_w / 2).ravel(), (yy + obj_w / 2).ravel()
    cp_, clp = _b_per_object(pz, plo, phi, np.full(pz.size, np.inf), v, p)            # (na,nd,n,P), (na,nd,P)
    coll = base_coll[..., None] | cp_
    clear = np.minimum(base_clear[..., None], clp)
    plans = _b_plan_from(_b_cost(coll, clear, v, prev, p, c))
    return (plans != base_plan).reshape(rr.shape)


def planner_b_pivots(t, v, prev, p, c) -> np.ndarray:
    """Objects whose removal alone changes Planner B's plan (exact; leave-one-out OR/min over per-object arrays)."""
    coll0, clear0 = _b_per_object(*t, v, p)
    N = coll0.shape[3]
    if N == 0:
        return np.zeros(0, int)
    base = _b_plan_from(_b_cost(coll0.any(axis=3), clear0.min(axis=2), v, prev, p, c))
    out = []
    for j in range(N):
        m = np.arange(N) != j
        cj = coll0[..., m].any(axis=3)
        clj = clear0[..., m].min(axis=2) if m.any() else np.full(coll0.shape[:2], np.inf)
        if _b_plan_from(_b_cost(cj, clj, v, prev, p, c)) != base:
            out.append(j)
    return np.array(out, int)
