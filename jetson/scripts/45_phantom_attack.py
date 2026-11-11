#!/usr/bin/env python
"""D38 / D67: the decision-sensitivity field as a threat model, and speculative verification as a defence.

Threat model: the attacker controls the CHEAP detector's output on a frame (e.g. a patch or spoof tuned to the
low-resolution model) and can add ONE phantom object or delete ONE detected object. The FULL model is not fooled
(transfer 0; the realistic range lies between this and full transfer, where no verifier can help).
Placements:
  phantom_dsf     at the grid cell whose insertion raises the braking demand most (the field's worst cell = the
                  minimax single-phantom attack, D67)
  phantom_random  at a uniformly drawn grid cell
  remove_support  delete the object with the highest demand (the decision's support)
  remove_random   delete a uniformly drawn detected object
Pipelines: cheap only; speculative (proved loop, exact-locality proxy with cached FULL detections, v_close 10);
uniform full (unaffected by construction). Metric: extra decision cost ΔJ caused by the attack (share of the
all-cheap loss without attack) and attack success rate (action changed in the attacker's direction). KITTI mono,
all DEEP units (8,008 frames).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import dsf, runmeta, speculative as SP

V_CLOSE = 10.0
pp, cp = P.PlannerParams(), P.CostParams()


def phantom(t, xyxy, conf, coarse, r, y, cam):
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    vb, uc, hw, h = cy + fy * cam.cam_h / r, cx - fx * y / r, fx * SP.OBJ_W / (2 * r), fy * SP.OBJ_H / r
    box = np.array([[uc - hw, vb - h, uc + hw, vb]])
    t2 = (np.r_[t[0], r], np.r_[t[1], y - SP.OBJ_W / 2], np.r_[t[2], y + SP.OBJ_W / 2], np.r_[t[3], np.inf])
    return t2, np.concatenate([xyxy.reshape(-1, 4), box]), np.r_[conf, 0.9], np.concatenate([coarse, np.array(["vehicle"])])


def main():
    rng = np.random.default_rng(0)
    dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    rows = []
    rr, yy = np.meshgrid(SP.R_GRID, SP.Y_GRID, indexing="ij")
    for s in sorted(set(dv.seq)):
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        prev_act = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            sc = c.scalars(i)
            W, H = float(sc["img_w"]), float(sc["img_h"])

            def pack(cache, idx):
                d, gg = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= 0.25
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "mono": tuple(np.asarray(gg[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))}

            ch, fu = pack(c, i), pack(f, f.index[fr])

            def detect_crop(crop):
                m = SP.centres_in(fu["xyxy"], [crop])
                return {"xyxy": fu["xyxy"][m], "conf": fu["conf"][m], "coarse": fu["coarse"][m]}

            def lift(full):
                idx = [int(np.flatnonzero((fu["xyxy"] == b).all(1))[0]) for b in full["xyxy"]]
                return tuple(a[idx] for a in fu["mono"])

            act = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            a_c = act(ch["mono"])
            base_spec = act(SP.run(ch, v, cam, W, H, detect_crop, lift, pp, v_close=V_CLOSE)[0]["mono"])
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c), "J_f": cost(act(fu["mono"])), "J_s": cost(base_spec)}
            # attacks on the cheap scene
            attacks = {}
            # phantom at the worst cell: the cell whose inserted object alone demands the most (in the corridor)
            cells = [(r, y) for r in SP.R_GRID for y in SP.Y_GRID if abs(y) - SP.OBJ_W / 2 < pp.corridor_half_w]
            dm = [P.required_decel(np.array([r]), np.array([y - SP.OBJ_W / 2]), np.array([y + SP.OBJ_W / 2]),
                                   np.array([np.inf]), v, pp)[0] for r, y in cells]
            r0, y0 = cells[int(np.argmax(dm))]
            attacks["phantom_dsf"] = ("up", phantom(ch["mono"], ch["xyxy"], ch["conf"], ch["coarse"], r0, y0, cam))
            r1, y1 = SP.R_GRID[rng.integers(len(SP.R_GRID))], SP.Y_GRID[rng.integers(len(SP.Y_GRID))]
            attacks["phantom_random"] = ("up", phantom(ch["mono"], ch["xyxy"], ch["conf"], ch["coarse"], r1, y1, cam))
            n = len(ch["conf"])
            if n:
                _, a_obj = P.required_decel(*ch["mono"], v, pp)
                for name, k in (("remove_support", int(np.argmax(a_obj))), ("remove_random", int(rng.integers(n)))):
                    m = np.arange(n) != k
                    attacks[name] = ("down", (tuple(a[m] for a in ch["mono"]), ch["xyxy"][m], ch["conf"][m], ch["coarse"][m]))
            for name, (direction, (t2, bx2, cf2, co2)) in attacks.items():
                att = {"xyxy": bx2, "conf": cf2, "coarse": co2, "mono": t2}
                a_ac = act(t2)
                a_as = act(SP.run(att, v, cam, W, H, detect_crop, lift, pp, v_close=V_CLOSE)[0]["mono"])
                succ = (lambda a, a0: a > a0) if direction == "up" else (lambda a, a0: a < a0)
                rec.update({f"J_c_{name}": cost(a_ac), f"J_s_{name}": cost(a_as),
                            f"succ_c_{name}": int(succ(a_ac, a_c)), f"succ_s_{name}": int(succ(a_as, base_spec))})
            rows.append(rec)
            prev_act = a_c
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(runmeta.FINAL / "d38_phantom_attack_kitti.pkl")
    L = d.J_c.sum()
    out = []
    for name in ("phantom_dsf", "phantom_random", "remove_support", "remove_random"):
        q = d.dropna(subset=[f"J_c_{name}"])
        out.append({"attack": name, "frames": len(q),
                    "success_cheap": q[f"succ_c_{name}"].mean(), "success_spec": q[f"succ_s_{name}"].mean(),
                    "dJ_cheap": (q[f"J_c_{name}"] - q.J_c).sum() / L, "dJ_spec": (q[f"J_s_{name}"] - q.J_s).sum() / L})
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / "d38_phantom_attack_kitti.csv", index=False)
    print(f"frames {len(d)}; no attack: G_full {(d.J_c - d.J_f).sum() / L:.4f}, G_spec {(d.J_c - d.J_s).sum() / L:.4f}")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
