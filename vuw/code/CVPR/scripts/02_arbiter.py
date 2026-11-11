#!/usr/bin/env python
"""Phase 0H stage 2: fit, select and score the arbiter on the ten core cells, exactly as pre-registered.

Inputs: the latest `flips_<dataset>_<geometry>` runs (stage 1, gate V1 passed) and DEEP's canonical decision values.
Outputs (run directory and results/final): per-cell test table, bootstrap draws, the verdict, and the secondary
budgeted comparison with DEEP's cached allocators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vtf import arbiter as A
from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
RAW_RAP = runmeta.RAP_REPO / "results" / "raw"
CELLS = [
    ("KITTI", "mono", "brake", "J"), ("KITTI", "oracle", "brake", "J"),
    ("KITTI", "mono", "traj", "JB"), ("KITTI", "oracle", "traj", "JB"),
    ("nuScenes", "mono", "brake", "J"), ("nuScenes", "oracle", "brake", "J"),
    ("nuScenes", "mono", "plan_ade", "JC_ade"), ("nuScenes", "mono", "plan_fde", "JC_fde"),
    ("nuScenes", "oracle", "plan_ade", "JC_ade"), ("nuScenes", "oracle", "plan_fde", "JC_fde"),
]
FAMILY = {"brake": "brake", "traj": "traj", "plan_ade": "plan", "plan_fde": "plan"}
MIN_AFFECTED = 10
NBOOT = 1000


def cell_name(c):
    return f"{c[0]} {c[1]} {c[2]}"


def latest_flips(ds, geo) -> pd.DataFrame:
    runs = sorted(runmeta.RAW.glob(f"*_flips_{ds}_{geo}"))
    for r in reversed(runs):
        g = r / "gate_v1.json"
        if g.exists() and all(v["pass"] for v in json.loads(g.read_text()).values()):
            return pd.read_pickle(r / f"flips__{ds}__{geo}.pkl"), r
    raise SystemExit(f"no flips run with gate V1 passed for {ds}/{geo}")


def feature_cols(df: pd.DataFrame, fam: str) -> list[str]:
    common = ["v_causal"] + [c for c in df.columns if c.startswith("fr_")]
    if fam == "brake":
        extra = ["brake_act_c", "brake_act_f", "brake_dir", "brake_a_c_mono", "brake_a_f_mono",
                 "brake_n_suff", "brake_n_nec", "brake_n_elems"] + [c for c in df.columns if c.startswith("brake_piv_")]
    elif fam == "traj":
        extra = ["traj_alon_c", "traj_alon_f", "traj_dlat_c", "traj_dlat_f", "traj_dir", "traj_dlat_change",
                 "traj_n_suff", "traj_n_nec", "brake_n_elems"] + [c for c in df.columns if c.startswith("traj_piv_")]
    else:
        extra = [c for c in df.columns if c.startswith("sal_")]
    cols = common + extra
    A.assert_deployable(cols)
    return cols


RULE_FEATS = {
    "brake": ["brake_piv_conf", "brake_piv_other_iou", "brake_piv_prev1_iou", "brake_piv_abs_log_zg_zh",
              "brake_dir", "fr_n_add"],
    "traj": ["traj_piv_conf", "traj_piv_other_iou", "traj_piv_prev1_iou", "traj_piv_abs_log_zg_zh",
             "traj_dir", "fr_n_add"],
    # the planner cells have no discrete flip direction; the salient element's being an addition stands in for it
    "plan": ["sal_conf", "sal_other_iou", "sal_prev1_iou", "sal_abs_log_zg_zh", "sal_add", "fr_n_add"],
}
GT_COL = {"brake": "gt_brake_piv_correct", "traj": "gt_traj_piv_correct", "plan": "gt_sal_correct"}


def build_cells():
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track.isin(["KITTI", "nuScenes"])].copy()       # nuPlan rows have no frame index
    dv["frame"] = dv.frame.astype(int)
    flips, runs = {}, {}
    out = {}
    for c in CELLS:
        ds, geo, sysname, target = c
        if (ds, geo) not in flips:
            flips[(ds, geo)], runs[(ds, geo)] = latest_flips(ds, geo)
        f = flips[(ds, geo)]
        ref = dv[(dv.track == ds) & (dv.geometry == geo) & (dv.system == sysname) & (dv.target == target)]
        d = ref[["seq", "frame", "unit", "split", "J_cheap", "V"]].merge(f, on=["seq", "frame"], how="left",
                                                                          validate="one_to_one")
        assert d.fr_n_cheap.notna().all(), f"{c}: frames without a flip record"
        fam = FAMILY[sysname]
        if fam in ("brake", "traj"):
            own = d[f"{fam}_V"].to_numpy(float)
            assert np.allclose(own, d.V.to_numpy(float), atol=1e-9), f"{c}: V differs from the canonical value"
            d["arb"] = d[f"{fam}_flip"].astype(bool)
        else:
            d["arb"] = np.abs(d.V.to_numpy(float)) > A.EPS
        d["unit_key"] = ds + ":" + d.unit.astype(str)
        d["ind_nuscenes"] = float(ds == "nuScenes")
        d["ind_oracle"] = float(geo == "oracle")
        d["cell"] = cell_name(c)
        out[c] = d
    return out, {f"{k[0]}/{k[1]}": str(v) for k, v in runs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    args = ap.parse_args()
    run = runmeta.new_run("arbiter", vars(args))
    cells, flip_runs = build_cells()
    (run / "inputs.json").write_text(json.dumps({"flip_runs": flip_runs, "decision_values": str(DV)}, indent=2))

    fams = {}
    for c, d in cells.items():
        fams.setdefault(FAMILY[c[2]], []).append(c)

    results, test_accept = [], {}
    for c, d in cells.items():
        fam = FAMILY[c[2]]
        cols = feature_cols(d, fam)
        pcols = cols + ["ind_nuscenes", "ind_oracle"]
        nt = d[(d.split != "test") & d.arb].reset_index(drop=True)
        te = d[d.split == "test"].reset_index(drop=True)
        pool = pd.concat([cells[o][(cells[o].split != "test") & cells[o].arb] for o in fams[fam]],
                         ignore_index=True)
        tmask = (pool.cell == cell_name(c)).to_numpy()

        # ---- selection by grouped 5-fold CV over the non-test units
        sel = {}
        for model, scope in A.CONFIGS:
            if scope == "cell":
                g, _ = A.oof_gain(model, nt, np.ones(len(nt), bool), cols)
            else:
                g, _ = A.oof_gain(model, pool, tmask, pcols)
            sel[f"{model}/{scope}"] = g
        best = max(sel, key=lambda k: (round(sel[k], 12), -list(sel).index(k)))
        model, scope = best.split("/")
        use_cols = cols if scope == "cell" else pcols
        train = nt if scope == "cell" else pool
        m = A.fit(model, train[use_cols].to_numpy(float), train.V.to_numpy(float))
        p = np.full(len(te), np.nan)
        arb = te.arb.to_numpy()
        p[arb] = A.proba(m, te.loc[arb, use_cols].to_numpy(float))
        acc = ~arb | (p >= 0.5)

        # ---- baselines
        rule = A.best_rule(nt, [f for f in RULE_FEATS[fam] if f in nt.columns])
        acc_rule = ~arb | A.apply_rule(te, rule)
        gt = te[GT_COL[fam]].to_numpy(float)
        acc_orc = ~arb | ~np.isfinite(gt) | (gt == 1.0)
        rng = np.random.default_rng(0)
        sh = train.copy()
        for u, idx in sh.groupby("unit_key").indices.items():
            sh.loc[sh.index[idx], "V"] = rng.permutation(sh.V.to_numpy()[idx])
        msh = A.fit(model, sh[use_cols].to_numpy(float), sh.V.to_numpy(float))
        psh = np.full(len(te), np.nan)
        psh[arb] = A.proba(msh, te.loc[arb, use_cols].to_numpy(float))
        acc_sh = ~arb | (psh >= 0.5)

        V = te.V.to_numpy(float)
        Lc = float(te.J_cheap.sum())
        Bp, Hn = float(V[V > 0].sum()), float(-V[V < 0].sum())
        n_aff = int((np.abs(V) > A.EPS).sum())

        def stats(a):
            g = float(V[a].sum())
            return {"G": g / Lc,
                    "benefit_retained": float(V[a & (V > 0)].sum()) / Bp if Bp > 0 else np.nan,
                    "harm_removed": float(-V[~a & (V < 0)].sum()) / Hn if Hn > 0 else np.nan}

        r = {"cell": cell_name(c), "dataset": c[0], "geometry": c[1], "system": c[2], "family": fam,
             "n_test": len(te), "n_test_units": int(te.unit.nunique()), "n_affected_test": n_aff,
             "n_arbitrated_test": int(arb.sum()), "n_train_arbitrated": int(len(nt)),
             "arb_vs_affected_agree": float(np.mean(arb == (np.abs(V) > A.EPS))),
             "selected": best, **{f"oof_gain_{k}": v for k, v in sel.items()},
             "rule": f"{rule[0]} {'>=' if rule[1] == 1 else '<='} {rule[2]:.4g}",
             "L_cheap": Lc, "G_full": float(V.sum()) / Lc, "G_max": Bp / Lc,
             "harm_share_of_affected": float((V < -A.EPS).sum()) / max(n_aff, 1)}
        for name, a in (("arb", acc), ("rule", acc_rule), ("verifier_oracle", acc_orc), ("shuffled", acc_sh)):
            s = stats(a)
            r.update({f"{name}_{k}": v for k, v in s.items()})
        den = r["G_max"] - r["G_full"]
        r["arb_capture"] = (r["arb_G"] - r["G_full"]) / den if den > 0 else np.nan
        r["verifier_oracle_capture"] = (r["verifier_oracle_G"] - r["G_full"]) / den if den > 0 else np.nan
        results.append(r)
        test_accept[c] = {"te": te, "acc": acc, "acc_rule": acc_rule, "acc_orc": acc_orc}
        print(f"{r['cell']:22s} sel={best:11s} aff={n_aff:4d} Gfull={r['G_full']:+.4f} Garb={r['arb_G']:+.4f} "
              f"Gmax={r['G_max']:+.4f} benefit={r['arb_benefit_retained']:.2f} harm_rm={r['arb_harm_removed']:.2f} "
              f"| rule {r['rule_G']:+.4f} | vOracle {r['verifier_oracle_G']:+.4f} | shuf {r['shuffled_G']:+.4f}",
              flush=True)

    # ---- paired bootstrap over test units, units drawn once per dataset and shared by that dataset's cells
    rng = np.random.default_rng(0)
    units = {ds: sorted({u for c, t in test_accept.items() if c[0] == ds for u in t["te"].unit.unique()})
             for ds in ("KITTI", "nuScenes")}
    idx = {c: {u: np.flatnonzero(t["te"].unit.to_numpy() == u) for u in units[c[0]]} for c, t in test_accept.items()}
    draws = {c: {"arb_full": np.full(args.nboot, np.nan), "arb_rule": np.full(args.nboot, np.nan)}
             for c in test_accept}
    for b in range(args.nboot):
        pick = {ds: rng.choice(units[ds], len(units[ds]), replace=True) for ds in units}
        for c, t in test_accept.items():
            ii = np.concatenate([idx[c][u] for u in pick[c[0]]])
            V = t["te"].V.to_numpy(float)[ii]
            Lc = float(t["te"].J_cheap.to_numpy(float)[ii].sum())
            if Lc <= 0:
                continue
            ga = float(V[t["acc"][ii]].sum())
            draws[c]["arb_full"][b] = (ga - V.sum()) / Lc
            draws[c]["arb_rule"][b] = (ga - float(V[t["acc_rule"][ii]].sum())) / Lc
    np.savez_compressed(run / "bootstrap.npz",
                        **{f"{cell_name(c)}|{k}": v for c, dd in draws.items() for k, v in dd.items()})

    def ci(x):
        x = x[np.isfinite(x)]
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)

    for r, c in zip(results, test_accept):
        r["arb_minus_full_lo"], r["arb_minus_full_hi"] = ci(draws[c]["arb_full"])
        r["arb_minus_rule_lo"], r["arb_minus_rule_hi"] = ci(draws[c]["arb_rule"])
        ok_n = r["n_affected_test"] >= MIN_AFFECTED
        r["C1"] = bool(ok_n and r["arb_harm_removed"] >= 0.50 and r["arb_benefit_retained"] >= 0.80)
        r["C2"] = bool(ok_n and r["arb_minus_full_lo"] > 0)
    pooled = np.nanmean(np.stack([draws[c]["arb_rule"] for c in test_accept]), axis=0)
    point_pooled = float(np.mean([r["arb_G"] - r["rule_G"] for r in results]))
    plo, phi = ci(pooled)
    mono = [r for r in results if r["geometry"] == "mono"]
    n1, n2 = sum(r["C1"] for r in results), sum(r["C2"] for r in results)
    m1, m2 = sum(r["C1"] for r in mono), sum(r["C2"] for r in mono)
    C = {"C1": n1 >= 6, "C2": n2 >= 6, "C3": plo > 0, "C4": m1 >= 3 and m2 >= 3}
    go = all(C.values())
    weak = (n2 >= 4) and C["C3"] and (m2 >= 2)
    verdict = "GO" if go else ("WEAK GO" if weak else "NO-GO")
    summary = {"verdict": verdict, "criteria": C, "C1_cells": n1, "C2_cells": n2, "C1_mono": m1, "C2_mono": m2,
               "C3_pooled_arb_minus_rule": point_pooled, "C3_ci": [plo, phi]}
    print(json.dumps(summary, indent=2), flush=True)

    tab = pd.DataFrame(results)
    tab.to_csv(run / "phase0h_arbiter_cells.csv", index=False)
    tab.to_csv(runmeta.FINAL / "phase0h_arbiter_cells.csv", index=False)
    (run / "verdict.json").write_text(json.dumps(summary, indent=2))
    (runmeta.FINAL / "phase0h_verdict.json").write_text(json.dumps(summary, indent=2))

    # ---- secondary: DEEP's cached allocators at a 20% selection budget, with V and with the arbitrated value
    from rap.scoring import topk_expect
    gates = sorted(RAW_RAP.glob("*_budget_gate_scores_ego"))[-1]
    routers = sorted(RAW_RAP.glob("*_routers_r1_ego"))[-1]
    sec = []
    for c, t in test_accept.items():
        te = t["te"]
        key = f"{c[0]}__{c[1]}__{c[2]}__{c[3]}"
        sc = {"ego speed": te.v_causal.to_numpy(float)}
        gz = np.load(gates / f"gates__{key}.npz", allow_pickle=True)
        gk = pd.DataFrame({"seq": gz["key_seq"].astype(str), "frame": gz["key_frame"].astype(int)})
        for n in ("gate_ridge", "gate_gbm"):
            gk[n] = gz[n]
        rz = np.load(routers / f"scores__{key}.npz", allow_pickle=True)
        rk = pd.DataFrame({"seq": rz["seq"].astype(str), "frame": rz["frame"].astype(int)})
        for n in ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"):
            rk[n] = rz[n]
        m = te[["seq", "frame"]].merge(gk, on=["seq", "frame"], how="left").merge(rk, on=["seq", "frame"], how="left")
        for n in ("gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"):
            if m[n].notna().all():
                sc[n] = m[n].to_numpy(float)
        V = te.V.to_numpy(float)
        Va = np.where(t["acc"], V, 0.0)
        Lc = float(te.J_cheap.sum())
        k = max(int(round(0.2 * len(V))), 1)
        prize = topk_expect(V, [V], [k])[0][0][0]
        row = {"cell": cell_name(c), "k": k, "oracle_G": prize / Lc,
               "random_G": k / len(V) * V.sum() / Lc, "random_arb_G": k / len(V) * Va.sum() / Lc}
        for n, s in sc.items():
            (g, ga), _ = topk_expect(s, [V, Va], [k])
            row[f"{n}_G"], row[f"{n}_arb_G"] = g[0] / Lc, ga[0] / Lc
        sec.append(row)
    sec = pd.DataFrame(sec)
    sec.to_csv(run / "phase0h_budget_secondary.csv", index=False)
    sec.to_csv(runmeta.FINAL / "phase0h_budget_secondary.csv", index=False)
    print(sec.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
