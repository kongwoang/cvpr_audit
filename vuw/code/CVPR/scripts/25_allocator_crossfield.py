#!/usr/bin/env python
"""D48-D50 (exploratory): allocator learners borrowed from other fields, on the same pre-escalation features.

Features (deployable): causal ego speed, cheap detections count and mean confidence, cheap required deceleration and
action, planner-probe VOI and reach (D4).  Fitted on non-test units, scored by nDG@20 on the test split.
  direct       GBM regression of V                                   (DEEP-style baseline)
  uplift T     two GBMs for J_cheap(x) and J_full(x), score = difference   (uplift modelling / CATE T-learner)
  sign |V|     GBM classifier of 1[V>0] weighted by |V|              (policy learning with signed outcomes)
  top-k        GBM classifier of membership in the oracle's top-20% set   (listwise learning-to-rank proxy)
  lower q25    GBM quantile regression, alpha = 0.25                  (risk-averse allocation, finance)
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR

from rap.decision import KittiAdapter
from rap.scoring import topk_expect
from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
KW = dict(max_depth=3, max_iter=200, learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0, max_bins=64,
          early_stopping=False, random_state=0)


def main():
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str})
    dv = dv[dv.track.isin(["KITTI", "nuScenes"])].copy()
    dv["frame"] = dv.frame.astype(int)
    vc = pd.read_csv(runmeta.RAP_REPO / "data/cache/nusc_v_ego_causal.csv", dtype={"seq": str})
    out = []
    for ds in ("KITTI", "nuScenes"):
        pr = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_voi_probe_{ds}" / f"probe__{ds}.pkl")))[-1])
        for (geo, system, target), ref in dv[dv.track == ds].groupby(["geometry", "system", "target"]):
            if system.startswith("plan"):
                continue
            d = ref.merge(pr, on=["seq", "frame"])
            if ds == "nuScenes":
                d = d.merge(vc[["seq", "frame", "v_ego_causal"]], on=["seq", "frame"])
            else:
                spd = {s: KittiAdapter.speeds(s) for s in d.seq.unique()}
                d["v_ego_causal"] = [float(spd[a][b]) if b < len(spd[a]) else float(spd[a][-1]) for a, b in zip(d.seq, d.frame)]
            fam = "traj" if system == "traj" else "brake"
            voi = f"voi_{fam}" if f"voi_{fam}" in d else "voi_brake"
            X = ["v_ego_causal", "n_cheap", "conf_mean", "a_c", "act_c", voi, voi.replace("voi", "reach")]
            nt, te = d[d.split != "test"], d[d.split == "test"]
            V, Lc = te.V.to_numpy(float), float(te.J_cheap.sum())
            k = max(int(round(0.2 * len(te))), 1)
            prize = topk_expect(V, [V], [k])[0][0][0]
            Xn, Xt, Vn = nt[X].to_numpy(float), te[X].to_numpy(float), nt.V.to_numpy(float)
            sc = {}
            sc["direct"] = HGR(**KW).fit(Xn, Vn).predict(Xt)
            sc["uplift T"] = HGR(**KW).fit(Xn, nt.J_cheap).predict(Xt) - HGR(**KW).fit(Xn, nt.J_full).predict(Xt)
            w = np.abs(Vn)
            keep = w > 1e-12
            sc["sign |V|"] = (HGC(**KW).fit(Xn[keep], (Vn[keep] > 0).astype(int), sample_weight=w[keep] / w[keep].mean())
                              .predict_proba(Xt)[:, 1])
            thr = np.quantile(Vn, 0.8)
            ytop = (Vn > max(thr, 1e-12)).astype(int)
            sc["top-k"] = HGC(**KW).fit(Xn, ytop, sample_weight=np.where(ytop, np.maximum(Vn, 1e-6) / max(Vn[ytop == 1].mean(), 1e-9), 1.0)).predict_proba(Xt)[:, 1] if ytop.sum() > 5 else np.zeros(len(te))
            sc["lower q25"] = HGR(loss="quantile", quantile=0.25, **KW).fit(Xn, Vn).predict(Xt)
            sc["upper q75"] = HGR(loss="quantile", quantile=0.75, **KW).fit(Xn, Vn).predict(Xt)
            sc["probe only"] = te[voi].to_numpy(float)
            row = {"cell": f"{ds} {geo} {system}", "oracle_share": prize / Lc}
            for n, s in sc.items():
                row[n] = topk_expect(s, [V], [k])[0][0][0] / prize if prize > 0 else np.nan
            out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / "d48_allocator_crossfield.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
