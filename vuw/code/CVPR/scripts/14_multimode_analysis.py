#!/usr/bin/env python
"""D5 and D6 (exploratory), from the multimode tables (11).

D5  decision regressions of perception *upgrades*: for pairs a -> b (resolution and model family), the change in
    misses and false positives, and the decision value of switching every input from a to b (share harmed, harm
    ratio, net loss change as a share of the all-a loss).
D6  a decision-convergence cascade: run 320 and 384; escalate to 640 only where their decisions disagree.  Compared
    at the same mean GPU cost with random and oracle frame escalation.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from rap.scoring import topk_expect
from vtf import runmeta

PAIRS = [("Y320", "Y640"), ("Y512", "Y640"), ("Y640", "Y960"), ("Y320", "R320"), ("Y640", "R640"),
         ("R320", "R640"), ("R480", "R640"), ("Y320", "R640")]
GPU_MS = {"Y320": 5.09, "Y384": 6.10, "Y512": 7.89, "Y640": 9.27}          # DEEP's measured FP16 TensorRT inference


def main():
    out5, out6 = [], []
    for f in sorted(glob.glob(str(runmeta.RAW / "*_multimode_*" / "multimode__KITTI__*.pkl"))):
        geo = f.rsplit("__", 1)[1][:-4]
        d = pd.read_pickle(f)
        for sysname, J, act in (("brake", "J", "act"), ("traj", "JB", "plan")):
            for a, b in PAIRS:
                V = (d[f"{J}_{a}"] - d[f"{J}_{b}"]).to_numpy(float)
                aff = np.abs(V) > 1e-9
                La = float(d[f"{J}_{a}"].sum())
                out5.append({"geometry": geo, "consumer": sysname, "pair": f"{a}->{b}",
                             "misses_change_%": 100 * (d[f"fn_{b}"].sum() / d[f"fn_{a}"].sum() - 1),
                             "fp_change_%": 100 * (d[f"fp_{b}"].sum() / max(d[f"fp_{a}"].sum(), 1) - 1),
                             "affected": int(aff.sum()), "harmed_share": float((V < -1e-9).sum() / max(aff.sum(), 1)),
                             "harm_ratio": float(-V[V < 0].sum() / max(V[V > 0].sum(), 1e-9)),
                             "net_loss_reduction": float(V.sum() / La), "max_if_arbitrated": float(V[V > 0].sum() / La)})
            # D6: cascade on 320 -> 384 disagreement, escalating to 640
            V = (d[f"{J}_Y320"] - d[f"{J}_Y640"]).to_numpy(float)
            L = float(d[f"{J}_Y320"].sum())
            dis = (d[f"{act}_Y320"] != d[f"{act}_Y384"]).to_numpy()
            ch = (d[f"{act}_Y320"] != d[f"{act}_Y640"]).to_numpy()
            p = dis.mean()
            cost = GPU_MS["Y384"] + p * GPU_MS["Y640"]                  # extra over always running 320
            alpha = min(cost / GPU_MS["Y640"], 1.0)                      # frame escalations the same cost buys
            k = max(int(round(alpha * len(V))), 1)
            out6.append({"geometry": geo, "consumer": sysname, "disagree_rate": p,
                         "P(640 changes | 320,384 agree)": float(ch[~dis].mean()),
                         "P(640 changes | disagree)": float(ch[dis].mean()),
                         "extra_ms_per_frame": cost, "equal_cost_escalation_rate": alpha,
                         "G_cascade": float(V[dis].sum() / L),
                         "G_cascade_using_384_when_agree": float(((d[f"{J}_Y320"] - np.where(dis, d[f"{J}_Y640"], d[f"{J}_Y384"])).sum()) / L),
                         "G_random_equal_cost": float(alpha * V.sum() / L),
                         "G_oracle_equal_cost": float(topk_expect(V, [V], [k])[0][0][0] / L),
                         "G_all_640": float(V.sum() / L)})
    r5, r6 = pd.DataFrame(out5), pd.DataFrame(out6)
    r5.to_csv(runmeta.FINAL / "d5_upgrade_regressions.csv", index=False)
    r6.to_csv(runmeta.FINAL / "d6_decision_cascade.csv", index=False)
    pd.set_option("display.width", 250)
    print(r5.round(3).to_string())
    print(r6.round(3).to_string())


if __name__ == "__main__":
    main()
