#!/usr/bin/env python
"""K32 automatic summary (no interpretation) from the tables of k32_test.py (tag test or partial) and k32_shadow.py, for
the hand-off while the analyst is away. The final REPORT.md (gate history, mechanisms) is written by hand from these.
-> results/k32/tables/REPORT_auto_<tag>.md
usage: k32_autoreport.py [--tag test]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k32"
T = K / "tables"
LOWER_BETTER = {"collision", "preventable_collision", "nonpreventable_collision", "n_collisions", "n_coll_per_km",
                "onset_delay_s", "unnecessary_hb_per_km", "missed_required_episodes", "unnecessary_hard_onsets",
                "flicker_per_100", "delay_rel", "missed", "route_time_s", "min_ttc_lt1"}
HIGHER_BETTER = {"min_ttc", "RC"}


def md(df, cols=None, nd=3):
    df = df[cols] if cols else df
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def ci(r):
    s = f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    return f"**{s}**" if r.excludes_0 else s


def done_counts():
    out = {}
    for d in ("test", "shadow"):
        n = c = 0
        for j in glob.glob(str(K / d / "*.json")):
            n += 1
            try:
                p = json.load(open(j))["_checkpoint"]["progress"]
                c += bool(p and p[0] >= p[1])
            except Exception:
                pass
        out[d] = (n, c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test")
    a = ap.parse_args()
    tag, pre = a.tag, ("" if a.tag == "test" else f"{a.tag}_")
    L = [f"# K32 automatic summary ({tag})", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} (VUW time) by k32_autoreport.py from k32_test.py / k32_shadow.py tables. "
         "No interpretation; the hand-written REPORT.md follows. Sign: v6-S − branch; **bold** = 95% paired bootstrap interval excludes 0."]
    dc = done_counts()
    L += ["", f"Checkpoints: test {dc['test'][1]} complete of {dc['test'][0]} started (of 360 registered incl. 48 unc); "
          f"shadow {dc['shadow'][1]} of {dc['shadow'][0]} (of 16)."]
    S = pd.read_csv(T / f"{tag}_summary.csv")
    cols = [c for c in ["branch", "setting", "routes", "collision", "preventable_collision", "nonpreventable_collision", "min_ttc",
                        "onset_delay_s", "missed_required_episodes", "unnecessary_hard_onsets", "unnecessary_hb_per_km", "RC",
                        "px_mean", "lat_ms_mean"] if c in S]
    L += ["", "## 1. Per branch and setting (unit = route, seeds averaged)", "", md(S.sort_values(["setting", "branch"]), cols)]
    g3 = T / f"{pre}g3_primary.csv"
    if g3.exists():
        G = pd.read_csv(g3)
        L += ["", "## 2. Primary (amendment G3): onset delay vs privileged-AEB on avoidable events (s; unit = route × seed)", "",
              md(G, ["stratum", "branch", "setting", "units", "events", "delay_rel_mean", "ci_lo", "ci_hi", "miss_rate"])]
        GI = pd.read_csv(T / f"{pre}g3_primary_intervals.csv")
        if len(GI):
            GI = GI[(GI.main == "v6S") & (GI.metric == "delay_rel")].copy()
            GI["v6S − branch"] = GI.apply(ci, axis=1)
            L += ["", "v6-S − branch (delay_rel; negative = v6-S earlier):", "", md(GI, ["stratum", "setting", "branch", "routes", "v6S − branch"])]
    rs = T / f"{tag}_intervals_route_seed.csv"
    if rs.exists():
        I = pd.read_csv(rs)
        I = I[I.main == "v6S"].copy()
        I["v6S − branch"] = I.apply(ci, axis=1)
        L += ["", "## 3. Secondary (route × seed bootstrap)", "",
              md(I[I.metric.isin(["preventable_collision", "unnecessary_hb_per_km", "min_ttc", "collision"])],
                 ["setting", "metric", "branch", "routes", "main_value", "branch_value", "v6S − branch"])]
    q = T / f"{tag}_min_ttc_quantiles.csv"
    if q.exists():
        L += ["", "Per-route-run min TTC quantiles:", "", md(pd.read_csv(q))]
    I2 = pd.read_csv(T / f"{tag}_intervals.csv")
    bad = []
    for _, r in I2[I2.main == "v6S"].iterrows():
        if not r.excludes_0:
            continue
        if (r.metric in LOWER_BETTER and r["diff"] > 0) or (r.metric in HIGHER_BETTER and r["diff"] < 0):
            bad.append(r)
    L += ["", "## 4. Every registered §4 / F metric where v6-S is significantly worse (route bootstrap)", ""]
    if bad:
        Bd = pd.DataFrame(bad)
        Bd["v6S − branch"] = Bd.apply(ci, axis=1)
        L += [md(Bd, ["setting", "branch", "metric", "routes", "main_value", "branch_value", "v6S − branch"])]
    else:
        L += ["none"]
    e1 = T / f"{pre}e1_events.csv"
    if e1.exists():
        E = pd.read_csv(e1)
        c = [x for x in ["branch", "setting", "kind", "n_correct", "n_wrong", "front_contact_H3", "preventable_front_contact_H3",
                         "min_ttc_H3", "decel_H3", "stop_H3"] if x in E]
        L += ["", "## 5. E1 decision-change events (outcome over the next 3 s)", "", md(E[E.kind.isin(["correct", "wrong", "no_change_required"])], c)]
    e2 = T / f"{pre}e2_forks.csv"
    if e2.exists():
        L += ["", "## 6. E2' model-based forks (X = branch's actions vs D = the draft's shadow actions, 3 s)", "", md(pd.read_csv(e2), nd=4)]
    sh = T / "shadow_fidelity.csv"
    if sh.exists():
        Sh = pd.read_csv(sh)
        L += ["", "## 7. Shadow mode (G4): decision fidelity on v6-S / u960 rollouts", "",
              md(Sh, [c for c in ["rollout", "branch", "routes", "agree_u960", "agree_gt", "flicker_per_100", "dis_run_mean",
                                  "onset_delay_vs_u960_s", "miss_rate_vs_u960"] if c in Sh])]
    lv = K / "latency_model_validation.csv"
    if lv.exists():
        L += ["", "## 8. G2 latency model validation (modelled vs measured)", "", md(pd.read_csv(lv))]
    C = pd.read_csv(T / f"{tag}_collisions.csv")
    if len(C):
        L += ["", "## 9. Collisions by contact type (D)", "",
              md(C.groupby(["branch", "setting", "contact_type"]).size().rename("n").reset_index())]
    (T / f"REPORT_auto_{tag}.md").write_text("\n".join(L) + "\n")
    print("written", T / f"REPORT_auto_{tag}.md")


if __name__ == "__main__":
    main()
