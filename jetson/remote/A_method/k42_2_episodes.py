#!/usr/bin/env python
"""K42-2 (post-hoc, CPU, stored actions; handoff/K42/PREREG.md): K31 episode statistics (k31_episodes.py imported
unchanged: level 1 = required >= DECELERATE primary, level 2 = hard brake secondary; deployed actions; merge gap 1 frame;
paired unit bootstrap 2,000, seed 0, of v6-S - row) for every row of main Tab. 2, all with refiner v3a.
Sources: K28 p3 instr (K31's 7 rows, heuristics @S), K28 v6basewf-yolo11m (u512 / u640 / u1280 + C1, YOLOv8s columns),
K33 pub / de (published, AnA, counterfactual; uncapped primary, @cap secondary), K33 rc routing sizes (BAR, router-G,
Chanakya: routed frame -> a_wf<s>|v3a of K33 pub, else a_cheap; gate = stored refined J reproduced on every frame).
Gate: the 7 K31 rows reproduce results/k31/tables/ep_summary.csv and ep_intervals.csv exactly.
usage: k42_2_episodes.py <out_dir>"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import k31_episodes as K31

B = Path(os.environ.get("B_REAL", os.environ["B"]))
KEY = ["unit", "frame"]
PUB = {"ada+1t1280": "AdaScale", "sahi640o1s10": "SAHI", "fovi1280b64a2": "FOVEA KDE_I", "ufp20s512": "UFP",
       "fovs1280b64a1": "FOVEA KDE_S (secondary)"}
DE = {"ana05hinfi": "AnA iterated", "cf40q20i": "counterfactual iterated",
      "ana05hinfo": "AnA one pass (secondary)", "cf40q20o": "counterfactual one pass (secondary)"}
RC = {"bar1280_0.5": "BAR router", "dvr1280_0.5": "router-G (decision-value router)", "ch128B_csc": "Chanakya"}
HEUR2 = {"nusc": ("unc4", "tcr3", "cor0"), "av2": ("unc2", "tcr0", "cor8"), "ahd": ("unc2", "tcr0", "cor8")}
FAMN = {"unc": "uncertainty", "tcr": "temporal criticality", "cor": "ego corridor"}


def recs(d, cols=None):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += [{c: r.get(c) for c in cols} if cols else r for r in pickle.load(open(f, "rb"))]
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def build(ds, checks):
    base = K31.load(ds)                                        # K31's own loader: instr GT frames, sorted
    base["unit"] = base.unit.astype(str)
    P = dict(K31.pipes(ds))
    extra = {}

    def merged(d, cols):
        d = d[d["gt"].astype(bool)] if "gt" in d else d
        m = base[KEY].merge(d[KEY + cols], on=KEY, how="left", validate="1:1")
        assert m[cols].notna().all().all(), (ds, cols[:3], int(m[cols].isna().any(axis=1).sum()))
        return m
    w = recs(B / "results/k28" / f"{ds}_test_v6basewf-yolo11m")
    for s in (512, 640, 1280):
        c = f"a_wf{s}+C1|v3a"
        extra[f"u{s} + C1 + v3a"] = merged(w, [c])[c].to_numpy(int)
    pub = recs(B / "results/k33" / f"{ds}_pub_test")
    de = recs(B / "results/k33" / f"{ds}_de_test")
    for src, table in ((pub, PUB), (de, DE)):
        for k, lab in table.items():
            for cap in ("", "@cap"):
                c = f"a_{k}{cap}|v3a"
                extra[f"{lab}{' @cap (secondary)' if cap else ''}"] = merged(src, [c])[c].to_numpy(int)
    rc = pd.read_pickle(B / "results/k33/rc" / f"test_{ds}.pkl").assign(unit=lambda x: x.unit.astype(str))
    pcols = [f"{x}_wf{s}|v3a" for s in (640, 768, 960, 1280) for x in ("a", "J")]
    pm = merged(pub, pcols + ["J_cheap", "a_cheap"])
    assert (pm.a_cheap.to_numpy(int) == base.a_cheap.to_numpy(int)).all()
    for k, lab in RC.items():
        for cap in ("", "@cap"):
            nm = k + cap
            r = base[KEY].merge(rc[KEY + [f"size_{nm}", f"J_{nm}"]], on=KEY, how="left", validate="1:1")
            sz = r[f"size_{nm}"].fillna(0).astype(int).to_numpy()
            J = pm.J_cheap.to_numpy(float).copy(); a = pm.a_cheap.to_numpy(int).copy()
            for s in (640, 768, 960, 1280):
                q = sz == s
                J[q] = pm[f"J_wf{s}|v3a"].to_numpy(float)[q]
                a[q] = pm[f"a_wf{s}|v3a"].to_numpy(int)[q]
            bad = int((~np.isclose(J, r[f"J_{nm}"].to_numpy(float), rtol=0, atol=1e-9)).sum())
            checks[f"{ds} rc {nm}: stored refined J reproduced from routing sizes (differing frames)"] = bad
            if bad == 0:
                extra[f"{lab}{' @cap (secondary)' if cap else ''}"] = a
            else:
                checks[f"{ds} rc {nm}"] = "reported missing (gate failed)"
    ins = recs(B / "results/k28/p3" / f"{ds}_test_instr", KEY + ["gt"] + [f"a_{c}@S+C1|v3a" for c in HEUR2[ds]])
    for c in HEUR2[ds]:
        col = f"a_{c}@S+C1|v3a"
        lab = f"{FAMN[c[:3]]} {c}@S + C1 + v3a (secondary, dev-best)"
        if col in P.values():
            continue
        extra[lab] = merged(ins, [col])[col].to_numpy(int)
    for lab, a in extra.items():
        col = "x_" + lab
        base[col] = a
        P[lab] = col
    return base, P


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    S, I, checks = [], [], {}
    orig = K31.pipes
    for ds in ("nusc", "av2", "ahd"):
        d, P = build(ds, checks)
        K31.pipes = lambda _ds, P=P: P
        s, i, _ = K31.analyse(ds, d)
        K31.pipes = orig
        S += s; I += i
        print(ds, "units", d.unit.nunique(), "frames", len(d), "pipelines", len(P), flush=True)
    S, I = pd.DataFrame(S), pd.DataFrame(I)
    old_s = pd.read_csv(B / "results/k31/tables/ep_summary.csv")
    old_i = pd.read_csv(B / "results/k31/tables/ep_intervals.csv")
    gate = {}
    for nm, old, new, key in (("ep_summary", old_s, S, ["dataset", "level", "pipeline"]),
                              ("ep_intervals", old_i, I, ["dataset", "level", "statistic", "pipeline"])):
        m = old.merge(new, on=key, how="left", suffixes=("_old", ""), validate="1:1")
        num = [c for c in old.columns if c not in key and old[c].dtype.kind in "fi"]
        diff = {c: float(np.nanmax(np.abs(m[c].astype(float) - m[c + "_old"].astype(float)))) if len(m) else np.nan for c in num}
        nanmis = {c: int((m[c].isna() != m[c + "_old"].isna()).sum()) for c in num}
        gate[nm] = {"rows": len(old), "missing": int(m[num[0]].isna().sum() - m[num[0] + "_old"].isna().sum()),
                    "max_abs_diff": max(diff.values()), "nan_pattern_mismatch": sum(nanmis.values())}
    gate["pass"] = all(g["max_abs_diff"] <= 1e-12 and g["nan_pattern_mismatch"] == 0 and g["missing"] == 0 for g in gate.values() if isinstance(g, dict))
    (od / "k42_2_checks.json").write_text(json.dumps({"gate_vs_K31": gate, "checks": checks}, indent=1) + "\n")
    print(json.dumps({"gate_vs_K31": gate, "checks": checks}, indent=1))
    assert gate["pass"], "K42-2 gate failed"
    I.to_csv(od / "k42_2_intervals.csv", index=False)
    S.to_csv(od / "k42_2_summary_all_statistics.csv", index=False)
    # compact table: missed_rate and false_per_1000 with DGSP - row intervals
    rows = []
    for _, r in S.iterrows():
        o = {"dataset": r.dataset, "level": r.level, "pipeline": r.pipeline, "units": r.units, "frames": r.frames,
             "episodes": r.episodes, "missed": r.missed, "missed_rate": r.missed_rate, "false_per_1000": r.false_per_1000}
        for st in ("missed_rate", "false_per_1000"):
            q = I[(I.dataset == r.dataset) & (I.level == r.level) & (I.pipeline == r.pipeline) & (I.statistic == st)]
            if len(q):
                q = q.iloc[0]
                o.update({f"DGSP_minus_row_{st}": q.v6S_minus_pipeline, f"lo_{st}": q.ci_lo, f"hi_{st}": q.ci_hi,
                          f"excludes_0_{st}": q.excludes_0})
        o["source"] = "K31 row (gate)" if r.pipeline in orig("nusc") or r.pipeline == "required action (act_gt)" else "K42-2 added"
        rows.append(o)
    E = pd.DataFrame(rows)
    E.to_csv(od / "k42_2_episodes.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 55)
    print(E[E.level.str.startswith("required")].drop(columns=["level", "units", "frames"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
