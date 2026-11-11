#!/usr/bin/env python
"""D62 x speculative: run the speculative loop only on the frames a training-free gate ranks highest.

Gate = ensemble decision sensitivity of the cheap scene (D62, `26_ensemble_sensitivity.py`): the component 'miss'
(insertion instability) and 'all'.  For budgets q (share of frames given the loop), decision value of
  gate -> speculative (+refiner)   vs   gate -> uniform 960   vs   random -> speculative,
with the cheap pipeline on all other frames.  KITTI test: the deployed v2 run (per-frame ms); nuScenes: v3 dev evals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
QS = (0.1, 0.2, 0.3, 0.5, 1.0)


def curves(d, jc, arms, score, name, rng):
    L = d[jc].sum()
    rows = []
    for q in QS:
        k = max(int(round(q * len(d))), 1)
        top = np.zeros(len(d), bool)
        top[np.lexsort((rng.random(len(d)), -d[score].to_numpy()))[:k]] = True
        rnd = np.zeros(len(d), bool)
        rnd[rng.choice(len(d), k, replace=False)] = True
        r = {"data": name, "gate": score, "q": q}
        for a, ja in arms.items():
            V = (d[jc] - d[ja]).to_numpy()
            r[f"G_{a}"] = V[top].sum() / L
            r[f"G_{a}_random"] = np.mean([V[np.isin(np.arange(len(d)), rng.choice(len(d), k, replace=False))].sum() / L
                                          for _ in range(200)])
            r[f"G_{a}_oracle"] = np.sort(V)[::-1][:k].clip(0).sum() / L
        rows.append(r)
    return rows


def main():
    rng = np.random.default_rng(0)
    rows = []
    ek = pd.read_pickle(RAW / "20260927_002658_ensemble_sens_KITTI" / "ensemble.pkl")
    ek["frame"] = ek.frame.astype(int)
    for run, tag in (("20260927_100945_e2e_v2_KITTI", "KITTI test v2"),):
        d = pd.read_pickle(RAW / run / "e2e.pkl").merge(ek, on=["seq", "frame"], how="left")
        assert d.ens_miss.notna().all()
        for g in ("ens_miss", "ens_all"):
            rows += curves(d, "J_c_trt", {"spec": "J_spec_trt", "spec_ref": "J_spec_ref_trt", "u960": "J_full960_trt"},
                           g, tag, rng)
        # latency of gate -> spec (+ref) at each budget: cheap on the rest
        for q in QS:
            k = int(round(q * len(d)))
            top = np.lexsort((rng.random(len(d)), -d.ens_miss.to_numpy()))[:k]
            ms = d.ms_cheap.copy()
            ms.iloc[top] = d.ms_speculative_ref.iloc[top]
            print(f"  {tag} q={q:.1f}: mean ms gate->spec+ref {ms.mean():.1f} (u960 {d.ms_full960.mean():.1f})")
    en = pd.read_pickle(RAW / "20260927_005228_ensemble_sens_nuScenes" / "ensemble.pkl")
    en["frame"] = en.frame.astype(int)
    for f, tag in (("eval__old_test_v3.csv.gz", "nuScenes old test v3"), ("eval__old_train+val_v3_h1.6.csv.gz", "nuScenes dev v3")):
        d = pd.read_csv(ROOT / "results" / "vuw" / f, dtype={"seq": str}).merge(en, on=["seq", "frame"], how="left")
        miss = d.ens_miss.isna().mean()
        d = d.dropna(subset=["ens_miss"])
        for g in ("ens_miss", "ens_all"):
            rows += curves(d, "J_c", {"spec": "J_spec", "spec_ref": "J_spec_ref_resnet18", "u960": "J_u960"}, g,
                           f"{tag} (gate missing {miss:.0%})", rng)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "final" / "d62_gate_spec.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["data", "gate", "q", "G_spec", "G_spec_random", "G_spec_ref", "G_u960", "G_u960_random", "G_spec_oracle"]
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
