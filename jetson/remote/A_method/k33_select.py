#!/usr/bin/env python
"""K33 dev selection (registered rule, handoff/K33/table1_design.md §1): per family, the candidate with the highest mean
dev G (nuScenes new val, AV2 CFG-DEV; refiner v3a) is evaluated once on test; a family is never run weaker than its
official / K28 form (that form is a candidate, so the maximum is at least it). Reads results/k33/<ds>_<part>_dev/;
writes results/k33/tables/dev_candidates.csv (every candidate: dev G per split, mean, px) and dev_selection.csv.
usage: k33_select.py [--parts de,pub]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k33"
OUT = K / "tables"
# family -> (row regex, official / K28 form)
FAMS = {
    "AnA one pass": (r"^ana\d\dh(3|5|inf)o$", "ana25hinfo"),
    "AnA iterated": (r"^ana\d\dh(3|5|inf)i$", "ana25hinfi"),
    "CF one round": (r"^cf\d\dq\d\do$", "cf20q10o"),
    "CF iterated": (r"^cf\d\dq\d\di$", "cf20q10i"),
    "AdaScale": (r"^ada[+-]\dt(960|1280|nat)$", "ada+0t960"),
    "SAHI": (r"^sahi\d+o\ds\d\d$", "sahi640o2s10"),
    "FOVEA KDE_I": (r"^fovi\d+b\d+a\d$", "fovi960b64a1"),
    "FOVEA KDE_S": (r"^fovs\d+b\d+a\d$", "fovs960b64a1"),
    "UFP": (r"^ufp\d\ds\d+$", "ufp15s1333"),
}


def load(ds, part):
    rows = []
    for f in sorted(glob.glob(str(K / f"{ds}_{part}_dev" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="de,pub")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G, PX, N = {}, {}, {}
    for part in a.parts.split(","):
        for ds in ("nusc", "av2"):
            d = load(ds, part)
            if d is None:
                continue
            d = d[d["gt"].astype(bool)]
            L = d.J_cheap.sum()
            N[(ds, part)] = (d.unit.nunique(), len(d))
            for c in d.columns:
                if c.startswith("J_") and c.endswith("|v3a"):
                    r = c[2:-4]
                    G[(ds, r)] = (L - d[c].sum()) / L
                    PX[(ds, r)] = d[f"px_{r}"].mean() if f"px_{r}" in d else np.nan
    rows, sel = [], []
    for fam, (rx, off) in FAMS.items():
        cands = sorted({r for (_, r) in G if re.match(rx, r)})
        if not cands:
            continue
        for r in cands:
            g = [G.get((ds, r), np.nan) for ds in ("nusc", "av2")]
            rows.append({"family": fam, "row": r, "official_form": r == off, "G_nusc_val": g[0], "G_av2_cfgdev": g[1],
                         "G_mean": np.mean(g) if not np.isnan(g).any() else np.nan,
                         "px_nusc_val": PX.get(("nusc", r)), "px_av2_cfgdev": PX.get(("av2", r))})
        fr = pd.DataFrame([x for x in rows if x["family"] == fam])
        complete = fr.G_mean.notna().all() and len(fr) == 12
        best = fr.loc[fr.G_mean.idxmax()] if fr.G_mean.notna().any() else None
        sel.append({"family": fam, "candidates": len(fr), "complete": bool(complete),
                    "selected": best.row if best is not None else None, "G_mean": best.G_mean if best is not None else np.nan,
                    "official_form": off, "official_G_mean": fr[fr.row == off].G_mean.iloc[0] if (fr.row == off).any() else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "dev_candidates.csv", index=False)
    S = pd.DataFrame(sel)
    S.to_csv(OUT / "dev_selection.csv", index=False)
    pd.set_option("display.width", 250)
    print({k: v for k, v in N.items()})
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
