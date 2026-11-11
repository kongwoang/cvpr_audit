#!/usr/bin/env python
"""K30-DEV analysis (dev splits only; selection rule pre-registered in handoff/K30/STATUS.md): G of every pipeline x
refiner variant on nuScenes val and AV2 CFG-DEV (results/k30/dev_devref/<ds>), each candidate combination of v6-S (refiner
variant x probe) minus deployed v6-S (v3a, frozen probe) per dataset with a paired unit bootstrap (2,000, seed 0), the macro
gain, and the selection decision. results/k30/tables/dev_devref.csv, dev_devref_selection.txt"""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
R30 = B / "results" / "k30"
REFS = ("none", "v3a", "v3a_n15", "v3a_n30", "v3a_hb")
NWS = {"nusc": 1600, "av2": 2048}


def load(ds):
    rows = []
    for f in sorted(glob.glob(str(R30 / "dev_devref" / ds / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    return d[d["gt"].astype(bool)].reset_index(drop=True) if len(d) else d


def boot(units, L, Ja, Jb, n=2000):
    """G(a) - G(b) = (sum Jb - sum Ja) / sum L"""
    uq, inv = np.unique(units, return_inverse=True)
    l, a, b = (np.bincount(inv, np.asarray(x, float), len(uq)) for x in (L, Ja, Jb))
    rng = np.random.default_rng(0)
    v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in (rng.integers(0, len(uq), len(uq)) for _ in range(n))]
    return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    out, gains = [], {}
    for ds in ("nusc", "av2"):
        d = load(ds)
        if d.empty:
            continue
        L = d.J_cheap.to_numpy(float)
        base = d["J_v6S|v3a"].to_numpy(float)
        for pipe in ("v6S", "v6S@vc5", "v6S@vc10", "wf960+C1", f"wf{NWS[ds]}+C1"):
            for r in REFS:
                c = f"J_{pipe}|{r}"
                if c not in d:
                    continue
                J = d[c].to_numpy(float)
                row = {"dataset": ds, "units": d.unit.nunique(), "frames": len(d), "pipeline": pipe, "refiner": r,
                       "G": float((L.sum() - J.sum()) / L.sum())}
                if pipe.startswith("v6S") and not (pipe == "v6S" and r == "v3a"):
                    pt, lo, hi = boot(d.unit.to_numpy(), L, J, base)          # G(candidate) - G(deployed v6-S)
                    row.update({"minus_deployed_v6S": pt, "ci_lo": lo, "ci_hi": hi})
                    gains.setdefault((pipe, r), {})[ds] = pt
                out.append(row)
    o = pd.DataFrame(out)
    (R30 / "tables").mkdir(parents=True, exist_ok=True)
    o.to_csv(R30 / "tables" / "dev_devref.csv", index=False)
    lines, best = [], None
    for k, g in gains.items():
        if len(g) < 2:
            continue
        macro = float(np.mean(list(g.values())))
        ok = macro >= 0.005 and min(g.values()) >= -0.005
        lines.append(f"{k[0]} | {k[1]}: nusc {g['nusc']:+.4f} av2 {g['av2']:+.4f} macro {macro:+.4f} {'QUALIFIES' if ok else ''}")
        if ok and (best is None or macro > best[1]):
            best = (k, macro)
    lines.append(f"SELECTED: {best[0][0]} with refiner {best[0][1]} (macro {best[1]:+.4f})" if best else "SELECTED: none -> v6-S stays")
    (R30 / "tables" / "dev_devref_selection.txt").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(o.round(4).to_string(index=False)); print("\n".join(lines)); print("K30 DEV ANALYZE DONE")


if __name__ == "__main__":
    main()
