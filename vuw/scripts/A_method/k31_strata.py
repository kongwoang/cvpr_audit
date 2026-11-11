#!/usr/bin/env python
"""K31 item 5 (post-hoc; stored records K28 phase 3, braking consumer; pre-registered in handoff/K31/STATUS.md
2026-10-19): distribution-shift strata. nuScenes new test: night / day and rain / no rain from the scene description;
AV2 val: city from the public log map file name (listed from the public AV2 bucket, cached in results/k31/meta);
Argoverse-HD val: city (city_info.json of each sequence's Argoverse 1.1 tracking log). Writes results/k31/tables/
strata.csv (per stratum: units, frames, G and px of every pipeline, v6-S minus each with the paired unit bootstrap inside
the stratum) and strata_units.csv (the stratum of every unit).
usage: k31_strata.py
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
OUT = B / "results" / "k31" / "tables"
META = B / "results" / "k31" / "meta"
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
DSL = {"nusc": "nuScenes new test", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val"}


def pipes(ds):
    n = NWS[ds]
    return {"cheap": ("cheap", "cheap"), "u960 + C1 + v3a": ("wf960+C1|v3a", "wf960"), "native + C1 + v3a": (f"wf{n}+C1|v3a", f"wf{n}"),
            "hybrid tsp8@S + C1 + v3a": ("tsp8@S+C1|v3a", "tsp8@S+C1"), "uncertainty unc2@S + C1 + v3a": ("unc2@S+C1|v3a", "unc2@S+C1"),
            "v6-S": ("v6S|v3a", "v6S")}


def load(ds):
    cols = ["unit", "frame", "gt", "J_cheap"] + [f"J_{j}" for j, _ in pipes(ds).values()] + [f"px_{p}" for _, p in pipes(ds).values()]
    rows = []
    for f in sorted(glob.glob(str(B / "results" / "k28" / "p3" / f"{ds}_test_instr" / "*.pkl"))):
        rows += [{c: r.get(c) for c in cols} for r in pickle.load(open(f, "rb"))]
    d = pd.DataFrame(rows)
    return d[d["gt"].astype(bool)].reset_index(drop=True)


def strata_nusc(units):
    sc = {s["name"]: s["description"] for s in json.load(open(B / "datasets" / "nuscenes" / "v1.0-trainval" / "scene.json"))}
    out = []
    for u in units:
        t = sc[u].lower()
        out.append({"unit": u, "description": sc[u], "lighting": "night" if "night" in t else "day",
                    "weather": "rain" if "rain" in t else "no rain"})
    return pd.DataFrame(out)


def strata_av2(units):
    META.mkdir(parents=True, exist_ok=True)
    f = META / "av2_val_city.csv"
    have = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["unit", "city"])
    known = dict(zip(have.unit, have.city))
    for u in units:
        if u in known:
            continue
        url = f"https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets/av2/sensor/val/{u}/map/log_map_archive_"
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        m = re.search(r"log_map_archive_[0-9a-f-]+____([A-Z]{3})_city_\d+\.json", xml)
        known[u] = m.group(1) if m else "unknown"
    pd.DataFrame({"unit": list(known), "city": list(known.values())}).to_csv(f, index=False)
    return pd.DataFrame({"unit": units, "city": [known[u] for u in units]})


def strata_ahd(units):
    sd = json.load(open(B / "datasets" / "argoverse_hd" / "Argoverse-HD" / "annotations" / "val.json"))["seq_dirs"]
    out = []
    for u in units:
        log = sd[int(u[3:])].split("/")[1]
        ci = json.load(open(B / "datasets" / "argoverse1" / "argoverse-tracking" / "val" / log / "city_info.json"))["city_name"]
        out.append({"unit": u, "log": log, "city": ci})
    return pd.DataFrame(out)


def boot_idx(nu, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, nu, nu) for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, units_out = [], []
    for ds in ("nusc", "av2", "ahd"):
        d = load(ds)
        units = sorted(d.unit.unique())
        st = {"nusc": strata_nusc, "av2": strata_av2, "ahd": strata_ahd}[ds](units)
        st.insert(0, "dataset", DSL[ds]); units_out.append(st)
        axes = ["lighting", "weather"] if ds == "nusc" else ["city"]
        for ax in axes:
            for val, su in st.groupby(ax):
                x = d[d.unit.isin(set(su.unit))].reset_index(drop=True)
                uq, inv = np.unique(x.unit.to_numpy(), return_inverse=True)
                bs = boot_idx(len(uq))
                sums = lambda c: np.bincount(inv, x[c].to_numpy(float), len(uq))   # noqa: E731
                L = sums("J_cheap")
                Jv = sums("J_v6S|v3a")
                for name, (jk, pk) in pipes(ds).items():
                    Jx = sums(f"J_{jk}")
                    r = {"dataset": DSL[ds], "axis": ax, "stratum": val, "units": len(uq), "frames": len(x),
                         "small_stratum (<5 units)": len(uq) < 5, "pipeline": name,
                         "G": float((L.sum() - Jx.sum()) / L.sum()) if L.sum() > 0 else np.nan, "px_mean": float(x[f"px_{pk}"].mean())}
                    if name != "v6-S" and L.sum() > 0:
                        v = [(Jx[s].sum() - Jv[s].sum()) / L[s].sum() for s in bs if L[s].sum() > 0]
                        r["v6S_minus_pipeline"] = float((Jx.sum() - Jv.sum()) / L.sum())
                        r["ci_lo"], r["ci_hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
                        r["excludes_0"] = bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)
                    rows.append(r)
        print(ds, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "strata.csv", index=False)
    pd.concat(units_out, ignore_index=True).to_csv(OUT / "strata_units.csv", index=False)
    print("K31 STRATA DONE")


if __name__ == "__main__":
    main()
