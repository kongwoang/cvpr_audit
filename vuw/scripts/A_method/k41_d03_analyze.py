#!/usr/bin/env python
"""K41-1 D03 analysis (PREREG 2026-11-01 D03 addendum): PiP-inspired rows (a) pipA all anchors, (b) pipB plan-selected
autoregressive, (c) pipC = (b) + DESP support, each under the nominal / K39-literal strict / observed strict caps, on the
three braking splits, natural draft (base) and K28 deletion setting (del_all). Rules and helpers as
k40r_d0102_analyze.py (K40 release Bootstrap, 2,000 draws, seed 0, Delta = DESP v6-S - row; p = 2 min(P(D<=0), P(D>=0));
Holm over the 9 rows per dataset x setting x refiner).
- Frozen parity: every unit's .done.json has no failures and 10 comparisons per frame (the detector compares the v6-S
  control with the K30 match records (base) / K28 phase-3 deletion records (del_all)); base additionally checks
  J_v6S|v3a against the released records.
- G = 1 - sum J_row / sum J_cheap on gt frames (del_all: J_cheap of the deleted draft).
- Repair (K39-6): agreement of the no-refiner action with pi(F) = u960 + C1 (released records; draft independent) on the
  frames where the cheap draft differs from pi(F).
- Recovery: base = natural misses (relevant objects the cheap draft misses, found by the row); del_all = deletion
  recovery (relevant objects whose draft detection was deleted, found by the row).
usage: k41_d03_analyze.py <out_dir> [base,del_all]   (on VUW, after . $B/env.sh)"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR")
REL = B / "tmp/desp_k40"
sys.path.insert(0, str(REL))
from analysis.common import Bootstrap, records  # noqa: E402  (read-only use of the K40 release helpers)

NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
RECS = REL / "artifacts/records"
ROWS = [f"pip{x}@{c}" for x in "ABC" for c in ("nominal", "k39_strict", "strict_observed")]


def load(setting, ds):
    rows, meta = [], []
    for f in sorted(glob.glob(str(B / "results/k41" / ("pip" if setting == "base" else "pip_" + setting) / ds / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
        meta.append(json.loads(Path(f).with_suffix(".done.json").read_text()))
    d = pd.DataFrame(rows)
    d["unit"] = d.unit.astype(str)
    return d[d["gt"].astype(bool)].sort_values(["unit", "frame"]).reset_index(drop=True), meta


def G(d, col):
    return float(1 - d[col].astype(float).sum() / d.J_cheap.astype(float).sum())


def draws(bt, L, Ja, Jb):
    l, a, b = bt.sums(L), bt.sums(Ja), bt.sums(Jb)
    return (b[bt.idx].sum(1) - a[bt.idx].sum(1)) / l[bt.idx].sum(1)


def holm_adj(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def recovery(d, row, setting):
    hit = n = 0
    for _, r in d.iterrows():
        fr = r.get(f"found_{row}")
        if not isinstance(fr, (list, np.ndarray)) or not len(fr):
            continue
        fr = np.asarray(fr, bool)
        if setting == "base":
            miss = ~np.asarray(r["found_cheap"], bool)
        else:
            dm = r.get("deleted_mask")
            if not isinstance(dm, (list, np.ndarray)) or not len(dm):
                continue
            miss = np.asarray(dm, bool)
        n += int(miss.sum()); hit += int((fr & miss).sum())
    return (hit / n if n else np.nan), n


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    settings = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "del_all"]
    checks, out = {}, []
    for setting in settings:
        for ds, nm in NAME.items():
            d, meta = load(setting, ds)
            R = records(nm, RECS)
            m = d[["unit", "frame", "J_v6S|v3a"]].merge(R[["unit", "frame", "J_v6S|v3a", "a_wf960+C1|none"]], on=["unit", "frame"],
                                                         how="left", suffixes=("", "@rel"), validate="1:1")
            c = {"frames": len(d), "released_frames": len(R), "units": len(meta),
                 "unit_failures": int(sum(len(x["failures"]) for x in meta)),
                 "comparisons": int(sum(x["comparisons"] for x in meta)), "rows": int(sum(x["rows"] for x in meta)),
                 "G_v6S": G(d, "J_v6S|v3a")}
            if setting == "base":
                c["max_abs_dJ_v6S_vs_release"] = float(np.nanmax(np.abs(m["J_v6S|v3a"] - m["J_v6S|v3a@rel"])))
                assert c["max_abs_dJ_v6S_vs_release"] < 1e-9, c
            assert len(d) == len(R) and c["unit_failures"] == 0 and c["comparisons"] == 10 * c["rows"], c
            checks[f"{setting}/{ds}"] = c
            bt = Bootstrap(d.unit)
            L = d.J_cheap.to_numpy(float)
            F = m["a_wf960+C1|none"].to_numpy(int)
            rep = d.a_cheap.to_numpy(int) != F
            e6 = d["a_v6S|none"].to_numpy(int) == F
            rec6 = recovery(d, "v6S", setting)
            for ref in ("v3a", "none"):
                block, Jd = [], d[f"J_v6S|{ref}"].to_numpy(float)   # control under the same refiner (fix 2026-11-02)
                for row in ROWS:
                    x = draws(bt, L, Jd, d[f"J_{row}|{ref}"].to_numpy(float))
                    eq = d[f"a_{row}|none"].to_numpy(int) == F
                    sd = bt.share_delta(e6, eq, rep)
                    rr, n = recovery(d, row, setting)
                    block.append({"setting": setting, "dataset": ds, "row": row, "refiner": ref, "units": len(bt.units), "frames": len(d),
                                  "G": G(d, f"J_{row}|{ref}"), "G_v6S": G(d, f"J_v6S|{ref}"),
                                  "px_mean": float(d[f"px_{row}"].mean()), "px_v6S": float(d.px_v6S.mean()),
                                  "px_ratio": float(d[f"px_{row}"].mean() / d.px_v6S.mean()),
                                  "rounds_mean": float(d[f"rounds_{row}"].mean()) if f"rounds_{row}" in d else np.nan,
                                  "delta": G(d, f"J_v6S|{ref}") - G(d, f"J_{row}|{ref}"),
                                  "ci_lo": float(np.percentile(x, 2.5)), "ci_hi": float(np.percentile(x, 97.5)),
                                  "p_boot": float(2 * min((x <= 0).mean(), (x >= 0).mean())),
                                  "repair_row": float(eq[rep].mean()), "repair_v6S": float(e6[rep].mean()),
                                  "repair_delta": sd[0], "repair_ci_lo": sd[1], "repair_ci_hi": sd[2], "repair_frames": int(rep.sum()),
                                  "recovery_row": rr, "recovery_v6S": rec6[0], "recovery_objects": n})
                for b_, q in zip(block, holm_adj([b_["p_boot"] for b_ in block])):
                    b_["p_holm"] = float(q)
                out += block
    t = pd.DataFrame(out)
    t.to_csv(od / f"k41_1_d03_{'_'.join(settings)}.csv", index=False)
    (od / f"k41_1_d03_{'_'.join(settings)}_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    pd.set_option("display.width", 250)
    print(t[t.refiner == "v3a"][["setting", "dataset", "row", "G", "G_v6S", "px_ratio", "delta", "ci_lo", "ci_hi", "p_holm",
                                 "repair_row", "repair_v6S", "recovery_row", "recovery_v6S", "recovery_objects"]].round(3).to_string(index=False))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
