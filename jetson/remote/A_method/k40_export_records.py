#!/usr/bin/env python
"""K40 (private helper): export the per-frame records of the paper's rows to one wide sanitized parquet per dataset
(release artifact records/<dataset>.parquet). Keys: unit, frame (GT frames of the test split). Columns: a_* / J_* /
px_* / ncrops_* / nguard_* / rounds_* of every stored row, plus v, a_gt, act_gt. Sources (merged on unit, frame;
a column present in several sources is shared when identical; a same-named column with different values (e.g. K33
"wf640|none" = whole frame + C1, K28 "wf640|none" = plain) is kept as <source>.<column>):
  K28 main + supp + phase-2 parts + p3 instr (k28_p3_p0.k24v6(ds, p3=True)), K33 de / pub / rc (k33_report.frames),
  K30 match, K39 vuw. Object arrays (found_*, crops_*, cov, ...) are dropped.
usage: k40_export_records.py <out_dir> [datasets]"""
import glob
import importlib.util
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
KEEP = re.compile(r"^(a|J|a2|J2|ac|Jc|px|cpx|ncrops|nguard|rounds|ncalls|size|skip_crop|skip_guard|budget|act_gt|J_cheap|field_cells|field_rmax|field_area_v6S|crop_area_tsp8)(_|@|$)|^(v|a_gt|act_gt|act2_gt|J_cheap|a_cheap|px_cheap|capv|kappa|decc|decc_in_straight_tube|nosup_cheap)$")


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def scalar_cols(d):
    return [c for c in d.columns if KEEP.match(c) and d[c].dtype != object]


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    dss = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nusc", "av2", "ahd"]
    AB = mod("k28p3p0", B / "scripts/A_method/k28_p3_p0.py")
    T1 = mod("k33r", B / "scripts/A_method/k33_report.py")
    report = {}
    for ds in dss:
        base = AB.k24v6(ds, p3=True)
        base["unit"] = base.unit.astype(str)
        if "gt" in base:
            base = base[base["gt"].astype(bool)]
        key = ["unit", "frame"]
        out = base[key + scalar_cols(base)].copy()
        out.columns = key + [c.replace("@k28", "@k28dup") for c in out.columns[2:]]
        srcs = {"instr": recs(B / "results/k28/p3" / f"{ds}_test_instr"), "tab1": T1.frames(ds), "match": recs(B / "results/k30/match" / ds),
                "strict": recs(B / "results/k39/vuw" / ds)}
        for nm, d_ in (("speed", B / "results/k30/speed" / ds), ("curved", B / "results/k31/r3" / f"{ds}_curv"),
                       ("ttc", B / "results/k31/r2" / f"{ds}_ttcpx"), ("thr", B / "results/k37/k1/test" / ds)):
            if glob.glob(str(d_ / "*.pkl")):
                srcs[nm] = recs(d_)
        rep = {"base_frames": len(out), "base_cols": out.shape[1] - 2}
        for nm, s in srcs.items():
            s = s.copy(); s["unit"] = s.unit.astype(str)
            if "gt" in s:
                s = s[s["gt"].astype(bool)]
            cols = scalar_cols(s)
            m = out[key].merge(s[key + cols], on=key, how="left", validate="1:1")
            miss = int((~m[cols].notna().any(axis=1)).sum())
            assert miss == 0 or nm not in ("instr", "tab1", "match", "strict"), (ds, nm, "frames missing", miss)
            same, new, clash = [], [], []
            for c in cols:
                if c in out:
                    a_, b_ = out[c].to_numpy(float), m[c].to_numpy(float)
                    bad = int((~np.isclose(a_, b_, rtol=0, atol=1e-12, equal_nan=True)).sum())
                    (same if bad == 0 else clash).append(c)
                else:
                    new.append(c)
            add = m[new + clash].rename(columns={c: f"{nm}.{c}" for c in clash})
            out = pd.concat([out, add], axis=1)
            rep[nm] = {"frames": len(s), "shared_cols_identical": len(same), "new_cols": len(new),
                       "same_name_different_values (namespaced)": clash}
        out = out.drop(columns=[c for c in out.columns if c.endswith("@k28dup")])
        # pi(F) transition flag on consecutive stored frames of a unit (non-GT frames included), as K39-6
        allf = recs(B / "results/k28/p3" / f"{ds}_test_instr").sort_values(key)
        pf = allf.groupby("unit")["a_wf960+C1|none"].shift(1)
        allf["F_transition"] = (pf.notna() & (pf != allf["a_wf960+C1|none"])).astype(bool)
        chain = [c for c in allf.columns if re.match(r"^(a|J)_ch_", c)]
        out = out.merge(allf[key + ["F_transition"] + [c for c in chain if c not in out]], on=key, how="left", validate="1:1")
        assert out.F_transition.notna().all(), ds
        # the annotated demand and the previous frame's cheap action (the cost's command-change term), K28 p0abl
        ab = recs(B / "results/k28" / f"p0abl_{ds}_p0")[key + ["a_gt", "prev_act"]].rename(columns={"a_gt": "a_gt_p0", "prev_act": "prev_cheap"})
        out = out.merge(ab, on=key, how="left", validate="1:1")
        if "a_gt" not in out:
            out["a_gt"] = out.a_gt_p0
        out = out.drop(columns="a_gt_p0")
        out = out.sort_values(key).reset_index(drop=True)
        for c in out.columns[2:]:
            if out[c].dtype == np.float64 and np.all(np.isnan(out[c]) | (out[c] == np.round(out[c]))) and c.startswith(("a_", "ncrops_", "nguard_", "rounds_", "size_", "act")):
                out[c] = out[c].astype("Int16")
        f = od / f"{NAME[ds]}.parquet"
        out.to_parquet(f, index=False, compression="zstd")
        rep.update({"rows": len(out), "cols": out.shape[1], "file_mb": round(f.stat().st_size / 2**20, 2)})
        report[ds] = rep
        print(ds, json.dumps(rep), flush=True)
    (od / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
