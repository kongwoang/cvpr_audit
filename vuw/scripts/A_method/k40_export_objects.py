#!/usr/bin/env python
"""K40 (private helper): object-level records for the missed-object items (tab:field top, tab:missed), sanitized
parquet artifact records/objects_<dataset>.parquet. One row per decision-relevant annotated object of a GT frame:
  setting   natural (unmodified draft; K28 instr, K33 de) | deleted (the draft's detections of the relevant objects
            deleted; K28 instr-del_all, K33 de_del_all)
  unit, frame, obj (index among the frame's relevant objects), deleted (setting deleted), deleted_gt_range (m, for the
  deleted objects), v; found_<row> (bool): the row's scene contains a detection matched to the object.
Frame-level fields needed with them (J_cheap_unmod, a_* / J_* of the deletion setting) go to records/deleted_<dataset>.parquet.
usage: k40_export_objects.py <out_dir> [datasets]"""
import glob
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
SCAL = re.compile(r"^(a|J|px)_|^(v|a_gt|act_gt|J_cheap|a_cheap|px_cheap|J_cheap_unmod|deleted)$")


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x[x["gt"].astype(bool)].reset_index(drop=True) if "gt" in x else x


def objects(d, setting, prefix):
    fcols = [c for c in d.columns if c.startswith("found_")]
    out = []
    for _, r in d.iterrows():
        n = None
        for c in fcols:
            x = r[c]
            if isinstance(x, (list, np.ndarray)) and len(x):
                n = len(x)
                break
        if not n:
            continue
        dm = np.asarray(r["deleted_mask"], bool) if setting == "deleted" and isinstance(r.get("deleted_mask"), (list, np.ndarray)) else np.zeros(n, bool)
        rng = list(r.get("deleted_gt_range") or []) if setting == "deleted" else []
        pos = list(np.flatnonzero(dm))
        for j in range(n):
            o = {"setting": setting, "unit": r.unit, "frame": int(r.frame), "obj": j, "v": float(r.v), "deleted": bool(dm[j]) if len(dm) else False,
                 "deleted_gt_range": float(rng[pos.index(j)]) if (len(dm) and dm[j] and pos.index(j) < len(rng)) else np.nan}
            for c in fcols:
                x = r[c]
                o[prefix + c] = bool(np.asarray(x, bool)[j]) if isinstance(x, (list, np.ndarray)) and len(x) == n else False
            out.append(o)
    return pd.DataFrame(out)


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    dss = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nusc", "av2", "ahd"]
    rep = {}
    key = ["setting", "unit", "frame", "obj"]
    for ds in dss:
        parts = []
        for setting, k28, k33 in (("natural", f"{ds}_test_instr", f"{ds}_de_test"), ("deleted", f"{ds}_test_instr-del_all", f"{ds}_de_test_del_all")):
            a = objects(recs(B / "results/k28/p3" / k28), setting, "")
            b = objects(recs(B / "results/k33" / k33), setting, "tab1.")
            m = a.merge(b.drop(columns=["v", "deleted", "deleted_gt_range"]), on=key, how="outer", validate="1:1", indicator=True)
            rep[f"{ds} {setting}"] = {"objects_k28": len(a), "objects_k33": len(b), "merge": m["_merge"].value_counts().to_dict()}
            parts.append(m.drop(columns=["_merge"]))
        o = pd.concat(parts, ignore_index=True)
        o.to_parquet(od / f"objects_{NAME[ds]}.parquet", index=False, compression="zstd")
        dl = recs(B / "results/k28/p3" / f"{ds}_test_instr-del_all")
        dl = dl[[c for c in dl.columns if SCAL.match(c) and dl[c].dtype != object] + ["unit", "frame"]]
        dl.to_parquet(od / f"deleted_{NAME[ds]}.parquet", index=False, compression="zstd")
        rep[ds] = {"objects": len(o), "deleted_frames": len(dl)}
        print(ds, json.dumps({k: v for k, v in rep.items() if k.startswith(ds)}, default=str), flush=True)
    (od / "export_objects_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
