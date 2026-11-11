#!/usr/bin/env python
"""K30 V6 (i): merge the PDMS rows of pdms_k30 (spec_pdm, native; and the superseded hybrid_pdm, renamed
hybrid_pdm_tubebug) with pdms_k30b (hybrid_pdm with the fixed tube) by token -> pdms_k30m/<log>.pkl, for
navsim_pdms.py --stage summary (NAVSIM_PDMS_OUT=pdms_k30m, NAVSIM_BRANCHES=spec_pdm,hybrid_pdm,native,hybrid_pdm_tubebug)."""
import os
import pickle
from pathlib import Path

X = Path(os.environ["NAVSIM_EXP_ROOT"])
A, Bf, M = X / "pdms_k30", X / "pdms_k30b", X / "pdms_k30m"
M.mkdir(parents=True, exist_ok=True)
n = miss = 0
for f in sorted(A.glob("*.pkl")):
    rows = pickle.load(open(f, "rb"))
    fb = Bf / f.name
    hb = {r["token"]: r for r in pickle.load(open(fb, "rb"))} if fb.exists() else {}
    out = []
    for r in rows:
        r = dict(r)
        for k in [k for k in r if k.endswith("_hybrid_pdm")]:
            r[k.replace("_hybrid_pdm", "_hybrid_pdm_tubebug")] = r.pop(k)
        q = hb.get(r["token"])
        if q is None:
            miss += 1
        else:
            r.update({k: v for k, v in q.items() if k.endswith("_hybrid_pdm")})
        out.append(r)
        n += 1
    pickle.dump(out, open(M / f.name, "wb"))
print(f"merged {n} rows, {miss} without a fixed-tube hybrid row; K30 NAV MERGE DONE")
