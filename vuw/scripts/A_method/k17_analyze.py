#!/usr/bin/env python
"""K17 tables (post-hoc analysis; AV2 with the K10 ego mask; Argoverse-HD frames with 3D labels). From k17_variants
(results/k17/<ds>_test_{base,del_all}) + K14 (uncertainty / temporal_crit per frame). FULL = frozen loop, SO = support
only (no field crop). Unit bootstrap 2,000, seed 0.
  field_contribution_<ds>.csv  sections:
    1 frames where FULL and SO act differently (without / with refiner): decision loss of FULL, SO, cheap, F (u960),
      split by what the field crop did (a true discovery, b spurious, c box / range of a draft object changed, d other)
    2 events per 1,000 frames vs the required action a*: correct HARD_BRAKE, correct DECELERATE, unnecessary DECELERATE
      (a* KEEP), unnecessary HARD_BRAKE (a* below HARD_BRAKE); interval of FULL minus SO
    3 the field's regime: a* >= DECELERATE while the draft has no object in the support of a* (a_cheap < a*); G on them
    4 equal compute: SO with enlarged support-crop context, SO + random extra crops up to FULL's per-frame tensor px
    5 closing speed (AV2, Argoverse-HD): FULL at v_close 0 / 5 / 10 (+ SO): G, px, agreement with pi(F), recovery of
      deleted decision-relevant objects (controlled deletion)
usage: k17_analyze.py
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
S = B / "k10_masked"
OUT = B / "results" / "k17_tables"


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def boots(units):
    uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    return [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]


def ci(x, bs, f=np.sum):
    v = [f(x[b]) for b in bs]
    return f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"


def dci(Va, Vb, L, bs):
    v = [(Va[b].sum() - Vb[b].sum()) / L[b].sum() for b in bs]
    return f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"


def recovery(dl, arm):
    n = f = 0
    for _, r in dl.iterrows():
        m, fo = r.get("deleted_mask"), r.get(f"found_{arm}")
        if isinstance(m, np.ndarray) and isinstance(fo, np.ndarray):
            n += int(m.sum()); f += int((fo & m).sum())
    return (f / n if n else np.nan), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds, root in (("nusc", B), ("av2", S), ("ahd", B)):
        d = load(root / "results" / "k17" / f"{ds}_test_base" / "*.pkl")
        if not len(d):
            continue
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        k14 = load(root / "results" / "k14" / f"{ds}__*.pkl")
        d = d.merge(k14[["unit", "frame"] + [f"{p}{h}{s}" for h in ("uncertainty", "temporal_crit") for p in ("J_", "a_") for s in ("", "_ref")]],
                    on=["unit", "frame"], how="left")
        L = d.J_cheap.to_numpy(float); bs = boots(d.unit.to_numpy())
        N = len(d); rows = []
        sec = lambda s, **k: rows.append({"dataset": ds, "section": s, **k})        # noqa: E731
        # ---- 1
        for ref in ("", "_ref"):
            m = (d[f"a_full{ref}"] != d[f"a_so{ref}"]).to_numpy()
            sec("1 frames where FULL and SO differ", refiner=bool(ref), group="all", frames=int(m.sum()), share=m.mean(),
                loss_full=d.loc[m, f"J_full{ref}"].sum(), loss_so=d.loc[m, f"J_so{ref}"].sum(), loss_cheap=d.loc[m, "J_cheap"].sum(),
                loss_F=d.loc[m, "J_u960"].sum(), full_raises=int((m & (d[f"a_full{ref}"] > d[f"a_so{ref}"])).sum()))
            for cat, nm in (("a", "a true discovery"), ("b", "b spurious detection"), ("c", "c draft object changed"), ("d", "d other")):
                mc = m & (d.field_cat == cat).to_numpy()
                sec("1 frames where FULL and SO differ", refiner=bool(ref), group=nm, frames=int(mc.sum()),
                    loss_full=d.loc[mc, f"J_full{ref}"].sum(), loss_so=d.loc[mc, f"J_so{ref}"].sum(),
                    loss_cheap=d.loc[mc, "J_cheap"].sum(), loss_F=d.loc[mc, "J_u960"].sum())
        # ---- 2
        ag = d.act_gt.to_numpy()
        ev = {"correct HARD_BRAKE": lambda a: (ag == 2) & (a == 2), "correct DECELERATE": lambda a: (ag == 1) & (a == 1),
              "unnecessary DECELERATE": lambda a: (ag == 0) & (a == 1), "unnecessary HARD_BRAKE": lambda a: (ag < 2) & (a == 2)}
        for ref in ("", "_ref"):
            af, aso = d[f"a_full{ref}"].to_numpy(), d[f"a_so{ref}"].to_numpy()
            for en, fn in ev.items():
                x = (fn(af).astype(float) - fn(aso).astype(float)) * 1000
                sec("2 events per 1,000 frames", refiner=bool(ref), group=en, full=1000 * fn(af).mean(), so=1000 * fn(aso).mean(),
                    full_minus_so_ci=ci(x, bs, np.mean))
        # ---- 3
        reg = ((ag >= 1) & (d.a_cheap.to_numpy() < ag))
        Lr = L[reg].sum()
        sec("3 regime: a* >= DECELERATE, draft has no object in the support of a*", group="frames", frames=int(reg.sum()), share=reg.mean())
        for p, nm in (("so", "support only"), ("uncertainty", "uncertainty crops"), ("temporal_crit", "temporal_crit crops"), ("full", "full")):
            if f"J_{p}" not in d:
                continue
            sec("3 regime: a* >= DECELERATE, draft has no object in the support of a*", group=nm, frames=int(reg.sum()),
                G_no_refiner=(Lr - d.loc[reg, f"J_{p}"].sum()) / Lr if Lr else np.nan,
                G_refiner=(Lr - d.loc[reg, f"J_{p}_ref"].sum()) / Lr if Lr and f"J_{p}_ref" in d else np.nan)
        # ---- 4
        Vf, Vfr = L - d.J_full.to_numpy(float), L - d.J_full_ref.to_numpy(float)
        for a in ["full", "so"] + sorted(c[2:] for c in d.columns if c.startswith("a_so_ctx") and not c.endswith("_ref")) + ["so_rand"]:
            if f"J_{a}" not in d:
                continue
            Vn, Vr = L - d[f"J_{a}"].to_numpy(float), L - d[f"J_{a}_ref"].to_numpy(float)
            r = {"group": a, "G_no_refiner": Vn.sum() / L.sum(), "G_refiner": Vr.sum() / L.sum(), "tensor_px": d[f"px_{a}"].mean(),
                 "px_ratio_to_full": d[f"px_{a}"].mean() / d.px_full.mean()}
            if a != "full":
                r["full_minus_ci_no_refiner"] = dci(Vf, Vn, L, bs)
                r["full_minus_ci_refiner"] = dci(Vfr, Vr, L, bs)
            sec("4 equal compute", **r)
        # ---- 5
        if ds in ("av2", "ahd"):
            dl = load(root / "results" / "k17" / f"{ds}_test_del_all" / "*.pkl")
            for a, vc in (("full", 0), ("vc5_car", 5), ("vc10_car", 10), ("so", None)):
                if f"J_{a}" not in d:
                    continue
                rec, n = recovery(dl, a) if len(dl) else (np.nan, 0)
                sec("5 closing speed", group=a, v_close=vc, G_no_refiner=(L - d[f"J_{a}"]).sum() / L.sum(),
                    G_refiner=(L - d[f"J_{a}_ref"]).sum() / L.sum(), tensor_px=d[f"px_{a}"].mean(),
                    agreement_u960=float((d[f"a_{a}"] == d.a_u960).mean()), recovery_deleted=rec, deleted_objects=n,
                    deletion_units=dl.unit.nunique() if len(dl) else 0)
        res = pd.DataFrame(rows)
        res.insert(1, "units", d.unit.nunique()); res.insert(2, "frames_total", N)
        res.to_csv(OUT / f"field_contribution_{ds}.csv", index=False)
        pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
