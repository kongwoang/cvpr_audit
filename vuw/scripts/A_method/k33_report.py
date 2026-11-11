#!/usr/bin/env python
"""K33 report tables (handoff/K33/REPORT.md; pre-registration table1_design.md + STATUS amendment 1). CPU. This is the
reporting step: it is the only K33 script that reads test costs. Rows with components = + C1 + refiner v3a ("|v3a").
  table1_v7.csv        per dataset x row:
                         G = (sum J_cheap - sum J) / sum J_cheap, px (tensor px / native, the cheap draft included when
                         the row runs it), delta = G(v6-S) - G(row) with the paired unit bootstrap 95% CI (2,000
                         resamples, seed 0; > 0 = v6-S better), the same for the row under v6-S's per-frame pixel cap
                         (@cap; v6-S itself uncapped), spurious braking boxes per 1,000 frames (K30 V7, from the _spur
                         re-run, whose costs must equal the first run's), Jetson latency (--latency csv; mean / p95 ms)
  missed_natural.csv   K28 K4 A: frames where the draft misses a decision-relevant object that u960 + comp finds;
                       recovery rate of those objects and G on those frames
  missed_deletion.csv  K28 K4 B (del_all): recovery of the deleted draft detections and loss vs the unmodified draft
                       (strata: all, ego-speed terciles; the deleted objects' ranges were not stored, so no range strata)
  ft_rows.csv          F (AV2 val, ego mask): the fine-tuned rows (+ comp) and their official forms ("_off": DEEP mono
                       lift, no refiner); delta vs v6-S-ft; HARD-on-KEEP share
  ft_mechanism.csv     F: the decisive object's estimated range per fine-tuned row over COCO u960 + comp's on the same
                       frame (median ratio, shares < 0.9 / > 1.1) and the decisive object's found rate
Routers / Chanakya (k33_rc.py): a frame routed to the whole frame at s takes that row's boxes (found / spurious of
wf<s> + comp), else the cheap draft's.
usage: k33_report.py [--latency csv] [--datasets nusc,av2,ahd]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k33"
OUT = K / "tables"
WF = (640, 768, 960, 1280)
# (family, row, source); source de / pub = k33_eval.py parts, rc = k33_rc.py, ref = reference rows
ROWS = [("reference", "cheap", "ref"), ("ours", "v6S", "de")] + [("whole frame", f"wf{s}", "pub") for s in WF] + [
    ("A AdaScale", "ada+1t1280", "pub"), ("A SAHI", "sahi640o1s10", "pub"), ("A FOVEA KDE_I", "fovi1280b64a2", "pub"),
    ("A FOVEA KDE_S", "fovs1280b64a1", "pub"), ("A BAR", "bar1280_0.5", "rc"), ("A decision-value router", "dvr1280_0.5", "rc"),
    ("B UFP", "ufp20s512", "pub"), ("C Chanakya", "ch128B_csc", "rc"),
    ("C Chanakya (dev-best, px unmatched; reference)", "ch128B_reinforce", "rc"),
    ("D AnA one pass", "ana05hinfo", "de"), ("D AnA iterated", "ana05hinfi", "de"),
    ("D AnA (t 0.25, reported in any case)", "ana25hinfo", "de"),
    ("E counterfactual one pass", "cf40q20o", "de"), ("E counterfactual iterated", "cf40q20i", "de")]
CAPPED = {r for f, r, s in ROWS if s in ("de", "pub", "rc") and r != "v6S" and not r.startswith("wf")}


def recs(d):
    rows = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows) if rows else None


def frames(ds, setting=""):
    """de + pub (+ the _spur re-run's spur columns) + rc, GT frames, keyed by (unit, frame)"""
    sfx = "_del_all" if setting else ""
    de, pub = recs(K / f"{ds}_de_test{sfx}"), recs(K / f"{ds}_pub_test{sfx}")
    de, pub = de[de["gt"].astype(bool)], pub[pub["gt"].astype(bool)]
    key = ["unit", "frame"]
    d = de.merge(pub[key + [c for c in pub.columns if c not in de.columns]], on=key, how="inner", validate="1:1")
    assert len(d) == len(de) == len(pub), (ds, setting, len(d), len(de), len(pub))
    chk = de[key + ["J_cheap"]].merge(pub[key + ["J_cheap"]], on=key)
    assert np.array_equal(chk.J_cheap_x.to_numpy(), chk.J_cheap_y.to_numpy()), (ds, "cheap differs between parts")
    rc = pd.read_pickle(K / "rc" / f"test_{ds}{sfx}.pkl")
    d = d.merge(rc[key + [c for c in rc.columns if c not in d.columns]], on=key, how="inner", validate="1:1")
    assert len(d) == len(de), (ds, setting, "rc frames", len(d), len(de))
    d.attrs["spur"] = False
    if not setting:
        sp = [recs(K / f"{ds}_{part}_test_spur") for part in ("de", "pub")]
        if all(x is not None for x in sp):
            s = sp[0].merge(sp[1][key + [c for c in sp[1].columns if c not in sp[0].columns]], on=key, how="inner")
            s = s[s["gt"].astype(bool)]
            m = d[key].merge(s, on=key, how="left", indicator=True)
            if (m["_merge"] == "both").all():
                Jc = [c for c in d.columns if c.startswith("J_") and c in m.columns]
                bad = [c for c in Jc if not np.allclose(d[c].to_numpy(float), m[c].to_numpy(float), equal_nan=True)]
                assert not bad, (ds, "the _spur re-run's costs differ from the first run's", bad[:5])
                for c in [c for c in m.columns if c.startswith("spur_")]:
                    d[c] = m[c].to_numpy()
                d.attrs["spur"] = True
    d.attrs["ds"] = ds
    return d.reset_index(drop=True)


def cols(d, row, cap=False):
    """per frame J, px, found (object arrays or None), spur (or None) of a row"""
    nm = row + ("@cap" if cap else "")
    if row == "cheap":
        return d.J_cheap.to_numpy(float), d.px_cheap.to_numpy(float), d.get("found_cheap"), d.get("spur_cheap")
    src = dict((r, s) for _, r, s in ROWS)[row]
    if src == "rc":
        sz = d[f"size_{nm}"].to_numpy()
        fd = [(r_[f"found_wf{z}"] if z else r_.get("found_cheap")) for z, (_, r_) in zip(sz, d.iterrows())] if "found_cheap" in d else None
        sp = None
        if d.attrs["spur"]:
            sp = np.array([(r_[f"spur_wf{z}|v3a"] if z else r_["spur_cheap"]) for z, (_, r_) in zip(sz, d.iterrows())], float)
        return d[f"J_{nm}"].to_numpy(float), d[f"px_{nm}"].to_numpy(float), pd.Series(fd, index=d.index) if fd is not None else None, sp
    return (d[f"J_{nm}|v3a"].to_numpy(float), d[f"px_{nm}"].to_numpy(float), d.get(f"found_{nm}"),
            d[f"spur_{nm}|v3a"].to_numpy(float) if d.attrs["spur"] and f"spur_{nm}|v3a" in d else None)


def boot_idx(units, n=2000, seed=0):
    return np.random.default_rng(seed).integers(0, len(units), (n, len(units)))


def delta(d, J, Jv, idx, units):
    """G(v6-S) - G(row) = (sum J - sum J_v6S) / sum J_cheap, paired unit bootstrap percentile CI"""
    g = d.assign(_j=J, _v=Jv).groupby("unit")[["J_cheap", "_j", "_v"]].sum().reindex(units)
    c, j, v = (g[x].to_numpy() for x in ("J_cheap", "_j", "_v"))
    pt = (j.sum() - v.sum()) / c.sum()
    bs = (j[idx].sum(1) - v[idx].sum(1)) / c[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi


def table1(ds, lat):
    d = frames(ds)
    units = np.array(sorted(d.unit.unique()))
    idx = boot_idx(units)
    Jc = d.J_cheap.to_numpy(float)
    Jv = cols(d, "v6S")[0]
    out = []
    for fam, row, src in ROWS:
        J, px, _, sp = cols(d, row)
        r = {"dataset": ds, "family": fam, "row": row, "frames": len(d), "units": len(units),
             "G": (Jc.sum() - J.sum()) / Jc.sum(), "px": px.mean()}
        r["delta_v6S_minus_row"], r["delta_lo"], r["delta_hi"] = delta(d, J, Jv, idx, units) if row != "v6S" else (np.nan,) * 3
        if row in CAPPED:
            Jk, pxk, _, spk = cols(d, row, cap=True)
            r.update({"G_cap": (Jc.sum() - Jk.sum()) / Jc.sum(), "px_cap": pxk.mean()})
            r["delta_cap"], r["delta_cap_lo"], r["delta_cap_hi"] = delta(d, Jk, Jv, idx, units)
            r["spur_per_1k_cap"] = 1e3 * spk.sum() / len(d) if spk is not None else np.nan
        r["spur_per_1k"] = 1e3 * sp.sum() / len(d) if sp is not None else np.nan
        if lat is not None:
            for _, lr in lat[(lat.dataset == ds) & (lat.row == row)].iterrows():
                r.update({f"jetson_{k_}": v_ for k_, v_ in lr.items() if k_ not in ("dataset", "row")})
        out.append(r)
    return out, d


def missed_natural(d):
    ds = d.attrs["ds"]
    has = d["found_cheap"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0) if "found_cheap" in d else pd.Series(False, index=d.index)
    nat, miss = [], {}
    for i in np.flatnonzero(has.to_numpy()):
        fc, fu = np.asarray(d.found_cheap.iat[i], bool), np.asarray(d["found_wf960"].iat[i], bool)
        m = (~fc) & fu
        if m.any():
            nat.append(i); miss[i] = m
    sub = d.iloc[nat]
    L = sub.J_cheap.to_numpy(float)
    rows = []
    for fam, row, _ in ROWS:
        J, _, fd, _ = cols(d, row)
        n = rec_ = 0
        for i in nat:
            f_ = fd.iat[i] if fd is not None else None
            fp = np.asarray(f_, bool) if isinstance(f_, (list, np.ndarray)) and len(f_) else np.zeros(len(miss[i]), bool)
            n += int(miss[i].sum()); rec_ += int((fp & miss[i]).sum())
        Js = J[nat]
        rows.append({"dataset": ds, "family": fam, "row": row, "frames": len(nat), "missed_objects": n,
                     "recovery_rate": rec_ / max(n, 1), "G_on_these_frames": (L.sum() - Js.sum()) / L.sum() if L.sum() > 0 else np.nan})
    return rows


def missed_deletion(ds):
    e = frames(ds, "del_all")
    e = e[e.deleted > 0].reset_index(drop=True)
    e.attrs["ds"], e.attrs["spur"] = ds, False
    tert = np.quantile(e.v, [1 / 3, 2 / 3])
    st = np.digitize(e.v, tert)
    rows = []
    for fam, row, _ in ROWS:
        J, _, fd, _ = cols(e, row)
        for sname, msk in [("all", np.ones(len(e), bool))] + [(f"speed tercile {k + 1}", st == k) for k in range(3)]:
            n = rec_ = 0
            for i in np.flatnonzero(msk):
                dm = np.asarray(e.deleted_mask.iat[i], bool)
                f_ = fd.iat[i] if fd is not None else None
                fp = np.asarray(f_, bool) if isinstance(f_, (list, np.ndarray)) and len(f_) else np.zeros(len(dm), bool)
                n += int(dm.sum()); rec_ += int(fp[dm].sum())
            Lu = e.J_cheap_unmod.to_numpy(float)[msk].sum()
            rows.append({"dataset": ds, "stratum": sname, "family": fam, "row": row, "frames": int(msk.sum()), "deleted_objects": n,
                         "recovery_rate": rec_ / max(n, 1) if row != "cheap" else 0.0,
                         "loss_vs_unmodified_draft": (J[msk].sum() - Lu) / Lu if Lu > 0 else np.nan})
    return rows


def ft_rows():
    d = recs(K / "av2_ft_test")
    d = d[d["gt"].astype(bool)].reset_index(drop=True)
    units = np.array(sorted(d.unit.unique()))
    idx = boot_idx(units)
    Jc = d.J_cheap.to_numpy(float)
    Jv = d["J_v6Sft|v3a"].to_numpy(float)
    out = []
    for c in sorted(c for c in d.columns if c.startswith("J_") and "|" in c):
        row, rf = c[2:].split("|")
        J = d[c].to_numpy(float)
        a = d[f"a_{row}|{rf}"].to_numpy()
        keep = d.act_gt.to_numpy() == 0
        r = {"row": row, "refiner": rf, "frames": len(d), "G": (Jc.sum() - J.sum()) / Jc.sum(),
             "px": d[f"px_{row}"].mean() if f"px_{row}" in d else np.nan, "hard_on_keep": float((a[keep] == 2).mean())}
        r["delta_v6Sft_minus_row"], r["delta_lo"], r["delta_hi"] = delta(d, J, Jv, idx, units)
        out.append(r)
    a = d["a_cheap"].to_numpy() if "a_cheap" in d else None
    if a is not None:
        out.append({"row": "cheap", "refiner": "-", "frames": len(d), "G": 0.0, "px": d.px_cheap.mean(),
                    "hard_on_keep": float((a[d.act_gt.to_numpy() == 0] == 2).mean())})
    return out


def f_mechanism():
    """F: range of the decisive object (the GT object with the largest braking demand) as estimated by each fine-tuned
    row (+ comp) over the COCO u960 + comp estimate on the same frame and object (both matched; AV2 val, ego mask)"""
    ft, pub = recs(K / "av2_ft_test"), recs(K / "av2_pub_test")
    d = ft[ft["gt"].astype(bool)].merge(pub[pub["gt"].astype(bool)][["unit", "frame", "dec_found_wf960", "dec_z_wf960"]],
                                        on=["unit", "frame"], how="inner")
    out = []
    for row in sorted(c[len("dec_found_"):] for c in d.columns if c.startswith("dec_found_")):
        m = d[f"dec_found_{row}"].fillna(False).astype(bool) & d["dec_found_wf960"].fillna(False).astype(bool)
        r = (d.loc[m, f"dec_z_{row}"].to_numpy(float) / d.loc[m, "dec_z_wf960"].to_numpy(float))
        r = r[np.isfinite(r) & (r > 0)]
        out.append({"row": row, "frames": int(m.sum()), "median_range_ratio_vs_coco_u960": float(np.median(r)) if len(r) else np.nan,
                    "share_ratio_below_0.9": float((r < 0.9).mean()) if len(r) else np.nan,
                    "share_ratio_above_1.1": float((r > 1.1).mean()) if len(r) else np.nan,
                    "decisive_found_rate": float(d[f"dec_found_{row}"].fillna(False).astype(bool)[d.dec >= 0].mean()) if "dec" in d else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", default="")
    ap.add_argument("--datasets", default="nusc,av2,ahd")
    a = ap.parse_args()
    lat = pd.read_csv(a.latency) if a.latency else None
    OUT.mkdir(parents=True, exist_ok=True)
    T, MN, MD = [], [], []
    for ds in a.datasets.split(","):
        t, d = table1(ds, lat)
        T += t
        MN += missed_natural(d)
        MD += missed_deletion(ds)
        print(f"{ds}: {len(d)} frames; spur columns {'yes' if d.attrs['spur'] else 'no'}", flush=True)
    pd.DataFrame(T).to_csv(OUT / "table1_v7.csv", index=False)
    pd.DataFrame(MN).to_csv(OUT / "missed_natural.csv", index=False)
    pd.DataFrame(MD).to_csv(OUT / "missed_deletion.csv", index=False)
    pd.DataFrame(ft_rows()).to_csv(OUT / "ft_rows.csv", index=False)
    pd.DataFrame(f_mechanism()).to_csv(OUT / "ft_mechanism.csv", index=False)
    print("K33 REPORT TABLES DONE", flush=True)


if __name__ == "__main__":
    main()
