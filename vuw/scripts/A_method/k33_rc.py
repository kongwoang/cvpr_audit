#!/usr/bin/env python
"""K33 routers (A) and Chanakya-style policy (C), post-hoc on per-frame costs (amendment 1, handoff/K33/STATUS.md).
Every action = the cheap draft (always run) or the cheap draft + a whole frame at long side s + C1 + refiner v3a with the
strong arm's continuous history (K28's router convention). Per-frame costs:
  dev    K33 part pub dev (rt: s_bar, s_dvr, feat) joined with K28's stored YOLOv8s dev rows results/k28/<ds>_r2basewf_r2
  train  K33 part pub --split train (rt + wf rows) [Chanakya only]
  test   K33 part pub test (rt + wf rows; --setting del_all: the deletion run); Argoverse-HD takes AV2's thresholds /
         policy. Rows: the dev selection (dev_selection_rc.csv) + ch128B_reinforce (the dev-best Chanakya candidate,
         unmatched px, reference). Pixel-cap rows (@cap, not under del_all): a frame whose action exceeds the per-frame cap
         (v6-S's, rec capv) takes the largest whole frame within it (routers: any s in STRONG up to the router's s;
         Chanakya: its own actions), else cheap only. Output: results/k33/rc/test_<ds>[_del_all].pkl, per frame
Routers (BAR = skip_offbin score, DVR = cond_V score): candidates strong s in {640, 768, 960, 1280} x routed share in
{px-matched to v6-S's mean dev px, 25%, 50%}; a frame is routed iff its score >= the dev quantile of the share (per
dataset; thresholds fixed on dev). Chanakya: actions {cheap, 640, 960, 1280}; softmax MLP on the causal state; objective
min E[J(a) + lambda px(a)] (cost-sensitive classification, exact expectation over the 4 actions) or REINFORCE (sampled,
Chanakya's policy gradient); lambda by bisection so that the argmax policy's mean dev px equals v6-S's mean dev px;
candidates hidden {32, 128, 512} x features {A: 21 statistics + speed, B: A + draft demand, draft action, previous
action} x learner {csc, reinforce}. Selection: the highest mean dev G (nuScenes val, AV2 CFG-DEV) per family.
usage: k33_rc.py --stage dev | test [--setting del_all]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k33"
OUT = K / "tables"
STRONG = (640, 768, 960, 1280)
ACTS = (0, 640, 960, 1280)


def recs(d):
    rows = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows) if rows else None


def frame_table(ds, split, setting=""):
    """unit, frame, J_cheap, px_cheap, s_bar, s_dvr, feat, [capv,] J_<s>, px_<s> for s in STRONG"""
    p = recs(K / f"{ds}_pub_{split}{'_' + setting if setting else ''}")
    if p is None:
        return None
    p = p[p["gt"].astype(bool)] if split != "train" else p
    cols = ["unit", "frame", "J_cheap", "px_cheap", "s_bar", "s_dvr", "feat"] + (["capv"] if "capv" in p else [])
    t = p[cols].copy()
    if split == "dev":
        w = recs(B / "results" / "k28" / f"{ds}_r2basewf_r2")
        w = w[["unit", "frame"] + [f"J_wf{s}+C1|v3a" for s in STRONG] + [f"px_wf{s}" for s in STRONG]]
        w.columns = ["unit", "frame"] + [f"J_{s}" for s in STRONG] + [f"px_{s}" for s in STRONG]
        t = t.merge(w, on=["unit", "frame"], how="inner")
        assert len(t) == len(p), (ds, len(t), len(p))
    else:
        for s in STRONG:
            t[f"J_{s}"] = p[f"J_wf{s}|v3a"].to_numpy()
            t[f"px_{s}"] = p[f"px_wf{s}"].to_numpy()
    return t.reset_index(drop=True)


def v6S_dev_px(ds):
    d = recs(K / f"{ds}_de_dev")
    d = d[d["gt"].astype(bool)]
    return float(d["px_v6S"].mean())


def G(Jc, J):
    return (Jc.sum() - J.sum()) / Jc.sum()


# ------------------------------------------------------------------------------------------------ routers
def router_cands():
    return [(s, sh) for s in STRONG for sh in ("pxS", 0.25, 0.5)]


def router_apply(t, score, s, thr):
    r = t[score].to_numpy() >= thr
    J = np.where(r, t[f"J_{s}"].to_numpy(), t.J_cheap.to_numpy())
    px = t.px_cheap.to_numpy() + r * t[f"px_{s}"].to_numpy()
    return J, px, r


# ------------------------------------------------------------------------------------------------ Chanakya
def feats(t, fs):
    X = np.stack(t.feat.to_numpy()).astype(np.float32)
    return X[:, :22] if fs == "A" else X


def costs(t):
    Jm = np.stack([t.J_cheap.to_numpy()] + [t[f"J_{s}"].to_numpy() for s in ACTS[1:]], 1).astype(np.float32)
    Pm = np.stack([t.px_cheap.to_numpy()] + [t.px_cheap.to_numpy() + t[f"px_{s}"].to_numpy() for s in ACTS[1:]], 1).astype(np.float32)
    return Jm, Pm


def train_policy(X, Jm, Pm, lam, hidden, learner, seed=0, epochs=30):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy((X - mu) / sd)
    C = torch.from_numpy(Jm + lam * Pm)
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            logp = torch.log_softmax(net(Xt[i]), 1)
            if learner == "csc":
                loss = (logp.exp() * C[i]).sum(1).mean()
            else:
                a = torch.multinomial(logp.exp(), 1, generator=g).squeeze(1)
                r = -C[i].gather(1, a[:, None]).squeeze(1)
                loss = -((r - r.mean()) * logp.gather(1, a[:, None]).squeeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net, mu, sd


def act_policy(net, mu, sd, X):
    import torch
    with torch.no_grad():
        return net(torch.from_numpy((X - mu) / sd)).argmax(1).numpy()


def chanakya_cands():
    return [(h, fs, ln) for h in (32, 128, 512) for fs in ("A", "B") for ln in ("csc", "reinforce")]


def fit_matched(tr, dv, h, fs, ln, target):
    """bisection on log lambda so that the argmax policy's mean dev px equals the target"""
    Xtr, Jtr, Ptr = feats(tr, fs), *costs(tr)
    Xdv, Jdv, Pdv = feats(dv, fs), *costs(dv)
    lo, hi, best = -4.0, 4.0, None
    for _ in range(12):
        mid = (lo + hi) / 2
        net, mu, sd = train_policy(Xtr, Jtr, Ptr, 10 ** mid, h, ln)
        a = act_policy(net, mu, sd, Xdv)
        px = Pdv[np.arange(len(a)), a].mean()
        if best is None or abs(px - target) < abs(best[1] - target):
            best = (mid, px, net, mu, sd)
        if px > target:
            lo = mid
        else:
            hi = mid
    mid, px, net, mu, sd = best
    a = act_policy(net, mu, sd, Xdv)
    return {"log10_lambda": mid, "px_dev": float(px), "J_dev": Jdv[np.arange(len(a)), a], "net": (net, mu, sd),
            "share": np.bincount(a, minlength=4) / len(a)}


def select(D, tgt):
    """the highest mean dev G per family; a Chanakya candidate counts only if its dev px is within 5% of v6-S's on both
    dev splits (STATUS 2026-10-20 VUW 14:10, fixed before any Chanakya test number)"""
    ok = np.ones(len(D), bool)
    ch = (D.family == "Chanakya").to_numpy()
    for ds in ("nusc", "av2"):
        ok &= ~ch | (np.abs(D[f"px_{ds}"].to_numpy() / tgt[ds] - 1) <= 0.05)
    E = D[ok]
    return E.loc[E.groupby("family").G_mean.idxmax(), ["family", "row", "G_nusc", "G_av2", "G_mean", "px_nusc", "px_av2"]]


def stage_dev():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    (K / "rc").mkdir(parents=True, exist_ok=True)
    rows, thr = [], {}
    T = {ds: frame_table(ds, "dev") for ds in ("nusc", "av2")}
    tgt = {ds: v6S_dev_px(ds) for ds in T}
    for fam, score in (("BAR (skip_offbin)", "s_bar"), ("decision-value router (cond_V)", "s_dvr")):
        for s, sh in router_cands():
            r = {"family": fam, "row": f"{score[2:]}{s}_{sh}"}
            for ds, t in T.items():
                rho = (tgt[ds] - t.px_cheap.mean()) / t[f"px_{s}"].mean() if sh == "pxS" else sh
                rho = float(np.clip(rho, 0.0, 1.0))
                th = float(np.quantile(t[score], 1 - rho)) if rho > 0 else np.inf
                J, px, rt = router_apply(t, score, s, th)
                r.update({f"G_{ds}": G(t.J_cheap.to_numpy(), J), f"px_{ds}": px.mean(), f"routed_{ds}": rt.mean(), f"thr_{ds}": th})
                thr[(r["row"], ds)] = th
            r["G_mean"] = np.mean([r["G_nusc"], r["G_av2"]])
            rows.append(r)
    for ds in ("nusc", "av2"):
        T[ds + "_train"] = frame_table(ds, "train")
    for h, fs, ln in chanakya_cands():
        r = {"family": "Chanakya", "row": f"ch{h}{fs}_{ln}"}
        for ds in ("nusc", "av2"):
            m = fit_matched(T[ds + "_train"], T[ds], h, fs, ln, tgt[ds])
            r.update({f"G_{ds}": G(T[ds].J_cheap.to_numpy(), m["J_dev"]), f"px_{ds}": m["px_dev"],
                      f"lambda_{ds}": m["log10_lambda"], f"share_{ds}": json.dumps(np.round(m["share"], 4).tolist())})
            net, mu, sd = m["net"]
            torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "hidden": h, "fs": fs},
                       K / "rc" / f"chanakya_{r['row']}_{ds}.pt")
        r["G_mean"] = np.mean([r["G_nusc"], r["G_av2"]])
        rows.append(r)
        print(r, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "dev_candidates_rc.csv", index=False)
    S = select(D, tgt)
    S.to_csv(OUT / "dev_selection_rc.csv", index=False)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in thr.items()}, open(K / "rc" / "router_thresholds.json", "w"))
    json.dump(tgt, open(K / "rc" / "v6S_dev_px.json", "w"))
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))


def capped(t, acts, sz):
    """per frame: the largest size in acts (0 = cheap) not above the uncapped size sz and within the cap"""
    out = np.zeros(len(t), int)
    for s in sorted(x for x in set(acts) if x):
        fit = t.px_cheap.to_numpy() + t[f"px_{s}"].to_numpy() <= t.capv.to_numpy()
        out = np.where((sz >= s) & fit, s, out)
    return out


def apply_sizes(t, sz):
    J, px = t.J_cheap.to_numpy().copy(), t.px_cheap.to_numpy().copy()
    for s in STRONG:
        m = sz == s
        J[m] = t[f"J_{s}"].to_numpy()[m]
        px[m] += t[f"px_{s}"].to_numpy()[m]
    return J, px


def stage_test(setting):
    import torch
    sel = select(pd.read_csv(OUT / "dev_candidates_rc.csv"), json.load(open(K / "rc" / "v6S_dev_px.json")))
    sel.to_csv(OUT / "dev_selection_rc.csv", index=False)
    rows = list(dict.fromkeys(list(sel.row) + ["ch128B_reinforce"]))
    thr = json.load(open(K / "rc" / "router_thresholds.json"))
    for ds in ("nusc", "av2", "ahd"):
        src = "av2" if ds == "ahd" else ds
        t = frame_table(ds, "test", setting)
        out = t[["unit", "frame", "J_cheap", "px_cheap"] + (["capv"] if "capv" in t else [])].copy()
        for row in rows:
            if row.startswith("ch"):
                ck = torch.load(K / "rc" / f"chanakya_{row}_{src}.pt", weights_only=False)
                X = feats(t, ck["fs"])
                net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], ck["hidden"]), torch.nn.ReLU(),
                                          torch.nn.Linear(ck["hidden"], ck["hidden"]), torch.nn.ReLU(), torch.nn.Linear(ck["hidden"], 4))
                net.load_state_dict(ck["state"])
                sz = np.asarray(ACTS)[act_policy(net, ck["mu"], ck["sd"], X)]
                own = ACTS
            else:
                score, s = "s_" + row[:3], int(row[3:].split("_")[0])
                sz = np.where(t[score].to_numpy() >= thr[f"{row}|{src}"], s, 0)
                own = [x for x in STRONG if x <= s]
            for nm, z in ((row, sz),) + (((row + "@cap", capped(t, own, sz)),) if "capv" in t else ()):
                out[f"J_{nm}"], out[f"px_{nm}"] = apply_sizes(t, z)
                out[f"size_{nm}"] = z
        out.to_pickle(K / "rc" / f"test_{ds}{'_' + setting if setting else ''}.pkl")
        print(f"{ds}: {len(out)} frames, rows {rows}", flush=True)
    print("K33 RC TEST DONE", setting, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dev", "test"])
    ap.add_argument("--setting", default="")
    a = ap.parse_args()
    stage_dev() if a.stage == "dev" else stage_test(a.setting)


if __name__ == "__main__":
    main()
