#!/usr/bin/env python
"""K20 tables (post-hoc analysis) from results/k20 (k20_pdm / k20_fast / k20_variants) and the PDMS runs.
  pdm_cost.csv      planner calls per frame (probe / support), wall clock per call / per frame, stage split, parity
  pdm_variants.csv  frozen spec_pdm and its identical-output implementations (fast, N workers) and approximations
                    (g5, g3, corridor of the plan; closed-form braking field from the paper): PDMS on navtest, paired
                    log bootstrap (2,000) of the variant minus frozen spec_pdm, ms per frame, planner calls, IoU of the
                    sensitive image footprint with the 7 x 7 field
  pdm_pairs.csv     redundancy pairs and probe + removal interactions on the 1,000-token subset
  b2d_calls.csv     Bench2Drive (K6): planner queries of the perception field and wall clock per step
usage: k20_analyze.py
"""
from __future__ import annotations

import os
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
R = B / "results" / "k20"
X = B / "results" / "navsim_exp"


def load(prefix):
    rows = []
    for f in sorted(R.glob(f"{prefix}__*.pkl")):
        rows += pickle.load(open(f, "rb"))
    return rows


def boot(units, x, n=2000):
    uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    v = [np.nanmean(x[np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))])]) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    cpu = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.split(":")[-1].strip()
    # ---------------- (1) cost
    cost = [r for r in load("cost") if "calls" in r]
    calls = pd.DataFrame([dict(c, token=r["token"]) for r in cost for c in r["calls"]])
    tok = pd.DataFrame([{k: v for k, v in r.items() if k != "calls"} for r in cost])
    st = ["setup", "observation", "proposal_manager", "generation", "simulation", "scoring", "extract"]
    calls["other_planner"] = calls.planner - calls[["observation", "proposal_manager", "generation", "simulation", "scoring"]].sum(1)
    rows = [{"quantity": "tokens (1,000 drawn with default_rng(0) from the frozen field's navtest tokens)", "value": len(tok)},
            {"quantity": "CPU", "value": f"{cpu}, 1 thread per process (OMP/MKL = 1), 12 processes on a shared 32-thread host"},
            {"quantity": "planner calls per frame, mean", "value": (1 + tok.n_probe + tok.n_support_calls).mean()},
            {"quantity": "planner calls per frame, p95", "value": (1 + tok.n_probe + tok.n_support_calls).quantile(.95)},
            {"quantity": "probe calls per frame, mean / p95", "value": f"{tok.n_probe.mean():.1f} / {tok.n_probe.quantile(.95):.0f}"},
            {"quantity": "support calls per frame, mean / p95", "value": f"{tok.n_support_calls.mean():.1f} / {tok.n_support_calls.quantile(.95):.0f}"},
            {"quantity": "wall clock per frame, mean / p95 (s)", "value": f"{tok.wall_token.mean():.2f} / {tok.wall_token.quantile(.95):.2f}"},
            {"quantity": "parity with the stored frozen field (sensitive cells / support), share of tokens",
             "value": f"{tok.parity_field.mean():.4f} / {tok.parity_support.mean():.4f}"}]
    for k in ("base", "probe", "support"):
        c = calls[calls.kind == k]
        rows.append({"quantity": f"wall clock per {k} call, mean / p95 (ms)", "value": f"{1000 * c.wall.mean():.1f} / {1000 * c.wall.quantile(.95):.1f}"})
    c = calls[calls.kind != "base"]
    for s in st + ["other_planner"]:
        rows.append({"quantity": f"share of a probe/support call: {s}", "value": round(float(c[s].sum() / c.wall.sum()), 4)})
    pd.DataFrame(rows).to_csv(R / "pdm_cost.csv", index=False)
    # ---------------- (2) + (3) variants
    var = []
    fast = pd.DataFrame([r for r in load("fast") if "err" not in r])
    if len(fast):
        var.append({"variant": "frozen spec_pdm (navsim_field, per call scenario rebuild)", "identical_output": "reference",
                    "ms_per_frame": 1000 * tok.wall_token.mean(), "planner_calls": (1 + tok.n_probe + tok.n_support_calls).mean()})
        var.append({"variant": "fast: planner state built once per token, roll-outs reused for identical proposals",
                    "identical_output": f"poses bit-identical on {int((fast.pose_maxdiff == 0).sum())}/{int(fast.pose_maxdiff.notna().sum())} "
                                        f"checked tokens; field parity {fast.parity_field.mean():.4f}, support parity {fast.parity_support.mean():.4f}",
                    "ms_per_frame": 1000 * fast.wall_token.mean(), "planner_calls": fast.n_calls.mean(),
                    "rollouts_reused_per_frame": fast.sim_reused.mean()})
    for w in (4, 8, 16):
        f = R / f"parallel_w{w}.pkl"
        if f.exists():
            p = pd.DataFrame(pickle.load(open(f, "rb")))
            var.append({"variant": f"fast, calls of a frame over {w} worker processes (200 tokens)", "identical_output": "same calls as fast",
                        "ms_per_frame": 1000 * p.wall_token.mean(), "planner_calls": p.n_calls.mean()})
    fin = pd.read_csv(X / "pdms_states_final.csv") if (X / "pdms_states_final.csv").exists() else None
    cells = pd.read_csv(R / "variant_cells.csv") if (R / "variant_cells.csv").exists() else None
    per_call = calls[calls.kind != "base"].wall.mean()
    for v, name in (("g5", "5 x 5 grid"), ("g3", "3 x 3 grid (ego lane +- one lane)"), ("corridor", "cells in the corridor of the current plan")):
        f = X / f"pdms_states_k20_{v}.csv"
        if fin is None or not f.exists():
            continue
        d = pd.read_csv(f)[["log", "token", "score_spec_pdm"]].merge(fin[["log", "token", "score_spec_pdm"]], on=["log", "token"], suffixes=("", "_frozen"))
        diff = (d.score_spec_pdm - d.score_spec_pdm_frozen).to_numpy()
        lo, hi = boot(d.log.to_numpy(), diff)
        calls_v = (1 + cells[f"probed_{v}"] + cells.support_calls_frozen).mean()
        var.append({"variant": f"approximation: {name}", "identical_output": "no", "PDMS": d.score_spec_pdm.mean(),
                    "PDMS_frozen_same_tokens": d.score_spec_pdm_frozen.mean(), "minus_frozen": np.nanmean(diff), "ci_lo": lo, "ci_hi": hi,
                    "planner_calls": calls_v, "ms_per_frame": 1000 * per_call * calls_v,
                    "ms_note": "planner calls x the mean frozen probe/support call (derived)",
                    "iou_sensitive_footprint": cells[f"iou_{v}"].mean(), "tokens": len(d)})
    for v, name in (("pairs", "support + objects of redundancy pairs (1,000-token subset)"),
                    ("pairs_probe", "support + pair objects + probe-interacting objects (1,000-token subset)")):
        f = X / f"pdms_states_k20_{v}.csv"
        if fin is None or not f.exists():
            continue
        d = pd.read_csv(f)[["log", "token", "score_spec_pdm"]].merge(fin[["log", "token", "score_spec_pdm"]], on=["log", "token"], suffixes=("", "_frozen"))
        diff = (d.score_spec_pdm - d.score_spec_pdm_frozen).to_numpy()
        lo, hi = boot(d.log.to_numpy(), diff)
        var.append({"variant": name, "identical_output": "no", "PDMS": d.score_spec_pdm.mean(), "PDMS_frozen_same_tokens": d.score_spec_pdm_frozen.mean(),
                    "minus_frozen": np.nanmean(diff), "ci_lo": lo, "ci_hi": hi, "tokens": len(d),
                    "tokens_changed_score": int((np.abs(diff) > 1e-9).sum())})
    if fin is not None:
        var.append({"variant": "frozen spec_pdm on navtest", "PDMS": fin.score_spec_pdm.mean(), "tokens": int(fin.score_spec_pdm.notna().sum())})
    var.append({"variant": "closed-form braking field as proxy (paper, existing number)", "PDMS": 0.8926, "planner_calls": 0, "identical_output": "no"})
    pd.DataFrame(var).to_csv(R / "pdm_variants.csv", index=False)
    # ---------------- (4) pairs
    pr = pd.DataFrame([r for r in load("pairs") if "err" not in r])
    if len(pr):
        pairs_rows = [{"quantity": "tokens", "value": len(pr)},
                      {"quantity": "object pairs tested (both singles do not move the plan)", "value": int(pr.n_pairs_tested.sum())},
                      {"quantity": "pairs whose joint removal moves the plan > 0.5 m", "value": int(pr.pairs_moving.map(len).sum())},
                      {"quantity": "share of tested pairs that move the plan", "value": pr.pairs_moving.map(len).sum() / max(pr.n_pairs_tested.sum(), 1)},
                      {"quantity": "tokens with at least one such pair", "value": int((pr.pairs_moving.map(len) > 0).sum())},
                      {"quantity": "probe + removal combinations tested (neither moves alone)", "value": int(pr.n_probe_drop_tested.sum())},
                      {"quantity": "probe + removal combinations that move the plan", "value": int(pr.probe_drop_moving.map(len).sum())},
                      {"quantity": "share of probe + removal combinations that move the plan",
                       "value": pr.probe_drop_moving.map(len).sum() / max(pr.n_probe_drop_tested.sum(), 1)},
                      {"quantity": "tokens with at least one probe + removal interaction", "value": int((pr.probe_drop_moving.map(len) > 0).sum())}]
        pd.DataFrame(pairs_rows).to_csv(R / "pdm_pairs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(var).round(4).to_string(index=False))
    print(platform.node())


if __name__ == "__main__":
    main()
