#!/usr/bin/env python
"""K40 (private helper): Jetson AGX Xavier per-frame timing records -> <out>/jetson_<window>.parquet (long format: one row
per run x pipeline x timed frame: window, dataset, config, rep, run (index), tag, unit, frame, warm, pipeline, ms, a, J,
J_cheap, px, execs). Run folders come from the windows' own manifests:
  latency    handoff/K29/latency_runs.csv (the K29 test window: trt32 fast path and the reference path)
  hybrid     results/raw/*_k30j_timing_<ds>_v6S_j1x_r<rep> (the hybrid and heuristics at DESP's per-frame pixels, nominal cap)
  baselines  results/raw/*_k33j_timing_<ds>_v6S_k33sel<group>_r<rep> (the K33 timing window of the compared methods)
  bounded    handoff/K34/jetson_frames.csv (the K34 timing window: v6-S and its <= 2 batch variants; timed frames only)
  deadline   experiments/k38_deadline_20261026 + experiments/k39_jetson_20261027 (logs/state.json, staged repetitions):
             wide per-frame records (source deadline | frontier; a_ / ms_ / px_ / execs_ / dl_stopped_<pipeline>, a_gt from
             the stored GT demand, prev_cheap = previous cheap action of the run, warm-up frames included)
  energy     handoff/K30/J/j2_energy_marginal.csv (run-level INA3221 energy; one row per dataset x run)
usage: scripts/py scripts/k40_export_jetson.py <out_dir> [windows]"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def runs_latency():
    t = pd.read_csv(ROOT / "docs/paper/handoff/K29/latency_runs.csv")
    for run in sorted(t.run.unique()):
        cfg = json.loads((RAW / run / "config.json").read_text())
        yield run, cfg.get("dataset"), cfg.get("config"), cfg.get("tag", "")


def runs_baselines():
    import re
    for d in sorted(RAW.glob("*_k33j_timing_*_v6S_k33sel*_r[0-9]")):
        m = re.search(r"k33j_timing_(nusc_test20|av2)_v6S_k33sel([a-z]+)_r(\d)$", d.name)
        if m and (d / "frames.pkl").exists():
            yield d.name, m[1], "v6S", f"r{m[3]}_{m[2]}"


def runs_hybrid():
    import re
    for d in sorted(RAW.glob("*_k30j_timing_*_v6S_j1x_r[0-9]")):
        m = re.search(r"k30j_timing_(nusc_test20|av2)_v6S_j1x_r(\d)$", d.name)
        if m and (d / "frames.pkl").exists():
            yield d.name, m[1], "v6S", f"r{m[2]}"


WINDOWS = {"latency": runs_latency, "baselines": runs_baselines, "hybrid": runs_hybrid}
HO = ROOT / "docs/paper/handoff"


def bounded():
    t = pd.read_csv(HO / "K34/jetson_frames.csv")
    x = pd.DataFrame({"dataset": t.dataset, "config": "v6S", "tag": "r" + t.rep.astype(str), "run": t.rep - 1,
                      "unit": t.unit.astype(str), "frame": t.frame.astype(int), "warm": False, "pipeline": t.pipeline,
                      "ms": t.latency_ms.astype(float), "a": t.action.astype(float), "J": t.decision_loss.astype(float),
                      "J_cheap": float("nan"), "px": t.tensor_px.astype(float),
                      "execs": (t.detector_engine_executions + t.crop_engine_executions).astype(float)})
    return x


def energy():
    t = pd.read_csv(HO / "K30/J/j2_energy_marginal.csv")
    return t.drop(columns=["wall_ms_above_cheap_run"]).rename(columns={"run": "pipeline"})


def deadline():
    gt = pd.read_csv(ROOT / "experiments/k39_jetson_20261027/gt_k30.csv")
    gt["unit"] = gt.unit.astype(str)
    out = []
    for src, e in (("deadline", "k38_deadline_20261026"), ("frontier", "k39_jetson_20261027")):
        st = json.loads((ROOT / "experiments" / e / "logs/state.json").read_text())
        for r in st["runs"]:
            if r["smoke"] or r["runtime"] != "staged" or not r["tag"].startswith("r"):
                continue
            d = pd.read_pickle(Path(r["run"]) / "frames.pkl")
            d["prev_cheap"] = d.groupby("unit").a_cheap.shift(1)
            d = d[~d.warm].reset_index(drop=True)
            d["unit"] = d.unit.astype(str)
            g = gt[gt.dataset == {"nusc_test20": "nusc", "av2t40": "av2"}[r["dataset"]]][["unit", "frame", "a_gt"]]
            d = d.drop(columns=["a_gt"], errors="ignore").merge(g, on=["unit", "frame"], how="left", validate="1:1")
            assert d.a_gt.notna().all(), (src, r["dataset"])
            keep = ["unit", "frame", "a_gt", "act_gt", "prev_cheap"] + [c for c in d.columns if c.startswith(("a_", "ms_", "px_", "execs_", "dl_stopped_")) and c != "a_gt"]
            x = d[keep].copy()
            x.insert(0, "rep", int(r["rep"])); x.insert(0, "dataset", r["dataset"]); x.insert(0, "source", src)
            out.append(x.copy())
    return pd.concat(out, ignore_index=True)


SPECIAL = {"bounded": bounded, "energy": energy, "deadline": deadline}


def long_frames(run, ds, cfg, tag, k):
    d = pd.read_pickle(RAW / run / "frames.pkl")
    d["unit"] = d.unit.astype(str)
    out = []
    for c in [c for c in d.columns if c.startswith("ms_")]:
        nm = c[3:]
        x = pd.DataFrame({"unit": d.unit, "frame": d.frame.astype(int), "warm": d.warm.astype(bool), "pipeline": nm,
                          "ms": d[c].astype(float),
                          "a": d.get(f"a_{nm}", pd.Series(index=d.index, dtype=float)).astype(float),
                          "J": d.get(f"J_{nm}", pd.Series(index=d.index, dtype=float)).astype(float),
                          "J_cheap": d.J_cheap.astype(float) if "J_cheap" in d else float("nan"),
                          "px": d.get(f"px_{nm}", pd.Series(index=d.index, dtype=float)).astype(float),
                          "execs": d.get(f"execs_{nm}", pd.Series(index=d.index, dtype=float)).astype(float)})
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    x.insert(0, "run", k); x.insert(0, "tag", tag); x.insert(0, "config", cfg); x.insert(0, "dataset", ds)
    return x


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    wins = sys.argv[2].split(",") if len(sys.argv) > 2 else list(WINDOWS) + list(SPECIAL)
    for w in wins:
        if w in SPECIAL:
            x = SPECIAL[w]()
        else:
            x = pd.concat([long_frames(run, ds, cfg, tag, k) for k, (run, ds, cfg, tag) in enumerate(WINDOWS[w]())], ignore_index=True)
        x.insert(0, "window", w)
        f = od / f"jetson_{w}.parquet"
        x.to_parquet(f, index=False, compression="zstd")
        print(w, x.shape, round(f.stat().st_size / 2**20, 1), "MB", len(x.drop_duplicates([c for c in ("source", "dataset", "run", "rep", "pipeline") if c in x])), "runs")


if __name__ == "__main__":
    main()
