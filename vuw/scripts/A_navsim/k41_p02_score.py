#!/usr/bin/env python
"""K41-2 P02 scoring + summary (PREREG K41-2: "PDMS for each joint effect"), CPU only.
  score    every P02 token with at least one joint effect: official NAVSIM v1.1 PDMS of the draft (base) plan and of
           every stored joint-effect trajectory, on a metric cache rebuilt from the original scene (as P01 / D07; the
           temporary cache is removed). -> results/k41/p02_score/<log>.pkl
  summary  per family (removal_triple, insertion_pair, insertion_triple, k2_pair): tokens tested, combinations tested
           / available, moving combinations, tokens with >= 1 moving combination, Wilson 95% CI of the per-combination
           and per-token rates, planner calls and wall time per family; PDMS of every joint effect vs its base plan.
           -> <out_dir>/k41_2_p02_families.csv, k41_2_p02_joint_pdms.csv, k41_2_p02_cost.csv
usage (env.sh; scripts/common/env_navsim.sh; CUDA hidden): k41_p02_score.py --stage score [--shard i/n] | --stage summary --out_dir D
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import lzma
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
P02 = B / "results" / "k41" / os.environ.get("K41_P02_IN", "p02")            # P03: K41_P02_IN=p03
OUT = B / "results" / "k41" / os.environ.get("K41_P02_SCORE", "p02_score")  # P03: K41_P02_SCORE=p03_score
FAM = ("removal_triple", "insertion_pair", "insertion_triple", "k2_pair")


def count(r, k):
    """(tested, available) combinations of family k on token row r"""
    return (r["k2_pairs"],) * 2 if k == "k2_pair" else tuple(r["removal_triples" if k == "removal_triple" else k])


def rows():
    return sorted((r for f in sorted(P02.glob("*.pkl")) for r in pickle.load(open(f, "rb"))), key=lambda r: (r["log"], r["token"]))


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing")
    with open(tmp, "wb") as f:
        pickle.dump(data, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def stage_score(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    data = Path(os.environ["OPENSCENE_DATA_ROOT"])
    cfg = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    params = OmegaConf.load(cfg); simulator = instantiate(params.simulator); scorer = instantiate(params.scorer)
    ts = TrajectorySampling(time_horizon=4, interval_length=.5)
    si, sn = (int(x) for x in args.shard.split("/"))
    by_log = {}
    for r in rows():
        if "err" not in r and any(r["joint"].get(k) for k in FAM):
            by_log.setdefault(r["log"], []).append(r)
    temp = OUT / "temporary_metric_cache" / str(os.getpid()); temp.mkdir(parents=True, exist_ok=True)
    for li, log in enumerate(sorted(by_log)):
        fo = OUT / f"{log}.pkl"
        if li % sn != si or fo.exists():
            continue
        assert shutil.disk_usage(B).free >= 200 * 1024 ** 3
        rs = by_log[log]
        sl = SceneLoader(data / "navsim_logs" / "test", data / "sensor_blobs" / "test",
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log],
                                     tokens=[r["token"] for r in rs]), SensorConfig.build_no_sensors())
        out = []
        for r in rs:
            scene = sl.get_scene_from_token(r["token"])
            proc = MetricCacheProcessor(cache_path=str(temp), force_feature_computation=False)   # fresh: no state leak
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
            proc.compute_metric_cache(scenario)
            cp = temp / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"
            with lzma.open(cp, "rb") as f:
                cache = pickle.load(f)

            def score(poses):
                poses = np.asarray(poses, np.float32); assert poses.shape == (8, 3)
                return {k: float(v) for k, v in dataclasses.asdict(pdm_score(cache, Trajectory(poses, ts), simulator.proposal_sampling,
                                                                               simulator, scorer)).items()}
            joint = {k: [{**{m: v for m, v in j.items() if m != "poses"}, "score": score(j["poses"]),
                          "shift": float(np.max(np.hypot(*(np.asarray(j["poses"])[:, :2] - r["base_poses"][:, :2]).T)))}
                         for j in r["joint"].get(k, [])] for k in FAM}
            assert temp in cp.parents; cp.unlink()
            out.append({"log": log, "token": r["token"], "base": score(r["base_poses"]), "joint": joint})
        dump(fo, out)
        print(f"  [{li + 1}/{len(by_log)}] {log}: {len(out)} tokens", flush=True)
    print("K41 P02 SCORE DONE", flush=True)


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stage_summary(args):
    rs = rows(); ok = [r for r in rs if "err" not in r]
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    fam = []
    for k in FAM:
        tested = [r for r in ok if count(r, k)[0] > 0]
        nt = sum(count(r, k)[0] for r in ok); na = sum(count(r, k)[1] for r in ok)
        mv = sum(len(r["joint"].get(k, [])) for r in ok); tk = sum(bool(r["joint"].get(k)) for r in ok)
        lo, hi = wilson(mv, nt); tlo, thi = wilson(tk, len(tested))
        fam.append({"family": k, "tokens": len(ok), "tokens_tested": len(tested), "combinations_tested": nt,
                    "combinations_available": na, "moving": mv, "rate": mv / nt if nt else float("nan"), "rate_lo": lo, "rate_hi": hi,
                    "tokens_moving": tk, "token_rate": tk / len(tested) if tested else float("nan"), "token_rate_lo": tlo,
                    "token_rate_hi": thi})
    cost = collections.defaultdict(lambda: [0, 0.0])
    for r in ok:
        for c in r["calls"]:
            cost[c["kind"]][0] += 1; cost[c["kind"]][1] += c["wall"]
    sc = {(s["log"], s["token"]): s for f in sorted(OUT.glob("*.pkl")) for s in pickle.load(open(f, "rb"))}
    need = [(r["log"], r["token"]) for r in ok if any(r["joint"].get(k) for k in FAM)]
    missing = [t for t in need if t not in sc]
    jp = []
    for t in need:
        if t in sc:
            s = sc[t]
            for k in FAM:
                for j in s["joint"][k]:
                    jp.append({"log": t[0], "token": t[1], "family": k, "members": repr(j.get("objs", j.get("cells"))),
                               "shift_m": j["shift"], "pdms_base": s["base"]["score"], "pdms_joint": j["score"]["score"],
                               "delta": j["score"]["score"] - s["base"]["score"]})
    for name, data in (("k41_2_p02_families.csv", fam), ("k41_2_p02_joint_pdms.csv", jp),
                       ("k41_2_p02_cost.csv", [{"kind": k, "calls": v[0], "wall_s": v[1]} for k, v in sorted(cost.items())])):
        with open(od / name, "w", newline="") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print("tokens", len(rs), "ok", len(ok), "errors", len(rs) - len(ok), "| joint-effect tokens", len(need), "scored", len(need) - len(missing))
    for x in fam:
        print(f"  {x['family']:17s} tested {x['combinations_tested']:7d}/{x['combinations_available']:9d}  moving {x['moving']:4d}"
              f"  rate {x['rate']:.2e} [{x['rate_lo']:.2e},{x['rate_hi']:.2e}]  tokens {x['tokens_moving']}/{x['tokens_tested']}")
    if jp:
        d = np.array([x["delta"] for x in jp])
        print(f"  joint effects scored {len(jp)}: PDMS delta mean {d.mean():+.4f}, <0: {(d < 0).sum()}, =0: {(d == 0).sum()}, >0: {(d > 0).sum()}")
    print("  wall h", sum(v[1] for v in cost.values()) / 3600, "calls", sum(v[0] for v in cost.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("score", "summary"), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out_dir", default=str(B / "results" / "k41" / "p02_summary"))
    args = ap.parse_args()
    (stage_score if args.stage == "score" else stage_summary)(args)


if __name__ == "__main__":
    main()
