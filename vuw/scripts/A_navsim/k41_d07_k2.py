#!/usr/bin/env python
"""D07 (K41-2 P1, PREREG K41-2 + D07 addendum): real-crop evaluation of the k = 2 field of PDM-Closed on the saved K20
seed-0 subset (1,000 navtest tokens), from the P02 interaction records (results/k41/p02).
  field   CPU  per token: k2 field = the frozen singleton field (field_test) + the image boxes of the cells of every
               moving k = 2 pair (P02 joint['k2_pair']) whose two cells lie in the corridor of the draft plan (K20(3):
               lateral offset from the P02 base plan <= 1.75 m + half the probe width); k2all = the same with every moving
               adjacent pair (superset, secondary). -> results/k41/d07/field_{k2,k2all}/<log>.pkl (navsim_field format)
  detect  GPU  tokens whose k2 / k2all field differs from the singleton field: spec_pdm_dets (navsim_detect, the final
               run's crop / guard / merge, SPEC_CROP_SCALE 0.5) on CAM_F0; plus the singleton recomputed on the same
               tokens as detector parity against the stored spec_pdm detections. -> results/k41/d07/dets/<log>.pkl
  score   CPU  all 1,000 tokens: PDM-Closed (navsim_pdms.plan) on the scene of every row - cheap, full960 (full
               perception), singleton (stored spec_pdm), k2, k2all, geometric (stored K30 hybrid_pdm: route tube +
               support) - agreement with the full960 plan (max pose distance <= 0.5 m over 4 s, the field's MOVE_M) and
               official NAVSIM v1.1 PDMS on a metric cache rebuilt from the original scene (as P01; the temporary cache
               is removed). Rows with identical detections share the plan / score. -> results/k41/d07/score/<log>.pkl
usage (env_navsim.sh; detect with the GPU wrapper, field / score with CUDA hidden):
  k41_d07_k2.py --stage field|detect|score [--shard i/n] [--limit N]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import lzma
import os
import pickle
import shutil
import socket
import time
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
X = B / "results" / "navsim_exp"
OUT = B / "results" / "k41" / os.environ.get("K41_D07_DIR", "d07")
P02 = B / "results" / "k41" / "p02"
ROWS = ("cheap", "full960", "singleton", "k2", "k2all", "geometric")
TAG = {"cheap": "cheap", "full960": "full960", "singleton": "spec_pdm", "k2": "spec_pdm", "k2all": "spec_pdm",
       "geometric": "hybrid_pdm"}                              # the original branch tags (fp track names)


def load_mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing")
    with open(tmp, "wb") as f:
        pickle.dump(data, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def p02_rows(limit=0):
    rows = [r for f in sorted(P02.glob("*.pkl")) for r in pickle.load(open(f, "rb"))]
    rows = sorted(rows, key=lambda r: (r["log"], r["token"]))
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- field (CPU)
def stage_field(args):
    NF = load_mod("nf", B / "scripts/A_navsim/navsim_field.py")
    NP = NF.NP
    gz = float((X / "dets" / "ground_z.txt").read_text())
    priors = pickle.load(open(X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    rows = p02_rows(args.limit)
    assert args.limit or len(rows) == 1000, len(rows)
    by_log, summary = {}, []
    for r in rows:
        by_log.setdefault(r["log"], []).append(r)
    for log, rs in by_log.items():
        frozen = pickle.load(open(X / "field_test" / f"{log}.pkl", "rb"))
        dets = pickle.load(open(X / "dets_k30b" / f"{log}.pkl", "rb"))
        out = {"k2": {}, "k2all": {}}
        for r in rs:
            tok = r["token"]
            if "err" in r:
                summary.append({"log": log, "token": tok, "err": r["err"]})
                continue
            fz, cam = frozen[tok], NP.Cam(*dets[tok]["cam"])
            p = np.asarray(r["base_poses"], float)
            px, py = np.r_[0.0, p[:, 0]], np.r_[0.0, p[:, 1]]
            o = np.argsort(px)

            def in_corr(x, y):
                return abs(y - np.interp(x, px[o], py[o])) <= 1.75 + Wd / 2
            moving = [tuple(map(tuple, j["cells"])) for j in r["joint"].get("k2_pair", [])]
            cells = {"k2": sorted({c for pr in moving if all(in_corr(*c) for c in pr) for c in pr}),
                     "k2all": sorted({c for pr in moving for c in pr})}
            rec = {"log": log, "token": tok, "moving_pairs": len(moving), "k2_pairs_tested": r["k2_pairs"],
                   "singleton_cells": len(fz.get("field_boxes", []))}
            for v, cs in cells.items():
                boxes = list(fz.get("field_boxes", []))
                for x, y in cs:
                    ib = cam.project(np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0]))
                    assert ib is not None, (tok, x, y)
                    if not any(np.allclose(ib, b) for b in boxes):
                        boxes.append(ib)
                out[v][tok] = {"field_boxes": boxes, "support_boxes": fz.get("support_boxes", []),
                               "added_cells": cs, "changed": len(boxes) > len(fz.get("field_boxes", []))}
                rec[f"added_cells_{v}"] = len(cs)
                rec[f"changed_{v}"] = out[v][tok]["changed"]
            summary.append(rec)
        for v in out:
            dump(OUT / f"field_{v}" / f"{log}.pkl", out[v])
    import pandas as pd
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "field_summary.csv", index=False)
    print(s[[c for c in s if c.startswith(("changed_", "added_", "moving"))]].sum().to_string())
    print("K41 D07 FIELD DONE", flush=True)


# ---------------------------------------------------------------- detect (GPU)
def stage_detect(args):
    import cv2
    ND = load_mod("nd", B / "scripts/A_navsim/navsim_detect.py")
    K = load_mod("k31", B / "scripts/A_navsim/k31_nav_p2.py")
    assert abs(ND.NS.CROP_SCALE - 0.5) < 1e-12
    si, sn = (int(x) for x in args.shard.split("/"))
    logs = sorted(p.stem for p in (OUT / "field_k2").glob("*.pkl"))
    for li, log in enumerate(logs):
        fo = OUT / "dets" / f"{log}.pkl"
        if li % sn != si or fo.exists():
            continue
        fl = {v: pickle.load(open(OUT / f"field_{v}" / f"{log}.pkl", "rb")) for v in ("k2", "k2all")}
        frozen = pickle.load(open(X / "field_test" / f"{log}.pkl", "rb"))
        dets = pickle.load(open(X / "dets_k30b" / f"{log}.pkl", "rb"))
        frames = {fr["token"]: fr for fr in pickle.load(open(K.DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        res = {}
        for tok in sorted(fl["k2"]):
            if not (fl["k2"][tok]["changed"] or fl["k2all"][tok]["changed"]):
                continue
            img = cv2.imread(str(K.DATA / "sensor_blobs" / "test" / frames[tok]["cams"]["CAM_F0"]["data_path"]))
            assert img is not None, tok
            dd, item = dets[tok], {}
            for v, fld in (("singleton", frozen[tok]), ("k2", fl["k2"][tok]), ("k2all", fl["k2all"][tok])):
                if v != "singleton" and not fld["changed"]:
                    continue
                t0 = time.perf_counter()
                det, area = ND.spec_pdm_dets(dd, img, fld)
                item[v] = det; item[f"px_{v}"] = area; item[f"wall_{v}"] = time.perf_counter() - t0
            st = dd["spec_pdm"]
            item["singleton_parity"] = bool(len(st["conf"]) == len(item["singleton"]["conf"]) and
                                            np.allclose(np.asarray(st["xyxy"], float).reshape(-1, 4),
                                                        np.asarray(item["singleton"]["xyxy"], float).reshape(-1, 4), atol=1e-3)
                                            and np.isclose(dd["px_spec_pdm"], item["px_singleton"]))
            res[tok] = item
        dump(fo, res)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(res)} changed tokens, singleton parity "
              f"{sum(v['singleton_parity'] for v in res.values())}/{len(res)}", flush=True)
    print("K41 D07 DETECT DONE", flush=True)


# ---------------------------------------------------------------- score (CPU)
def stage_score(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    NP = load_mod("np_", B / "scripts/A_navsim/navsim_pdms.py")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    gz = float((X / "dets" / "ground_z.txt").read_text())
    priors = pickle.load(open(X / "dets" / "size_priors.pkl", "rb"))
    cfg = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    params = OmegaConf.load(cfg); simulator = instantiate(params.simulator); scorer = instantiate(params.scorer)
    si, sn = (int(x) for x in args.shard.split("/"))
    rows = [r for r in p02_rows(args.limit) if "err" not in r]
    base_poses = {r["token"]: np.asarray(r["base_poses"]) for r in rows}
    by_log = {}
    for r in rows:
        by_log.setdefault(r["log"], []).append(r["token"])
    od = OUT / "score"; od.mkdir(parents=True, exist_ok=True)
    temp = od / "temporary_metric_cache" / str(os.getpid()); temp.mkdir(parents=True, exist_ok=True)
    for li, log in enumerate(sorted(by_log)):
        fo = od / f"{log}.pkl"
        if li % sn != si or fo.exists():
            continue
        assert shutil.disk_usage(B).free >= 200 * 1024 ** 3
        toks = by_log[log]
        dets = pickle.load(open(X / "dets_k30b" / f"{log}.pkl", "rb"))
        stored = {r_["token"]: r_ for r_ in pickle.load(open(X / "pdms_final" / f"{log}.pkl", "rb"))}
        new = pickle.load(open(OUT / "dets" / f"{log}.pkl", "rb")) if (OUT / "dets" / f"{log}.pkl").exists() else {}
        sl = SceneLoader(NP.DATA / "navsim_logs" / "test", NP.DATA / "sensor_blobs" / "test",
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log],
                                     tokens=toks), SensorConfig.build_no_sensors())
        assert set(sl.tokens) == set(toks), (log, "scene manifest")
        out = []
        for tok in sorted(toks):
            t_tok = time.perf_counter()
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = NP.Cam(*dets[tok]["cam"])
            src = {"cheap": dets[tok]["cheap"], "full960": dets[tok]["full960"], "singleton": dets[tok]["spec_pdm"],
                   "geometric": dets[tok]["hybrid_pdm"]}
            nt = new.get(tok, {})
            src["k2"] = nt.get("k2", src["singleton"])
            src["k2all"] = nt.get("k2all", src["singleton"])
            plans, keyof = {}, {}
            for row in ROWS:
                key = pickle.dumps((TAG[row], {k: np.asarray(v).tobytes() for k, v in sorted(src[row].items())}))
                if key not in keyof:
                    ann = NP.branch_annotations(scene.frames[idx].annotations, src[row], cam, gz, priors, TAG[row])
                    fr = list(scene.frames); fr[idx] = dataclasses.replace(fr[idx], annotations=ann)
                    keyof[key] = NP.plan(MetricCacheProcessor(None, False), dataclasses.replace(scene, frames=fr))
                plans[row] = keyof[key]
            # P02 reused one PDM-Closed instance across tokens; NAVSIM v1.1 PDMObservation never resets
            # _collided_track_ids, so red-light ids leak between scenes of the same map. Every plan / cache here uses a
            # fresh instance; the distance to the P02 draft plan is recorded, not asserted (STATUS 2026-11-02).
            p02_dist = float(np.max(np.hypot(*(plans["cheap"].poses[:, :2] - base_poses[tok][:, :2]).T)))
            proc = MetricCacheProcessor(cache_path=str(temp), force_feature_computation=False)
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
            proc.compute_metric_cache(scenario)
            cp = temp / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"
            with lzma.open(cp, "rb") as f:
                cache = pickle.load(f)
            scores, memo = {}, {}
            for row in ROWS:
                k = plans[row].poses.tobytes()
                if k not in memo:
                    memo[k] = {kk: float(vv) for kk, vv in dataclasses.asdict(pdm_score(cache, plans[row], simulator.proposal_sampling,
                                                                                        simulator, scorer)).items()}
                scores[row] = memo[k]
            assert temp in cp.parents; cp.unlink()
            full = plans["full960"].poses
            out.append({"log": log, "token": tok, "scores": scores,
                        "poses": {r_: plans[r_].poses for r_ in ROWS},
                        "agree_full": {r_: bool(np.max(np.hypot(*(plans[r_].poses[:, :2] - full[:, :2]).T)) <= 0.5) for r_ in ROWS},
                        "px": {"singleton": float(dets[tok]["px_spec_pdm"]), "geometric": float(dets[tok]["px_hybrid_pdm"]),
                               "k2": float(nt.get("px_k2", dets[tok]["px_spec_pdm"])), "k2all": float(nt.get("px_k2all", dets[tok]["px_spec_pdm"]))},
                        "changed": {"k2": "k2" in nt, "k2all": "k2all" in nt}, "p02_base_dist": p02_dist,
                        "stored": {r_: float(stored[tok][f"score_{TAG[r_]}"]) for r_ in ("cheap", "full960", "singleton", "geometric")
                                   if f"score_{TAG[r_]}" in stored.get(tok, {})},
                        "wall": time.perf_counter() - t_tok})
        dump(fo, out)
        print(f"  [{li + 1}/{len(by_log)}] {log}: {len(out)} tokens", flush=True)
    print("K41 D07 SCORE DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["field", "detect", "score"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"field": stage_field, "detect": stage_detect, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
