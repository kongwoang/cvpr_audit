#!/usr/bin/env python
"""K30 V8a navhard: the devkit's two-stage EPDMS (navsim/planning/script/run_pdm_score.py, v2.2) with two changes that do
not touch any score:
  1. the per-token rows (run_pdm_score.run_pdm_score through worker_map, unchanged) are pickled to <output_dir>/raw_rows.pkl
     BEFORE aggregation;
  2. failed scenarios are left out of the aggregation. In v2.2 a failed token gets NaN results; the devkit's
     pseudo-closed-loop aggregation then raises, its fallback raises again (NaN metric arrays), and no CSV is written. On
     navhard about 100 synthetic stage-two scenarios fail inside the devkit's IDM traffic agents ("Agent's baseline does not
     intersect the agent itself"), for the privileged reference too. aggregate() drops the failed tokens from the
     reactive_all_mapping groups (a pair with a failed token; a group whose stage-one token failed) and from the rows, then
     runs the devkit's own create_scene_aggregators / compute_final_scores / calculate_individual_mapping_scores and the
     same summary rows as main().
k30_navhard_agg.py re-aggregates every branch on the union of the failed tokens of all branches (common token set).
Hydra: same config as run_pdm_score.py (+agent.* as for k30_nav_v2agent).
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

import navsim.planning.script.run_pdm_score as R
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)
CONFIG_PATH = str(Path(R.__file__).parent / R.CONFIG_PATH)


def aggregate(df: pd.DataFrame, raw_mapping, scene_tokens, proposal_sampling, exclude=None):
    """the devkit's main() aggregation on the tokens that did not fail (and are not in `exclude`)"""
    failed = set(df.loc[~df["valid"].astype(bool), "token"]) | set(exclude or [])
    ok = set(df.loc[df["valid"].astype(bool), "token"]) - failed       # scored, valid and not excluded
    all_mappings, dropped_groups, dropped_pairs = {}, 0, 0
    for orig_token, prev_token, two_stage_pairs in raw_mapping:
        if not (prev_token in scene_tokens or orig_token in scene_tokens):
            continue
        keep = [tuple(p) for p in two_stage_pairs if all(t in ok for t in p)]
        if orig_token not in ok or prev_token not in ok or not keep:     # SceneAggregator needs both stage-one rows and >= 1 pair
            dropped_groups += 1
            dropped_pairs += len(two_stage_pairs)
            continue
        dropped_pairs += len(two_stage_pairs) - len(keep)
        all_mappings[(orig_token, prev_token)] = keep
    used = {t for (o, p), pairs in all_mappings.items() for t in (o, p, *[x for pr in pairs for x in pr])}
    d = df[df["token"].isin(used)].copy()
    d = R.create_scene_aggregators(all_mappings, d, proposal_sampling)
    d = R.compute_final_scores(d)
    score_cols = [c for c in d.columns if ((any(s.name in c for s in fields(PDMResults)) or c == "two_frame_extended_comfort"
                                            or c == "score") and c != "pdm_score")]
    g, s1, s2 = R.calculate_individual_mapping_scores(d[score_cols + ["token", "weight"]], all_mappings)
    for col in score_cols:
        m1, m2 = d["frame_type"] == SceneFrameType.ORIGINAL, d["frame_type"] == SceneFrameType.SYNTHETIC
        d.loc[m1, f"{col}_stage_one"] = d.loc[m1, col]
        d.loc[m2, f"{col}_stage_two"] = d.loc[m2, col]
    d.drop(columns=score_cols, inplace=True)
    d["score"] = d["score_stage_one"].combine_first(d["score_stage_two"])
    d.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)
    cols = [f"{c}_stage_one" for c in score_cols if c != "score"] + [f"{c}_stage_two" for c in score_cols if c != "score"] + ["score"]
    d = d[["token", "valid"] + cols]
    rows = []
    for name, ser, sfx in (("extended_pdm_score_stage_one", s1, ["_stage_one"]), ("extended_pdm_score_stage_two", s2, ["_stage_two"]),
                           ("extended_pdm_score_combined", g, None)):
        r = pd.Series(index=d.columns, dtype=object)
        r["token"], r["valid"] = name, True
        r["score"] = ser.get("score", np.nan)
        for sf, src in ((["_stage_one"], s1), (["_stage_two"], s2)) if sfx is None else ((sfx, ser),):
            for c in src.index:
                if c not in ("token", "valid", "score"):
                    r[f"{c}{sf[0]}"] = src[c]
        rows.append(r)
    out = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    info = {"failed_or_excluded_tokens": len(failed), "dropped_groups": dropped_groups, "dropped_pairs": dropped_pairs,
            "groups_used": len(all_mappings), "tokens_used": len(used)}
    return out, info


@hydra.main(config_path=CONFIG_PATH, config_name=R.CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)
    scene_loader = SceneLoader(synthetic_sensor_path=None, original_sensor_path=None, data_path=Path(cfg.navsim_log_path),
                               synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
                               scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=SensorConfig.build_no_sensors())
    MetricCacheLoader(Path(cfg.metric_cache_path))
    data_points = [{"cfg": cfg, "log_file": lf, "tokens": tl} for lf, tl in scene_loader.get_tokens_list_per_log().items()]
    df = pd.concat(worker_map(worker, R.run_pdm_score, data_points))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_dir / "raw_rows.pkl")
    n_fail = int((~df["valid"].astype(bool)).sum())
    res, info = aggregate(df, cfg.train_test_split.reactive_all_mapping, set(scene_loader.tokens),
                          instantiate(cfg.simulator.proposal_sampling))
    ts = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    res.to_csv(out_dir / f"{ts}.csv")
    logger.info(f"K30 NAVHARD SCORE: rows {len(df)}, failed {n_fail}, {info}; EPDMS (failed tokens left out) "
                f"{res.loc[res.token == 'extended_pdm_score_combined', 'score'].iloc[0]:.4f} -> {out_dir / f'{ts}.csv'}")
    print("K30 NAVHARD SCORE DONE", flush=True)


if __name__ == "__main__":
    main()
