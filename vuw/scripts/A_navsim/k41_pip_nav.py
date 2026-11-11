#!/usr/bin/env python
"""D04 (K41-1 P1, PREREG 2026-11-01 D04 addendum): real-crop PiP-inspired rows for the fixed K31 bottleneck selector on
NAVSIM navtest (15 candidates = the anchors). Per token, on CAM_F0 with YOLOv8s crops at SPEC_CROP_SCALE 0.5, truncation
guard and the K31 merge rule (merge_real of k31_nav_p2.run, copied unchanged):
  field             the K31 DESP loop with real crops (parity row: pi / px must equal results/k31/p2 pi_real / px_real)
  all               (a) one round, the step boxes of all 15 candidates (draft-score order, steps in time order)
  selected          (b) <= 8 rounds, one 0.5 s step per round of the top M = 3 candidates by s(tau, current scene)
  selected_support  (c) (b) + the DESP support crops of the current scene each round
  geometric         DESP support + all step boxes (support first), iterated as the field loop (<= 3 rounds, stops when nothing new)
Every new row is capped at the token's field crop area (native px, ordinary crops; cap rule of the crop heuristics: a box
that does not fit the remaining budget is skipped). Step box of candidate c, step t = union of the CAM_F0 boxes of the
ego rectangle (EGO_L x EGO_W x 1.6 m) at the 0.1 s poses 5(t-1)..5t with 8 <= x <= 55 m (the probing field's range).
Detections are cached per crop within a token (same crop -> same detections); each row is charged its own crops
(native px, detector tensor px, calls, detector seconds). -> results/k41/pip_nav/<log>.pkl, <log>.done.json
usage: k41_pip_nav.py [--shard i/n] [--only_log L] [--max_tokens N] [--out results/k41/pip_nav]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import socket
import time
from pathlib import Path

import numpy as np
import k31_nav_p2 as K

B = Path(os.environ["B"])
EXP = K.EXP
ROWS = ("field", "all", "selected", "selected_support", "geometric")
TOP_M, STEPS, H_OBJ = 3, 8, 1.6


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trajectory_poses(sel, index):
    """exactly k41_selector_metrics.trajectory(...).poses"""
    poses = sel.P[index, 5:41:5, :].copy()
    poses[:, :2] -= K.EGO_RA * np.stack([np.cos(poses[:, 2]), np.sin(poses[:, 2])], 1)
    return poses.astype(np.float32)


def step_boxes(sel, cam, gz):
    """(15, 8) image boxes (or None) of the candidates' 0.5 s steps"""
    out = [[None] * STEPS for _ in range(len(sel.P))]
    for c in range(len(sel.P)):
        for t in range(1, STEPS + 1):
            bs = []
            for i in range(5 * (t - 1), 5 * t + 1):
                x, y, hd = sel.P[c, i]
                if not 8.0 <= x <= 55.0:
                    continue
                ib = cam.project(np.array([x, y, gz + H_OBJ / 2, K.EGO_L, K.EGO_W, H_OBJ, hd]))
                if ib is not None:
                    bs.append(ib)
            if bs:
                bs = np.asarray(bs, float)
                out[c][t - 1] = np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
    return out


def carea(c):
    return float((c[2] - c[0]) * (c[3] - c[1]))


def run(args):
    NS = K.load_mod("nsr", B / "scripts" / "A_nuplan" / "nuplan_spec_real.py")
    VS, W, H = NS.VS, NS.W, NS.H
    import cv2
    from rap.detect import Mode
    from rap.mono import box_iou
    assert abs(NS.CROP_SCALE - 0.5) < 1e-12, "SPEC_CROP_SCALE must be 0.5 (final run)"
    gz = float((EXP / "dets" / "ground_z.txt").read_text())
    priors = pickle.load(open(EXP / "dets" / "size_priors.pkl", "rb"))
    si, sn = (int(x) for x in args.shard.split("/"))
    out = B / args.out
    out.mkdir(parents=True, exist_ok=True)
    logs = sorted(p.stem for p in (B / "results/k31/p2").glob("*.pkl"))
    assert len(logs) == 136
    if args.only_log:
        logs = [args.only_log]
    script = digest(__file__)
    for li, log in enumerate(logs):
        if li % sn != si or (out / f"{log}.done.json").exists():
            continue
        t_log, c_log = time.monotonic(), time.process_time()
        ref = {r["token"]: r for r in pickle.load(open(B / "results/k31/p2" / f"{log}.pkl", "rb"))}
        dets = pickle.load(open(EXP / "dets_k30b" / f"{log}.pkl", "rb"))
        cen = pickle.load(open(EXP / "k30_center" / f"{log}.pkl", "rb"))
        frames = {fr["token"]: fr for fr in pickle.load(open(K.DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        unit = out / f"{log}.pkl"
        rows = pickle.load(open(unit, "rb")) if unit.exists() else []
        have = {r["token"] for r in rows}
        n_new = 0
        for tok in sorted(ref):
            if tok in have:
                continue
            r0, rec, fr = ref[tok], dets[tok], frames[tok]
            ann, cam = fr["anns"], K.NP.Cam(*rec["cam"])
            cl = cen.get(tok)
            cl = cl if isinstance(cl, np.ndarray) and len(cl) >= 2 else None
            v0 = float(np.hypot(*np.asarray(fr["ego_dynamic_state"], float)[:2]))
            assert v0 == r0["v0"]
            sel = K.Selector(cl, v0)
            prb = K.probes(cam, gz, priors)
            bC, pC = K.branch_objects(ann, rec["cheap"], cam, gz, priors)
            xC = np.asarray(rec["cheap"]["xyxy"], float).reshape(-1, 4)
            draft = K.Scene(ann, bC, zip(pC, xC))
            assert sel.pi(draft.o) == r0["pi_cheap"], (log, tok, "selector parity")
            item = {"log": log, "token": tok, "v0": v0, "pi_cheap": r0["pi_cheap"], "pi_full960": r0["pi_full960"],
                    "selected": {}, "trajectories": {}, "stats": {}}
            img = cv2.imread(str(K.DATA / "sensor_blobs" / "test" / fr["cams"]["CAM_F0"]["data_path"]))
            if img is None or r0.get("pi_real", -1) < 0:
                item["err"] = "no image"
                rows.append(item)
                continue
            # ---- detector with a per-token crop cache; per-row charges
            cache, charge = {}, {}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * NS.CROP_SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * NS.CROP_SCALE / 32)) * 32)
                key = (x1, y1, x2, y2, nh, nw)
                if key not in cache:
                    t0 = time.perf_counter()
                    d, _ = NS._detector().detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                    kk = d["conf"] >= 0.25
                    cache[key] = ({"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                                   "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]},
                                  time.perf_counter() - t0)
                charge["calls"] += 1
                charge["tensor_px"] += nh * nw
                charge["det_s"] += cache[key][1]
                return cache[key][0]
            gdet = VS.guarded(detect_crop, W, H)
            gcache = {}

            def merge_real(R):                                       # k31_nav_p2.run.merge_real, unchanged
                full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                grown = []
                for c_ in R:
                    key = tuple(np.round(c_, 2))
                    if key not in gcache[charge["row"]]:
                        gcache[charge["row"]][key] = gdet(c_)
                    d_, ex_ = gcache[charge["row"]][key]
                    grown += ex_
                    full = {kk: np.concatenate([full[kk], np.asarray(d_[kk])]) for kk in full}
                if len(full["conf"]) > 1:
                    o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
                    kp = []
                    for i_ in o_:
                        if not kp or iou_[i_, kp].max() < 0.5:
                            kp.append(i_)
                    full = {kk: full[kk][np.sort(np.array(kp))] for kk in full}
                o2 = VS.centres_in(xC, R) if R else np.zeros(len(xC), bool)
                if grown and len(full["conf"]) and len(xC):
                    o2 |= box_iou(xC, full["xyxy"]).max(1) >= 0.5
                det = {kk: np.concatenate([np.asarray(rec["cheap"][kk])[~o2], full[kk]]) for kk in ("xyxy", "conf", "coarse")}
                bm, pm = K.branch_objects(ann, det, cam, gz, priors)
                merge_real.grown = grown
                return K.Scene(ann, bm, zip(pm, np.asarray(det["xyxy"], float).reshape(-1, 4)))

            def start(row):
                charge.update(row=row, calls=0, tensor_px=0, det_s=0.0)
                gcache[row] = {}
                merge_real.grown = []
                return time.perf_counter()

            def finish(row, a, R, rounds, t0, **extra):
                grown = list(merge_real.grown)
                item["selected"][row] = int(a)
                item["trajectories"][row] = trajectory_poses(sel, int(a))
                item["stats"][row] = dict(px=K.area(R + grown, W, H), px_crops=K.area(R, W, H), n_crops=len(R),
                                          n_grown=len(grown), rounds=rounds, calls=charge["calls"],
                                          tensor_px=charge["tensor_px"], det_seconds=charge["det_s"],
                                          wall_seconds=time.perf_counter() - t0, **extra)

            # ---- field (parity row)
            t0 = start("field")
            a, k, R, _, capped, _ = K.loop(sel, prb, merge_real, draft, VS, W, H)
            finish("field", a, R, k, t0, capped=capped)
            st = item["stats"]["field"]
            item["field_parity"] = dict(pi=bool(a == r0["pi_real"]), px=bool(np.isclose(st["px"], r0["px_real"], rtol=0, atol=1e-12)),
                                        rounds=bool(k == r0["rounds_real"]))
            budget = st["px_crops"] * W * H                          # native area of the field's ordinary crops
            boxes = step_boxes(sel, cam, gz)
            s0 = sel.score(draft.o)[0]
            prior = [int(c) for c in np.lexsort((np.arange(len(s0)), -s0))]

            def fit(R, cand):
                """cap rule: requests not covered by R or by earlier requests that fit the remaining budget"""
                used, new, skipped = sum(carea(c) for c in R), [], 0
                for c in cand:
                    if c is None or VS.covered(c, R + new):
                        continue
                    if used + carea(c) > budget + 1e-6:
                        skipped += 1
                        continue
                    new.append(c)
                    used += carea(c)
                return new, skipped

            def support_crops(M, R):
                _, _, sup = K.field_support(sel, M, prb)
                return [VS.box_crop(np.asarray(b, float), W, H) for b in sup]

            # ---- (a) all anchors, one round
            t0 = start("all")
            R, skipped = fit([], [boxes[c][t] for c in prior for t in range(STEPS)])
            M = merge_real(R) if R else draft
            finish("all", sel.pi(M.o), R, int(bool(R)), t0, skipped=skipped)

            # ---- (b) / (c) plan-selected, autoregressive
            for row, with_sup in (("selected", False), ("selected_support", True)):
                t0 = start(row)
                R, M, t, rounds, skipped = [], draft, 0, 0, 0
                while t < STEPS:
                    s = sel.score(M.o)[0]
                    top = [int(c) for c in np.lexsort((np.arange(len(s)), -s))[:TOP_M]]
                    new, sk = fit(R, ([boxes[c][t] for c in top]) + (support_crops(M, R) if with_sup else []))
                    skipped += sk
                    t += 1
                    while not new and t < STEPS:                      # step covered / empty: next step, same round
                        new, sk = fit(R, [boxes[c][t] for c in top] + (support_crops(M, R) if with_sup else []))
                        skipped += sk
                        t += 1
                    if not new:
                        break
                    R = R + new
                    M = merge_real(R)
                    rounds += 1
                finish(row, sel.pi(M.o), R, rounds, t0, skipped=skipped)

            # ---- geometric corridor of all candidates + support, iterated as the field loop
            t0 = start("geometric")
            R, M, skipped, rounds = [], draft, 0, 0
            geo = [boxes[c][t] for c in prior for t in range(STEPS)]
            for k in range(K.MAX_ROUNDS + 1):
                new, sk = fit(R, support_crops(M, R) + geo)             # support first (PREREG correction)
                skipped += sk
                if not new or k == K.MAX_ROUNDS:
                    break
                R = R + new
                M = merge_real(R)
                rounds += 1
            finish("geometric", sel.pi(M.o), R, rounds, t0, skipped=skipped, capped=bool(new))
            item["budget_px"] = budget
            item["distinct_detector_calls"] = len(cache)
            rows.append(item)
            n_new += 1
            if n_new % 10 == 0:
                tmp = unit.with_suffix(".writing.pkl")
                with open(tmp, "wb") as fh:
                    pickle.dump(rows, fh)
                tmp.replace(unit)
            if args.max_tokens and n_new >= args.max_tokens:
                break
        tmp = unit.with_suffix(".writing.pkl")
        with open(tmp, "wb") as fh:
            pickle.dump(rows, fh)
        tmp.replace(unit)
        ok = [r for r in rows if "err" not in r]
        par = {k: int(sum(r["field_parity"][k] for r in ok)) for k in ("pi", "px", "rounds")}
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens, field parity {par}", flush=True)
        if args.max_tokens:
            break
        meta = dict(log=log, tokens=len(rows), errors=len(rows) - len(ok), field_parity=par, host=socket.gethostname(),
                    wall_seconds=time.monotonic() - t_log, cpu_seconds=time.process_time() - c_log, script_sha256=script,
                    cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))
        (out / f"{log}.done.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("K41 D04 PIP NAV DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only_log", default="")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--out", default="results/k41/pip_nav")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
