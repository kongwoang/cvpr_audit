"""K38-1 harness: scripts/88_k38.py from the K35-4 K29 harness (87_k35_k29.py: trt32 fast path + FAST_RECOVERY helper
+ RUNTIME staging) with:
- rows: v6-S (ours) and B2K-A (ours_b2A) at D = inf, and under each deadline D (ours_dl<D>, ours_b2A_dl<D>), all in
  the same process, each with its own history; v6-S / B2K-A run through deadline.py (verbatim loops + hooks);
- deadline rule: before each round, stop if elapsed + median(next round) + median(finish = refiner + action) > D, the
  medians per dataset / pipeline / round index from --medians (measured on the dev prefixes by --calibrate, fixed
  before the test run);
- --calibrate: record the per-round times of ours / ours_b2A (no stopping);
- --wf u960 --wf_c1only: only u960 + C1 + v3a among the whole-frame rows (hold is applied to its own run offline);
- dataset av2t40: the first 40 sorted AV2 test (val) logs, av2_test40_logs.txt, ~/datasets/k38/av2/sensor/val.
Frozen sources, weights and engines unchanged."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'k35_k29_20261026'


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (SRC/'scripts/87_k35_k29.py').read_text().replace(str(SRC), str(HERE))
    s = once(s, 'from staged import StagedCrops, StagedOneCall\n', 'from staged import StagedCrops, StagedOneCall\nimport deadline as DL\n')
    s = once(s, 'choices=["av2", "nusc", "nusc_test20", "nusc_val", "av2_dev"])', 'choices=["av2", "nusc", "nusc_test20", "nusc_val", "av2_dev", "av2t40"])')
    s = once(s, '    ap.add_argument("--tag", default="")',
             '    ap.add_argument("--tag", default="")\n'
             '    ap.add_argument("--deadlines", default="", help="comma list of D in ms")\n'
             '    ap.add_argument("--medians", default="", help="json of the dev round medians")\n'
             '    ap.add_argument("--calibrate", action="store_true")\n'
             '    ap.add_argument("--wf_c1only", action="store_true")')
    s = once(s, '''        if args.dataset == "av2":
            gen = frames_k34_av2(args)
        else:''', '''        if args.dataset == "av2":
            gen = frames_k34_av2(args)
        elif args.dataset == "av2t40":
            gen = frames_t40(args)
        else:''')
    s = once(s, 'def frames_k34_av2(args):', '''def frames_t40(args):
    """K38: the first 40 sorted AV2 test (val) logs, all annotated frames in order (as K29's AV2 set)"""
    from vtf import av2 as AV
    root = Path.home() / "datasets/k38/av2/sensor/val"
    logs = [l for l in Path("/home/kongwoang/research/CVPR/experiments/k38_deadline_20261026/av2_test40_logs.txt").read_text().split() if l]
    assert len(logs) == 40
    for log in logs:
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        for fr in range(len(L.frame_ts)):
            path = L.image_path(fr)
            assert Path(path).exists(), ("missing frame", path)
            g = geom[geom["frame"] == fr] if len(geom) else geom
            yield log, fr, str(path), float(speeds[fr]), (g[(g["y2"] - g["y1"]) >= 10] if len(g) else g), L, (R, t)


def frames_k34_av2(args):''')
    s = once(s, 'f"k35_k29_{args.runtime}', 'f"k38_{args.runtime}')
    # whole frame: u960 + C1 + v3a only
    s = once(s, '''        for nm, m in M.items():
            if args.parity:''', '''        for nm, m in M.items():
            if args.wf_c1only and not args.parity:
                pass
            elif args.parity:''')
    s = once(s, '''                for kind in ("comp", "c1comp"):''', '''                for kind in (("c1comp",) if args.wf_c1only else ("comp", "c1comp")):''')
    # rows
    s = once(s, 'variants = [("ours",None),("ours_b2A",2),("ours_b2A_retro",2)]',
             'variants = [("ours",None),("ours_b2A",2)] + [(f"{b}_dl{D_:g}", n_) for D_ in DLS for b, n_ in (("ours",None),("ours_b2A",2))]')
    s = once(s, '''        crop51 = None
        for p in (PATHS if args.ours else []):''', '''        crop51 = None
        DLS = [float(x) for x in args.deadlines.split(",") if x]
        MED = json.loads(Path(args.medians).read_text())["nusc" if DS == "nusc" else "av2"] if args.medians else None

        def hooks(nm0, t0):
            """(stop_fn, mark, log) of pipeline nm0: deadline rows stop by the dev medians; calibrate records times"""
            log = {"top": [], "end": [], "stopped": False}
            def mark(ev, r):
                sync(); log[ev].append(1e3 * (time.perf_counter() - t0))
            if "_dl" in nm0:
                base, D_ = nm0.split("_dl")[0], float(nm0.split("_dl")[1])
                med = MED[base]
                def stop_fn(r):
                    sync()
                    el = 1e3 * (time.perf_counter() - t0)
                    nxt = med["round"][min(r, len(med["round"]) - 1)]
                    if el + nxt + med["finish"] > D_:
                        log["stopped"] = True
                        return True
                    return False
                return stop_fn, None, log
            return None, (mark if args.calibrate else None), log
        for p in (PATHS if args.ours else []):''')
    s = once(s, '''                if N is None:
                    scene, crops, rounds = SP.run(dd, v, cam, W, H, lambda cr: mu([cr])[0], lf, pp,
                                                  v_close=args.v_close, max_rounds=3, guard=True, detect_many=mu, batch_round=True)''',
             '''                stop_fn, mark, hlog = hooks(nm0, t0)
                if N is None:
                    scene, crops, rounds = DL.run(dd, v, cam, W, H, lambda cr: mu([cr])[0], lf, pp,
                                                  v_close=args.v_close, max_rounds=3, guard=True, detect_many=mu, batch_round=True,
                                                  stop_fn=stop_fn, mark=mark)''')
    s = once(s, '''                        scene, calls, hit = bounded_keep(dd,v,cam,W,H,traced_one if is_recovery and (args.helper=='original' or eligible_frame) else one_p,lf,pp,args.v_close,N,seed=seed)''',
             '''                        scene, calls, hit = DL.bounded_keep(dd,v,cam,W,H,traced_one if is_recovery and (args.helper=='original' or eligible_frame) else one_p,lf,pp,args.v_close,N,seed=seed,stop_fn=stop_fn,mark=mark)''')
    s = once(s, '''                ms = stop(nm, t0)''', '''                ms = stop(nm, t0)
                rec[f"fin_{nm}"] = ms - 1e3 * (t_loop - t0)
                rec[f"dl_stopped_{nm}"] = hlog["stopped"]
                if mark is not None:
                    rec[f"rt_top_{nm}"], rec[f"rt_end_{nm}"] = list(hlog["top"]), list(hlog["end"])''')
    out = HERE/'scripts/88_k38.py'
    out.parent.mkdir(exist_ok=True)
    out.write_text(s)
    print('built', out)


if __name__ == '__main__':
    main()
