"""K39-1 harness (handoff/K39/PREREG.md): scripts/89_k39.py from the frozen K38-1 harness 88_k38.py (sources of
experiments/k38_deadline_20261026 imported unchanged) with:
- rows: v6-S (ours), B2K-A (ours_b2A) and B1K-A (ours_b1A, <= 1 crop batch, pending truncated detections kept: the same
  DL.bounded_keep with N = 1) at D = inf and in-loop under every --deadlines D (K38-1 rule, --medians from dev);
- whole frame + C1 + v3a at --wf sides (static engines) and at --wf_dyn sides (u800, u960 control) on the existing
  dynamic crop engine yolov8s_dyn (one whole-image call; canvas ceil32(side scale)), names u<side>d+c1comp;
- the hybrid tsp8S+C1 (YOLOv8s, support + TUBE[8], v3a) under v6-S's per-frame crop-detector px incl. guard
  (STATS tpx of the D = inf v6-S row = K30 V1a / K39-2 nominal budget), via --heur --heur_only tsp8S+C1;
- engine executions per row (graphs.EXEC: draft / whole-frame / crop / C1 encoder / refiner executions), crop
  rectangles of v6-S rows, and --graphs none|static|all (A1, graphs.py).
Frozen sources, weights and engines unchanged."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'k38_deadline_20261026'


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (SRC / 'scripts/88_k38.py').read_text()
    s = once(s, 'from staged import StagedCrops, StagedOneCall\n',
             f'from staged import StagedCrops, StagedOneCall\nsys.path.insert(0, "{HERE}")\nimport graphs as GR\n')
    s = once(s, 'HEUR = {"tsp2S+C1": ("yolov8s", 2, "v3a_vuw"),', 'HEUR = {"tsp8S+C1": ("yolov8s", 8, "v3a_vuw"), "tsp2S+C1": ("yolov8s", 2, "v3a_vuw"),')
    s = once(s, '    ap.add_argument("--wf_c1only", action="store_true")',
             '    ap.add_argument("--wf_c1only", action="store_true")\n'
             '    ap.add_argument("--wf_dyn", default="", help="comma list of u<side> on the dynamic engine")\n'
             '    ap.add_argument("--graphs", default="none", choices=["none", "static", "all"])')
    s = once(s, '    args = ap.parse_args()\n', '    args = ap.parse_args()\n    GR.install(args.graphs)\n')
    s = once(s, 'f"k38_{args.runtime}', 'f"k39_{args.graphs}_{args.runtime}')
    s = once(s, '        assert args.config == "v51S", "the heuristics\' budget is v5.1-S\'s crop px of the frame"',
             '        pass                                                       # K39: budget = v6-S crop px incl. guard')
    s = once(s, 'variants = [("ours",None),("ours_b2A",2)] + [(f"{b}_dl{D_:g}", n_) for D_ in DLS for b, n_ in (("ours",None),("ours_b2A",2))]',
             'variants = [("ours",None),("ours_b2A",2),("ours_b1A",1)] + [(f"{b}_dl{D_:g}", n_) for D_ in DLS for b, n_ in (("ours",None),("ours_b2A",2),("ours_b1A",1))]')
    s = once(s, '                    if nm0.startswith("ours_b2A"):', '                    if nm0.startswith(("ours_b2A", "ours_b1A")):')
    s = once(s, '''                    info = {"n_crops": len(crops), "rounds": rounds}
                    if nm0 == "ours" and crop51 is None:
                        crop51 = sum(canvas(c, W, H, 960 / NWS[DS]) for c in crops)''',
             '''                    info = {"n_crops": len(crops), "rounds": rounds}
                    rec[f"crops_{nm}"] = [np.asarray(c, float) for c in crops]
                    if nm0 == "ours" and crop51 is None:
                        crop51 = STATS["tpx"]                      # K39: v6-S crop-detector px incl. guard''')
    s = once(s, '''        def start(name):
            if name in banks:''', '''        def start(name):
            EX0[name] = GR.EXEC[0]
            if name in banks:''')
    s = once(s, '''            rec[f"ms_{name}"] = ms
''', '''            rec[f"ms_{name}"] = ms
            rec[f"execs_{name}"] = GR.EXEC[0] - EX0.pop(name, GR.EXEC[0])
''')
    s = once(s, '    rows, hist, banks, cur, prev_act, imgs = [], {}, {}, None, None, []\n',
             '''    rows, hist, banks, cur, prev_act, imgs = [], {}, {}, None, None, []
    EX0, DYN = {}, {}

    def dyn_runner(sc):
        """StagedCrops on the shared dynamic engine at scale sc (whole image as one crop)"""
        r = StagedCrops.__new__(StagedCrops)
        r.G, r.s, r.eng, r.Mode, r.threshold, r.stats = G, sc, many.eng, Mode, THR, STATS
        return r
''')
    s = once(s, '        # ---- ours (the configuration), <= 3 rounds and bounded, in every path\n', '''        # ---- K39: whole frame + C1 + v3a on the dynamic engine (u800; u960 control)
        for nm_d in [x for x in args.wf_dyn.split(",") if x]:
            run_d = DYN.setdefault(nm_d, dyn_runner(int(nm_d[1:]) / NWS[DS]))
            it_d = run_d.items([[0, 0, W, H]])[0]
            for p in PATHS:
                name = f"{nm_d}d+c1comp{sfx(p)}"
                t0 = start(name)
                G.frame(img)
                dw2 = keep(run_d([[0, 0, W, H]])[0])
                sc_ = {"xyxy": dw2["xyxy"], "conf": dw2["conf"], "coarse": dw2["coarse"], "mono": lift_c(name, p)(dw2),
                       "from_full": np.ones(len(dw2["conf"]), bool)}
                tt2, _ = refine(sc_, refiner(p, "v3a_vuw", name))
                a2 = act(tt2)
                stop(name, t0)
                rec[f"a_{name}"], rec[f"J_{name}"], rec[f"px_{name}"] = a2, J(a2), it_d[2] * it_d[3] / NATIVE
                push(name, dw2["xyxy"])
        # ---- ours (the configuration), <= 3 rounds and bounded, in every path
''')
    out = HERE / 'scripts/89_k39.py'
    out.parent.mkdir(exist_ok=True)
    out.write_text(s)
    print('built', out)


if __name__ == '__main__':
    main()
