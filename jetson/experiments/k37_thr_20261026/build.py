"""K37-1 harness: isolated copy of the frozen scripts/A_method/k30_eval.py with one new part 'thr' (K34-1 grid): the
consumer threshold thr_decel in {0.6, 0.8, 1.0 (base), 1.2, 1.5} m/s^2, thr_hard 3.5 fixed, varied in the controller,
the v6-S loop's field and support, the refiner support and the hybrid's support. Per setting: v6-S (loop re-run),
hybrid tsp8 at DESP's crop px of the frame at that setting (x1.00, as K30 V1a), whole frame u960 / native + C1 + v3a
(one detection, scored per setting, as K30 part cons), cheap (per setting, from the base code).
GT, required action and the cost stay at the deployed definition (PlannerParams(): tau1 1.0, tau2 3.5): GT[setting] =
GT[base] and the cost uses the deployed parameters and the deployed cheap's previous action.
--split dev (k27_dev.units: nuScenes new val 152, AV2 CFG-DEV 300) | test (k24_test.units). Output
results/k37/k1/<split>/<ds>/<unit>.pkl. No other change (checked by build: exact once-replacements)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (B/'scripts/A_method/k30_eval.py').read_text()
    s = once(s, 'ap.add_argument("--part", required=True, choices=["match", "axis", "idm", "cons"])',
             'ap.add_argument("--part", required=True, choices=["thr"])\n    ap.add_argument("--split", required=True, choices=["dev", "test"])')
    s = once(s, '    od = REAL / "results" / "k30" / (part + (f"_{args.tag}" if args.tag else "")) / ds',
             '    od = REAL / "results" / "k37" / "k1" / (args.split + (f"_{args.tag}" if args.tag else "")) / ds')
    s = once(s, '    SETTINGS = {"base": PP0} if part != "cons" else {"base": PP0, **{k: replace(PP0, **v) for k, v in CONS.items()}}',
             '    SETTINGS = {"base": PP0, **{f"t{t:g}": replace(PP0, thr_decel=t) for t in (0.6, 0.8, 1.2, 1.5)}}\n'
             '    assert all(p_.thr_hard == PP0.thr_hard == 3.5 for p_ in SETTINGS.values()) and PP0.thr_decel == 1.0')
    s = once(s, '    for ui, (unit, cam, gen) in enumerate(K24.units(ds, args.max_units)):',
             '    for ui, (unit, cam, gen) in enumerate((K27.units if args.split == "dev" else K24.units)(ds, args.max_units)):')
    s = once(s, '''                a_gt, a_obj = P.required_decel(*gt_t, v, pp_) if len(g) else (0.0, np.zeros(0))
                rel = np.flatnonzero(np.asarray(a_obj) >= pp_.thr_decel) if len(g) else np.zeros(0, int)''',
             '''                a_gt, a_obj = P.required_decel(*gt_t, v, PP0) if len(g) else (0.0, np.zeros(0))   # K37: deployed GT
                rel = np.flatnonzero(np.asarray(a_obj) >= PP0.thr_decel) if len(g) else np.zeros(0, int)''')
    s = once(s, '''                rec.update({f"a_gt{sfx}": a_gt, f"act_gt{sfx}": P.discrete_action(a_gt, pp_), f"n_relevant{sfx}": len(rel), f"dec{sfx}": dec})''',
             '''                if sn_ == "base":
                    rec.update({"a_gt": a_gt, "act_gt": P.discrete_action(a_gt, PP0), "n_relevant": len(rel), "dec": dec})''')
    s = once(s, '''                return P.decision_cost(a, GT[sn_][0], prev_a[sn_], SETTINGS[sn_], cp)["J"]''',
             '''                return P.decision_cost(a, GT["base"][0], prev_a["base"], PP0, cp)["J"]     # K37: deployed cost''')
    s = once(s, '''            elif part == "cons":
                for sn_ in SETTINGS:''', '''            elif part == "thr":
                for sn_ in SETTINGS:
                    crop6 = loop("v6S", 2.0, 960, sn_=sn_)
                    heur("tsp8", "x1.00", 1.0 * (cheap_px + crop6) - cheap_px, sn_=sn_, c2=False)
                for L in (960, NWS[ds]):
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    tc1 = lift_c(nm, "C1", ms)(d)
                    for sn_ in SETTINGS:
                        score(f"{nm}+C1", scene_dict(d, tc1), ("none", "v3a"), sn_, c2=False)
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    push(nm, scene_dict(d, None))
            elif part == "cons":
                for sn_ in SETTINGS:''')
    (HERE/'k30_thr.py').write_text(s)
    print('built k30_thr.py')


if __name__ == '__main__' and __import__('sys').argv[1:] == []:
    main()


def holdout():
    """k30_thr_h.py: the same harness on the K35-4 holdout (unit source k24_holdout.py, K35_AHD_SPLIT=train, as K35-4);
    output results/k37/k1/hold/<ds>. Second (and last) use of the holdout; see handoff/K37/K37-1/PREREG.md."""
    s = (HERE/'k30_thr.py').read_text()
    s = once(s, 'B / "scripts" / "A_method" / "k24_test.py"', 'Path("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/experiments/k35_holdout_20261025/k24_holdout.py")')
    s = once(s, '    od = REAL / "results" / "k37" / "k1" / (args.split + (f"_{args.tag}" if args.tag else "")) / ds',
             '    od = REAL / "results" / "k37" / "k1" / ("hold" + (f"_{args.tag}" if args.tag else "")) / ds\n'
             '    assert os.environ["K35_AHD_SPLIT"] == "train" and args.split == "test" and ds == "ahd"')
    (HERE/'k30_thr_h.py').write_text(s)
    print('built k30_thr_h.py')


if __name__ == '__main__' and __import__('sys').argv[1:] == ['holdout']:
    holdout()
