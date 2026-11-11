"""K42 VUW harness: isolated copy of experiments/k39_vuw_20261027/k39_eval.py with one new part 'k42'
(handoff/K42/PREREG.md §A2). Everything else is unchanged. Per frame, on the K28 test units:
  v6S    frozen v6-S; every call of the stopping test SP.covered(crop, crops) (tol 0.95, max over SINGLE earlier crops)
         is logged together with the exact union-containment decision (candidate area inside the union of earlier
         crops >= area); refiner none / v3a
  v6U    v6-S with the stopping test replaced by exact containment in the union (own history, own crops); none / v3a
  wf960+C1   pi(F); none / v3a
  sa..sf     the K39-4 single interventions, now scored with none AND v3a (unchanged otherwise)
  ch_a..ch_f the K28 p3 / K16 counterfactual chain (k28_p3.py CHAIN, detect_F / lift_F), scored with none AND v3a
Output results/k42/vuw/<ds>/<unit>.pkl."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "k39_eval_src.py").read_text()

HELPERS = '''

# ---------------------------------------------------------------------------------------------------- K42 helpers
_COV_ORIG = SP.covered
COVLOG = {"on": False, "calls": []}


def union_frac(crop, crops):
    """exact share of crop's area inside the union of crops (coordinate compression)"""
    x1, y1, x2, y2 = (float(a) for a in crop)
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    R = []
    for c in crops:
        a, b, cc, d = max(x1, float(c[0])), max(y1, float(c[1])), min(x2, float(c[2])), min(y2, float(c[3]))
        if cc > a and d > b:
            R.append((a, b, cc, d))
    if not R:
        return 0.0
    xs = sorted({r[0] for r in R} | {r[2] for r in R})
    ys = sorted({r[1] for r in R} | {r[3] for r in R})
    tot = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mx, my = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(r[0] <= mx <= r[2] and r[1] <= my <= r[3] for r in R):
                tot += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return tot / area


def covered_logged(crop, crops, tol=0.95):
    r = _COV_ORIG(crop, crops, tol)
    if COVLOG["on"]:
        uf = union_frac(crop, crops) if crops else 0.0
        COVLOG["calls"].append((bool(r), uf, len(crops)))
    return r


def covered_union(crop, crops, tol=0.95):
    """K42-4(b) strict rule: skip only if the candidate lies entirely inside the union of earlier crops"""
    return bool(crops) and union_frac(crop, crops) >= 1.0 - 1e-9


SP.covered = covered_logged
'''
anchor = "\ndef idm_demand("
assert src.count(anchor) == 1
src = src.replace(anchor, HELPERS + anchor)
a = 'ap.add_argument("--part", required=True, choices=["k39"])'
assert src.count(a) == 1
src = src.replace(a, 'ap.add_argument("--part", required=True, choices=["k39", "k42"])')
a = 'od = REAL / "results" / "k39" / ("vuw" + (f"_{args.tag}" if args.tag else "")) / ds'
assert src.count(a) == 1
src = src.replace(a, 'od = REAL / "results" / ("k42" if part == "k42" else "k39") / ("vuw" + (f"_{args.tag}" if args.tag else "")) / ds')

PART = '''            elif part == "k42":
                # ---- v6-S frozen, stopping test logged
                COVLOG["on"], COVLOG["calls"] = True, []
                try:
                    loop("v6S", 2.0, 960, c2=False) if False else loop("v6S", 2.0, 960)
                finally:
                    COVLOG["on"] = False
                cl = COVLOG["calls"]
                rec["cov_calls_v6S"] = len(cl)
                rec["cov_skip_v6S"] = sum(1 for r_, u_, n_ in cl if r_)
                rec["cov_skip_union_req_v6S"] = sum(1 for r_, u_, n_ in cl if r_ and u_ < 1.0 - 1e-9)   # skipped now, union would require
                rec["cov_req_union_skip_v6S"] = sum(1 for r_, u_, n_ in cl if (not r_) and n_ and u_ >= 1.0 - 1e-9)  # run now, union would skip
                rec["cov_skip_union_fracs_v6S"] = [u_ for r_, u_, n_ in cl if r_ and u_ < 1.0 - 1e-9]
                # ---- v6-S with the strict union-containment stopping test (own history)
                SP.covered = covered_union
                try:
                    loop("v6U", 2.0, 960)
                finally:
                    SP.covered = covered_logged
                # ---- pi(F)
                md, ms = wf_mode(ds, 960)
                dF = keep(det.detect(img, md)[0])
                FC1 = lift_c("wf960", "C1", ms)(dF)
                score("wf960+C1", scene_dict(dF, FC1), ("none", "v3a"), c2=False)
                rec["px_wf960"] = md.net_h * md.net_w / NATIVE
                push("wf960", scene_dict(dF, None))
                fxy = np.asarray(dF["xyxy"], float).reshape(-1, 4)

                def detect_F(cr):
                    k_ = SP.centres_in(fxy, [cr]) if len(fxy) else np.zeros(0, bool)
                    return {"xyxy": fxy[k_], "conf": np.asarray(dF["conf"])[k_], "coarse": np.asarray(dF["coarse"])[k_]}
                # ---- K39-4 single interventions, none AND v3a
                sd_, _, _, rd_, cv_ = KL.run_g(base, v, cam, W, H, detect_F, lift_c("sa", "C1", 960 / NW), PP0)
                score("sa", sd_, ("none", "v3a"), c2=False)
                rec["rounds_sa"], rec["conv_sa"] = rd_, cv_
                push("sa", sd_)

                def lift_Fm(name, sc):
                    own = lift_c(name, "C1", sc)

                    def f(full):
                        o = own(full)
                        X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                        if not len(fxy) or not len(X):
                            return o
                        iou = box_iou(X, fxy)
                        j, ok = iou.argmax(1), iou.max(1) >= 0.5
                        return tuple(np.where(ok, np.asarray(fq, float)[j], np.asarray(oq, float)) for fq, oq in zip(FC1, o))
                    return f
                tpx_b = [0.0]
                sb_, crb, grb, rdb, cvb = KL.run_g(base, v, cam, W, H, make_detect(tpx_b, 960 / NW), lift_Fm("sb", 960 / NW), PP0,
                                                   v_closes=(0.0,), ctx=2.0, max_calls=None)
                score("sb", sb_, ("none", "v3a"), c2=False)
                rec["px_sb"], rec["rounds_sb"] = (cheap_px + tpx_b[0]) / NATIVE, rdb
                push("sb", sb_)
                loop("sc1", 2.0, 960, v_closes=(10.0,))
                loop("sc2", 2.0, 960, v_closes=(20.0,))
                loop("sd", 2.0, 960, r_grid=R_EXT, y_grid=Y_EXT)
                loop("se", 2.0, 960, probes=(KL.PED, KL.CAR, KL.TRUCK))
                loop("sf", 2.0, 960, max_rounds=10)
                # ---- K28 p3 / K16 counterfactual chain (k28_p3.py CHAIN), none AND v3a

                def lift_F(full):
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    ii = [int(np.argmax(box_iou(X[n:n + 1], fxy)[0])) for n in range(len(X))]
                    return tuple(np.asarray(a_, float)[ii] for a_ in FC1)
                for cn_, kw_, fl_ in CHAIN42:
                    cn = f"ch_{cn_}"
                    sdc, _, _, rdc, cvc = KL.run_g(base, v, cam, W, H, detect_F, lift_F if fl_ else lift_c(cn, "C1", 960 / NW), PP0, **kw_)
                    score(cn, sdc, ("none", "v3a"), c2=False)
                    rec[f"rounds_{cn}"], rec[f"conv_{cn}"] = rdc, cvc
                    push(cn, sdc)
'''
PART = PART.replace('                    loop("v6S", 2.0, 960, c2=False) if False else loop("v6S", 2.0, 960)\n', '                    loop("v6S", 2.0, 960)\n')
a = '            elif part == "cons":'
assert src.count(a) == 1
src = src.replace(a, PART + a)
CH = '''
CHAIN42 = (("a", {}, False), ("b", {}, True), ("c1", dict(v_closes=(10.0,)), True), ("c2", dict(v_closes=(20.0,)), True),
           ("d", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT), True),
           ("e", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK)), True),
           ("f", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK), max_rounds=10), True))
'''
a = "\nSMOKE_CHECK ="
assert src.count(a) == 1
src = src.replace(a, CH + a)
src = src.replace('"""K30 (post-hoc;', '"""K42 harness (k42_vuw_20261111/build.py from k39_eval.py; part k42 added). K30 (post-hoc;', 1)
(HERE / "k42_eval.py").write_text(src)
print("written", len(src))
