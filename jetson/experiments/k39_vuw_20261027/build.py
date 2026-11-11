"""K39 VUW harness: isolated copy of the frozen scripts/A_method/k30_eval.py (md5 55471d34) with one new part 'k39'
(test splits; handoff/K39/PREREG.md K39-2 and K39-4):
  v6-S (frozen loop; its realised crop px incl. guard = the per-frame budget), u960 + C1 (pi(F)),
  K39-2  the 8 S heuristics + C1 at the nominal m = 1 cap (K30 V1a rule: parity with results/k30/match) and under the
         STRICT post-guard cap: a crop-detector call whose canvas would push the cumulative crop px above v6-S's is
         skipped; a skipped guard re-detection keeps the pending truncated detection as detected,
  K39-4  v6-S with one chain intervention ALONE (k28_p3 CHAIN, real crops unless stated): sa exact locality (= ch_a),
         sb F's geometry (IoU >= 0.5 match, else own C1), sc1 / sc2 closing probe 10 / 20 m/s, sd extended grid,
         se three probe templates, sf round cap 10 (refiner none).
Output results/k39/vuw/<ds>/<unit>.pkl. Everything else in k30_eval.py is unchanged."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')

PART = '''            elif part == "k39":
                crop6 = loop("v6S", 2.0, 960)
                md, ms = wf_mode(ds, 960)
                dF = keep(det.detect(img, md)[0])
                FC1 = lift_c("wf960", "C1", ms)(dF)
                score("wf960+C1", scene_dict(dF, FC1), ("none", "v3a"), c2=False)
                rec["px_wf960"] = md.net_h * md.net_w / NATIVE
                push("wf960", scene_dict(dF, None))
                rec["budget_k39"] = crop6
                # ---- K39-2: nominal m = 1 (K30 parity) and strict post-guard cap, budget = v6-S's crop px of the frame
                for cfg in HEUR8:
                    heur(cfg, "x1.00", 1.0 * (cheap_px + crop6) - cheap_px)
                    heur_strict(cfg, "strict", 1.0 * (cheap_px + crop6) - cheap_px, crop6)
                    if SMOKE_CHECK:                               # strict machinery with an unlimited px cap = nominal
                        heur_strict(cfg, "chk", 1.0 * (cheap_px + crop6) - cheap_px, float("inf"))
                # ---- K39-4: single interventions (no refiner)
                fxy = np.asarray(dF["xyxy"], float).reshape(-1, 4)

                def detect_F(cr):
                    k_ = SP.centres_in(fxy, [cr]) if len(fxy) else np.zeros(0, bool)
                    return {"xyxy": fxy[k_], "conf": np.asarray(dF["conf"])[k_], "coarse": np.asarray(dF["coarse"])[k_]}
                sd_, _, _, rd_, cv_ = KL.run_g(base, v, cam, W, H, detect_F, lift_c("sa", "C1", 960 / NW), PP0)
                score("sa", sd_, ("none",), c2=False)
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
                score("sb", sb_, ("none",), c2=False)
                rec["px_sb"], rec["rounds_sb"] = (cheap_px + tpx_b[0]) / NATIVE, rdb
                push("sb", sb_)
                loop("sc1", 2.0, 960, refs=("none",), v_closes=(10.0,))
                loop("sc2", 2.0, 960, refs=("none",), v_closes=(20.0,))
                loop("sd", 2.0, 960, refs=("none",), r_grid=R_EXT, y_grid=Y_EXT)
                loop("se", 2.0, 960, refs=("none",), probes=(KL.PED, KL.CAR, KL.TRUCK))
                loop("sf", 2.0, 960, refs=("none",), max_rounds=10)
            elif part == "cons":
                for sn_ in SETTINGS:'''

STRICT = '''            def strict_guarded(tpx, bud_px, skips):
                """SP.guarded (max_grow 2, relevant None) with the strict cap: a call whose canvas would push tpx above
                bud_px is skipped; a skipped growth keeps its pending truncated detection as detected"""
                dcrop = make_detect(tpx, scale9)

                def over(cr):
                    k_ = ckey(cr, scale9)
                    return tpx[0] + k_[4] * k_[5] > bud_px + 1e-6

                def det_(crop):
                    if over(crop):
                        skips[0] += 1
                        return {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}, []
                    d = {k: np.asarray(x) for k, x in dcrop(crop).items()}
                    if not len(d["conf"]):
                        return d, []
                    top, other = SP.truncated(d["xyxy"], crop, W, H, top_only=True)
                    tr = other | top
                    out = {k: x[~tr] for k, x in d.items()}
                    extra = []
                    live = [({k: d[k][i:i + 1] for k in d}, crop) for i in np.flatnonzero(tr)]
                    for step in range(2):
                        if not live:
                            break
                        nxt = []
                        for row, c in live:
                            b = row["xyxy"][0]
                            g_ = SP.grow(b, c, W, H)
                            if over(g_):
                                skips[1] += 1
                                out = {k: np.concatenate([out[k], row[k]]) for k in out}
                                continue
                            extra.append(g_)
                            dg = {k: np.asarray(x) for k, x in dcrop(g_).items()}
                            if not len(dg["conf"]):
                                continue
                            ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                            iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                            io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                            j = int(np.argmax(io))
                            if io[j] < 0.5:
                                continue
                            rj = {k: dg[k][j:j + 1] for k in dg}
                            if not SP.truncated(dg["xyxy"][j:j + 1], g_, W, H)[0] or step == 1:
                                out = {k: np.concatenate([out[k], rj[k]]) for k in out}
                            else:
                                nxt.append((rj, g_))
                        live = nxt
                    return out, extra
                return det_

            def heur_strict(cfg, tag, bud, bud_px):
                """heur() with the strict post-guard px cap (selection as heur at budget bud); name '<cfg>@<tag>+C1'"""
                fam, i = cfg[:3], int(cfg[3:])
                name = f"{cfg}@{tag}+C1"
                sup_d = SP.support(ch["mono"], v, PP0)
                sel_, tb, spc = select(fam, i, name, bud, sup_d, PP0)
                tpx, skips = [0.0], [0, 0]
                sh, allc = IE.merge_once(base, sel_, strict_guarded(tpx, bud_px, skips), lift_c(name, "C1", scale9))
                score(name, sh, ("none", "v3a"), c2=False)
                rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                rec[f"cpx_{name}"], rec[f"skip_crop_{name}"], rec[f"skip_guard_{name}"] = tpx[0], skips[0], skips[1]
                push(name, sh)
                if fam == "tcr":
                    st[name]["state"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            # ================================================================================================= parts
'''


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (B / 'scripts/A_method/k30_eval.py').read_text()
    s = once(s, 'ap.add_argument("--part", required=True, choices=["match", "axis", "idm", "cons"])',
             'ap.add_argument("--part", required=True, choices=["k39"])')
    s = once(s, '    od = REAL / "results" / "k30" / (part + (f"_{args.tag}" if args.tag else "")) / ds',
             '    od = REAL / "results" / "k39" / ("vuw" + (f"_{args.tag}" if args.tag else "")) / ds')
    s = once(s, '''            elif part == "cons":
                for sn_ in SETTINGS:''', PART)
    s = once(s, '            # ================================================================================================= parts\n', STRICT)
    s = once(s, 'from rap.mono import predicted_geometry ', 'from rap.mono import box_iou, predicted_geometry ')
    s = once(s, 'MULTS = (0.75, 1.0, 1.25, 1.5)', 'MULTS = (0.75, 1.0, 1.25, 1.5)\n'
             'R_EXT = np.array([8.0, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0, 80.0])   # = k16_attr / k28_p3 (K16 d)\n'
             'Y_EXT = np.arange(-9.0, 9.01, 1.5)\n'
             'SMOKE_CHECK = os.environ.get("K39_SMOKE_CHECK") == "1"')
    (HERE / 'k39_eval.py').write_text(s)
    print('built k39_eval.py')


if __name__ == '__main__':
    main()
