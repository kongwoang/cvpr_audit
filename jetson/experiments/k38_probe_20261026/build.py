"""K38-2 harness: isolated copy of the frozen scripts/A_method/k30_eval.py with one new part 'probe' (test splits):
v6-S (frozen loop) and v6-S with the deployable probe changes of tab:locality together (K16 / k28_p3 CHAIN c1 + d + e):
closing probe 10 m/s (v_closes=(10.0,)), extended grid (ranges 8, 12.5, ..., 70, 80 m; lateral -9..9 m in 1.5 m),
three probe sizes (pedestrian 0.6 x 1.8, car, truck 2.6 x 3.6 m; footprint = union). Real crop detections (no oracle),
round cap, context, guard, C1, v3a unchanged. u960 + C1 (pi(F), no refiner) on the same frames for the agreement.
Output results/k38/k2/<ds>/<unit>.pkl."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (B/'scripts/A_method/k30_eval.py').read_text()
    s = once(s, 'ap.add_argument("--part", required=True, choices=["match", "axis", "idm", "cons"])',
             'ap.add_argument("--part", required=True, choices=["probe"])')
    s = once(s, '    od = REAL / "results" / "k30" / (part + (f"_{args.tag}" if args.tag else "")) / ds',
             '    od = REAL / "results" / "k38" / ("k2" + (f"_{args.tag}" if args.tag else "")) / ds')
    s = once(s, '''            elif part == "cons":
                for sn_ in SETTINGS:''', '''            elif part == "probe":
                loop("v6S", 2.0, 960, dec=True)
                loop("v6Sp", 2.0, 960, dec=True, v_closes=(10.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK))
                for L in (960,):
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    score(f"{nm}+C1", scene_dict(d, lift_c(nm, "C1", ms)(d)), ("none", "v3a"), c2=False, dec=True)
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    push(nm, scene_dict(d, None))
            elif part == "cons":
                for sn_ in SETTINGS:''')
    s = once(s, 'MULTS = (0.75, 1.0, 1.25, 1.5)', 'MULTS = (0.75, 1.0, 1.25, 1.5)\n'
             'R_EXT = np.array([8.0, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0, 80.0])   # = k16_attr / k28_p3 (K16 d)\n'
             'Y_EXT = np.arange(-9.0, 9.01, 1.5)')
    (HERE/'k30_probe.py').write_text(s)
    print('built k30_probe.py')


if __name__ == '__main__':
    main()
