"""K41-5 / J04 harness (handoff/K41/PREREG.md, J04 addendum): scripts/92_k41.py generated from the frozen K39-1 harness
experiments/k39_jetson_20261027/scripts/89_k39.py (read only; its imports of k38_deadline_20261026 / graphs.py unchanged)
with, in the same process and on the same backend (staged fast path, trt32, --graphs static):
- the dev-selected K33 adaptive rows (vtf.k33sel / vtf.k33pub, code of scripts/69_k33_jetson.py --k33 de,pub + C rows):
  AnA-style one pass / iterated (ana05hinfo / ana05hinfi), counterfactual one pass / iterated (cf40q20o / cf40q20i),
  AdaScale (ada+1t1280), FOVEA KDE_S / KDE_I (fovs1280b64a1 / fovi1280b64a2), decision-value router (dvr1280_0.5),
  BAR router (bar1280_0.5), Chanakya (ch128B_csc); every row + C1 (fast path) + refiner v3a, own history;
- uncertainty crops unc2S+C1 (K28 unc2 = context 2, lowest draft confidence first; ties by index: the Jetson draft has
  no class entropy) under the K39 heuristic budget (v6-S crop px incl. guard of the frame), via --heur;
- in-loop deadline variants <row>_dl<D> of the iterated rows (ana05hinfi, cf40q20i): before each crop round stop if
  elapsed + median(round) + median(finish) > D, medians per dataset / row / round from --medians41 (dev prefixes, no
  stopping, --calibrate); stopped rows return the action of the last completed round's merged scene;
- K41 rows store a, J, px, rounds, crops and, with --calibrate, round time marks; their order rotates by --rep.
One-pass rows do not depend on D: their deadline rows are their own D = inf run with the K38-1 hold rule (exact).
usage: python build.py  (writes scripts/92_k41.py and prints its sha256)"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "k39_jetson_20261027" / "scripts" / "89_k39.py"


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


ARGS = '''    ap.add_argument("--graphs", default="none", choices=["none", "static", "all"])
    ap.add_argument("--k41", default="", help="comma list of K41 families: de, pub, rc")
    ap.add_argument("--k41_rows", default="ana05hinfo,ana05hinfi,cf40q20o,cf40q20i,ada+1t1280,fovs1280b64a1,fovi1280b64a2,"
                    "dvr1280_0.5,bar1280_0.5,ch128B_csc")
    ap.add_argument("--medians41", default="", help="json of the dev round medians of the iterated K41 rows")'''

SETUP = '''    # ---- K41-5: models of the K33 rows (as scripts/69_k33_jetson.py)
    K41 = {x for x in args.k41.split(",") if x}
    K41_ROWS = [x for x in args.k41_rows.split(",") if x]
    if K41 & {"de"} or "unc2S+C1" in HEUR:
        assert args.config == "v6S" and abs(960 / NWS[DS] - scale) < 1e-12
        hmany["yolov8s"] = many                                    # the crops of D / E / unc: v6-S's detector and scale
    RTR = None
    if K41 & {"pub", "rc"}:
        import torchvision
        from PIL import Image as PILImage
        ada_net = torchvision.models.resnet18(weights=None); ada_net.fc = torch.nn.Linear(512, 5)

        class _Ada(torch.nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
                self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x):
                return self.m((x - self.mu) / self.sd)
        ADA = _Ada(ada_net)
        ADA.load_state_dict(torch.load(ENG / "k33" / ("adascale_nusc.pt" if DS == "nusc" else "adascale_av2.pt"), map_location="cpu"))
        ADA = ADA.cuda().eval()
        FGG = KP.fovea_gg(ROOT / "third_party" / "fovea")
        FSAL = {(bw, am): torch.load(ENG / "k33" / "fovsal" / f"{'nusc' if DS == 'nusc' else 'av2'}_bw{bw}_amp{am}.pt").cuda()
                for bw in (32, 64) for am in (1, 2)}
        FOVG = {}
        if "rc" in K41:
            src_ = "nusc" if DS == "nusc" else "av2"
            mb_ = torchvision.models.mobilenet_v2(weights=None)
            BARN = torch.nn.Sequential(mb_.features[:14], torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(96, 1))
            BARN.load_state_dict(torch.load(ENG / "k33" / f"skip_offbin_{src_}.pt", map_location="cpu"))
            BARN = BARN.cuda().eval()
            fpy = ROOT / "third_party" / "bgt-ada" / "src" / "features.py"
            BGF = type(sys)("bgt_features")
            exec(compile("from __future__ import annotations\\n" + fpy.read_text(), str(fpy), "exec"), BGF.__dict__)
            ckc = torch.load(ENG / "k33" / "router" / f"chanakya_ch128B_csc_{src_}_t.pt")
            hc_ = ckc["hidden"]
            CHN = torch.nn.Sequential(torch.nn.Linear(len(ckc["mu"]), hc_), torch.nn.ReLU(), torch.nn.Linear(hc_, hc_), torch.nn.ReLU(),
                                      torch.nn.Linear(hc_, 4))
            CHN.load_state_dict(ckc["state"])
            RTR = {"src": src_, "xgb": KP.xgb_json(ENG / "k33" / "router" / f"condV_{src_}.json"), "chn": CHN.eval(),
                   "mu": ckc["mu"].numpy(), "sd": ckc["sd"].numpy(), "fs": ckc["fs"],
                   "thr": json.load(open(ENG / "k33" / "router" / "router_thresholds.json")),
                   "tf": torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    MED41 = json.loads(Path(args.medians41).read_text())["nusc" if DS == "nusc" else "av2"] if args.medians41 else None
'''

UNC = '''                    if hn.startswith("unc"):                  # K41: K28 unc2 (context 2, lowest confidence first)
                        order = np.argsort(base["conf"], kind="stable")
                        sel = fill_px([SP.box_crop(base["xyxy"][k], W, H) for k in order], bud)[0] if bud > 0 else []
                    else:
                        sp_, used_ = fill_px([SP.box_crop(base["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)
                        sel = sp_ + (tube(bud - used_, near, far, wm) if bud - used_ > 0 else [])
'''

ROWS = '''        # ---- K41-5: the dev-selected K33 adaptive rows (same backend, own histories, order rotated by --rep)
        if K41:
            hs = 960 / NWS[DS]
            lo_keep = [None]

            def k41_done(nm, a_k, px, t0, **extra):
                stop(nm, t0)
                rec.update({f"a_{nm}": a_k, f"J_{nm}": J(a_k), f"px_{nm}": px / NATIVE, **{f"{k_}_{nm}": v_ for k_, v_ in extra.items()}})

            def merge_rounds(base, crops, res, nm, p):
                full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                grown = []
                for d_, ex in res:
                    grown += ex
                    full = {k: np.concatenate([full[k], np.asarray(d_[k])]) for k in full}
                if len(full["conf"]) > 1:
                    o = np.argsort(-full["conf"], kind="stable"); iou = K61.box_iou(full["xyxy"], full["xyxy"]); kk = []
                    for i in o:
                        if not kk or iou[i, kk].max() < 0.5:
                            kk.append(i)
                    full = {k: full[k][np.sort(np.array(kk))] for k in full}
                out = SP.centres_in(base["xyxy"], crops) if crops else np.zeros(len(base["xyxy"]), bool)
                if grown and len(full["conf"]) and len(base["xyxy"]):
                    out |= K61.box_iou(np.asarray(base["xyxy"], float), full["xyxy"]).max(1) >= 0.5
                fz = lift_c(nm, p)(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
                return {"xyxy": np.r_[base["xyxy"][~out], full["xyxy"]], "conf": np.r_[base["conf"][~out], full["conf"]],
                        "coarse": np.r_[base["coarse"][~out], full["coarse"]],
                        "mono": tuple(np.r_[a[~out], b] for a, b in zip(base["mono"], fz)),
                        "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}, grown

            def row_de(nm0, p, D_=None):
                fam = "ana" if nm0.startswith("ana") else "cf"
                par = next((t_, th_) for t_, th_ in KS.ana_cands() if f"ana{int(round(t_ * 100)):02d}h{'inf' if not np.isfinite(th_) else int(th_)}" == nm0[:-1]) \\
                    if fam == "ana" else next((p_, q_) for p_, q_ in KS.cf_cands() if f"cf{int(round(p_ * 100)):02d}q{int(round(q_ * 10)):02d}" == nm0[:-1])
                iters = 3 if nm0.endswith("i") else 1
                nm = nm0 + (f"_dl{D_:g}" if D_ is not None else "") + sfx(p)
                STATS["tpx"] = 0
                t0 = start(nm)
                G.frame(img)
                xb, r_, pad = letterbox(G.x, CH.net_h, CH.net_w)
                raw_ = det_s.forward(xb[None], CH)
                lo_d = GP.post_lean(GP.host_columns(raw_, CH, 0.05)[0], CH, r_, pad, G.shape, min_conf=0.05)
                lo_d = {"xyxy": np.asarray(lo_d["xyxy"], float).reshape(-1, 4), "conf": np.asarray(lo_d["conf"], float),
                        "coarse": np.asarray(lo_d["coarse"])}
                dd = keep(lo_d)
                base = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"],
                        "mono": lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev")), "from_full": np.zeros(len(dd["conf"]), bool)}
                sup0 = SP.support(base["mono"], v, pp)
                if fam == "ana":
                    lo_mono = lift(lo_d["xyxy"], lo_d["coarse"], hist.get("lo_prev"))
                    lo_keep[0] = lo_d
                crops, res, scene, grown, rd, stopped, tops, ends = [], [], base, [], 0, False, [], []
                for rd in range(iters):
                    if D_ is not None:
                        sync()
                        med = MED41[nm0]
                        if 1e3 * (time.perf_counter() - t0) + med["round"][min(rd, len(med["round"]) - 1)] + med["finish"] > D_:
                            stopped = True
                            break
                    if args.calibrate:
                        sync(); tops.append(1e3 * (time.perf_counter() - t0))
                    if fam == "ana":
                        if rd == 0:
                            xt, idx, xs, sup = lo_d["xyxy"], KS.ana_targets(lo_d["conf"], lo_mono, v, pp, *par), base["xyxy"], sup0
                        else:
                            xt, idx, xs, sup = (scene["xyxy"], KS.ana_targets(scene["conf"], scene["mono"], v, pp, *par),
                                                scene["xyxy"], SP.support(scene["mono"], v, pp))
                    else:
                        xt, idx, xs, sup = (scene["xyxy"], KS.cf_targets(scene["mono"], v, pp, *par), scene["xyxy"],
                                            SP.support(scene["mono"], v, pp))
                    new = KS.crops_for(xt, idx, [], W, H, crops)
                    new += KS.crops_for(xs, [], sup, W, H, crops + new)
                    if not new:
                        break
                    crops = crops + new
                    res = res + list(SP.guarded_batch(hmany_p["yolov8s"], W, H)(new))
                    scene, grown = merge_rounds(base, crops, res, nm, p)
                    if args.calibrate:
                        sync(); ends.append(1e3 * (time.perf_counter() - t0))
                else:
                    rd = iters
                t_fin = time.perf_counter()
                tt, _ = refine(scene, refiner(p, "v3a_vuw", nm))
                a_k = act(tt)
                k41_done(nm, a_k, CH.net_h * CH.net_w + sum(canvas(c, W, H, hs) for c in crops + grown), t0,
                         ncrops=len(crops), rounds=rd, dl_stopped=stopped)
                rec[f"fin_{nm}"] = 1e3 * (time.perf_counter() - t_fin)
                if args.calibrate:
                    rec[f"rt_top_{nm}"], rec[f"rt_end_{nm}"] = tops, ends
                push(nm, np.asarray(scene["xyxy"], float).reshape(-1, 4)[np.asarray(scene["from_full"], bool)])

            def draft41():
                dd = keep(gdetect(det_s, CH))
                dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
                return dd

            def finish41(nm, dd, px, p, t0, **extra):
                dd = {"xyxy": np.asarray(dd["xyxy"], float).reshape(-1, 4), "conf": np.asarray(dd["conf"], float),
                      "coarse": np.asarray(dd["coarse"])}
                mono = lift_c(nm, p)(dd) if len(dd["conf"]) else tuple(np.zeros(0) for _ in range(4))
                sc_ = {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                       "from_full": np.ones(len(dd["conf"]), bool)}
                tt, _ = refine(sc_, refiner(p, "v3a_vuw", nm))
                k41_done(nm, act(tt), px, t0, **extra)
                push(nm, dd["xyxy"])

            def canv(sd, ada=False):
                if ada and sd <= 960:
                    return (int(round(sd * 9 / 16 / 32)) * 32, sd) if DS == "nusc" else (sd, int(round(0.8 * sd / 32)) * 32)
                if DS == "nusc":
                    return {640: (384, 640), 960: (576, 960), 1280: (736, 1280)}.get(sd, (int(math.ceil(sd * 900 / 1600 / 32)) * 32, sd))
                return {640: (640, 512), 960: (960, 768), 1280: (1280, 992)}.get(sd, (sd, int(math.ceil(sd * 1550 / 2048 / 32)) * 32))

            def whole(sd, ada=False):
                h_, w_ = canv(sd, ada)
                if max(h_, w_) > 1024:                              # the static engine of exactly that shape (K28 modes)
                    return keep(gdetect(det_s, Mode(f"{h_}x{w_}", h_, w_))), h_ * w_
                run_d = DYN.setdefault(f"k41_{sd}", dyn_runner(sd / max(H, W)))
                return run_d([[0, 0, W, H]])[0], h_ * w_

            def row_ada(nm0, p):
                k_, top = int(nm0[3:5]), int(nm0.split("t")[1])
                LAD = [320, 480, 640, 800, 960, 1120, 1280]
                nm = nm0 + sfx(p)
                t0 = start(nm)
                G.frame(img)
                dd = draft41()
                with torch.no_grad():
                    xi = torch.from_numpy(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255.0
                    i0 = int(ADA(xi).argmax(1).item())
                sd = LAD[int(np.clip(i0 + k_, 0, LAD.index(top)))]
                if sd == 320:
                    finish41(nm, dd, CH.net_h * CH.net_w, p, t0, size=sd)
                else:
                    dw, px_ = whole(sd, ada=True)
                    finish41(nm, dw, CH.net_h * CH.net_w + px_, p, t0, size=sd)

            def row_fov(nm0, p):
                var, cl, bw, am = nm0[3], int(nm0[4:8]), int(nm0.split("b")[1].split("a")[0]), int(nm0.split("a")[-1])
                h_, w_ = canv(cl)
                nm = nm0 + sfx(p)
                gk = (h_, w_, bw, am)
                if gk not in FOVG:
                    FOVG[gk] = FGG.PlainKDEGrid(output_shape=(h_, w_), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                                anti_crop=True, bandwidth_scale=bw, amplitude_scale=am)
                gen_ = FOVG[gk]
                t0 = start(nm)
                G.frame(img)
                meta = [{"pad_shape": (H, W, 3)}]
                xt = G.x[None] * 255.0
                with torch.no_grad():
                    if var == "i":
                        dd = draft41()
                        grid = gen_(xt, meta, [torch.tensor(dd["xyxy"], dtype=torch.float32, device="cuda")])
                    else:
                        sal = FSAL[(bw, am)]
                        grid = gen_.separable_saliency_to_grid(xt, meta, sal.sum(dim=2), sal.sum(dim=3), xt.device)
                    warped = torch.nn.functional.grid_sample(G.x[None], grid, align_corners=True)[0]
                G.x, G.shape = warped.contiguous(), (h_, w_, 3)
                if max(h_, w_) > 1024:
                    dw = keep(gdetect(det_s, Mode(f"{h_}x{w_}", h_, w_)))
                else:
                    dw = DYN.setdefault("k41_canvas", dyn_runner(1.0))([[0, 0, w_, h_]])[0]
                if len(dw["conf"]):
                    bb = FGG.unwarp_bboxes(torch.tensor(dw["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W))
                    dw = dict(dw, xyxy=bb.cpu().numpy().astype(float))
                finish41(nm, dw, (CH.net_h * CH.net_w if var == "i" else 0) + h_ * w_, p, t0)

            def row_rc(nm0, p):
                nm = nm0 + sfx(p)
                t0 = start(nm)
                G.frame(img)
                dd = draft41()
                if nm0.startswith("bar"):
                    with torch.no_grad():
                        xi = RTR["tf"](PILImage.fromarray(np.ascontiguousarray(img[:, :, ::-1])))[None].cuda()
                        sc = float(BARN(xi).item())
                    sd = 1280 if sc >= RTR["thr"][f"{nm0}|{RTR['src']}"] else 0
                else:
                    bx_ = np.asarray(dd["xyxy"], float).reshape(-1, 4)
                    e_ = BGF.extract_detection_features([(float(b_[0]), float(b_[1]), float(b_[2] - b_[0]), float(b_[3] - b_[1]), float(c_), str(k_))
                                                         for b_, c_, k_ in zip(bx_, dd["conf"], dd["coarse"])])
                    f_ = [float(e_[k_]) for k_ in BGF.BASE_FEATURES]
                    if nm0.startswith("dvr"):
                        sc = RTR["xgb"](f_ + [v])
                        sd = 1280 if sc >= RTR["thr"][f"{nm0}|{RTR['src']}"] else 0
                    else:
                        xf = np.array(f_ + [v, float(P.required_decel(*dd["mono"], v, pp)[0]), float(act(dd["mono"])),
                                            float(-1 if prev_act is None else prev_act)], np.float32)
                        with torch.no_grad():
                            sc = int(RTR["chn"](torch.from_numpy((xf - RTR["mu"]) / RTR["sd"])[None]).argmax(1).item())
                        sd = (0, 640, 960, 1280)[sc]
                if sd == 0:
                    k41_done(nm, act(dd["mono"]), CH.net_h * CH.net_w, t0, size=0, score=sc)
                else:
                    dw, px_ = whole(sd)
                    finish41(nm, dw, CH.net_h * CH.net_w + px_, p, t0, size=sd, score=sc)

            jobs = []
            for nm0 in K41_ROWS:
                if nm0.startswith(("ana", "cf")) and "de" in K41:
                    jobs.append((row_de, nm0, ()))
                    if nm0.endswith("i") and MED41 is not None:
                        jobs += [(row_de, nm0, (D_,)) for D_ in DLS]
                elif nm0.startswith("ada") and "pub" in K41:
                    jobs.append((row_ada, nm0, ()))
                elif nm0.startswith("fov") and "pub" in K41:
                    jobs.append((row_fov, nm0, ()))
                elif nm0.startswith(("bar", "dvr", "ch")) and "rc" in K41:
                    jobs.append((row_rc, nm0, ()))
            rot = (args.rep - 1) % max(len(jobs), 1)
            for p in PATHS:
                for fn, nm0, extra in jobs[rot:] + jobs[:rot]:
                    fn(nm0, p, *extra)
            if lo_keep[0] is not None:
                hist["lo_prev"] = {"xyxy": lo_keep[0]["xyxy"], "coarse": lo_keep[0]["coarse"]}
'''


def main():
    s = SRC.read_text()
    s = once(s, 'HEUR = {"tsp8S+C1": ("yolov8s", 8, "v3a_vuw"),', 'HEUR = {"unc2S+C1": ("yolov8s", 2, "v3a_vuw"), "tsp8S+C1": ("yolov8s", 8, "v3a_vuw"),')
    s = once(s, 'import graphs as GR\n', 'import graphs as GR\nfrom vtf import k33sel as KS\nfrom vtf import k33pub as KP\n')
    s = once(s, '    ap.add_argument("--graphs", default="none", choices=["none", "static", "all"])', ARGS)
    s = once(s, 'f"k39_{args.graphs}', 'f"k41_{args.graphs}')
    s = once(s, '    if args.parity:\n        many_p, one_p = CachedCrops(many, "v"), CachedOne(one, "o")', SETUP + '    if args.parity:\n        many_p, one_p = CachedCrops(many, "v"), CachedOne(one, "o")')
    s = once(s, '''                    sp_, used_ = fill_px([SP.box_crop(base["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)
                    sel = sp_ + (tube(bud - used_, near, far, wm) if bud - used_ > 0 else [])
''', UNC)
    s = once(s, "        chist=hist.setdefault('cheap_history',{})\n", ROWS + "        chist=hist.setdefault('cheap_history',{})\n")
    out = HERE / "scripts" / "92_k41.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(s)
    print(out, hashlib.sha256(s.encode()).hexdigest())


if __name__ == "__main__":
    main()
