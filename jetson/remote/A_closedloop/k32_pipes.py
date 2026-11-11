"""K32 perception branches for one CARLA front camera (1600 x 900), frozen components, per-frame code as k30_eval.py /
k28_p3.py (draft, crop keys, C1 lift on the branch's own same-scale history, refiner v3a on the braking support, the frozen
loop k_loop.run_g, the bounded loop k_loop.run_bounded, the K28 hybrid tsp8 with the guard). Branches:
  cheap    YOLOv8s 192 x 320 draft, box-growth TTC on its previous frame (the datasets' draft)
  u960     whole frame 576 x 960 (+ comp.)          native   whole frame 928 x 1600 (+ comp.)
  v6S      frozen v6-S (v_close 0, crops at 960 / 1600, support context 2, guard, <= 3 rounds, + comp.)
  v6Sb2    v6-S with at most 2 detector rounds (k_loop.run_bounded, N = 2)
  hybrid   tsp8 (draft support boxes + TUBE[8] = near 10 m, far 70 m, fill) at v6-S's crop px of the frame, guard, + comp.
  unc      K28 unc2 (amendment G1): crops around the draft's detections, lowest confidence first (ties: higher entropy),
           context 2.0, at v6-S's crop px of the frame, guard, K28 merge, + comp.
"+ comp." = C1 looming TTC and refiner v3a (each switchable for the dev rule). Without C1 the TTC is the box-growth TTC of
the branch's previous final scene (as the plain pipelines of K28); with AEB_TTC_TICKS = k (amendment C) the box growth
is taken against the scene k ticks back (dt = k x 0.05 s), as the datasets' frame intervals.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

from rap import planner as P
from rap.detect import Mode
from rap.kitti import Calib
from rap.mono import box_iou, predicted_geometry
from vtf import speculative as SP

B = Path(os.environ["B"])
_l = importlib.util.spec_from_file_location("kl", B / "scripts" / "A_method" / "k_loop.py")
KL = importlib.util.module_from_spec(_l); _l.loader.exec_module(KL)
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
THR = 0.25
TUBE8 = (10.0, 70.0, "fill")
NW = 1600


def crop64(img, b):
    from nusc_refiner_crop import crop64 as c
    return c(img, b)


def assoc_idx(curr, prev, thr=0.3):
    """k27_dev.assoc_idx (verbatim)"""
    out = -np.ones(len(curr), int)
    if len(prev) == 0 or len(curr) == 0:
        return out
    m = box_iou(curr, prev)
    taken = set()
    for ci in np.argsort(-m.max(axis=1)):
        for pj in np.argsort(-m[ci]):
            if m[ci, pj] < thr:
                break
            if pj not in taken:
                taken.add(int(pj)); out[ci] = pj
                break
    return out


def chains(X, s1, s2):
    """k28_tdata.chains"""
    if s1 is None or s2 is None or not len(s1) or not len(s2) or not len(X):
        return []
    j1 = assoc_idx(X, s1)
    a12 = assoc_idx(s1, s2)
    return [(i, int(j1[i]), int(a12[j1[i]])) for i in range(len(X)) if j1[i] >= 0 and a12[j1[i]] >= 0]


def merge_once(cheap, crops, gdet, lift):
    """instr_eval.merge_once (verbatim)"""
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    grown = []
    for c in crops:
        d, ex = gdet(c)
        grown += ex
        full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
    if len(full["conf"]) > 1:
        o = np.argsort(-full["conf"], kind="stable"); iou = box_iou(full["xyxy"], full["xyxy"]); keep = []
        for i in o:
            if not keep or iou[i, keep].max() < 0.5:
                keep.append(i)
        full = {k: full[k][np.sort(np.array(keep))] for k in full}
    out = SP.centres_in(cheap["xyxy"], crops) if crops else np.zeros(len(cheap["xyxy"]), bool)
    if grown and len(full["conf"]) and len(cheap["xyxy"]):
        out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
    fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
    return {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
            "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
            "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
            "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}, crops + grown


class TempHead(torch.nn.Module):
    """k28_ttrain.TempHead without the ImageNet download (the state dict overwrites every weight)"""
    def __init__(self, n_feat):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = torch.nn.Sequential(torch.nn.Linear(3 * 512 + n_feat, 256), torch.nn.ReLU(inplace=True), torch.nn.Linear(256, 1))

    def forward(self, x, f):
        n = x.shape[0]
        x = x.reshape(n * 3, 3, 64, 64)
        x = torch.nn.functional.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        e = self.cnn((x - self.mu) / self.sd).reshape(n, 3 * 512)
        return self.head(torch.cat([e, f], 1))[:, 0]


class Res18x3(torch.nn.Module):
    """k23_train_refiner.Res18x3 without the ImageNet download"""
    def __init__(self, n_feat, n_out=3):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = torch.nn.Sequential(torch.nn.Linear(512 + n_feat, 256), torch.nn.ReLU(inplace=True), torch.nn.Linear(256, n_out))

    def forward(self, x, f):
        x = torch.nn.functional.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        return self.head(torch.cat([self.cnn((x - self.mu) / self.sd), f], 1))


class Pipes:
    """all perception branches of one rollout; state (histories) persists across frames"""

    def __init__(self, det, K, W, H, cam_xyz, dt, use_c1=True, use_ref=True, dev="cuda:0"):
        self.det, self.W, self.H, self.dt, self.dev = det, W, H, dt, dev
        self.use_c1, self.use_ref = use_c1, use_ref
        P2 = np.zeros((3, 4)); P2[:3, :3] = K
        self.cam = type("Cam", (), {})()
        self.cam.calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
        self.cam.cam_h, self.cam.dt = cam_xyz[2], dt
        self.R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)          # OpenCV camera -> ego (x fwd, y left, z up)
        self.t = np.asarray(cam_xyz, float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ckh = torch.load(B / "out" / "k28_temporal" / "temporal_head_np1.pt", map_location=dev, weights_only=False)
        self.head = TempHead(ckh["n_feat"]); self.head.load_state_dict(ckh["state"]); self.head = self.head.to(dev).eval()
        self.hmu, self.hsd = np.asarray(ckh["fmu"], np.float32), np.asarray(ckh["fsd"], np.float32)
        ck = torch.load(B / "out" / "resnet18_v3a" / "image_refiner_np1.pt", map_location=dev, weights_only=False)
        self.ref = Res18x3(len(ck["fmu"])); self.ref.load_state_dict(ck["state"]); self.ref = self.ref.to(dev).eval()
        self.fmu, self.fsd = np.asarray(ck["fmu"], np.float32), np.asarray(ck["fsd"], np.float32)
        self.pp = P.PlannerParams(standoff=P.PlannerParams().standoff + float(os.environ.get("AEB_FRONT", "0")))   # amendment B
        self.CH = Mode("b2d_c320", 192, 320)
        self.WF = {"u960": (Mode("b2d_u960", 576, 960), 960 / NW), "native": (Mode("b2d_u1600", 928, 1600), 1.0)}
        self.st, self.imgs, self.prev_c = {}, [], None
        self.KT = int(os.environ.get("AEB_TTC_TICKS", "1"))            # amendment C: box-growth TTC over KT ticks
        self.c_hist = []
        self.cheap_px = 192 * 320

    # ------------------------------------------------------------------ helpers
    def keep(self, d):
        k = d["conf"] >= THR
        return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

    def geom(self, dd, pv=None, dt=None):
        g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, self.cam.calib, self.cam.cam_h, dt or self.dt,
                               cam_to_ego=(self.R, self.t))
        return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

    def act(self, mono):
        return P.discrete_action(P.required_decel(*mono, self.v, self.pp)[0], self.pp)

    def ckey(self, cr, scale):
        W, H = self.W, self.H
        x1, y1 = int(cr[0]), int(cr[1])
        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
        xe, ye = min(x2, W), min(y2, H)
        nw = max(32, int(math.ceil((xe - x1) * scale / 32)) * 32)
        nh = max(32, int(math.ceil((ye - y1) * scale / 32)) * 32)
        return (x1, y1, x2, y2, nh, nw)

    def make_detect(self, tpx, scale, cnt=None):
        def detect_crop(cr):
            key = self.ckey(cr, scale)
            tpx[0] += key[4] * key[5]
            if cnt is not None:
                cnt[0] += 1
            if key not in self.cache:
                x1, y1, x2, y2, nh, nw = key
                d = self.keep(self.det.detect(self.img[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0])
                self.cache[key] = {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}
            return {k: x.copy() for k, x in self.cache[key].items()}
        return detect_crop

    def lift(self, name, sc):
        """C1 lift (k30_eval.lift_c, mode C1) of pipeline `name`, or the box-growth TTC when C1 is off; both TTCs logged"""
        h = self.st.get(name, {})
        img, imgs, W, H, dt = self.img, self.imgs, self.W, self.H, self.dt

        def f(full):
            X = np.asarray(full["xyxy"], float).reshape(-1, 4)
            z, lo, hi, _ = self.geom(full)
            hh = h.get("hist", [])                                    # C: the scene KT ticks back (or the oldest kept)
            ttc_g = self.geom(full, hh[0], self.dt * len(hh))[3] if hh else np.full(len(X), np.inf)
            ttc = np.full(len(X), np.inf)
            chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
            if chn:
                idx = np.array([c[0] for c in chn])
                zz = predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, self.cam.calib,
                                        self.cam.cam_h, dt, cam_to_ego=(self.R, self.t))
                cr, fe = [], []
                for n, (i, j1, j2) in enumerate(chn):
                    b1, b2 = h["s1"][j1], h["s2"][j2]
                    x1, y1, x2, y2 = X[i]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                    cr.append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                    fe.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                               float(CLS[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                               float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))), dt, 1.0])
                xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().to(self.dev) / 255.0
                fin = torch.from_numpy(((np.array(fe, np.float32) - self.hmu) / self.hsd).astype(np.float32)).to(self.dev)
                with torch.no_grad():
                    rh = np.clip(self.head(xin, fin).float().cpu().numpy(), 0, 2)
                for n, (i, _, _) in enumerate(chn):
                    r = rh[n]
                    ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
            self._ttc_alt[name] = (X.copy(), ttc.copy(), ttc_g.copy())
            return (z, lo, hi, ttc) if self.use_c1 else (z, lo, hi, ttc_g)
        return f

    def push(self, name, scene):
        h = self.st.setdefault(name, {})
        ff = np.asarray(scene["from_full"], bool)
        h["s2"] = h.get("s1")
        h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
        h["prev"] = {"xyxy": np.asarray(scene["xyxy"], float).reshape(-1, 4), "coarse": np.asarray(scene["coarse"])}
        h["hist"] = (h.get("hist", []) + [h["prev"]])[-self.KT:]

    def refine(self, scene, on=None):
        """refiner v3a on the braking support objects that came from a high-resolution pass (k30_eval.refine)"""
        tt = [np.asarray(a, float).copy() for a in scene["mono"]]
        if not (self.use_ref if on is None else on):
            return tuple(tt), []
        sup = [p_ for p_ in SP.support(scene["mono"], self.v, self.pp) if scene["from_full"][p_]]
        if not sup:
            return tuple(tt), []
        W, H = self.W, self.H
        zz = predicted_geometry({"xyxy": scene["xyxy"][sup], "coarse": scene["coarse"][sup]}, None, self.cam.calib, self.cam.cam_h,
                                self.dt, cam_to_ego=(self.R, self.t))
        Fm = []
        for n, p_ in enumerate(sup):
            x1, y1, x2, y2 = scene["xyxy"][p_]
            zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
            Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                       float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
        Fm = ((np.array(Fm, np.float32) - self.fmu) / self.fsd).astype(np.float32)
        x = torch.from_numpy(np.stack([crop64(self.img, scene["xyxy"][p_]) for p_ in sup])).permute(0, 3, 1, 2).float().to(self.dev) / 255.0
        with torch.no_grad():
            o = self.ref(x, torch.from_numpy(Fm).to(self.dev)).float().cpu().numpy().reshape(len(sup), -1)
        for n, p_ in enumerate(sup):
            c = float(np.exp(o[n, 0]))
            for a in tt[:3]:
                a[p_] *= c
        return tuple(tt), sup

    def tube(self, bud, scale9):
        near, far, _ = TUBE8
        W, H = self.W, self.H
        vb_n = min(self.cy + self.fy * self.cam.cam_h / near, H)
        top = max(self.cy + self.fy * self.cam.cam_h / far - self.fy * 1.6 / far, 0)
        best = []
        for half_m in np.arange(0.5, 12.01, 0.25):
            half = self.fx * half_m / near
            c = np.array([max(self.cx - half, 0), top, min(self.cx + half, W), vb_n])
            k = self.ckey(c, scale9)
            if k[4] * k[5] > bud:
                break
            best = [c]
        return best

    # ------------------------------------------------------------------ one frame
    def frame(self, img, v, names):
        """run the branches in `names` on this frame; returns {name: dict(scene, mono, mono_noref, action, px, rounds, ...)}"""
        self.img, self.v, self.cache, self._ttc_alt = img, v, {}, {}
        W, H = self.W, self.H
        NATIVE = W * H
        out = {}
        raw = self.det.detect(img, self.CH)[0]
        ch = self.keep(raw)
        ch["mono"] = self.geom(ch, self.c_hist[0], self.dt * len(self.c_hist)) if self.c_hist else self.geom(ch, None)
        self.prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
        self.c_hist = (self.c_hist + [self.prev_c])[-self.KT:]
        base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
        ch_scene = dict(base, from_full=np.zeros(len(ch["conf"]), bool))
        if "cheap" in names:
            out["cheap"] = {"scene": ch_scene, "mono": ch["mono"], "mono_noref": ch["mono"], "px": self.cheap_px / NATIVE,
                            "rounds": 0, "ncrops": 0, "sup_ref": []}
        for nm in ("u960", "native"):
            if nm in names:
                md, sc = self.WF[nm]
                d = self.keep(self.det.detect(img, md)[0])
                sc_ = {"xyxy": d["xyxy"], "conf": d["conf"], "coarse": d["coarse"], "mono": self.lift(nm, sc)(d),
                       "from_full": np.ones(len(d["conf"]), bool)}
                out[nm] = {"scene": sc_, "px": md.net_h * md.net_w / NATIVE, "rounds": 1, "ncrops": 0}
        scale = 960 / NW
        v6px = None
        if names & {"v6S", "hybrid", "unc"}:
            tpx, cnt = [0.0], [0]
            sc_, crops, grown, rounds, _ = KL.run_g(base, v, self.cam, W, H, self.make_detect(tpx, scale, cnt), self.lift("v6S", scale),
                                                    self.pp, v_closes=(0.0,), ctx=2.0, max_calls=None)
            v6px = tpx[0]
            if "v6S" in names:
                out["v6S"] = {"scene": sc_, "px": (self.cheap_px + tpx[0]) / NATIVE, "rounds": rounds, "ncrops": len(crops),
                              "nguard": len(grown), "calls": cnt[0]}
            self._v6S_scene = sc_
        if "v6Sb2" in names:
            tpx, cnt = [0.0], [0]
            sc_, crops, grown, rounds, conv = KL.run_g(base, v, self.cam, W, H, self.make_detect(tpx, scale, cnt), self.lift("v6Sb2", scale),
                                                       self.pp, v_closes=(0.0,), ctx=2.0, max_calls=2)
            out["v6Sb2"] = {"scene": sc_, "px": (self.cheap_px + tpx[0]) / NATIVE, "rounds": rounds, "ncrops": len(crops),
                            "nguard": len(grown), "calls": cnt[0], "bound_hit": not conv}
        if "hybrid" in names:
            sup_d = SP.support(ch["mono"], v, self.pp)
            sel, used = [], 0
            for k in sup_d:
                c = SP.box_crop(ch["xyxy"][k], W, H)
                kk = self.ckey(c, scale)
                if used + kk[4] * kk[5] <= v6px:
                    sel.append(c); used += kk[4] * kk[5]
            tb = self.tube(v6px - used, scale) if v6px - used > 0 else []
            tpx = [0.0]
            sc_, allc = merge_once(base, sel + tb, SP.guarded(self.make_detect(tpx, scale), W, H), self.lift("hybrid", scale))
            out["hybrid"] = {"scene": sc_, "px": (self.cheap_px + tpx[0]) / NATIVE, "rounds": 1, "ncrops": len(allc),
                             "budget_px": (self.cheap_px + v6px) / NATIVE}
        if "unc" in names:                                       # amendment G1: K28 unc2 at v6-S's crop px
            ent = np.asarray(raw.get("entropy", np.zeros(len(raw["conf"]))), float)[raw["conf"] >= THR]
            order = np.lexsort((-ent, ch["conf"]))
            sel, used = [], 0
            for k in order:
                c = KL.box_crop_ctx(ch["xyxy"][k], W, H, 2.0)
                kk = self.ckey(c, scale)
                if used + kk[4] * kk[5] <= v6px:
                    sel.append(c); used += kk[4] * kk[5]
            tpx, cnt = [0.0], [0]
            sc_, allc = merge_once(base, sel, SP.guarded(self.make_detect(tpx, scale, cnt), W, H), self.lift("unc", scale))
            out["unc"] = {"scene": sc_, "px": (self.cheap_px + tpx[0]) / NATIVE, "rounds": 1, "ncrops": len(allc),
                          "calls": cnt[0], "crop_px": tpx[0], "budget_px": (self.cheap_px + v6px) / NATIVE}
        for nm, r in out.items():
            if nm != "cheap":
                r["mono"], r["sup_ref"] = self.refine(r["scene"])
                r["mono_noref"] = tuple(np.asarray(a, float) for a in r["scene"]["mono"])
                r["ttc_alt"] = self._ttc_alt.get(nm)
            r["action"] = self.act(r["mono"])
        for nm in out:
            if nm != "cheap":
                self.push(nm, out[nm]["scene"])
        if names & {"hybrid", "unc"} and "v6S" not in names:
            self.push("v6S", self._v6S_scene)                     # the hidden budget loop keeps its own history
        self.imgs = (self.imgs + [img])[-2:]
        return out
