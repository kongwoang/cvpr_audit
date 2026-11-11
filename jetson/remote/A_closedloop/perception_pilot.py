"""PDM-Lite (privileged rule-based expert of carla_garage, Bench2Drive copy) with PERCEPTION IN THE LOOP.

Convention (as our NAVSIM / nuPlan experiments, DEEP 114): every vehicle / walker whose 3D box projects into the front
camera (1600x900, FOV 70, Bench2Drive's CAM_FRONT pose) is kept only if a detection of the active perception branch
matches it (Hungarian, class-compatible, IoU >= 0.3); matched actors keep their true state, unmatched in-camera actors
are hidden from the planner, and unmatched detections become static pseudo-obstacles lifted on the ground plane (box
bottom centre, class-median sizes, heading along the viewing ray, zero speed). Actors outside the camera are untouched
(the planner keeps its privileged view there, as in the NAVSIM setup). The planner, map, route, traffic lights and
scenario information stay privileged — only object perception in the front camera is replaced.

Branch (env PERC_BRANCH): privileged | cheap (YOLOv8s 192x320) | u960 (576x960) | u1600 (928x1600, native) |
ours (cheap draft -> braking decision-sensitivity field + support crops at scale 0.6, guard, <= 3 rounds; the nuScenes
v3 configuration, frozen COCO YOLOv8s) | ours_v6 (K28 v6-S without the refiner, which cannot act here: v_close 0, crops
at 960 / 1600 = 0.6, guard, <= 3 rounds, crop objects lifted with the learned looming cue C1 on the branch's own same-scale
history, as k28_test.lift_c mode C1 / 61_k28_v6_latency.lift_c1; head out/k28_temporal/temporal_head_np1.pt = the frozen
head re-saved without numpy-2 objects). Per-frame records (n actors in camera, hidden, phantoms, crops) -> SAVE_PATH.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import carla
import numpy as np
from scipy.optimize import linear_sum_assignment

from autopilot import AutoPilot

W, H, FOV = 1600, 900, 70.0
CAM_POSE = dict(x=0.80, y=0.0, z=1.60, roll=0.0, pitch=0.0, yaw=0.0)
IOU_THR, THR = 0.3, 0.25
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def assoc_idx(curr, prev, thr=0.3):
    """k27_dev.assoc_idx (verbatim)"""
    from rap.mono import box_iou
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
    """k28_tdata.chains: same-scale 3-observation chains (i, j1, j2)"""
    if s1 is None or s2 is None or not len(s1) or not len(s2) or not len(X):
        return []
    j1 = assoc_idx(X, s1)
    a12 = assoc_idx(s1, s2)
    return [(i, int(j1[i]), int(a12[j1[i]])) for i in range(len(X)) if j1[i] >= 0 and a12[j1[i]] >= 0]


def temp_head(n_feat):
    """k28_ttrain.TempHead without the ImageNet weight download (the state dict overwrites every weight)"""
    import torch
    import torchvision

    class TempHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=None)
            m.fc = torch.nn.Identity()
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
    return TempHead()
PRIORS = {"vehicle": (4.5, 1.9, 1.6), "person": (0.6, 0.6, 1.75), "cyclist": (1.8, 0.7, 1.7)}
TYPE = {"vehicle": "vehicle.pseudo.phantom", "person": "walker.pedestrian.phantom", "cyclist": "vehicle.pseudo.phantom"}


def get_entry_point():
    return "PerceptionPilot"


def actor_class(a):
    if a.type_id.startswith("walker"):
        return "person"
    nw = int(a.attributes.get("number_of_wheels", 4)) if hasattr(a, "attributes") else 4
    return "cyclist" if nw == 2 else "vehicle"


COMPAT = {"vehicle": {"vehicle"}, "person": {"person"}, "cyclist": {"cyclist", "person", "vehicle"}}


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                    # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


class PseudoActor:
    """A static phantom obstacle with the actor API PDM-Lite uses."""
    _next = 10_000_000

    def __init__(self, loc, yaw, size, cls):
        PseudoActor._next += 1
        self.id = PseudoActor._next
        self.type_id = TYPE[cls]
        self.attributes = {"role_name": "phantom", "number_of_wheels": "4"}
        L, Wd, Ht = size
        self._tf = carla.Transform(loc, carla.Rotation(yaw=yaw))
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, Ht / 2), carla.Vector3D(L / 2, Wd / 2, Ht / 2))
        self.is_alive = True

    def get_location(self):
        return self._tf.location

    def get_transform(self):
        return self._tf

    def get_velocity(self):
        return carla.Vector3D(0, 0, 0)

    def get_acceleration(self):
        return carla.Vector3D(0, 0, 0)

    def get_angular_velocity(self):
        return carla.Vector3D(0, 0, 0)

    def get_control(self):
        return carla.WalkerControl() if self.type_id.startswith("walker") else carla.VehicleControl()


class PerceivedActors(list):
    """ActorList stand-in: iterable, filter(pattern) with CARLA's fnmatch semantics on type_id."""

    def filter(self, pattern):
        import fnmatch
        return PerceivedActors([a for a in self if fnmatch.fnmatch(a.type_id, pattern)])

    def find(self, aid):
        return next((a for a in self if a.id == aid), None)


class WorldProxy:
    """carla.World with get_actors() / get_actor() answering from the perceived scene; everything else delegated."""

    def __init__(self, world, actors):
        self._w, self._actors = world, actors

    def get_actors(self, *a):
        return self._actors

    def get_actor(self, aid):
        return self._actors.find(aid) or self._w.get_actor(aid)

    def __getattr__(self, k):
        return getattr(self._w, k)


class PerceptionPilot(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.branch = os.environ.get("PERC_BRANCH", "privileged")
        self.records = []
        self.K = np.array([[W / (2 * math.tan(math.radians(FOV) / 2)), 0, W / 2],
                           [0, W / (2 * math.tan(math.radians(FOV) / 2)), H / 2], [0, 0, 1]])
        if self.branch != "privileged":
            from rap.detect import Mode, TwoFidelityDetector
            self.det = TwoFidelityDetector(os.environ.get("PERC_WEIGHTS", str(Path(os.environ["B"]) / "models" / "yolov8s.pt")),
                                           device="cuda:0", backend="torch")
            self.modes = {"cheap": Mode("b2d_c320", 192, 320), "u960": Mode("b2d_u960", 576, 960),
                          "u1600": Mode("b2d_u1600", 928, 1600)}
            self.prev_scene = None
            self.prev_cheap = None
            if self.branch == "ours_v6":
                import torch
                ck = torch.load(Path(os.environ["B"]) / "out" / "k28_temporal" / "temporal_head_np1.pt", map_location="cuda:0",
                                weights_only=False)
                self.head = temp_head(ck["n_feat"]); self.head.load_state_dict(ck["state"]); self.head = self.head.to("cuda:0").eval()
                self.hmu, self.hsd = np.asarray(ck["fmu"], np.float32), np.asarray(ck["fsd"], np.float32)
                self.hist, self.imgs = {}, []

    def sensors(self):
        s = list(super().sensors())
        s.append({"type": "sensor.camera.rgb", **CAM_POSE, "width": W, "height": H, "fov": FOV, "id": "rgb_front_perc"})
        return s

    # ------------------------------------------------------------------------------------------- geometry helpers
    def _cam_world(self):
        """camera -> world 4x4 (UE4 axes: x forward, y right, z up)"""
        tf = self._vehicle.get_transform()
        return np.array(tf.get_matrix()) @ np.array(carla.Transform(carla.Location(CAM_POSE["x"], CAM_POSE["y"], CAM_POSE["z"])).get_matrix())

    def _project(self, actor, Twc_inv):
        bb = actor.bounding_box
        corners = bb.get_world_vertices(actor.get_transform()) if not isinstance(actor, PseudoActor) else []
        if not corners:
            return None
        P = np.array([[c.x, c.y, c.z, 1.0] for c in corners]) @ Twc_inv.T          # world -> camera (UE4 axes)
        x, y, z = P[:, 1], -P[:, 2], P[:, 0]                                       # -> OpenCV (right, down, forward)
        if (z > 0.1).sum() < 4:
            return None
        z = np.maximum(z, 0.1)
        u, v = self.K[0, 0] * x / z + self.K[0, 2], self.K[1, 1] * y / z + self.K[1, 2]
        x1, y1, x2, y2 = max(u.min(), 0), max(v.min(), 0), min(u.max(), W - 1), min(v.max(), H - 1)
        return None if x2 - x1 < 2 or y2 - y1 < 2 else (x1, y1, x2, y2)

    def _ground(self, u, v, Twc):
        """image point -> world point on the ego's ground plane"""
        d = np.array([1.0, (u - self.K[0, 2]) / self.K[0, 0], -(v - self.K[1, 2]) / self.K[1, 1]])   # UE4 camera axes
        dw = Twc[:3, :3] @ d
        o = Twc[:3, 3]
        gz = self._vehicle.get_transform().location.z
        if dw[2] >= -1e-6:
            return None
        s = (gz - o[2]) / dw[2]
        p = o + s * dw
        return p, math.degrees(math.atan2(dw[1], dw[0]))

    # ------------------------------------------------------------------------------------------- perception branches
    def _detect(self, img):
        from rap.mono import box_iou
        keep = lambda d: {"xyxy": d["xyxy"][d["conf"] >= THR].astype(float), "conf": d["conf"][d["conf"] >= THR].astype(float),  # noqa: E731
                          "coarse": d["coarse"][d["conf"] >= THR]}
        if self.branch in self.modes:
            return keep(self.det.detect(img, self.modes[self.branch])[0]), 0
        # ours: cheap draft -> braking field + support crops (vtf.speculative.run, the nuScenes v3 configuration)
        from rap import planner as Pl
        from rap.detect import Mode
        from rap.kitti import Calib
        from rap.mono import predicted_geometry
        from vtf import speculative as SP
        ch = keep(self.det.detect(img, self.modes["cheap"])[0])
        P2 = np.zeros((3, 4)); P2[:3, :3] = self.K
        cam = type("Cam", (), {})()
        cam.calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
        cam.cam_h, cam.dt = CAM_POSE["z"], 0.05
        R_ce = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)                # OpenCV camera -> ego (x fwd, y left, z up)
        t_ce = np.array([CAM_POSE["x"], CAM_POSE["y"], CAM_POSE["z"]])

        def lift(dd, pv):
            g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R_ce, t_ce))
            return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
        ch["mono"] = lift(ch, self.prev_cheap)
        self.prev_cheap = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}

        def crop_det(cr):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = img[y1:y2, x1:x2]
            nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
            nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
            d = keep(self.det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
            d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
            return d
        v = float(self._get_forward_speed())
        if self.branch == "ours_v6":
            return self._detect_v6(img, ch, v, cam, R_ce, t_ce, lift, crop_det, Pl, SP, predicted_geometry)
        scene, crops, _ = SP.run(ch, v, cam, W, H, crop_det, lambda ff: lift(ff, self.prev_scene), Pl.PlannerParams(),
                                 v_close=10.0, guard=True)
        self.prev_scene = {"xyxy": scene["xyxy"]}
        return {"xyxy": np.asarray(scene["xyxy"], float).reshape(-1, 4), "conf": np.asarray(scene["conf"], float),
                "coarse": np.asarray(scene["coarse"])}, len(crops)

    def _detect_v6(self, img, ch, v, cam, R_ce, t_ce, lift, crop_det, Pl, SP, predicted_geometry):
        """ours_v6: v6-S loop (v_close 0, crops 0.6, guard) with the C1 lift on this branch's own history"""
        import torch
        from nusc_refiner_crop import crop64
        h, imgs, dt = self.hist, self.imgs, cam.dt

        def lift_c1(full):
            X = np.asarray(full["xyxy"], float).reshape(-1, 4)
            z, lo, hi, _ = lift({"xyxy": X, "coarse": full["coarse"]}, None)
            ttc = np.full(len(X), np.inf)
            chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
            if not chn:
                return z, lo, hi, ttc
            idx = np.array([c[0] for c in chn])
            zz = predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                    cam_to_ego=(R_ce, t_ce))
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
            xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().to("cuda:0") / 255.0
            fin = torch.from_numpy(((np.array(fe, np.float32) - self.hmu) / self.hsd).astype(np.float32)).to("cuda:0")
            with torch.no_grad():
                rh = np.clip(self.head(xin, fin).float().cpu().numpy(), 0, 2)
            for n, (i, _, _) in enumerate(chn):
                r = rh[n]
                ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
            return z, lo, hi, ttc
        scene, crops, _ = SP.run(ch, v, cam, W, H, crop_det, lift_c1, Pl.PlannerParams(), v_close=0.0, guard=True)
        ff = np.asarray(scene["from_full"], bool)
        h["s2"] = h.get("s1")
        h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
        self.imgs = (imgs + [img])[-2:]
        return {"xyxy": np.asarray(scene["xyxy"], float).reshape(-1, 4), "conf": np.asarray(scene["conf"], float),
                "coarse": np.asarray(scene["coarse"])}, len(crops)

    def _perceived(self, img):
        actors = self._world.get_actors()
        others = [a for a in list(actors.filter("*vehicle*")) + list(actors.filter("*walker*")) if a.id != self._vehicle.id]
        Twc = self._cam_world()
        Twc_inv = np.linalg.inv(Twc)
        proj = {a.id: self._project(a, Twc_inv) for a in others}
        inc = [a for a in others if proj[a.id] is not None
               and a.get_location().distance(self._vehicle.get_location()) < self.config.detection_radius]
        det, n_crops = self._detect(img)
        dx, dc = det["xyxy"], np.asarray(det["coarse"]).astype(str)
        matched, used = set(), set()
        if inc and len(dx):
            M = iou([proj[a.id] for a in inc], dx)
            for i, a in enumerate(inc):
                M[i, [j for j in range(len(dx)) if dc[j] not in COMPAT[actor_class(a)]]] = 0
            r, c = linear_sum_assignment(-M)
            for i, j in zip(r, c):
                if M[i, j] >= IOU_THR:
                    matched.add(inc[i].id); used.add(j)
        hidden = {a.id for a in inc} - matched
        out = PerceivedActors([a for a in actors if a.id not in hidden])
        n_ph = 0
        for j in range(len(dx)):
            if j in used:
                continue
            b = dx[j]
            g = self._ground(0.5 * (b[0] + b[2]), b[3], Twc)
            if g is None:
                continue
            p, yaw = g
            rng = math.hypot(p[0] - Twc[0, 3], p[1] - Twc[1, 3])
            if rng > 80.0 or rng < 0.5:
                continue
            cls = dc[j] if dc[j] in PRIORS else "vehicle"
            L, Wd, Ht = PRIORS[cls]
            cx, cy = p[0] + math.cos(math.radians(yaw)) * L / 2, p[1] + math.sin(math.radians(yaw)) * L / 2
            out.append(PseudoActor(carla.Location(cx, cy, p[2]), yaw, PRIORS[cls], cls))
            n_ph += 1
        self.records.append({"step": self.step if hasattr(self, "step") else len(self.records), "in_cam": len(inc),
                             "hidden": len(hidden), "phantoms": n_ph, "crops": n_crops})
        return out

    def _get_control(self, input_data, plant):
        if self.branch == "privileged" or "rgb_front_perc" not in input_data:
            return super()._get_control(input_data, plant)
        img = np.ascontiguousarray(input_data["rgb_front_perc"][1][:, :, :3])          # BGRA -> BGR
        real = self._world
        self._world = WorldProxy(real, self._perceived(img))
        try:
            return super()._get_control(input_data, plant)
        finally:
            self._world = real

    def destroy(self, results=None):
        # one JSON line per route (appended): per-frame means of in-camera actors, hidden actors, unmatched detections
        # (phantoms), crops — the agent is re-created for every route, so a per-chunk file would keep only the last route
        if self.branch != "privileged" and os.environ.get("SAVE_PATH"):
            p = Path(os.environ["SAVE_PATH"]) / f"perception_{self.branch}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(json.dumps({"branch": self.branch, "routes": Path(os.environ.get("ROUTES", "r")).stem,
                                     "route_index": getattr(self, "route_index", None), "frames": len(self.records),
                                     "mean": {k: float(np.mean([r[k] for r in self.records])) if self.records else 0.0
                                              for k in ("in_cam", "hidden", "phantoms", "crops")}}) + "\n")
        try:
            super().destroy(results)
        except TypeError:
            super().destroy()
