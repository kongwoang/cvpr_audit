"""Argoverse 1.1 tracking logs (the Argoverse-HD benchmark's sequences) in DEEP's conventions — mirrors vtf.av2.AV2Log.

Frames: the ring_front_center images of the sequence that Argoverse-HD annotates (30 Hz, 1920x1200), in order.
Ground-truth geometry (for decision value) comes from the 3D amodal sweep labels (per_sweep_annotations_amodal, 10 Hz
LiDAR): a camera frame gets the labels of the sweep closest in time if it is within 50 ms (else no geometry for that
frame); boxes are motion-compensated LiDAR -> camera time through the city poses, projected into the camera for the 2D
box, footprint in the ego frame for range / lateral extent, range rate by DEEP's `_fill_range_rate`. Classes: vehicles
(VEHICLE, LARGE_VEHICLE, BUS, TRAILER, EMERGENCY_VEHICLE, SCHOOL_BUS, ...), PEDESTRIAN, BICYCLIST / MOTORCYCLIST.
Camera: pinhole K (radial distortion ignored, as for AV2 / nuScenes), camera height = camera z in the ego frame.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rap.geometry import GEOM_FIELDS, _fill_range_rate
from rap.kitti import Calib

CAM = "ring_front_center"
COARSE = {"VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BUS": "vehicle", "TRAILER": "vehicle", "SCHOOL_BUS": "vehicle",
          "EMERGENCY_VEHICLE": "vehicle", "MOPED": "cyclist", "MOTORCYCLE": "cyclist", "BICYCLE": "cyclist",
          "BICYCLIST": "cyclist", "MOTORCYCLIST": "cyclist", "PEDESTRIAN": "person"}


def _R(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


class AV1Log:
    def __init__(self, root, log, image_names):
        """root = .../argoverse-tracking/val; image_names = the sequence's Argoverse-HD image file names in fid order."""
        self.root, self.log = Path(root), log
        d = self.root / log
        cal = json.loads((d / "vehicle_calibration_info.json").read_text())
        c = next(x["value"] for x in cal["camera_data_"] if x["key"].endswith(CAM))
        q = c["vehicle_SE3_camera_"]["rotation"]["coefficients"]
        self.Rc, self.tc = _R(*q), np.asarray(c["vehicle_SE3_camera_"]["translation"], float)
        self.K = np.array([[c["focal_length_x_px_"], 0, c["focal_center_x_px_"]],
                           [0, c["focal_length_y_px_"], c["focal_center_y_px_"]], [0, 0, 1]], float)
        self.W, self.H = 1920, 1200
        self.img_dir = d / CAM
        self.names = list(image_names)
        self.frame_ts = np.array([int(n.rsplit("_", 1)[1].split(".")[0]) for n in self.names], np.int64)
        self.dt = float(np.median(np.diff(self.frame_ts))) * 1e-9 if len(self.frame_ts) > 1 else 1 / 30
        ps = sorted((d / "poses").glob("city_SE3_egovehicle_*.json"))
        self.pose_ts = np.array([int(p.stem.rsplit("_", 1)[1]) for p in ps], np.int64)
        self._pose_files = ps
        lab = d / "per_sweep_annotations_amodal"
        ls = sorted(lab.glob("tracked_object_labels_*.json")) if lab.exists() else []
        self.sweep_ts = np.array([int(p.stem.rsplit("_", 1)[1]) for p in ls], np.int64)
        self._label_files = ls

    @functools.lru_cache(maxsize=None)
    def _pose_i(self, i):
        p = json.loads(self._pose_files[i].read_text())
        return _R(*p["rotation"]), np.asarray(p["translation"], float)

    def _pose(self, t):
        i = int(np.clip(np.searchsorted(self.pose_ts, t), 1, len(self.pose_ts) - 1))
        j = i if abs(self.pose_ts[i] - t) < abs(self.pose_ts[i - 1] - t) else i - 1
        return self._pose_i(j)

    def image_path(self, fr):
        return self.img_dir / self.names[fr]

    def speeds(self):
        pos = np.array([self._pose(t)[1] for t in self.frame_ts])
        if len(pos) < 2:
            return np.zeros(len(pos))
        # central difference over ~0.1 s (3 frames) smooths the 30 Hz pose jitter
        k = 3
        v = np.zeros(len(pos))
        for i in range(len(pos)):
            a, b = max(0, i - k), min(len(pos) - 1, i + k)
            v[i] = np.linalg.norm(pos[b, :2] - pos[a, :2]) / max((self.frame_ts[b] - self.frame_ts[a]) * 1e-9, 1e-3)
        return v

    @functools.cached_property
    def calib(self):
        P2 = np.zeros((3, 4))
        P2[:3, :3] = self.K
        return Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))

    @property
    def cam_h(self):
        return float(self.tc[2])

    def transform(self, frame="ego"):
        assert frame == "ego"
        return self.Rc, self.tc

    def gt_frames(self, tol_ns=50_000_000):
        """frame index -> sweep index for frames with a sweep within tol (the frames with ground-truth geometry)."""
        out = {}
        if not len(self.sweep_ts):
            return out
        for fr, t in enumerate(self.frame_ts):
            j = int(np.argmin(np.abs(self.sweep_ts - t)))
            if abs(int(self.sweep_ts[j]) - int(t)) <= tol_ns:
                out[fr] = j
        return out

    @functools.lru_cache(maxsize=1)
    def geometry(self):
        rows, tracks = [], {}
        for fr, j in self.gt_frames().items():
            t_l, t_c = int(self.sweep_ts[j]), int(self.frame_ts[fr])
            Rl, tl = self._pose(t_l)
            Rm, tm = self._pose(t_c)
            for o in json.loads(self._label_files[j].read_text()):
                cls = COARSE.get(o.get("label_class", ""))
                if cls is None:
                    continue
                c = np.array([o["center"]["x"], o["center"]["y"], o["center"]["z"]], float)
                r = o["rotation"]
                Rb = _R(r["w"], r["x"], r["y"], r["z"])
                l, w, h = o["length"], o["width"], o["height"]
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in corners])
                allc = np.vstack([foot, foot + Rb @ np.array([0, 0, h])])
                comp = lambda p: ((Rl @ p.T).T + tl - tm) @ Rm                       # noqa: E731
                foot_e, allc_e = comp(foot), comp(allc)
                cam = (allc_e - self.tc) @ self.Rc
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (self.K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = np.clip(uv.min(0), 0, [self.W - 1, self.H - 1])
                x2, y2 = np.clip(uv.max(0), 0, [self.W - 1, self.H - 1])
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                tid = tracks.setdefault(o.get("track_label_uuid", id(o)), len(tracks))
                rows.append((fr, tid, o["label_class"].lower(), cls, x1, y1, x2, y2, foot_e[:, 0].min(), foot_e[:, 0].mean(),
                             foot_e[:, 1].mean(), foot_e[:, 1].min(), foot_e[:, 1].max()))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = self.log[:16], r[0], r[1]
            out[i]["type"], out[i]["coarse"] = r[2], r[3]
            for k, col in enumerate(("x1", "y1", "x2", "y2")):
                out[i][col] = r[4 + k]
            out[i]["truncated"], out[i]["occluded"] = 0.0, 0
            for k, col in enumerate(("long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
                out[i][col] = r[8 + k]
            out[i]["range_rate"], out[i]["ttc"] = 0.0, np.inf
        # frames are 30 Hz but labelled sweeps ~10 Hz (every 3rd frame): dt per frame, window trimmed to ±0.117 s
        _fill_range_rate(out, halfwidth=3, dt=1 / 30)
        return out
