"""Argoverse 2 (sensor dataset) in DEEP's conventions: ground-truth geometry, ego speeds and a front-camera object.

Frames: the annotation (LiDAR) timestamps, every `stride`-th (10 Hz / stride), each paired with the nearest
ring_front_center image. Ground truth mirrors `rap.nusc.NuScenesDB.sequence_geometry`: road users (vehicle / person /
cyclist classes with >= 1 interior LiDAR point) as ego-frame footprints (long_near = nearest footprint x, lateral
extent from footprint y) and 2D boxes from the 3D corners projected into the camera; boxes are motion-compensated from
the LiDAR to the camera timestamp through the city poses; range rate by DEEP's `_fill_range_rate`.
Camera: pinhole K (radial distortion ignored, as DEEP's nuScenes path), camera height = the camera's z in the ego
frame, `transform("ego")` = camera -> ego (R, t), `dt` = frame interval.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from rap.geometry import GEOM_FIELDS, _fill_range_rate
from rap.kitti import Calib

CAM = "ring_front_center"
COARSE = {"REGULAR_VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BUS": "vehicle", "BOX_TRUCK": "vehicle",
          "TRUCK": "vehicle", "TRUCK_CAB": "vehicle", "VEHICULAR_TRAILER": "vehicle", "SCHOOL_BUS": "vehicle",
          "ARTICULATED_BUS": "vehicle", "MOTORCYCLE": "cyclist", "BICYCLE": "cyclist", "BICYCLIST": "cyclist",
          "MOTORCYCLIST": "cyclist", "WHEELED_RIDER": "cyclist", "PEDESTRIAN": "person"}


def _R(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


class AV2Log:
    def __init__(self, root: str | Path, log: str, stride: int = 2):
        self.root, self.log, self.stride = Path(root), log, stride
        d = self.root / log
        self.ann = pd.read_feather(d / "annotations.feather")
        ext = pd.read_feather(d / "calibration" / "egovehicle_SE3_sensor.feather").set_index("sensor_name").loc[CAM]
        intr = pd.read_feather(d / "calibration" / "intrinsics.feather").set_index("sensor_name").loc[CAM]
        self.Rc, self.tc = _R(ext.qw, ext.qx, ext.qy, ext.qz), np.array([ext.tx_m, ext.ty_m, ext.tz_m], float)
        self.K = np.array([[intr.fx_px, 0, intr.cx_px], [0, intr.fy_px, intr.cy_px], [0, 0, 1]], float)
        self.W, self.H = int(intr.width_px), int(intr.height_px)
        pose = pd.read_feather(d / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
        self.pose_ts = pose.timestamp_ns.to_numpy(np.int64)
        self.pose = pose
        self.img_dir = d / "sensors" / "cameras" / CAM
        self.img_ts = np.array(sorted(int(p.stem) for p in self.img_dir.glob("*.jpg")), np.int64)
        ts = np.sort(self.ann.timestamp_ns.unique())
        self.frame_ts = ts[::stride]
        self.dt = float(np.median(np.diff(self.frame_ts))) * 1e-9 if len(self.frame_ts) > 1 else 0.2

    # ---- poses ---------------------------------------------------------------------------------------------------
    def _pose(self, t):
        i = int(np.clip(np.searchsorted(self.pose_ts, t), 1, len(self.pose_ts) - 1))
        j = i if abs(self.pose_ts[i] - t) < abs(self.pose_ts[i - 1] - t) else i - 1
        r = self.pose.iloc[j]
        return _R(r.qw, r.qx, r.qy, r.qz), np.array([r.tx_m, r.ty_m, r.tz_m], float)

    def image_ts(self, fr: int) -> int:
        t = self.frame_ts[fr]
        return int(self.img_ts[np.argmin(np.abs(self.img_ts - t))])

    def image_path(self, fr: int) -> Path:
        return self.img_dir / f"{self.image_ts(fr)}.jpg"

    def speeds(self) -> np.ndarray:
        pos = np.array([self._pose(t)[1] for t in self.frame_ts])
        if len(pos) < 2:
            return np.zeros(len(pos))
        v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / (np.diff(self.frame_ts) * 1e-9)
        return np.r_[v[0], v]

    # ---- camera --------------------------------------------------------------------------------------------------
    @functools.cached_property
    def calib(self) -> Calib:
        P2 = np.zeros((3, 4))
        P2[:3, :3] = self.K
        return Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))

    @property
    def cam_h(self) -> float:
        return float(self.tc[2])

    def transform(self, frame: str = "ego"):
        assert frame == "ego"
        return self.Rc, self.tc

    # ---- ground truth ----------------------------------------------------------------------------------------------
    @functools.lru_cache(maxsize=1)
    def geometry(self) -> np.ndarray:
        rows = []
        tracks = {}
        for fr, t in enumerate(self.frame_ts):
            a = self.ann[(self.ann.timestamp_ns == t) & self.ann.category.isin(list(COARSE)) &
                         (self.ann.num_interior_pts >= 1)]
            if not len(a):
                continue
            # motion compensation: ego(t_lidar) -> city -> ego(t_camera)
            Rl, tl = self._pose(t)
            Rm, tm = self._pose(self.image_ts(fr))
            for r in a.itertuples():
                Rb = _R(r.qw, r.qx, r.qy, r.qz)
                c = np.array([r.tx_m, r.ty_m, r.tz_m])
                l, w, h = r.length_m, r.width_m, r.height_m
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in corners])
                top = foot + Rb @ np.array([0, 0, h])
                allc = np.vstack([foot, top])
                comp = lambda p: ((Rl @ p.T).T + tl - tm) @ Rm           # noqa: E731  ego(lidar) -> ego(camera)
                foot_e, allc_e = comp(foot), comp(allc)
                cam = (allc_e - self.tc) @ self.Rc
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (self.K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, self.W - 1)
                y1, y2 = np.clip([y1, y2], 0, self.H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                tid = tracks.setdefault(r.track_uuid, len(tracks))
                rows.append((self.log, fr, tid, r.category.lower(), COARSE[r.category], x1, y1, x2, y2, 0.0, 0,
                             foot_e[:, 0].min(), foot_e[:, 0].mean(), foot_e[:, 1].mean(), foot_e[:, 1].min(),
                             foot_e[:, 1].max(), 0.0))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0][:16], r[1], r[2]
            out[i]["type"], out[i]["coarse"] = r[3], r[4]
            for j, col in enumerate(("x1", "y1", "x2", "y2", "truncated", "occluded", "long_near", "long_cen",
                                     "lat_cen", "lat_min", "lat_max", "range_rate")):
                out[i][col] = r[5 + j]
            out[i]["ttc"] = np.inf
        _fill_range_rate(out, halfwidth=1, dt=self.dt)
        return out
