"""K30 V4: perception_pilot.PerceptionPilot with per-frame LOGGING only (the driving is unchanged: the same perceived world,
the same PDM-Lite control). Per frame it records, from the simulator's ground truth:
  ego        speed (m/s), location (x, y), control brake / throttle, PDM-Lite's hazard flags (vehicle / walker / traffic
             light / stop sign) and the vehicle / walker causing the hazard -> whether it is a phantom (an unmatched
             detection the branch turned into a static obstacle; ids >= 10,000,000)
  actors     every real vehicle / walker within 60 m that is ahead and laterally inside the ego corridor (|y| <= 1.2 m + the
             actor's half width, ego frame), with its time to collision (gap from the ego front to the actor's near face
             over the closing speed; inf if not closing), whether it projects into the front camera within the detection
             radius, and whether the branch hid it (in camera and not matched by a detection)
destroy() appends one JSON line per route to $SAVE_PATH/inloop_<branch>.jsonl: the route's town and start point (the first
point of the global plan; the leaderboard calls setup() without a route index, so the route id is recovered offline by
matching town + start point to the routes file) and the per-frame records.
Metrics derived offline (k30_b2d_inloop.py): recall of actors within 3 s TTC (in camera, not hidden), hard-brake events
(brake >= 0.5, rising edge) per km and the share caused by a phantom ("unnecessary"), minimum TTC to the closest relevant
actor per route.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from perception_pilot import PerceptionPilot, PseudoActor

EGO_FRONT = 2.5
CORR_HALF = 1.2


def get_entry_point():
    return "PerceptionPilotLog"


class PerceptionPilotLog(PerceptionPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.inloop, self._hidden, self._inc = [], set(), set()
        self._start = None

    def _perceived(self, img):
        out = super()._perceived(img)
        kept = {a.id for a in out}
        actors = self._world.get_actors()
        others = [a for a in list(actors.filter("*vehicle*")) + list(actors.filter("*walker*")) if a.id != self._vehicle.id]
        Twc_inv = np.linalg.inv(self._cam_world())
        self._inc = {a.id for a in others if self._project(a, Twc_inv) is not None
                     and a.get_location().distance(self._vehicle.get_location()) < self.config.detection_radius}
        self._hidden = {i for i in self._inc if i not in kept}
        return out

    def _get_control(self, input_data, plant):
        self._hidden, self._inc = set(), set()
        out = super()._get_control(input_data, plant)                # AutoPilot returns (control, driving_data)
        try:
            self._log(out[0] if isinstance(out, tuple) else out)
        except Exception as ex:                                                      # noqa: BLE001 (logging never breaks driving)
            self.inloop.append({"err": f"{type(ex).__name__}: {ex}"[:160]})
        return out

    def _log(self, ctl):
        if self._start is None and self._global_plan_world_coord:
            p0 = self._global_plan_world_coord[0][0].location
            self._start = {"town": self._world.get_map().name.split("/")[-1], "x": round(p0.x, 2), "y": round(p0.y, 2)}
        tf = self._vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)                 # forward
        rx, ry = -math.sin(yaw), math.cos(yaw)                # right (UE4: y right)
        ev = self._vehicle.get_velocity()
        speed = math.hypot(ev.x, ev.y)
        if self.branch == "privileged":
            Twc_inv = np.linalg.inv(self._cam_world())
        acts = []
        for a in list(self._world.get_actors().filter("*vehicle*")) + list(self._world.get_actors().filter("*walker*")):
            if a.id == self._vehicle.id or isinstance(a, PseudoActor):
                continue
            loc = a.get_location()
            dx, dy = loc.x - tf.location.x, loc.y - tf.location.y
            x, y = dx * fx + dy * fy, dx * rx + dy * ry
            if x <= 0 or math.hypot(x, y) > 60.0:
                continue
            ext = a.bounding_box.extent
            if abs(y) > CORR_HALF + ext.y:
                continue
            av = a.get_velocity()
            closing = (ev.x - av.x) * fx + (ev.y - av.y) * fy
            gap = x - EGO_FRONT - ext.x
            ttc = gap / closing if closing > 0.1 and gap > 0 else (0.0 if gap <= 0 else math.inf)
            if self.branch == "privileged":
                in_cam = self._project(a, Twc_inv) is not None and loc.distance(tf.location) < self.config.detection_radius
            else:
                in_cam = a.id in self._inc
            acts.append([int(a.id), round(float(x), 2), round(float(y), 2), round(float(ttc), 3) if math.isfinite(ttc) else None,
                         bool(in_cam), bool(a.id in self._hidden)])
        haz_id = getattr(self, "vehicle_affecting_id", None) if getattr(self, "vehicle_hazard", False) else None
        wid = getattr(self, "walker_affecting_id", None) if getattr(self, "walker_hazard", False) else None
        self.inloop.append({"v": round(speed, 3), "xy": [round(tf.location.x, 2), round(tf.location.y, 2)],
                            "brake": round(float(ctl.brake), 3), "throttle": round(float(ctl.throttle), 3),
                            "haz": [bool(getattr(self, "vehicle_hazard", False)), bool(getattr(self, "walker_hazard", False)),
                                    bool(getattr(self, "traffic_light_hazard", False)), bool(getattr(self, "stop_sign_hazard", False))],
                            "haz_id": haz_id, "walker_id": wid,
                            "haz_phantom": bool((haz_id is not None and int(haz_id) >= 10_000_000) or (wid is not None and int(wid) >= 10_000_000)),
                            "actors": acts})

    def destroy(self, results=None):
        if os.environ.get("SAVE_PATH"):
            p = Path(os.environ["SAVE_PATH"]) / f"inloop_{self.branch}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(json.dumps({"branch": self.branch, "routes": Path(os.environ.get("ROUTES", "r")).stem,
                                     "start": self._start, "frames": self.inloop}) + "\n")
        super().destroy(results)
