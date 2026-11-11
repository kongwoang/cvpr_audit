"""K32 agent (pre-registered in handoff/K32/STATUS.md): PDM-Lite route following WITHOUT actors + the paper's braking
controller on the scene of a perception branch, commanded deceleration = max(route follower's, controller's).

env  AEB_BRANCH   privileged | cheap | u960 | native | hybrid | v6S | v6Sb2 | unc (amendment G1: K28 unc2 crops)
     AEB_LAT      L0 (no latency) | L1 (K29 per-frame Jetson latency, results/k32/latency_samples.csv; unc: the G2 model
                  t = a + b * calls + c * crop Mpx + a residual draw, results/k32/latency_model.json)
     AEB_SEED     latency draws (with the route's start point); the traffic-manager seed is the evaluator's
     AEB_C1 / AEB_REF   1 (default) | 0: C1 / refiner v3a on or off for every branch (dev rule)
     AEB_LOGALL   1: dev log run — every perception branch is computed on the same frames (the applied action stays the
                  AEB_BRANCH one), with the lift error of every matched actor
     AEB_BRAKE_MAP  json {"2.5": b, "6.0": b}: brake-pedal value per commanded deceleration (dev calibration)
destroy() writes $SAVE_PATH/aeb_<branch>_<setting>.jsonl, one JSON line per route (start point, per-tick records).
Amendment A (handoff/K32/STATUS.md, 2026-10-20): standstill hold (A1) for every branch, logged per tick ("hold"), and a
contact log ("contacts": first contact of the ego box with each vehicle / walker, with type and ego speed).
Amendment B: AEB_FRONT (m) = the ego's front ahead of its centre; every branch's range is counted from the front for the
braking controller (stand-off r0 + AEB_FRONT; the hold radius follows).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import carla
import numpy as np
from scipy.optimize import linear_sum_assignment

from autopilot import AutoPilot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rap import planner as P

W, H, FOV = 1600, 900, 70.0
CAM_POSE = dict(x=0.80, y=0.0, z=1.60, roll=0.0, pitch=0.0, yaw=0.0)
DT = 0.05
EGO_FRONT = 2.5
BRANCHES = ("cheap", "u960", "native", "v6S", "v6Sb2", "hybrid", "unc")
COMPAT = {"vehicle": {"vehicle"}, "person": {"person"}, "cyclist": {"cyclist", "person", "vehicle"}}


def get_entry_point():
    return "AEBPilot"


def actor_class(a):
    if a.type_id.startswith("walker"):
        return "person"
    nw = int(a.attributes.get("number_of_wheels", 4)) if hasattr(a, "attributes") else 4
    return "cyclist" if nw == 2 else "vehicle"


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                    # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


class _Blind(list):
    def filter(self, pattern):
        import fnmatch
        return _Blind([a for a in self if fnmatch.fnmatch(a.type_id, pattern)])

    def find(self, aid):
        return next((a for a in self if a.id == aid), None)


class BlindWorld:
    """carla.World whose actor list has no vehicle / walker / static prop (traffic lights and signs stay)"""

    def __init__(self, world):
        self._w = world
        acts = world.get_actors()
        self._a = _Blind([a for a in acts if not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")
                                                   or a.type_id.startswith("static.prop"))])

    def get_actors(self, *a):
        return self._a

    def get_actor(self, aid):
        return self._a.find(aid) or self._w.get_actor(aid)

    def __getattr__(self, k):
        return getattr(self._w, k)


def f3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


class AEBPilot(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self.branch = os.environ.get("AEB_BRANCH", "privileged")
        self.setting = os.environ.get("AEB_LAT", "L0")
        self.logall = os.environ.get("AEB_LOGALL", "0") == "1"
        self.seed = int(os.environ.get("AEB_SEED", "0"))
        self.pp = P.PlannerParams(standoff=P.PlannerParams().standoff + float(os.environ.get("AEB_FRONT", "0")))   # amendment B
        bm = json.loads(os.environ.get("AEB_BRAKE_MAP", '{"2.5": 0.3, "6.0": 0.75}'))
        self.bmap = {1: float(bm["2.5"]), 2: float(bm["6.0"])}
        self.K = np.array([[W / (2 * math.tan(math.radians(FOV) / 2)), 0, W / 2],
                           [0, W / (2 * math.tan(math.radians(FOV) / 2)), H / 2], [0, 0, 1]])
        self.tick, self.q, self.applied, self.recs, self._start = 0, [], 0, [], None
        self.near_applied, self.hold, self.hold_clear, self.contacts, self.n_ph = False, False, 0, {}, 0
        self.rng = np.random.default_rng(self.seed)
        self.names = set()
        if self.branch != "privileged" or self.logall:
            from rap.detect import TwoFidelityDetector
            import k32_pipes
            det = TwoFidelityDetector(str(Path(os.environ["B"]) / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
            self.pipes = k32_pipes.Pipes(det, self.K, W, H, (CAM_POSE["x"], CAM_POSE["y"], CAM_POSE["z"]), DT,
                                         use_c1=os.environ.get("AEB_C1", "1") == "1", use_ref=os.environ.get("AEB_REF", "1") == "1")
            if self.logall:
                self.names = set(BRANCHES)
            else:
                # E1: the cheap draft's action on the same frame (shadow; the draft is computed by every branch anyway)
                self.names = {self.branch, "cheap"} | ({"u960"} if self.branch == "v6S" else set())
        self.lat = None
        if self.setting == "L1" and self.branch == "unc":                         # amendment G2
            self.lat = json.load(open(Path(os.environ["B"]) / "results" / "k32" / "latency_model.json"))
        elif self.setting == "L1" and self.branch != "privileged":
            import pandas as pd
            d = pd.read_csv(Path(os.environ["B"]) / "results" / "k32" / "latency_samples.csv")
            self.lat = d[d.branch == self.branch].reset_index(drop=True)

    def sensors(self):
        s = list(super().sensors())
        s.append({"type": "sensor.camera.rgb", **CAM_POSE, "width": W, "height": H, "fov": FOV, "id": "rgb_front_perc"})
        return s

    def _manage_route_obstacle_scenarios(self, target_speed, ego_speed, route_waypoints, list_vehicles, route_points):
        return target_speed, False, [target_speed, None, None, None]            # actor-blind route follower: no bypasses

    # ---------------------------------------------------------------------------------------------- geometry
    def _cam_world(self):
        tf = self._vehicle.get_transform()
        return np.array(tf.get_matrix()) @ np.array(carla.Transform(carla.Location(CAM_POSE["x"], CAM_POSE["y"], CAM_POSE["z"])).get_matrix())

    def _project(self, actor, Twc_inv):
        corners = actor.bounding_box.get_world_vertices(actor.get_transform())
        Pc = np.array([[c.x, c.y, c.z, 1.0] for c in corners]) @ Twc_inv.T
        x, y, z = Pc[:, 1], -Pc[:, 2], Pc[:, 0]
        if (z > 0.1).sum() < 4:
            return None
        z = np.maximum(z, 0.1)
        u, v = self.K[0, 0] * x / z + self.K[0, 2], self.K[1, 1] * y / z + self.K[1, 2]
        x1, y1, x2, y2 = max(u.min(), 0), max(v.min(), 0), min(u.max(), W - 1), min(v.max(), H - 1)
        return None if x2 - x1 < 2 or y2 - y1 < 2 else (x1, y1, x2, y2)

    def _annotated(self):
        """annotated states of every vehicle / walker: ego frame (x fwd, y left), in-camera flag and 2D box"""
        tf = self._vehicle.get_transform()
        Tew_inv = np.linalg.inv(np.array(tf.get_matrix()))
        Twc_inv = np.linalg.inv(self._cam_world())
        yaw = math.radians(tf.rotation.yaw)
        fx_, fy_ = math.cos(yaw), math.sin(yaw)
        ev = self._vehicle.get_velocity()
        out = []
        for a in self._world.get_actors():
            if a.id == self._vehicle.id or not (a.type_id.startswith("vehicle") or a.type_id.startswith("walker")):
                continue
            loc = a.get_location()
            if loc.distance(tf.location) > 90.0:
                continue
            cs = a.bounding_box.get_world_vertices(a.get_transform())
            Pe = np.array([[c.x, c.y, c.z, 1.0] for c in cs]) @ Tew_inv.T
            x, y = Pe[:, 0], -Pe[:, 1]                                               # UE4 y right -> left
            av = a.get_velocity()
            closing = (ev.x - av.x) * fx_ + (ev.y - av.y) * fy_
            lat_rate = (av.x - ev.x) * (-fy_) + (av.y - ev.y) * fx_       # actor's lateral speed rel. to the ego (+left)
            near = float(x.min())
            ttc = near / closing if closing > 0.1 and near > 0 else (0.0 if near <= 0 and x.max() > 0 else math.inf)
            box = self._project(a, Twc_inv)
            inc = box is not None and loc.distance(tf.location) <= self.pp.max_range       # B2: the controller's range
            out.append({"id": int(a.id), "cls": actor_class(a), "near": near, "far": float(x.max()), "lmin": float(y.min()),
                        "lmax": float(y.max()), "ttc": ttc, "closing": closing, "lat_rate": lat_rate, "box": box, "inc": inc})
        return out

    # ---------------------------------------------------------------------------------------------- control
    def _latency(self, r):
        if self.lat is None:
            return 0.0
        if self.branch == "unc":
            m = self.lat
            t = m["a"] + m["b_per_call"] * float(r.get("ncrops", 0) > 0) + m["c_per_mpx"] * float(r.get("crop_px", 0.0)) / 1e6
            return float(max(t + m["residuals"][int(self.rng.integers(0, len(m["residuals"])))], m["min_ms"]))
        d = self.lat
        if self.branch in ("v6S", "v6Sb2"):
            k = int(r.get("rounds", 0))
            while k > 0 and (d.calls == k).sum() < 50:
                k -= 1
            d = d[d.calls == k] if (d.calls == k).sum() else d
        return float(d.ms.iloc[int(self.rng.integers(0, len(d)))])

    def _get_control(self, input_data, plant):
        self.tick += 1
        real = self._world
        self._world = BlindWorld(real)
        try:
            out = super()._get_control(input_data, plant)
        finally:
            self._world = real
        ctl = out[0] if isinstance(out, tuple) else out
        try:
            self._aeb(ctl, input_data)
        except Exception as ex:                                                      # noqa: BLE001
            self.recs.append({"t": self.tick, "err": f"{type(ex).__name__}: {ex}"[:200]})
            raise
        return out

    def _aeb(self, ctl, input_data):
        if self._start is None and self._global_plan_world_coord:
            p0 = self._global_plan_world_coord[0][0].location
            self._start = {"town": self._world.get_map().name.split("/")[-1], "x": round(p0.x, 2), "y": round(p0.y, 2)}
            self.rng = np.random.default_rng([self.seed, int(abs(p0.x) * 10), int(abs(p0.y) * 10)])
        tf = self._vehicle.get_transform()
        ev = self._vehicle.get_velocity()
        v = math.hypot(ev.x, ev.y)
        ann = self._annotated()
        inc = [a for a in ann if a.get("inc")]
        gt_t = tuple(np.array([a[k] for a in inc], float) for k in ("near", "lmin", "lmax", "ttc")) if inc else tuple(np.zeros(0) for _ in range(4))
        a_gt = P.discrete_action(P.required_decel(*gt_t, v, self.pp)[0], self.pp)
        corr = [(a["near"] - EGO_FRONT) / a["closing"] if a["closing"] > 0.1 else math.inf
                for a in ann if EGO_FRONT < a["near"] < 60.0 and P.in_corridor([a["lmin"]], [a["lmax"]], self.pp.corridor_half_w)[0]]
        # E2': true states of the actors that a short-horizon forward model needs (ahead within 60 m, |lateral| < 8 m)
        act_ = sorted([a for a in ann if -3.0 < a["far"] and a["near"] < 60.0 and a["lmax"] > -8.0 and a["lmin"] < 8.0],
                      key=lambda a: a["near"])[:8]
        rec = {"act": [[a["id"], round(a["near"], 2), round(a["far"], 2), round(a["lmin"], 2), round(a["lmax"], 2),
                        round(a["closing"], 2), round(a["lat_rate"], 2)] for a in act_],
               "t": self.tick, "v": round(v, 3), "xy": [round(tf.location.x, 2), round(tf.location.y, 2)],
               "yaw": round(tf.rotation.yaw, 2), "a_gt": a_gt, "min_ttc": f3(max(min(corr), 0.0)) if corr else None,
               "pdm_brake": round(float(ctl.brake), 3), "pdm_throttle": round(float(ctl.throttle), 3)}
        hold_r = self.pp.standoff + 3.0                                              # A1: r0 + 3 m
        if self.branch == "privileged":
            a_frame, lat_ms, res = a_gt, 0.0, None
            near_obj = bool(len(gt_t[0]) and np.any(P.in_corridor(gt_t[1], gt_t[2], self.pp.corridor_half_w)
                                                    & (gt_t[0] <= hold_r) & (np.array([a["far"] for a in inc]) > 0)))
        if self.names:
            img = np.ascontiguousarray(input_data["rgb_front_perc"][1][:, :, :3])
            res = self.pipes.frame(img, v, self.names)
            if self.branch != "privileged":
                r = res[self.branch]
                a_frame = int(r["action"])
                z_, lo_, hi_ = (np.asarray(x_, float) for x_ in r["mono"][:3])
                near_obj = bool(len(z_) and np.any(P.in_corridor(lo_, hi_, self.pp.corridor_half_w) & (z_ <= hold_r)))
                if a_frame > P.KEEP:                    # the decisive object of the branch (phantom attribution)
                    _, aob = P.required_decel(*r["mono"], v, self.pp)
                    j = int(np.argmax(aob))
                    sc = r["scene"]
                    rec["dec_obj"] = [round(float(x_), 1) for x_ in np.asarray(sc["xyxy"], float)[j]] + [
                        round(float(sc["conf"][j]), 3), str(sc["coarse"][j]), round(float(z_[j]), 2), round(float(lo_[j]), 2),
                        round(float(hi_[j]), 2), round(float(np.asarray(r["mono"][3], float)[j]), 2)]
                    if a_gt == P.KEEP and self.n_ph < int(os.environ.get("AEB_SAVE_PH", "0")) and os.environ.get("SAVE_PATH"):
                        import cv2
                        im = np.ascontiguousarray(img.copy())
                        x1, y1, x2, y2 = (int(q) for q in np.asarray(sc["xyxy"], float)[j])
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite(str(Path(os.environ["SAVE_PATH"]) / f"ph_{self.branch}_{self.tick:05d}.jpg"), im)
                        self.n_ph += 1
                lat_ms = self._latency(r)
                rec.update({"px": round(float(r["px"]), 5), "rounds": int(r.get("rounds", 0)), "ncrops": int(r.get("ncrops", 0)),
                            "lat_ms": round(lat_ms, 2)})
                if self.branch == "v6S" and "u960" in res:
                    rec["a_shadow_u960"] = int(res["u960"]["action"])
                if self.branch != "cheap" and "cheap" in res:
                    rec["a_shadow_cheap"] = int(res["cheap"]["action"])
            if self.logall:
                rec["all"] = {nm: [int(r["action"]), round(float(r["px"]), 5), int(r.get("rounds", 0))] for nm, r in res.items()}
                rec["err"] = self._errors(res, inc)
        # latency: the action of frame t is applied from tick t + ceil(lat / dt) on; the newest applicable frame wins
        self.q.append((self.tick + (math.ceil(lat_ms / 1000.0 / DT - 1e-9) if self.setting == "L1" else 0), self.tick, a_frame,
                       near_obj))
        ready = [e for e in self.q if e[0] <= self.tick]
        if ready:
            newest = max(ready, key=lambda e: e[1])
            self.applied, self.near_applied = newest[2], newest[3]
            self.q = [e for e in self.q if e[1] > newest[1]]
        rec["a_frame"], rec["a_applied"] = a_frame, self.applied
        if self.applied > P.KEEP:
            ctl.throttle = 0.0
            ctl.brake = float(max(ctl.brake, self.bmap[self.applied]))
        # A1 standstill hold: full brake while an object of the branch's scene stays in the corridor within r0 + 3 m
        if not self.hold:
            if v < 0.5 and self.near_applied:
                self.hold, self.hold_clear = True, 0
        else:
            self.hold_clear = 0 if self.near_applied else self.hold_clear + 1
            if self.hold_clear >= 10:
                self.hold = False
        if self.hold:
            ctl.throttle, ctl.brake = 0.0, 1.0
        rec["near5"], rec["hold"] = int(near_obj), int(self.hold)
        rec["brake"] = round(float(ctl.brake), 3)
        self._contacts(ann, v)
        self.recs.append(rec)

    def _contacts(self, ann, v, m=0.1):
        """first contact of the ego box with each actor (axis-aligned in the ego frame, margin m), typed by where it is"""
        ex = self._vehicle.bounding_box.extent
        for a in ann:
            if a["id"] in self.contacts:
                continue
            if a["far"] < -ex.x - m or a["near"] > ex.x + m or a["lmax"] < -ex.y - m or a["lmin"] > ex.y + m:
                continue
            inc_ = bool(P.in_corridor([a["lmin"]], [a["lmax"]], self.pp.corridor_half_w)[0])
            if a["near"] >= ex.x - 0.5:
                typ = "front_corridor" if inc_ else "front_outside"
            elif a["far"] <= -ex.x + 0.5:
                typ = "rear"
            else:
                typ = "side"
            self.contacts[a["id"]] = {"t": self.tick, "id": a["id"], "cls": a["cls"], "v": round(v, 3), "type": typ,
                                      "near": round(a["near"], 2), "far": round(a["far"], 2), "lmin": round(a["lmin"], 2),
                                      "lmax": round(a["lmax"], 2), "closing": round(a["closing"], 2), "hold": int(self.hold)}

    def _errors(self, res, inc):
        """per branch, every in-camera actor matched by a detection of the final scene: estimated vs annotated range / TTC"""
        out = {}
        boxes = [a for a in inc if a["box"] is not None]
        if not boxes:
            return out
        gb = np.array([a["box"] for a in boxes], float)
        for nm, r in res.items():
            sc = r["scene"]
            dx, dc = np.asarray(sc["xyxy"], float).reshape(-1, 4), np.asarray(sc["coarse"]).astype(str)
            if not len(dx):
                continue
            M = iou(gb, dx)
            for i, a in enumerate(boxes):
                M[i, [j for j in range(len(dx)) if dc[j] not in COMPAT[a["cls"]]]] = 0
            ri, ci = linear_sum_assignment(-M)
            alt = r.get("ttc_alt")
            rows = []
            for i, j in zip(ri, ci):
                if M[i, j] < 0.3:
                    continue
                a = boxes[i]
                ff = bool(sc["from_full"][j])
                t_c1 = t_g = float(r["mono_noref"][3][j])
                if ff and alt is not None:
                    k = np.flatnonzero(np.all(np.abs(alt[0] - dx[j]) < 1e-6, axis=1))
                    if len(k):
                        t_c1, t_g = float(alt[1][k[0]]), float(alt[2][k[0]])
                rows.append([a["id"], round(a["near"], 3), f3(a["ttc"]), round(float(r["mono_noref"][0][j]), 3),
                             round(float(r["mono"][0][j]), 3), f3(t_c1), f3(t_g), ff, bool(j in r.get("sup_ref", []))])
            out[nm] = rows
        return out

    def destroy(self, results=None):
        if os.environ.get("SAVE_PATH"):
            tag = f"{self.branch}{'_logall' if self.logall else ''}_{self.setting}"
            p = Path(os.environ["SAVE_PATH"]) / f"aeb_{tag}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(json.dumps({"branch": self.branch, "setting": self.setting, "seed": self.seed, "logall": self.logall,
                                     "c1": os.environ.get("AEB_C1", "1"), "ref": os.environ.get("AEB_REF", "1"),
                                     "front": os.environ.get("AEB_FRONT", "0"),
                                     "routes": Path(os.environ.get("ROUTES", "r")).stem, "start": self._start,
                                     "contacts": list(self.contacts.values()), "ticks": self.recs}) + "\n")
        try:
            super().destroy(results)
        except TypeError:
            super().destroy()
