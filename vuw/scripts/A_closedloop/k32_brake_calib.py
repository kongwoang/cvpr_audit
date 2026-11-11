#!/usr/bin/env python
"""K32 brake calibration (dev, pre-registered): the Bench2Drive ego (vehicle.lincoln.mkz_2020, role hero) on a straight
Town12 road without other actors, synchronous mode at 20 Hz. For each initial speed in {5, 10, 15} m/s and brake pedal in
{0.1, 0.2, ..., 1.0}: set the speed, then throttle 0 / steer 0 / brake b for up to 3 s; deceleration = speed drop over
the first 1.0 s (or until the speed falls below 0.5 m/s) / elapsed time. Writes results/k32/brake_calib.csv and
results/k32/brake_map.json = {"2.5": b, "6.0": b}: the smallest pedal whose speed-averaged deceleration reaches the
commanded one (linear interpolation between pedal steps; 1.0 if never reached).
usage (carla env, CARLA server running on --port): k32_brake_calib.py --port P
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))


def straight_start(wmap, need=200.0):
    for sp in wmap.get_spawn_points():
        wp = wmap.get_waypoint(sp.location)
        if wp.is_junction:
            continue
        yaw0, ok, w = wp.transform.rotation.yaw, True, wp
        for _ in range(int(need / 5)):
            nx = w.next(5.0)
            if not nx or nx[0].is_junction or abs(((nx[0].transform.rotation.yaw - yaw0 + 180) % 360) - 180) > 2.0:
                ok = False
                break
            w = nx[0]
        if ok:
            return wp.transform
    raise RuntimeError("no straight road")


def main():
    import carla
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    client = carla.Client("localhost", args.port)
    client.set_timeout(600.0)
    world = client.load_world("Town12")
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    wmap = world.get_map()
    tf0 = straight_start(wmap)
    tf0.location.z += 0.5
    bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
    bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(bp, tf0)
    for _ in range(40):
        world.tick()
    fwd = tf0.get_forward_vector()
    rows = []
    for v0 in (5.0, 10.0, 15.0):
        for b in np.round(np.arange(0.1, 1.01, 0.1), 2):
            ego.set_transform(tf0)
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            for _ in range(10):
                world.tick()
            ego.set_target_velocity(carla.Vector3D(fwd.x * v0, fwd.y * v0, 0.0))
            world.tick()
            sp = []
            for k in range(60):
                ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=float(b)))
                world.tick()
                vv = ego.get_velocity()
                sp.append(math.hypot(vv.x, vv.y))
            sp = np.array(sp)
            n = int(np.argmax(sp < 0.5)) if (sp < 0.5).any() else len(sp)
            n = max(1, min(n, 20))
            a = (sp[0] - sp[n - 1]) / (n - 1) / 0.05 if n > 1 else float("nan")
            rows.append({"v0": v0, "brake": float(b), "v_first": float(sp[0]), "decel": float(a), "ticks": n})
            print(rows[-1], flush=True)
    ego.destroy()
    st.synchronous_mode = False
    world.apply_settings(st)
    d = pd.DataFrame(rows)
    out = B / "results" / "k32"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "brake_calib.csv", index=False)
    m = d.groupby("brake").decel.mean().sort_index()
    bm = {}
    for a_cmd in (2.5, 6.0):
        if (m >= a_cmd).any():
            hi = m.index[np.argmax(m.values >= a_cmd)]
            lo_idx = list(m.index).index(hi) - 1
            if lo_idx < 0:
                bm[str(a_cmd)] = float(hi)
            else:
                lo = m.index[lo_idx]
                bm[str(a_cmd)] = float(lo + (a_cmd - m[lo]) / (m[hi] - m[lo]) * (hi - lo))
        else:
            bm[str(a_cmd)] = 1.0
    json.dump(bm, open(out / "brake_map.json", "w"))
    print("brake map", bm, "\nK32 BRAKE CALIB DONE")


if __name__ == "__main__":
    main()
