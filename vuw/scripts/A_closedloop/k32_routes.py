#!/usr/bin/env python
"""K32 route lists (pre-registered in handoff/K32/STATUS.md, written before any run):
  results/k32/test_routes.xml  bench2drive220.xml restricted to the test scenario types (one scenario per route)
  results/k32/dev_routes.xml   short single-scenario routes cut from routes_training.xml (Town12): the training route is
                               densified on the road network (CARLA GlobalRoutePlanner on the offline Town12 OpenDRIVE map,
                               2 m), and cut from 80 m before to 80 m after the waypoint nearest the scenario trigger; up to
                               4 per type in file order; a trigger within 100 m of an evaluation trigger of the same type is
                               dropped
  results/k32/routes_meta.csv  every route: split, id, type, trigger, parent route, length
usage (carla env): k32_routes.py
"""
from __future__ import annotations

import copy
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
D = B / "third_party" / "Bench2Drive" / "leaderboard" / "data"
OUT = B / "results" / "k32"
EB = ["BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute", "OppositeVehicleTakingPriority",
      "OppositeVehicleRunningRedLight", "ParkingCutIn", "PedestrianCrossing", "ParkingCrossingPedestrian", "StaticCutIn",
      "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "ControlLoss"]
COCO = ["CrossingBicycleFlow", "EnterActorFlow", "EnterActorFlowV2", "HighwayCutIn", "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow", "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurn",
        "SignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionRightTurn", "T_Junction", "VanillaNonSignalizedTurn",
        "VanillaNonSignalizedTurnEncounterStopsign", "VanillaSignalizedTurnEncounterGreenLight",
        "VanillaSignalizedTurnEncounterRedLight", "ParkingExit"]
TYPES = EB + COCO
PER_TYPE, BEFORE, AFTER, EXCL = 4, 80.0, 80.0, 100.0


def trig(sc):
    t = sc.find("trigger_point")
    return np.array([float(t.get("x")), float(t.get("y"))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    # ---- test
    ev = ET.parse(D / "bench2drive220.xml").getroot()
    keep = ET.Element("routes")
    ev_trig = []
    for r in ev.findall("route"):
        scs = r.find("scenarios").findall("scenario")
        assert len(scs) == 1, r.get("id")
        ty = scs[0].get("type")
        ev_trig.append((ty, trig(scs[0])))
        if ty in TYPES:
            keep.append(copy.deepcopy(r))
            meta.append({"split": "test", "route_id": r.get("id"), "town": r.get("town"), "type": ty,
                         "group": "Emergency_Brake" if ty in EB else "COCO, longitudinal", "parent": "", "length_m": np.nan})
    ET.ElementTree(keep).write(OUT / "test_routes.xml", encoding="utf-8", xml_declaration=True)
    # ---- dev
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    xodr = (B / "tools" / "carla_0.9.15" / "CarlaUE4" / "Content" / "Carla" / "Maps" / "Town12" / "OpenDrive" / "Town12.xodr").read_text()
    wmap = carla.Map("Town12", xodr)
    grp = GlobalRoutePlanner(wmap, 2.0)
    tr = ET.parse(D / "routes_training.xml").getroot()
    count = {t: 0 for t in TYPES}
    dense_cache = {}
    dev = ET.Element("routes")
    k = 0
    for r in tr.findall("route"):
        for sc in r.find("scenarios").findall("scenario"):
            ty = sc.get("type")
            if ty not in TYPES or count[ty] >= PER_TYPE:
                continue
            p = trig(sc)
            if any(t == ty and np.linalg.norm(p - q) < EXCL for t, q in ev_trig):
                continue
            rid = r.get("id")
            if rid not in dense_cache:
                kp = [carla.Location(float(q.get("x")), float(q.get("y")), float(q.get("z"))) for q in r.find("waypoints").findall("position")]
                pts = []
                for a, b in zip(kp[:-1], kp[1:]):
                    seg = grp.trace_route(a, b)
                    pts += [(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w, _ in seg]
                P = np.array(pts)
                keepi = np.r_[True, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) > 0.5]
                dense_cache[rid] = P[keepi]
            P = dense_cache[rid]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
            i0 = int(np.argmin(np.linalg.norm(P[:, :2] - p, axis=1)))
            if np.linalg.norm(P[i0, :2] - p) > 10.0:
                continue                                           # trigger not on the densified route
            sel = (s >= s[i0] - BEFORE) & (s <= s[i0] + AFTER)
            Q = P[sel]
            if s[sel][-1] - s[sel][0] < 100.0:
                continue
            nr = ET.SubElement(dev, "route", {"id": f"{9000 + k}", "town": "Town12"})
            if r.find("weathers") is not None:
                nr.append(copy.deepcopy(r.find("weathers")))
            wp = ET.SubElement(nr, "waypoints")
            for x, y, z in Q:
                ET.SubElement(wp, "position", {"x": f"{x:.1f}", "y": f"{y:.1f}", "z": f"{z:.1f}"})
            scs = ET.SubElement(nr, "scenarios")
            scs.append(copy.deepcopy(sc))
            meta.append({"split": "dev", "route_id": f"{9000 + k}", "town": "Town12", "type": ty,
                         "group": "Emergency_Brake" if ty in EB else "COCO, longitudinal", "parent": rid,
                         "trigger_x": p[0], "trigger_y": p[1], "length_m": float(s[sel][-1] - s[sel][0])})
            count[ty] += 1
            k += 1
            print(ty, rid, k, flush=True)
    ET.ElementTree(dev).write(OUT / "dev_routes.xml", encoding="utf-8", xml_declaration=True)
    pd.DataFrame(meta).to_csv(OUT / "routes_meta.csv", index=False)
    print("test", sum(m["split"] == "test" for m in meta), "dev", k, {t: c for t, c in count.items() if c})
    print("K32 ROUTES DONE")


if __name__ == "__main__":
    sys.exit(main())
