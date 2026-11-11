#!/usr/bin/env python3
"""K29 diagnostics: input-voltage (VDD_IN, read as the INA3221 bus voltage of the GPU channel) and rail currents, read at
~50 Hz from sysfs; one line per second (min / mean voltage, max per-rail current, total power), appended and fsynced so the
last second before a hard reset survives. usage: k29_vin_logger.py <out.log>"""
import os
import sys
import time

A, B = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4", "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon5"
CH = [("GPU", A, 1), ("CPU", A, 2), ("SOC", A, 3), ("CV", B, 1), ("VDDRQ", B, 2)]


def rd(p):
    with open(p) as f:
        return int(f.read())


fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.write(fd, b"time vmin_mV vmean_mV vmax_mV P_W Pmax_W " + " ".join(f"I{n}max_mA" for n, _, _ in CH).encode() + b"\n")
while True:
    t_end = time.time() + 1.0
    vs, ps, imax = [], [], {n: 0 for n, _, _ in CH}
    while time.time() < t_end:
        v = rd(f"{A}/in1_input")
        tot = 0.0
        for n, h, i in CH:
            c = rd(f"{h}/curr{i}_input")
            imax[n] = max(imax[n], c)
            tot += c * v / 1e6
        vs.append(v); ps.append(tot)
        time.sleep(0.02)
    line = f"{time.strftime('%T')} {min(vs)} {sum(vs) / len(vs):.0f} {max(vs)} {sum(ps) / len(ps):.1f} {max(ps):.1f} " + \
        " ".join(str(imax[n]) for n, _, _ in CH) + "\n"
    os.write(fd, line.encode()); os.fsync(fd)
