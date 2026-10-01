#!/usr/bin/env python3
"""Kinematic test of the gimbal through an overpass (no ROS, no Gazebo).

A parked car at the origin facing +y; the drone flies the overpass shot
(20 m behind, 4 m up -> 15 m peak over the car -> 20 m in front, 13.3 s,
eased) with a sideways sway (the A1 runs swayed a few metres). Body yaw turns
to face the car at 90 deg/s except within 5 m (held, as the node does).
The gimbal runs GimbalController exactly as the node does: the image-space
PD on each detection (15 Hz, car inside the 46 x 26 deg view and >= 9.4 m),
world pointing at the car after 0.5 s without one; the gimbal slews at
180 deg/s. Compares nadir_band 0 (before 2026-10-01) with the new default.

    python3 scripts/test_gimbal_nadir.py
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'circumnavigation_controller', 'circumnavigation_controller'))
from gimbal_controller import GimbalController   # noqa: E402

HALF_FOV = (math.radians(23.0), math.radians(13.0))
SLEW = math.radians(180.0)
CAR = np.array([0.0, 0.0, 0.6])


def camera_axes(drone_yaw, g_yaw, g_pitch):
    """World unit vectors (right, down, forward) of the optical frame. Gimbal
    yaw is positive to the right of the nose; pitch negative down, past -90
    looking backwards."""
    heading = drone_yaw - g_yaw
    ch, sh = math.cos(heading), math.sin(heading)
    fwd = np.array([math.cos(g_pitch) * ch, math.cos(g_pitch) * sh, math.sin(g_pitch)])
    right = np.array([sh, -ch, 0.0])
    down = np.cross(fwd, right)
    return right, down, fwd


def drone_path(t, sway, T=13.33, r=20.0, h=4.0, peak=15.0):
    u = min(max(t / T, 0.0), 1.0)
    s = u * u * (3 - 2 * u)
    y = -r + 2 * r * s
    z = h + 4 * s * (1 - s) * (peak - h)
    x = sway * math.sin(2 * math.pi * t / 4.0)
    return np.array([x, y, z])


def run(nadir_band_deg, sway, seed=0, T=13.33):
    rs = np.random.default_rng(seed)
    g = GimbalController(nadir_band_deg=nadir_band_deg)
    g_yaw, g_pitch = 0.0, math.radians(-10.0)
    cmd_yaw, cmd_pitch = g_yaw, g_pitch
    drone_yaw = math.pi / 2               # facing +y (the car) at the start
    dt, t = 0.01, 0.0
    next_det, next_ctl = 0.0, 0.0
    last_det_t, det = -1e9, None
    used = None
    seen, n, max_yaw = 0, 0, 0.0
    while t <= T + 2.0:
        p = drone_path(t, sway, T)
        d = CAR - p
        if math.hypot(d[0], d[1]) > 5.0:  # body faces the car, held when close
            want = math.atan2(d[1], d[0])
            err = math.atan2(math.sin(want - drone_yaw), math.cos(want - drone_yaw))
            drone_yaw += max(-math.radians(90) * dt, min(math.radians(90) * dt, err))
        right, down, fwd = camera_axes(drone_yaw, g_yaw, g_pitch)
        cam = np.array([d @ right, d @ down, d @ fwd])
        in_view = (cam[2] > 0 and abs(math.atan2(cam[0], cam[2])) < HALF_FOV[0]
                   and abs(math.atan2(cam[1], cam[2])) < HALF_FOV[1])
        n += 1
        seen += in_view
        if t >= next_det:
            if in_view and np.linalg.norm(d) >= 9.4:
                det = cam + 0.03 * np.linalg.norm(d) * rs.standard_normal(3)
                last_det_t = t
            next_det += 1.0 / 15.0
        if t >= next_ctl:                 # node gimbal timer, 10 Hz
            if det is not None and t - last_det_t <= 0.5:
                if used != last_det_t:
                    c = g.update_image_pd(det[0], det[1], det[2], g_pitch, g_yaw, t)
                    cmd_yaw, cmd_pitch = c.yaw, c.pitch
                    used = last_det_t
            else:
                c = g.update(p[0], p[1], p[2], drone_yaw, CAR[0], CAR[1], CAR[2],
                             current_yaw=g_yaw if nadir_band_deg > 0 else None)
                cmd_yaw, cmd_pitch = c.yaw, c.pitch
            next_ctl += 0.1
        step = SLEW * dt
        g_yaw += max(-step, min(step, cmd_yaw - g_yaw))
        g_pitch += max(-step, min(step, cmd_pitch - g_pitch))
        max_yaw = max(max_yaw, abs(math.degrees(g_yaw)))
        t += dt
    return 100.0 * seen / n, max_yaw


def main():
    print(f"{'sway':>5} | {'before (band 0)':>26} | {'after (band 25 deg)':>26}")
    for sway in (0.5, 1.5, 3.0):
        b = [run(0.0, sway, s) for s in range(5)]
        a = [run(25.0, sway, s) for s in range(5)]
        fmt = lambda r: (f"in view {np.mean([x[0] for x in r]):5.1f} %, "     # noqa: E731
                         f"max yaw {np.max([x[1] for x in r]):5.0f} deg")
        print(f"{sway:4.1f}m | {fmt(b)} | {fmt(a)}")


if __name__ == '__main__':
    main()
