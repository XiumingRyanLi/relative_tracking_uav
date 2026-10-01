#!/usr/bin/env python3
"""Offline benchmark of the target estimator (TargetEstimator: gating, flip
correction, CV KF or CTRA EKF) against a clean ground truth.

Truth: the car driven along a racing line (config/race_lines/<world>.csv) at
race_driver's speed profile. The logged Gazebo car truth isn't usable for
this: it only updates every ~0.2 s and is stamped with the CSV row time, so
it carries up to ~1.8 m of timing error at race speed.

Detections, modelled on DOPE in the race runs of 2026-09-28:
  - 15 Hz with 10 % dropped frames, 70 ms capture -> use latency
  - position error = white + a bias drifting with a 1.5 s time constant, each
    3.5 % of the drone->car range (range taken from a real flight, looped)
  - heading error = 2 deg white + 3 deg bias (0.6 s); measured sd ~3.6 deg
  - nose-to-tail flips on 3 % of frames beyond 40 m
  - outliers: 2 % of frames 5-15 m off in a random direction

Scores the position error at "now" (capture + latency) and 0.5 / 1 / 2 s
ahead (straight-line extrapolation of the estimate, as the controller does),
velocity and heading error, and the gate: good frames rejected, outliers let
through.

    python3 scripts/benchmark_target_filter.py                      # held-out tracks
    python3 scripts/benchmark_target_filter.py --worlds iris_monza
    python3 scripts/benchmark_target_filter.py --gates 9.21 13.8 20   # sweep the CTRA gate
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "src", "circumnavigation_controller")
sys.path.insert(0, os.path.join(PKG, "circumnavigation_controller"))
from race_line import RaceLine                      # noqa: E402
from target_estimator import TargetEstimator        # noqa: E402

RANGE_LOG = os.path.join(REPO, "logs", "relative_pid_20260928_150220.csv")


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def race_truth(world, csv=None, **line_kwargs):
    """spline(t) -> car xy (spline(t, 1) velocity, (t, 2) acceleration).
    csv: a racing-line file instead of the world's; line_kwargs go to
    RaceLine (e.g. max_speed=17)."""
    pts = np.loadtxt(csv or os.path.join(PKG, "config", "race_lines", f"{world}.csv"), delimiter=",", comments="#")
    line = RaceLine(pts, **line_kwargs)
    xy = np.vstack([line.pts[-20:], line.pts, line.pts[:20]])
    xy = savgol_filter(xy, 15, 3, axis=0)[20:-20]          # smooth the ~1 m points
    closed = np.vstack([xy, xy[:1]])
    s_pts = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    path = CubicSpline(s_pts, closed, bc_type="periodic")
    length = s_pts[-1]
    dt, t, s, v = 0.01, [0.0], [0.0], 3.0
    while s[-1] < length:
        target = float(np.interp(s[-1] % length, line.s, line.speed, period=line.length))
        v += float(np.clip(target - v, -4.0 * dt, 2.0 * dt))  # race_driver brake / accel
        s.append(s[-1] + v * dt)
        t.append(t[-1] + dt)
    return CubicSpline(np.array(t), path(np.array(s) % length)), t[-1]


def range_profile():
    t, d = [], []
    if os.path.exists(RANGE_LOG):
        for row in csv.DictReader(open(RANGE_LOG)):
            try:
                tt, dd = float(row["time"]), float(row["gt_dist_m"])
            except (ValueError, KeyError):
                continue
            if t and tt <= t[-1]:
                continue
            t.append(tt)
            d.append(min(dd, 79.0))
    if len(t) < 10:                                         # no log: a plausible profile
        t = list(np.arange(0.0, 200.0, 0.5))
        d = [25.0 + 12.0 * math.sin(tt / 9.0) for tt in t]
    return np.array(t) - t[0], np.array(d)


def detections(spline, t_end, rng_tab, seed, rate=15.0, drop=0.1, pos_frac=0.035, bias_frac=0.035,
               bias_tau=1.5, h_white=math.radians(2.0), h_bias=math.radians(3.0), h_tau=0.6,
               flip_prob=0.03, flip_range=40.0, outlier_prob=0.02, outlier_m=(5.0, 15.0)):
    rs = np.random.default_rng(seed)
    dt = 1.0 / rate
    bias, hb = np.zeros(2), 0.0
    period = rng_tab[0][-1]
    for t in np.arange(2.0, t_end - 3.0, dt):
        rng = float(np.interp(t % period, *rng_tab))
        a = math.exp(-dt / bias_tau)
        bias = a * bias + math.sqrt(1 - a * a) * bias_frac * rng * rs.standard_normal(2)
        ah = math.exp(-dt / h_tau)
        hb = ah * hb + math.sqrt(1 - ah * ah) * h_bias * rs.standard_normal()
        if rs.random() < drop:
            continue
        xy = spline(t) + bias + pos_frac * rng * rs.standard_normal(2)
        outlier = rs.random() < outlier_prob
        if outlier:
            ang = rs.uniform(-math.pi, math.pi)
            xy = xy + rs.uniform(*outlier_m) * np.array([math.cos(ang), math.sin(ang)])
        v = spline(t, 1)
        psi = math.atan2(v[1], v[0]) + hb + h_white * rs.standard_normal()
        if rng > flip_range and rs.random() < flip_prob:
            psi += math.pi
        yield t, xy, wrap(psi), rng, outlier


def make_estimator(filter_name, gate_nis=25.0):
    return TargetEstimator(
        heading_axis=0, min_distance=0.2, max_distance=80.0, max_position_jump=3.0, jump_per_m=0.10,
        max_consecutive_rejects=5, noise_per_m=0.02, use_filtered_position=True, kf_coast_timeout_sec=1.0,
        kf_accel_noise_std=1.5, kf_measurement_noise_std=0.15, kf_initial_pos_std=1.0, kf_initial_vel_std=3.0,
        kf_max_target_speed=15.0, kf_enable_adaptive_q=True, kf_adaptive_q_max_scale=12.0,
        kf_adaptive_q_decay=0.5, heading_alpha=0.25, yaw_rate_tau_sec=1.0, yaw_rate_min_speed=1.0,
        max_yaw_rate=math.radians(30.0), max_heading_jump=math.radians(60.0), max_heading_distance=60.0,
        enable_heading_filter=True, heading_course_min_speed=3.0, heading_flip_threshold=math.radians(110.0),
        heading_course_memory_sec=3.0, enable_heading_ukf=False, ukf_coast_timeout_sec=1.0, ukf_std_a=1.5,
        ukf_std_yawdd=0.5, ukf_std_pos=0.15, ukf_std_yaw=0.2, ukf_max_yaw_rate=3.0,
        use_ctra=(filter_name == "ctra"), ctra_gate_nis=gate_nis,
    )


def pose_matrix(x, y, psi):
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = math.cos(psi), -math.sin(psi), math.sin(psi), math.cos(psi)
    T[0, 3], T[1, 3], T[2, 3] = x, y, 0.6
    return T


def run(filter_name, worlds, seeds, gate_nis=25.0, latency=0.07, horizons=(0.0, 0.5, 1.0, 2.0)):
    pos = {h: [] for h in horizons}
    vel, hdg = [], []
    gate = {"good": 0, "good_rejected": 0, "outliers": 0, "outliers_accepted": 0}
    rng_tab = range_profile()
    for world in worlds:
        spline, t_end = race_truth(world)
        for seed in seeds:
            est = make_estimator(filter_name, gate_nis)
            for k, (t, xy, psi, rng, outlier) in enumerate(detections(spline, t_end, rng_tab, seed)):
                accepted = est.update(pose_matrix(xy[0], xy[1], psi), rng, t) is None
                if k >= 30:
                    key = "outliers" if outlier else "good"
                    gate[key] += 1
                    gate[key + ("_accepted" if outlier else "_rejected")] += accepted if outlier else not accepted
                if not accepted or k < 30:
                    continue
                now = t + latency
                p, v = np.array([est.x, est.y]), np.array([est.vx, est.vy])
                for h in horizons:
                    pos[h].append(float(np.hypot(*(p + v * (latency + h) - spline(now + h)))))
                tv = spline(now, 1)
                vel.append(float(np.hypot(*(v - tv))))
                hdg.append(abs(math.degrees(wrap(est.heading - math.atan2(tv[1], tv[0])))))
    return pos, np.array(vel), np.array(hdg), gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", nargs="+", default=["iris_silverstone", "iris_oschersleben"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gates", type=float, nargs="+", default=[25.0],
                    help="CTRA chi-square gate(s) on the position NIS (2 DOF: 9.21 = 99 %%, 13.8 = 99.9 %%)")
    args = ap.parse_args()
    print(f"worlds: {', '.join(args.worlds)}; {args.seeds} noise seeds each; median / p90")
    print(f"{'filter':12} " + " ".join(f"{'pos +' + str(h) + 's':>13}" for h in (0.0, 0.5, 1.0, 2.0))
          + f" {'vel m/s':>11} {'heading deg':>12}  good rejected / outliers accepted")
    for name, g in [("cv", None)] + [("ctra", g) for g in args.gates]:
        pos, vel, hdg, gate = run(name, args.worlds, range(args.seeds), gate_nis=g or 25.0)
        cells = " ".join(f"{np.median(v):5.2f} / {np.percentile(v, 90):5.2f}" for v in pos.values())
        label = name if g is None else f"ctra {g:g}"
        print(f"{label:12} {cells} {np.median(vel):4.2f} / {np.percentile(vel, 90):4.2f}"
              f" {np.median(hdg):4.1f} / {np.percentile(hdg, 90):4.1f}"
              f"  {100 * gate['good_rejected'] / max(1, gate['good']):5.1f} % / "
              f"{100 * gate['outliers_accepted'] / max(1, gate['outliers']):5.1f} %")


if __name__ == "__main__":
    main()
