#!/usr/bin/env python3
"""Offline closed-loop test of the tracking controllers (no ROS, no Gazebo).

Car: a racing line at race_driver's speed profile (see benchmark_target_filter.py).
Camera/gimbal: the car is detected only inside the camera view (+-23 x +-13 deg)
    around where the gimbal points -- the current estimate, or the coast
    prediction while the car is lost -- with the gimbal slewing at 40 deg/s;
    and only 9.4-80 m away (DOPE).
Detections: 15 Hz, 10 % dropped, 70 ms latency, noise scaled with the actual
    range (white + drifting bias), heading noise, flips beyond 40 m, 2 %
    outliers. -> TargetEstimator (CTRA EKF).
    DOPE's input scale (dope_detector.py): it only detects the car at ~90 px;
    while tracking it picks the scale from the last distance, but after 1 s
    without a detection it cycles its 9 scales one per frame, and a frame
    only detects if its scale is within one step of the one the range needs.
Occlusions (--occlusions): blackouts of 1.5 / 3 / 4.5 s every ~20 s (trees).
Drone plant (fitted on the race logs of 2026-09-28): transport delay (xy 0.8,
    z 0.7, yaw 0.25 s), first-order lag (0.45 / 0.3 / 0.3 s), 5 m/s^2 limit.
The node's logic, through the same modules: coast after 0.5 s without a
    detection (target_prediction.py), lost -> search after target_timeout.
    When lost the sim records it and re-seeds the car, so later occlusions
    are still tested.

Variants:
    pid        the PID path
    mpc_old    MPC as flown on 2026-09-29: EKF with acceleration (CTRA); coast in a
               straight line, heading frozen, 2 s full speed, 20 m cap; EKF
               restarted after gaps > 1 s; no range scenarios; search after 10 s
    mpc_coast  EKF without acceleration (CTRV); new coast (EKF prediction, turn
               fading, 3 s, 40 m), predicts through gaps <= 5 s, search after
               5 s; no range scenarios
    mpc_new    mpc_coast + the brake-while-turning / accelerate range scenarios
    mpc_scan   mpc_new + the gimbal yaw sweep around the prediction while coasting
    mpc_scan_fast  a wider (2.5 sigma) sweep with a 90 deg/s gimbal (what-if: the real
               gimbal's slew rate is unknown; the sim default is 40 deg/s)
    mpc_now    the defaults as flown 2026-09-29 12:44 (CTRV, scenarios, 2.5 sigma scan)
    mpc_maxr   + soft maximum range 45 m to the car (cut inside instead of losing it)
    mpc_fade   + the horizon car keeps turning (turn rate fading over 1.5 s)
    mpc_both   + both
    no_scan / sweep / stare   current defaults with no gimbal scan, the continuous sweep,
               and stop-and-stare (180 deg/s gimbal, as measured)
Shots (--shot): back (18 m behind, the race-run shot), right, left (18 m beside).

    python3 scripts/mpc_closed_loop_sim.py                          # no occlusions
    python3 scripts/mpc_closed_loop_sim.py --occlusions             # trees
    python3 scripts/mpc_closed_loop_sim.py --variants mpc_old mpc_new --occlusions --worlds iris_silverstone
"""
import argparse
import math
import os
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src", "circumnavigation_controller", "circumnavigation_controller"))
from benchmark_target_filter import race_truth, make_estimator, pose_matrix, wrap   # noqa: E402
from pid_controller import PIDRelativeController                                    # noqa: E402
from drone_mpc import DroneMPC                                                       # noqa: E402
from mpc_tracker import MpcTracker                                                   # noqa: E402
from target_prediction import CoastParams, CoastScan, predict_car, predict_horizon, range_scenarios  # noqa: E402

DT_SIM, DT_CTRL, DT_DET = 0.01, 0.05, 1.0 / 15.0
DELAYS = (0.8, 0.7, 0.25)
TAUS = (0.45, 0.3, 0.3)
ACCEL_MAX = 5.0
OFFSET = np.array([-18.0, 0.0, 4.0])     # shot: 18 m behind, 4 m up
CAR_Z = 0.6
HALF_FOV = (math.radians(23.0), math.radians(13.0))
GIMBAL_RATE = math.radians(40.0)
LATENCY = 0.07
COAST_AFTER = 0.5                        # feedforward_timeout_sec

VARIANTS = {
    "pid":       dict(controller="pid", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=-1.0, accel=False),
    "mpc_old":   dict(controller="mpc", coast=CoastParams(2.0, 1.0, 20.0, 0.0), reacquire=1.0, timeout=10.0, window=-1.0, accel=True),
    "mpc_coast": dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=-1.0, accel=False),
    "mpc_new":   dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0, accel=False),
    "mpc_scan":  dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0, accel=False,
                      scan=dict(sigma_k=1.5, rate_deg=40.0, max_deg=40.0)),
    "mpc_scan_fast": dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                          accel=False, scan=dict(sigma_k=2.5, rate_deg=90.0, max_deg=50.0), gimbal_rate_deg=90.0),
    # current defaults (as flown on 2026-09-29 12:44) and the right-side-loss fixes
    "mpc_now":  dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(), gimbal_rate_deg=90.0, max_range=0.0, track_fade=0.0),
    "mpc_maxr": dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(), gimbal_rate_deg=90.0, max_range=45.0, track_fade=0.0),
    "mpc_fade": dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(), gimbal_rate_deg=90.0, max_range=0.0, track_fade=1.5),
    "mpc_both": dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(), gimbal_rate_deg=90.0, max_range=45.0, track_fade=1.5),
    # scan comparison (current defaults otherwise; gimbal 180 deg/s as measured on 2026-09-29)
    "no_scan":  dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, gimbal_rate_deg=180.0, max_range=45.0, track_fade=1.5, w=(90.0, 10.0)),
    "sweep":    dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(mode="sweep"), gimbal_rate_deg=180.0, max_range=45.0, track_fade=1.5,
                     w=(90.0, 10.0)),
    "stare":    dict(controller="mpc", coast=CoastParams(3.0, 1.0, 40.0, 1.5), reacquire=5.0, timeout=5.0, window=1.0,
                     accel=False, scan=dict(mode="stare"), gimbal_rate_deg=180.0, max_range=45.0, track_fade=1.5,
                     w=(90.0, 10.0)),
}


class Plant:
    """Drone: delayed, lagged response to [vx vy vz yaw_rate] commands."""
    def __init__(self, p, psi):
        self.p = np.array(p, dtype=float); self.v = np.zeros(3); self.psi = psi; self.r = 0.0
        self.hist = deque([(-1e9, np.zeros(4))])

    def command(self, t, u):
        self.hist.append((t, np.asarray(u, dtype=float)))

    def _u(self, t, d):
        u = self.hist[0][1]
        for ts, uu in self.hist:
            if ts <= t - d:
                u = uu
            else:
                break
        return u

    def step(self, t, dt):
        uxy, uz, ur = self._u(t, DELAYS[0]), self._u(t, DELAYS[1]), self._u(t, DELAYS[2])
        a = (uxy[:2] - self.v[:2]) / TAUS[0]
        n = np.hypot(*a)
        if n > ACCEL_MAX:
            a *= ACCEL_MAX / n
        self.v[:2] += a * dt
        self.v[2] += (uz[2] - self.v[2]) / TAUS[1] * dt
        self.p += self.v * dt
        self.r += (ur[3] - self.r) / TAUS[2] * dt
        self.psi = wrap(self.psi + self.r * dt)
        while len(self.hist) > 2 and self.hist[1][0] < t - 2.0:
            self.hist.popleft()


class Gimbal:
    """Boresight (world azimuth, elevation), slewing toward an aim point."""
    def __init__(self):
        self.az, self.el = 0.0, -0.2
        self.rate = GIMBAL_RATE

    def aim(self, drone_p, target_xy, dt, az_offset=0.0):
        d = np.r_[target_xy - drone_p[:2], CAR_Z - drone_p[2]]
        az, el = math.atan2(d[1], d[0]) + az_offset, math.atan2(d[2], np.hypot(d[0], d[1]))
        step = self.rate * dt
        self.az = wrap(self.az + float(np.clip(wrap(az - self.az), -step, step)))
        self.el += float(np.clip(el - self.el, -step, step))

    def sees(self, drone_p, car_xy):
        d = np.r_[car_xy - drone_p[:2], CAR_Z - drone_p[2]]
        az, el = math.atan2(d[1], d[0]), math.atan2(d[2], np.hypot(d[0], d[1]))
        return abs(wrap(az - self.az)) <= HALF_FOV[0] and abs(el - self.el) <= HALF_FOV[1]


SCALE_PYRAMID = [0.16, 0.2, 0.24, 0.28, 0.34, 0.4, 0.48, 0.556, 0.7]
FX_FULL = 640.0 / math.tan(0.4)          # 1280 px, 0.8 rad HFOV


def dope_scale_ok(rng, frame_idx, since_last_det):
    """DOPE's adaptive input scale: fine while tracking; lost (> 1 s), only
    the frames whose scale suits the range can detect."""
    if since_last_det <= 1.0:
        return True
    wanted = 90.0 * rng / (FX_FULL * 4.4)
    i_want = min(range(len(SCALE_PYRAMID)), key=lambda i: abs(SCALE_PYRAMID[i] - wanted))
    return abs(frame_idx % len(SCALE_PYRAMID) - i_want) <= 1


def shot_point(car_p, car_psi):
    c, s = math.cos(car_psi), math.sin(car_psi)
    return np.array([car_p[0] + c * OFFSET[0] - s * OFFSET[1], car_p[1] + s * OFFSET[0] + c * OFFSET[1], OFFSET[2]])


def blackouts(t_start, t_end, seed):
    rs = np.random.default_rng(seed + 1000)
    out, t, i = [], t_start + 10.0, 0
    while t < t_end - 8.0:
        dur = (1.5, 3.0, 4.5)[i % 3]
        s = t + rs.uniform(-5.0, 5.0)
        out.append((s, s + dur))
        t += 20.0; i += 1
    return out


class ScaledSpline:
    """The car's path driven `k` times faster (k > 1: faster than race_driver's
    15 m/s profile, e.g. the 16 m/s seen in Gazebo on 2026-09-29)."""
    def __init__(self, spline, k):
        self.spline, self.k = spline, k

    def __call__(self, t, nu=0):
        return self.spline(np.asarray(t) * self.k, nu) * self.k ** nu


def run(variant, world, seconds, seed, mpc, occlusions, speed_scale=1.0, trace=None):
    cfg = VARIANTS[variant]
    spline, t_end = race_truth(world)
    if speed_scale != 1.0:
        spline, t_end = ScaledSpline(spline, speed_scale), t_end / speed_scale
    t_end = min(t_end - 3.0, seconds + 3.0)
    rs = np.random.default_rng(seed)
    car_psi_at = lambda t: math.atan2(*spline(t, 1)[::-1])
    t = 2.0
    plant = Plant(shot_point(spline(t), car_psi_at(t)), car_psi_at(t))
    gimbal = Gimbal()
    scan = CoastScan(**cfg["scan"]) if cfg.get("scan") is not None else None
    gimbal.rate = math.radians(cfg.get("gimbal_rate_deg", 40.0))
    def new_estimator():
        e = make_estimator("ctra")
        e.ctra_reacquire_sec = cfg["reacquire"]
        e.ctra.estimate_accel = cfg["accel"]
        return e

    est = new_estimator()
    pid = PIDRelativeController(nominal_dt=DT_CTRL, speed_margin_xy=8.0, max_speed_xy=15.0, max_speed_z=3.0,
                                max_accel_xy=5.0, max_accel_z=2.0, derivative_tau=0.3, kp_xy=0.6, ki_xy=0.02,
                                max_integral_xy=3.0)
    tracker = None
    if cfg["controller"] == "mpc":
        mpc.set_max_range(cfg.get("max_range", 0.0))
        wr, wt = cfg.get("w", (30.0, 30.0))
        mpc.set_weights(wr, wt, 5.0, 10.0, 10.0, 0.1, 1.0, 2.0, 2.0)
        tracker = MpcTracker(mpc, DELAYS, TAUS, scenario_window_sec=cfg["window"])
        tracker.reset()
    blk = blackouts(t, t_end, seed) if occlusions else []
    bias, hb = np.zeros(2), 0.0
    det = None
    dope_frame, dope_last = 0, -1e9       # DOPE's own frame counter / last detection time
    next_det = next_ctrl = t
    cmd = np.zeros(4)
    m = {"err": [], "rad": [], "ang": [], "yaw": [], "rng": [], "du": [], "ms": [],
         "occl": [], "lost": 0, "lost_outside_occlusion": 0}
    blk_state = {i: None for i in range(len(blk))}   # None -> pending; dict after the blackout ends

    def seed_car(now):
        """(Re)start: the car seen right where it is (after a loss)."""
        nonlocal det
        for k in range(3):
            tc = now - LATENCY - (2 - k) * DT_DET
            c = spline(tc); ps = car_psi_at(tc)
            est.update(pose_matrix(c[0], c[1], ps), float(np.hypot(*(c - plant.p[:2]))), tc)
        det = SimpleNamespace(stamp_sec=now - LATENCY, received_time=now)

    seed_car(t)
    while t < t_end:
        car_p, car_psi = spline(t), car_psi_at(t)
        occluded = any(a <= t < b for a, b in blk)
        # ---- detection ----
        if t >= next_det:
            cap = t - LATENCY
            c_cap, psi_cap = spline(cap), car_psi_at(cap)
            rng = float(np.linalg.norm(np.r_[c_cap - plant.p[:2], CAR_Z - plant.p[2]]))
            a = math.exp(-DT_DET / 1.5); ah = math.exp(-DT_DET / 0.6)
            bias = a * bias + math.sqrt(1 - a * a) * 0.035 * rng * rs.standard_normal(2)
            hb = ah * hb + math.sqrt(1 - ah * ah) * math.radians(3.0) * rs.standard_normal()
            visible = (9.4 <= rng <= 80.0 and not occluded and gimbal.sees(plant.p, c_cap)
                       and rs.random() >= 0.1 and dope_scale_ok(rng, dope_frame, t - dope_last))
            dope_frame += 1
            if visible:
                dope_last = t
                xy = c_cap + bias + 0.035 * rng * rs.standard_normal(2)
                if rs.random() < 0.02:
                    ang = rs.uniform(-math.pi, math.pi)
                    xy = xy + rs.uniform(5, 15) * np.array([math.cos(ang), math.sin(ang)])
                psi = psi_cap + hb + math.radians(2.0) * rs.standard_normal()
                if rng > 40.0 and rs.random() < 0.03:
                    psi += math.pi
                if est.update(pose_matrix(xy[0], xy[1], wrap(psi)), rng, cap) is None:
                    det = SimpleNamespace(stamp_sec=cap, received_time=t)
                    for i, (a0, b0) in enumerate(blk):
                        st = blk_state[i]
                        if st is not None and st.get("reacq") is None and not st.get("lost") and t >= b0:
                            st["reacq"] = t - b0
            next_det += DT_DET
        # ---- control (the node's logic) ----
        if t >= next_ctrl:
            age = t - det.received_time
            if age > cfg["timeout"]:                       # lost -> search (the node would search, then land)
                m["lost"] += 1
                if not occluded and not any(b <= t < b + cfg["timeout"] + 1 for a, b in blk):
                    m["lost_outside_occlusion"] += 1
                for i, (a0, b0) in enumerate(blk):
                    st = blk_state[i]
                    if st is not None and st.get("reacq") is None and not st.get("lost"):
                        st["lost"] = True
                plant = Plant(shot_point(car_p, car_psi), car_psi)     # re-seed and carry on
                est = new_estimator()
                if tracker: tracker.reset()
                seed_car(t)
                age = 0.0
            coasting = age > COAST_AFTER
            cp = cfg["coast"]
            xs, ys, psi_now, sig_now, rate = predict_car(est, det, [t], cp)
            az_off = 0.0
            if scan is not None:
                if coasting:
                    az_off = scan.update(DT_CTRL, float(sig_now[0]),
                                         float(np.hypot(xs[0] - plant.p[0], ys[0] - plant.p[1])))
                else:
                    scan.reset()
            gimbal.aim(plant.p, np.array([xs[0], ys[0]]), DT_CTRL, az_off)
            prev = cmd.copy()
            if cfg["controller"] == "pid":
                cmd = pid_command(est, pid, plant, t, np.array([xs[0], ys[0]]), float(psi_now[0]),
                                  float(rate[0]), coasting)
            else:
                times = tracker.stage_times(t)
                hx, hy, hpsi, hsig = predict_horizon(est, det, times, coasting, cp, cfg.get("track_fade", 0.0))
                n1 = len(times)
                x_meas = np.r_[plant.p, plant.v, plant.psi, plant.r]
                scen = None if coasting else range_scenarios(est, det, times, t)
                cmd, info = tracker.command(t, x_meas, np.c_[hx, hy], CAR_Z, hpsi, hsig,
                                            np.tile(OFFSET, (n1, 1)), np.full(n1, OFFSET[2]),
                                            car_speed=math.hypot(est.vx, est.vy), scenarios=scen)
                m["ms"].append(info["solve_ms"])
            plant.command(t, cmd)
            if t > 12.0:
                m["du"].append(np.hypot(*(cmd[:2] - prev[:2])) / DT_CTRL)
            # blackout just ended: score the prediction there
            for i, (a0, b0) in enumerate(blk):
                if blk_state[i] is None and t >= b0:
                    blk_state[i] = {"pred_err": float(np.hypot(xs[0] - car_p[0], ys[0] - car_p[1])),
                                    "reacq": None, "lost": False}
            next_ctrl += DT_CTRL
        plant.step(t, DT_SIM)
        t += DT_SIM
        if t > 12.0:
            sp = shot_point(car_p, car_psi)
            d = plant.p[:2] - car_p
            m["err"].append(float(np.hypot(*(plant.p[:2] - sp[:2]))))
            rad_s = np.hypot(*d) - math.hypot(OFFSET[0], OFFSET[1])
            ang_s = math.degrees(wrap(math.atan2(d[1], d[0]) - (car_psi + math.atan2(OFFSET[1], OFFSET[0]))))
            m["rad"].append(abs(rad_s))
            m["ang"].append(abs(ang_s))
            if trace is not None:
                trace.append((t, rad_s, ang_s, float(np.linalg.norm(np.r_[d, CAR_Z - plant.p[2]])),
                              float(np.hypot(*spline(t, 1)))))
            m["yaw"].append(abs(math.degrees(wrap(math.atan2(-d[1], -d[0]) - plant.psi))))
            m["rng"].append(float(np.linalg.norm(np.r_[d, CAR_Z - plant.p[2]])))
    m["occl"] = [(round(blk[i][1] - blk[i][0], 1), st) for i, st in blk_state.items() if st is not None]
    return m


def pid_command(est, pid, plant, now, tgt, psi_pred, rate, coasting):
    yaw_rate = 0.0 if coasting else est.yaw_rate
    h = psi_pred if coasting else est.heading + yaw_rate * 0.35
    c, s = math.cos(h), math.sin(h)
    rel = np.array([c * OFFSET[0] - s * OFFSET[1], s * OFFSET[0] + c * OFFSET[1]])
    e = np.r_[tgt + rel - plant.p[:2], OFFSET[2] - plant.p[2]]
    d = tgt - plant.p[:2]
    eyaw = 0.0 if np.hypot(*d) < 5.0 else wrap(math.atan2(d[1], d[0]) - plant.psi)
    ff = np.array([est.vx, est.vy]) * rate
    if not coasting:
        rot = np.array([-yaw_rate * rel[1], yaw_rate * rel[0]])
        n = np.hypot(*rot)
        ff = ff + (rot * 5.0 / n if n > 5.0 else rot)
    out = pid.update(e[0], e[1], e[2], eyaw, now, ff_vx=ff[0], ff_vy=ff[1])
    return np.array([out.vx, out.vy, out.vz, out.yaw_rate])


def catch_ups(ang, dt=DT_SIM, lost_deg=20.0, back_deg=10.0):
    """Episodes where the angle round the car slipped past lost_deg: (recovered?,
    seconds until it was back within back_deg)."""
    eps, i, n = [], 0, len(ang)
    while i < n:
        if ang[i] > lost_deg:
            j = i
            while j < n and ang[j] > back_deg:
                j += 1
            eps.append((j < n, (j - i) * dt))
            i = j + 1
        else:
            i += 1
    return eps


def summary(name, ms):
    a = {k: np.array(v) for k, v in ms.items() if k not in ("occl", "lost", "lost_outside_occlusion", "eps")}
    band = np.mean((a["rad"] <= 5.0) & (a["ang"] <= 10.0)) * 100
    eps = ms.get("eps", [])
    rec = [d for ok, d in eps if ok]
    line = (f"{name:9} err {np.median(a['err']):5.2f} / {np.percentile(a['err'], 90):5.2f} m"
            f" | distance held (+-5 m) {np.mean(a['rad'] <= 5.0) * 100:5.1f} %, worst off {a['rad'].max():5.1f} m"
            f" | angle >20 deg: {len(eps)} times, back <10 deg {len(rec)}"
            f" (median {np.median(rec) if rec else float('nan'):4.1f} s, max {max(rec) if rec else float('nan'):4.1f} s)"
            f" | band {band:4.1f} %"
            f" | yaw {np.median(a['yaw']):4.1f} / {np.percentile(a['yaw'], 90):4.1f} deg"
            f" | too close (<9.4 m) {np.mean(a['rng'] < 9.4) * 100:4.1f} %"
            f" | lost {ms['lost']} ({ms['lost_outside_occlusion']} not after a blackout)")
    if len(a["ms"]):
        line += f" | solve {np.median(a['ms']):.2f} / {a['ms'].max():.1f} ms"
    print(line)
    if ms["occl"]:
        for dur in sorted({d for d, _ in ms["occl"]}):
            sts = [st for d, st in ms["occl"] if d == dur]
            got = [st["reacq"] for st in sts if st["reacq"] is not None]
            pe = [st["pred_err"] for st in sts]
            print(f"{'':9}   {dur:3.1f} s blackouts: picked up again {len(got)}/{len(sts)}"
                  f" (median {np.median(got) if got else float('nan'):.2f} s after it ended),"
                  f" prediction error at the end {np.median(pe):5.1f} m median")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", nargs="+", default=["iris_silverstone", "iris_oschersleben"])
    ap.add_argument("--variants", nargs="+", default=["pid", "mpc_old", "mpc_coast", "mpc_new", "mpc_scan"],
                    choices=list(VARIANTS))
    ap.add_argument("--seconds", type=float, default=110.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--occlusions", action="store_true", help="add 1.5 / 3 / 4.5 s blackouts every ~20 s")
    ap.add_argument("--shot", choices=["back", "right", "left"], default="back")
    ap.add_argument("--weights", nargs="+", default=[],
                    help="radial,tangential MPC weight pairs to compare (current defaults otherwise), e.g. 30,30 90,10")
    ap.add_argument("--car-speed-scale", type=float, default=1.0,
                    help="drive the car this much faster (1.07 ~ the 16 m/s seen in Gazebo)")
    ap.add_argument("--plot", default="", help="save distance/angle-vs-time plots (first world, seed 0) to this PNG")
    args = ap.parse_args()
    global OFFSET
    OFFSET = {"back": np.array([-18.0, 0.0, 4.0]), "right": np.array([0.0, -18.0, 4.0]),
              "left": np.array([0.0, 18.0, 4.0])}[args.shot]
    for pair in args.weights:
        wr, wt = (float(v) for v in pair.split(","))
        VARIANTS[f"w{wr:g}/{wt:g}"] = dict(VARIANTS["mpc_both"], w=(wr, wt))
    if args.weights:
        args.variants = [f"w{float(p.split(',')[0]):g}/{float(p.split(',')[1]):g}" for p in args.weights]
    mpc = DroneMPC()
    traces = {}
    print("error to the true shot point median / p90; band = +-5 m and +-10 deg around the car;"
          " yaw error to the car median / p90")
    for world in args.worlds:
        print(f"== {world}, {args.seconds:.0f} s x {args.seeds} seeds, occlusions {'on' if args.occlusions else 'off'},"
              f" shot {args.shot}")
        for name in args.variants:
            agg = {"lost": 0, "lost_outside_occlusion": 0, "occl": [], "eps": []}
            for seed in range(args.seeds):
                tr = [] if (args.plot and seed == 0 and world == args.worlds[0]) else None
                res = run(name, world, args.seconds, seed, mpc, args.occlusions, args.car_speed_scale, tr)
                for k, v in res.items():
                    if k in ("lost", "lost_outside_occlusion"):
                        agg[k] += v
                    else:
                        agg.setdefault(k, []).extend(v)
                agg["eps"].extend(catch_ups(np.array(res["ang"])))
                if tr is not None:
                    traces[name] = np.array(tr)
            summary(name, agg)
    if args.plot and traces:
        plot_traces(traces, args)


def plot_traces(traces, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r_ref = math.hypot(OFFSET[0], OFFSET[1])
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for name, tr in traces.items():
        ax[0].plot(tr[:, 0], tr[:, 3], lw=1, label=name)
        ax[1].plot(tr[:, 0], tr[:, 2], lw=1, label=name)
    ax[0].axhspan(r_ref - 5, r_ref + 5, color="g", alpha=0.12, label=f"shot distance +-5 m")
    ax[0].axhline(9.4, color="r", ls="--", lw=0.8, label="DOPE min (9.4 m)")
    ax[0].axhline(80, color="r", ls=":", lw=0.8, label="detections rejected (80 m)")
    ax[0].set_ylabel("range to car [m]"); ax[0].legend(fontsize=8, ncol=3); ax[0].grid(alpha=0.3)
    ax[1].axhspan(-10, 10, color="g", alpha=0.12)
    ax[1].set_ylabel("angle round the car vs shot [deg]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    first = next(iter(traces.values()))
    ax[2].plot(first[:, 0], first[:, 4], "k", lw=1, label="car speed")
    ax[2].axhline(15, color="r", ls="--", lw=0.8, label="drone speed limit")
    ax[2].set_ylabel("car speed [m/s]"); ax[2].set_xlabel("time [s]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle(f"{args.worlds[0]}, shot {args.shot}, car speed x{args.car_speed_scale:g}")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print(f"plot saved: {args.plot}")


if __name__ == "__main__":
    main()
