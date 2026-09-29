#!/usr/bin/env python3
"""Where is the car now / at future times, from the last estimate (no ROS).

Shared by the node (drone setpoint, gimbal aim, MPC horizon) and the offline
closed-loop sim, so both run the same coast logic.

After the last detection the car is carried on:
  * CTRA filter: along the EKF's own prediction -- current speed, turn rate
    fading with coast_turn_fade_sec, heading moving with it (a car hidden by
    a tree on a bend keeps turning);
  * CV filter: in a straight line at the last velocity, heading frozen.
For coast_full_speed_sec it runs at full speed, then slows to a stop over
coast_taper_sec and never moves more than coast_max_distance.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CoastParams:
    full_speed_sec: float = 3.0
    taper_sec: float = 1.0
    max_distance: float = 40.0
    turn_fade_sec: float = 1.5      # <= 0: straight line even with the CTRA filter


def coast_progress(dt, full_speed_sec, taper_sec):
    """(travel seconds at the last speed, share of the speed still applied)
    after dt seconds without a detection; works on scalars or arrays."""
    dt = np.maximum(0.0, np.asarray(dt, dtype=float))
    taper = max(taper_sec, 1e-3)
    e = np.clip(dt - full_speed_sec, 0.0, taper)
    travel = np.minimum(dt, full_speed_sec) + e - e * e / (2.0 * taper)
    rate = np.where(dt <= full_speed_sec, 1.0, np.clip(1.0 - (dt - full_speed_sec) / taper, 0.0, 1.0))
    return travel, rate


def base_time(detection, now):
    """Capture time of the last detection, or its reception time if the stamp
    is in another clock domain (use_sim_time mismatch)."""
    base = detection.stamp_sec
    if not 0.0 <= now - base <= now - detection.received_time + 1.0:
        base = detection.received_time
    return base


def predict_car(estimator, detection, times, params: CoastParams, extend_uncertainty=True):
    """Car at absolute `times` (ascending): arrays x, y, psi, sigma, rate.

    sigma: position std for weighting (the EKF's own growth, plus 1.5 m per
    second spent in the taper/stop, where the car is really still moving).
    rate: share of the car's velocity still applied (1 while predicting at
    full speed) -- scales the velocity feedforward.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    now = float(times[0])
    base = base_time(detection, now)
    dt = times - base
    travel, rate = coast_progress(dt, params.full_speed_sec, params.taper_sec)
    est = estimator
    if getattr(est, "use_ctra", False) and est.ctra.initialized:
        ekf = est.ctra
        fade = params.turn_fade_sec if params.turn_fade_sec > 0.0 else None
        # The EKF state is at its last update (the capture time = base).
        xs, ys, psi, sigma = ekf.predict_trajectory(travel, fade)
        x0, y0 = ekf.x, ekf.y
    else:
        xs = est.x + est.vx * travel
        ys = est.y + est.vy * travel
        psi = np.full(n, est.heading)
        sigma = 1.0 + 1.5 * travel
        x0, y0 = est.x, est.y
    # Never carry the estimate further than max_distance from where it was seen.
    d = np.hypot(xs - x0, ys - y0)
    over = d > params.max_distance
    if np.any(over):
        k = np.where(over, params.max_distance / np.maximum(d, 1e-9), 1.0)
        xs, ys = x0 + (xs - x0) * k, y0 + (ys - y0) * k
        rate = np.where(over, 0.0, rate)
    if extend_uncertainty:
        sigma = sigma + 1.5 * np.maximum(0.0, dt - travel)
    return xs, ys, psi, sigma, rate


def predict_horizon(estimator, detection, times, coasting, params: CoastParams, track_turn_fade_sec=0.0):
    """Car over an MPC horizon (absolute times): x, y, psi, sigma arrays.

    Tracking with the CTRA filter: the EKF rolled forward from its last
    update (straight line by default, track_turn_fade_sec > 0 keeps a fading
    turn). Coasting, or the CV filter: the coast prediction (predict_car)."""
    est = estimator
    if not coasting and getattr(est, "use_ctra", False) and est.ctra.initialized:
        base = base_time(detection, float(times[0]))
        fade = track_turn_fade_sec if track_turn_fade_sec > 0.0 else None
        return est.ctra.predict_trajectory(np.asarray(times, dtype=float) - base, fade)
    xs, ys, psi, sigma, _ = predict_car(est, detection, times, params)
    return xs, ys, psi, sigma


def range_scenarios(estimator, detection, times, now, brake_decel=4.0, car_accel=2.0, car_max_speed=15.0):
    """Car positions for the MPC's minimum-range scenarios at absolute `times`:
    (brake_xy, accel_xy), each (n, 2), or None if there is no CTRA state.

    brake: the car brakes hard from now AND keeps turning at its current turn
        rate -- what a car does into a hairpin. (Braking alone along the
        heading missed the offline too-close cases: all at 5-6.6 m/s hairpins,
        turning 22-28 deg/s, not braking.)
    accel: the car accelerates from now along its predicted (straight) path.
    """
    est = estimator
    if not (getattr(est, "use_ctra", False) and est.ctra.initialized):
        return None
    ekf = est.ctra
    base = base_time(detection, now)
    times = np.asarray(times, dtype=float)
    # Signed speed: a reversing car (v < 0) brakes / speeds up along its
    # backward path; it can reverse up to the EKF's max_reverse_speed.
    v_signed = ekf.speed
    v0 = abs(v_signed)
    vmax = car_max_speed if v_signed >= 0.0 else getattr(ekf, "max_reverse_speed", car_max_speed)
    tau = np.maximum(0.0, times - now)             # braking / accelerating starts now
    lead = max(0.0, now - base)                    # the car already drove this since the estimate
    b, a = max(brake_decel, 1e-3), max(car_accel, 1e-3)
    t_stop = v0 / b
    s_brake = np.where(tau < t_stop, v0 * tau - 0.5 * b * tau ** 2, v0 * v0 / (2.0 * b))
    t_sat = max(0.0, vmax - v0) / a
    s_acc = np.where(tau < t_sat, v0 * tau + 0.5 * a * tau ** 2,
                     v0 * t_sat + 0.5 * a * t_sat ** 2 + vmax * (tau - t_sat))
    if v0 < 0.5:
        xy = np.c_[np.full(len(times), ekf.x), np.full(len(times), ekf.y)]
        return xy, xy.copy()
    # Distance along a path at speed |v0| <-> time on that path (from the
    # estimate); predict_trajectory moves backwards for v < 0.
    bx, by, _, _ = ekf.predict_trajectory(lead + s_brake / v0, turn_fade_tau=1e9)   # turn held
    ax, ay, _, _ = ekf.predict_trajectory(lead + s_acc / v0, None)                 # straight
    return np.c_[bx, by], np.c_[ax, ay]


class CoastScan:
    """Gimbal yaw scan around the predicted car while it is hidden.

    The prediction drifts sideways by a few metres per hidden second; close
    to the car that is a big angle (offline misses after 3 s blackouts: the
    car came back 35-66 deg off the boresight at 12-21 m, camera +-23 deg).
    Scan half-width = what sigma_k * sigma looks like from the drone, minus
    what the camera already covers (+ a margin), up to max_deg.

    mode "stare" (default): stop-and-stare. Hold the prediction, then
        +step, -step, +2 step, -2 step ... within the half-width, each for
        dwell_sec, jumping between them at the gimbal's slew rate. Once DOPE
        has not seen the car for 1 s it tries ONE input scale per frame out
        of 9 (dope_detector scale_pyramid) and only 1-2 of them suit the
        range, so a view has to be held for a whole cycle (9 frames / 15 Hz
        = 0.6 s) for the right scale to be tried. The continuous sweep of
        run 20260929_142724 crossed the car at ~176 deg/s: 3-4 frames in
        view, no detection.
    mode "sweep": the continuous back-and-forth sweep at rate_deg.
    """

    def __init__(self, half_fov_deg=23.0, sigma_k=2.5, max_deg=50.0, rate_deg=90.0, margin_deg=5.0,
                 mode="stare", step_deg=30.0, dwell_sec=0.7):
        self.half_fov = math.radians(half_fov_deg)
        self.sigma_k = sigma_k
        self.max = math.radians(max_deg)
        self.rate = math.radians(rate_deg)
        self.margin = math.radians(margin_deg)
        self.mode = mode
        self.step = math.radians(step_deg)
        self.dwell = dwell_sec
        self.reset()

    def reset(self):
        self.offset, self.direction = 0.0, 1.0
        self._slot = 0              # stare: index into 0, +1, -1, +2, -2 ... (in steps)
        self._held = 0.0            # stare: time held at the current slot (after arriving)

    def amplitude(self, sigma, range_m):
        need = math.atan2(self.sigma_k * sigma, max(range_m, 1.0)) - self.half_fov + self.margin
        return min(max(need, 0.0), self.max)

    @staticmethod
    def _slot_offset(k):
        """0, +1, -1, +2, -2, ... (in steps)."""
        return 0 if k == 0 else (k + 1) // 2 * (1 if k % 2 else -1)

    def update(self, dt, sigma, range_m):
        """Yaw offset (rad) to add to the aim at the predicted car."""
        amp = self.amplitude(sigma, range_m)
        if self.mode == "stare":
            return self._update_stare(dt, amp)
        if amp <= 0.0:
            # Nothing to scan: drift back to the prediction.
            step = self.rate * dt
            self.offset -= max(-step, min(step, self.offset))
            return self.offset
        self.offset += self.direction * self.rate * dt
        if self.offset >= amp:
            self.offset, self.direction = amp, -1.0
        elif self.offset <= -amp:
            self.offset, self.direction = -amp, 1.0
        return self.offset

    def _update_stare(self, dt, amp):
        n_side = int(amp // self.step + (1 if amp % self.step > 0.5 * self.step else 0)) if amp > 0 else 0
        n_slots = 2 * n_side + 1
        target = self._slot_offset(self._slot % n_slots) * self.step
        target = max(-amp - 0.5 * self.step, min(amp + 0.5 * self.step, target))
        move = self.rate * dt
        if abs(target - self.offset) > 1e-6:
            self.offset += max(-move, min(move, target - self.offset))
            if abs(target - self.offset) > 1e-6:
                return self.offset          # still slewing to the next view
        self._held += dt
        if self._held >= self.dwell and n_slots > 1:
            self._held = 0.0
            self._slot = (self._slot + 1) % n_slots
        return self.offset
