#!/usr/bin/env python3
"""Planar CTRA EKF for the target car: position and heading from the detector.

State (world ENU):  s = [x, y, psi, v, w, a]   (a pinned to 0 unless estimate_accel)
    x, y  position (m)          psi  heading (rad)
    v     speed along psi (m/s) w    turn rate (rad/s)
    a     along-track accel (m/s^2)
Motion: constant turn rate and acceleration (CTRA), white noise on the turn
    rate derivative (s_wdot) and on jerk (s_jerk).
Measurement: z = [x, y, psi]  (psi optional).

Why this model: a car drives the way it points, so the detector heading is a
direct measurement of the direction of travel. With DOPE's noise (position
error ~3.5 % of range with a bias drifting over ~1.5 s, heading ~3.6 deg)
the heading-aided filter beat the position-only constant-velocity KF on
held-out race tracks (median, through the full TargetEstimator) by 13 % now,
31 % at 1 s and 29 % at 2 s ahead, 40 % on velocity and 31 % on heading
(scripts/benchmark_target_filter.py). Without the heading
it was worse than the CV KF, so feed it flip-resolved headings only (see
TargetEstimator._resolve_heading_flip).

The world-frame velocity and acceleration (velocity(), acceleration()) are
what a double-integrator target model in an MPC wants.
"""
import math

import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TargetCTRAEKF:
    def __init__(
        self,
        s_wdot: float = 0.2,
        s_jerk: float = 0.5,
        heading_std: float = math.radians(4.0),
        initial_speed_std: float = 6.0,
        initial_turn_rate_std: float = 0.3,
        initial_accel_std: float = 2.0,
        max_speed: float = 15.0,
        max_turn_rate: float = math.radians(60.0),
        max_step_sec: float = 0.02,
        estimate_accel: bool = False,
        s_speed: float = 2.0,
    ):
        self.s_wdot = s_wdot
        self.s_jerk = s_jerk
        self.heading_std = heading_std
        self.initial_speed_std = initial_speed_std
        self.initial_turn_rate_std = initial_turn_rate_std
        self.initial_accel_std = initial_accel_std
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.max_step_sec = max_step_sec
        # estimate_accel False (default): CTRV -- acceleration pinned to 0,
        # speed a random walk (s_speed m/s^2 white noise). With DOPE the
        # along-track acceleration only shows up ~1 s late, so an estimated
        # one carries the last straight's +2 m/s^2 into the braking zone and
        # pushes the speed UP while the car brakes (+1.6 m/s offline; the
        # MPC then kept its distance from a phantom car 15-25 m ahead).
        # CTRV: +0.7 m/s, and 10 % better p90 1-2 s ahead.
        self.estimate_accel = estimate_accel
        self.s_speed = s_speed
        self.s = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False

    # ------------------------------------------------------------------
    def reset(self, x: float, y: float, psi: float, speed: float = 0.0, speed_std=None):
        """Start at a measured position/heading. `speed` can carry the last
        known speed across a short loss (the KF re-seed case)."""
        self.s = np.array([x, y, _wrap(psi), max(0.0, speed), 0.0, 0.0])
        vs = self.initial_speed_std if speed_std is None else speed_std
        self.P = np.diag([1.0, 1.0, self.heading_std ** 2, vs ** 2,
                          self.initial_turn_rate_std ** 2,
                          self.initial_accel_std ** 2 if self.estimate_accel else 0.0])
        self.initialized = True

    @staticmethod
    def _step(s, dt):
        """One Euler step of the CTRA model and its Jacobian."""
        x, y, psi, v, w, a = s
        c, sn = math.cos(psi), math.sin(psi)
        nxt = np.array([x + v * c * dt, y + v * sn * dt, psi + w * dt, v + a * dt, w, a])
        F = np.eye(6)
        F[0, 2] = -v * sn * dt; F[0, 3] = c * dt
        F[1, 2] = v * c * dt;   F[1, 3] = sn * dt
        F[2, 4] = dt
        F[3, 5] = dt
        return nxt, F

    def _clamp(self):
        self.s[2] = _wrap(self.s[2])
        self.s[3] = min(max(self.s[3], 0.0), self.max_speed)
        self.s[4] = min(max(self.s[4], -self.max_turn_rate), self.max_turn_rate)

    def _propagate(self, dt: float, s=None, P=None):
        """(state, covariance) predicted dt ahead from (s, P) -- the filter's
        own by default -- without committing them."""
        n = max(1, int(math.ceil(dt / self.max_step_sec)))
        d = dt / n
        s = (self.s if s is None else s).copy()
        P = self.P if P is None else P
        if not self.estimate_accel:
            s[5] = 0.0
        F_total = np.eye(6)
        for _ in range(n):
            s, F = self._step(s, d)
            s[3] = max(s[3], 0.0)
            F_total = F @ F_total
        # Integrated white noise on w-dot (acting on psi, w) and jerk (on v, a).
        Q = np.zeros((6, 6))
        for (i, j), sg in (((2, 4), self.s_wdot), ((3, 5), self.s_jerk)):
            q = sg * sg
            Q[i, i] += q * dt ** 3 / 3.0
            Q[i, j] += q * dt ** 2 / 2.0
            Q[j, i] += q * dt ** 2 / 2.0
            Q[j, j] += q * dt
        if not self.estimate_accel:
            # Speed random walk instead of jerk-driven acceleration.
            Q[3, 5] = Q[5, 3] = Q[5, 5] = 0.0
            Q[3, 3] = self.s_speed ** 2 * dt
        P_new = F_total @ P @ F_total.T + Q
        if not self.estimate_accel:
            P_new[5, :] = 0.0
            P_new[:, 5] = 0.0
        return s, P_new

    def predict(self, dt: float):
        if not self.initialized or dt <= 0.0:
            return
        self.s, self.P = self._propagate(dt)
        self._clamp()

    def position_nis(self, x: float, y: float, pos_std: float, dt: float) -> float:
        """Normalized innovation squared of a position measurement dt after
        the last update: how many 'sigmas squared' it is from where the
        filter expects the car, given the filter's own uncertainty and the
        measurement noise. Chi-square with 2 DOF if the model is right."""
        s, P = self._propagate(dt) if dt > 0.0 else (self.s, self.P)
        innov = np.array([x - s[0], y - s[1]])
        S = P[:2, :2] + np.eye(2) * pos_std ** 2
        return float(innov @ np.linalg.solve(S, innov))

    def update(self, x: float, y: float, pos_std: float, psi=None):
        """Position (and heading, if given) measurement."""
        if psi is None:
            H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1.0
            innov = np.array([x - self.s[0], y - self.s[1]])
            R = np.eye(2) * pos_std ** 2
        else:
            H = np.zeros((3, 6)); H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            innov = np.array([x - self.s[0], y - self.s[1], _wrap(psi - self.s[2])])
            R = np.diag([pos_std ** 2, pos_std ** 2, self.heading_std ** 2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ innov
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form: stays symmetric PSD
        self._clamp()

    def predict_trajectory(self, times, turn_fade_tau=None):
        """Car at future times (seconds after the last update, ascending), no
        measurements: arrays x, y, psi and pos_std (sqrt of the mean position
        variance, growing with time -- for weighting an MPC horizon).

        turn_fade_tau None: straight line at the current speed and heading
        (the most robust 1-2 s predictor in the benchmark: median ~ fading
        turn, better p90). Otherwise the turn rate fades with that time
        constant. Along-track acceleration is not extrapolated: its estimate
        is about as noisy as the car's real acceleration.
        """
        times = np.asarray(times, dtype=float)
        out = np.empty((len(times), 4))
        s = self.s.copy()
        s[5] = 0.0                                   # no acceleration extrapolation
        w0 = s[4]
        if turn_fade_tau is None:
            s[4] = 0.0
        P = self.P.copy()
        t_prev = 0.0
        for i, t in enumerate(times):
            dt = max(0.0, t - t_prev)
            if dt > 0.0:
                steps = max(1, int(math.ceil(dt / 0.05)))
                d = dt / steps
                for k in range(steps):
                    if turn_fade_tau is not None:
                        s[4] = w0 * math.exp(-(t_prev + k * d) / turn_fade_tau)
                    s, P = self._propagate(d, s, P)
            out[i] = (s[0], s[1], _wrap(s[2]), math.sqrt(max(0.0, 0.5 * (P[0, 0] + P[1, 1]))))
            t_prev = t
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    # ------------------------------------------------------------------
    @property
    def x(self): return float(self.s[0])
    @property
    def y(self): return float(self.s[1])
    @property
    def heading(self): return float(self.s[2])
    @property
    def speed(self): return float(self.s[3])
    @property
    def turn_rate(self): return float(self.s[4])
    @property
    def along_accel(self): return float(self.s[5])

    def velocity(self):
        """World-frame (vx, vy)."""
        v, psi = self.s[3], self.s[2]
        return float(v * math.cos(psi)), float(v * math.sin(psi))

    def acceleration(self):
        """World-frame (ax, ay): along-track accel plus centripetal v * w."""
        psi, v, w, a = self.s[2], self.s[3], self.s[4], self.s[5]
        c, sn = math.cos(psi), math.sin(psi)
        return float(a * c - v * w * sn), float(a * sn + v * w * c)
