#!/usr/bin/env python3
"""
Linear Kalman Filter for target position/velocity tracking.

State (world frame):
    x = [px, py, pz, vx, vy, vz]

Motion model: constant velocity with white-noise acceleration.
Measurement model: direct position observation (H = identity on position).
This is intentionally a *linear* KF, not EKF/UKF -- the nonlinear
camera-optical -> world transform already happens upstream in
RelativePositionController._on_visual_odom(), so by the time a
measurement reaches this filter it is just a noisy world-frame xyz
point. No Jacobians needed. If a nonlinear turning-target motion model
(CTRV) is added later, an EKF/UKF becomes appropriate -- this class
does not attempt that.

Adaptive process noise (innovation-based):
    A fixed-Q constant-velocity KF has a single, permanent trade-off
    between smoothing (low Q -> slow to react, but rejects measurement
    noise well) and responsiveness (high Q -> reacts fast, but noisier
    velocity output). That fixed trade-off is what causes visible lag
    whenever the target actually accelerates/decelerates -- the filter
    is still discounting new measurements by the same amount it did
    while the target was cruising at constant velocity.

    Instead of picking one fixed point on that trade-off, this filter
    watches its own normalized innovation squared (NIS) each update --
    a chi-square-distributed statistic that says "how surprised should
    a correctly-tuned constant-velocity filter be by this measurement."
    When NIS is at its nominal level, the target really is moving at
    roughly constant velocity, and Q stays at its smooth baseline. When
    NIS spikes well above nominal, that's a signal the target is
    maneuvering and the constant-velocity assumption is temporarily
    wrong -- Q is scaled up so the next predict/update trusts the new
    measurement almost fully and the velocity estimate snaps to match.
    As soon as the residual returns to nominal, Q relaxes back down.
    This is a standard technique (innovation-based adaptive estimation)
    for maneuvering-target tracking with a single-model KF, short of a
    full IMM (interacting multiple model) filter.
"""

import math

import numpy as np


class TargetKalmanFilter:
    # Chi-square critical values for a 3-DOF innovation (x,y,z position
    # residual). _NIS_BASELINE is the median (50th percentile) value a
    # correctly-tuned constant-velocity filter should see under normal
    # tracking -- used to normalize the scale factor around 1.0.
    _NIS_BASELINE = 2.366

    def __init__(
        self,
        accel_noise_std: float = 1.5,
        measurement_noise_std: float = 0.15,
        initial_pos_std: float = 1.0,
        initial_vel_std: float = 3.0,
        max_speed: float = 8.0,
        enable_adaptive_q: bool = True,
        adaptive_q_max_scale: float = 12.0,
        adaptive_q_decay: float = 0.5,
    ):
        """
        accel_noise_std: baseline process noise, assumed target
            acceleration magnitude (m/s^2), drives process noise Q at
            the smooth/nominal operating point. Larger = filter trusts
            new measurements more / smooths velocity less, even at
            baseline.
        measurement_noise_std: assumed std dev of position measurements
            (m), drives R. Larger = filter trusts its own prediction
            more / smooths position more.
        max_speed: safety clamp (m/s) applied to the returned velocity
            estimate, so a single bad transform/detection outlier can't
            inject a huge feedforward velocity into the PID.
        enable_adaptive_q: kill switch for the innovation-based Q
            scaling described above. False reproduces the old fixed-Q
            behaviour exactly (for A/B testing responsiveness).
        adaptive_q_max_scale: hard ceiling on how far Q can be inflated
            above baseline during a detected maneuver. Higher = faster
            snap to a new velocity, but noisier during the transient.
        adaptive_q_decay: exponential smoothing factor (0-1) applied to
            the scale factor between updates, so a single noisy/outlier
            measurement doesn't fully open the gate on its own -- only
            sustained above-nominal innovation (i.e. an actual velocity
            change, not one noisy frame) drives the scale up. Higher =
            faster reaction to genuine changes but less outlier
            rejection.
        """
        self.q_accel = float(accel_noise_std)
        self.r_std = float(measurement_noise_std)
        self.initial_pos_std = float(initial_pos_std)
        self.initial_vel_std = float(initial_vel_std)
        self.max_speed = float(max_speed)

        self.enable_adaptive_q = bool(enable_adaptive_q)
        self.adaptive_q_max_scale = float(adaptive_q_max_scale)
        self.adaptive_q_decay = float(adaptive_q_decay)
        self._q_scale = 1.0  # current process-noise inflation factor

        self.x = np.zeros(6, dtype=float)  # px,py,pz,vx,vy,vz
        self.P = np.eye(6, dtype=float)

        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.eye(3, dtype=float) * (self.r_std ** 2)

        self.initialized = False

    def reset(self, position_meas):
        """Reinitialize filter at a measured position with zero velocity
        and wide uncertainty. Call this on first detection and whenever
        the target has been lost long enough that the coasted velocity
        estimate can no longer be trusted (e.g. after a search-mode
        dropout)."""
        px, py, pz = position_meas
        self.x = np.array([px, py, pz, 0.0, 0.0, 0.0], dtype=float)

        self.P = np.diag([
            self.initial_pos_std ** 2,
            self.initial_pos_std ** 2,
            self.initial_pos_std ** 2,
            self.initial_vel_std ** 2,
            self.initial_vel_std ** 2,
            self.initial_vel_std ** 2,
        ])
        self.initialized = True
        self._q_scale = 1.0  # don't carry a stale maneuver flag across a reset

    def predict(self, dt: float):
        if not self.initialized or dt <= 0.0:
            return

        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q = (self.q_accel ** 2)
        if self.enable_adaptive_q:
            q *= self._q_scale
        Q = np.zeros((6, 6), dtype=float)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        # per-axis white-noise-acceleration block, axes: x(0,3) y(1,4) z(2,5)
        for p_idx, v_idx in ((0, 3), (1, 4), (2, 5)):
            Q[p_idx, p_idx] = 0.25 * dt4 * q
            Q[p_idx, v_idx] = 0.5 * dt3 * q
            Q[v_idx, p_idx] = 0.5 * dt3 * q
            Q[v_idx, v_idx] = dt2 * q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, position_meas):
        if not self.initialized:
            self.reset(position_meas)
            return

        z = np.array(position_meas, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

        if self.enable_adaptive_q:
            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)
            except np.linalg.LinAlgError:
                nis = self._NIS_BASELINE

            # Scale is 1.0 at/below the nominal residual level a
            # correctly-tuned CV filter expects, and grows above that as
            # the residual grows -- i.e. as the constant-velocity
            # assumption looks more wrong. Capped so one huge outlier
            # can't blow Q up arbitrarily.
            raw_scale = max(1.0, nis / self._NIS_BASELINE)
            raw_scale = min(raw_scale, self.adaptive_q_max_scale)

            # Exponential smoothing: a genuine velocity change shows up
            # as *sustained* above-nominal innovation over a couple of
            # updates and will still ramp the scale up quickly; a single
            # noisy/outlier frame gets damped instead of fully opening
            # the gate on its own.
            self._q_scale = (
                (1.0 - self.adaptive_q_decay) * self._q_scale
                + self.adaptive_q_decay * raw_scale
            )

    def get_position(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_velocity(self):
        vx, vy, vz = float(self.x[3]), float(self.x[4]), float(self.x[5])
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > self.max_speed and speed > 0.0:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
            vz *= scale
        return vx, vy, vz

    def get_process_noise_scale(self) -> float:
        """Current Q inflation factor -- 1.0 at baseline, higher while a
        maneuver is being detected. Useful to log/plot to see the
        adaptive behaviour actually kicking in during testing."""
        return float(self._q_scale)