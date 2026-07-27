#!/usr/bin/env python3
"""
Unscented Kalman Filter, Constant Turn Rate and Velocity (CTRV) model,
for target heading / yaw-rate estimation.

This is deliberately a SEPARATE filter from target_kalman_filter.py.
That one is a linear KF over Cartesian [px,py,pz,vx,vy,vz] and stays
exactly as-is -- it is mathematically correct as a linear filter and
already feeds PID velocity feedforward.

This filter instead tracks state:
    x = [px, py, v, yaw, yawd]
        px, py : world-frame position (m)
        v      : scalar forward speed along heading (m/s)
        yaw    : heading (rad)
        yawd   : yaw rate (rad/s)

Why this one genuinely needs UKF and not a linear KF:
    The process model is
        px' = px + (v/yawd) * [sin(yaw + yawd*dt) - sin(yaw)]
        py' = py + (v/yawd) * [cos(yaw) - cos(yaw + yawd*dt)]
        yaw' = yaw + yawd*dt
    yaw appears inside sin/cos and is multiplied with v and yawd --
    genuinely nonlinear, and the yawd -> 0 (straight-line driving) case
    is a removable singularity that needs explicit handling. An EKF
    would need a hand-derived Jacobian with a special-cased linear
    branch at yawd ~ 0; the UKF's sigma-point propagation handles both
    regimes through the same code path.

Measurement: this target publishes full pose (position + orientation)
every detection, so two update modes are provided:
    update_position(px, py)              -- position only
    update_position_heading(px, py, yaw) -- position + heading

Use update_position() alone when the heading measurement is judged
unreliable for that frame (e.g. target too far away for a stable
solvePnP orientation) so the filter still tracks position/yaw via the
motion model without being corrupted by a bad heading reading.

This is the standard CTRV UKF formulation (as used in typical
pedestrian/vehicle tracking coursework and literature), not a fully
general scaled UKF with separate alpha/beta/kappa tuning -- that
generality is not needed for a single ground target.
"""

import math

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class TargetCTRVUKF:
    def __init__(
        self,
        std_a: float = 1.5,
        std_yawdd: float = 0.5,
        std_pos: float = 0.15,
        std_yaw: float = 0.20,
        initial_pos_std: float = 1.0,
        initial_v_std: float = 3.0,
        initial_yaw_std: float = math.pi,
        initial_yawd_std: float = 1.0,
        max_yaw_rate: float = 3.0,
    ):
        """
        std_a: process noise, longitudinal acceleration (m/s^2).
        std_yawdd: process noise, yaw angular acceleration (rad/s^2).
        std_pos: position measurement noise std (m).
        std_yaw: heading measurement noise std (rad).
        max_yaw_rate: safety clamp (rad/s) on the returned yaw-rate
            estimate, same rationale as the CV filter's max_speed clamp
            -- a single bad orientation reading shouldn't be able to
            inject a huge yaw-rate into downstream logic.
        """
        self.n_x = 5
        self.n_aug = 7

        self.std_a = float(std_a)
        self.std_yawdd = float(std_yawdd)
        self.std_pos = float(std_pos)
        self.std_yaw = float(std_yaw)

        self.initial_pos_std = float(initial_pos_std)
        self.initial_v_std = float(initial_v_std)
        self.initial_yaw_std = float(initial_yaw_std)
        self.initial_yawd_std = float(initial_yawd_std)

        self.max_yaw_rate = float(max_yaw_rate)

        self.lambda_ = 3.0 - self.n_aug
        n_sig = 2 * self.n_aug + 1
        self.weights = np.full(n_sig, 1.0 / (2.0 * (self.lambda_ + self.n_aug)))
        self.weights[0] = self.lambda_ / (self.lambda_ + self.n_aug)

        self.x = np.zeros(self.n_x, dtype=float)
        self.P = np.eye(self.n_x, dtype=float)
        self.Xsig_pred = np.zeros((self.n_x, n_sig), dtype=float)

        self.initialized = False

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def reset(self, px: float, py: float, yaw: float = 0.0):
        self.x = np.array([px, py, 0.0, _wrap_to_pi(yaw), 0.0], dtype=float)
        self.P = np.diag([
            self.initial_pos_std ** 2,
            self.initial_pos_std ** 2,
            self.initial_v_std ** 2,
            self.initial_yaw_std ** 2,
            self.initial_yawd_std ** 2,
        ])
        self.initialized = True

    # ------------------------------------------------------------
    # Sigma points
    # ------------------------------------------------------------
    def _generate_augmented_sigma_points(self):
        x_aug = np.zeros(self.n_aug, dtype=float)
        x_aug[: self.n_x] = self.x

        P_aug = np.zeros((self.n_aug, self.n_aug), dtype=float)
        P_aug[: self.n_x, : self.n_x] = self.P
        P_aug[self.n_x, self.n_x] = self.std_a ** 2
        P_aug[self.n_x + 1, self.n_x + 1] = self.std_yawdd ** 2

        # symmetrize + tiny jitter for numerical stability before cholesky
        P_aug = 0.5 * (P_aug + P_aug.T) + np.eye(self.n_aug) * 1e-9
        L = np.linalg.cholesky(P_aug)

        n_sig = 2 * self.n_aug + 1
        Xsig_aug = np.zeros((self.n_aug, n_sig), dtype=float)
        Xsig_aug[:, 0] = x_aug

        scale = math.sqrt(self.lambda_ + self.n_aug)
        for i in range(self.n_aug):
            Xsig_aug[:, i + 1] = x_aug + scale * L[:, i]
            Xsig_aug[:, i + 1 + self.n_aug] = x_aug - scale * L[:, i]

        return Xsig_aug

    def _predict_sigma_points(self, Xsig_aug, dt: float):
        n_sig = Xsig_aug.shape[1]
        Xsig_pred = np.zeros((self.n_x, n_sig), dtype=float)
        eps = 1e-3

        for i in range(n_sig):
            px, py, v, yaw, yawd, nu_a, nu_yawdd = Xsig_aug[:, i]

            if abs(yawd) > eps:
                px_p = px + (v / yawd) * (math.sin(yaw + yawd * dt) - math.sin(yaw))
                py_p = py + (v / yawd) * (math.cos(yaw) - math.cos(yaw + yawd * dt))
            else:
                # removable singularity at yawd ~ 0 (straight-line motion)
                px_p = px + v * dt * math.cos(yaw)
                py_p = py + v * dt * math.sin(yaw)

            v_p = v
            yaw_p = yaw + yawd * dt
            yawd_p = yawd

            # process noise
            px_p += 0.5 * dt * dt * math.cos(yaw) * nu_a
            py_p += 0.5 * dt * dt * math.sin(yaw) * nu_a
            v_p += dt * nu_a
            yaw_p += 0.5 * dt * dt * nu_yawdd
            yawd_p += dt * nu_yawdd

            Xsig_pred[:, i] = [px_p, py_p, v_p, _wrap_to_pi(yaw_p), yawd_p]

        return Xsig_pred

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    def predict(self, dt: float):
        if not self.initialized or dt <= 0.0:
            return

        Xsig_aug = self._generate_augmented_sigma_points()
        self.Xsig_pred = self._predict_sigma_points(Xsig_aug, dt)

        x = np.zeros(self.n_x, dtype=float)
        for i in range(self.Xsig_pred.shape[1]):
            x += self.weights[i] * self.Xsig_pred[:, i]
        x[3] = _wrap_to_pi(x[3])

        P = np.zeros((self.n_x, self.n_x), dtype=float)
        for i in range(self.Xsig_pred.shape[1]):
            diff = self.Xsig_pred[:, i] - x
            diff[3] = _wrap_to_pi(diff[3])
            P += self.weights[i] * np.outer(diff, diff)

        self.x = x
        self.P = P

    # ------------------------------------------------------------
    # Update: position only (2D measurement)
    # ------------------------------------------------------------
    def update_position(self, px_meas: float, py_meas: float):
        if not self.initialized:
            self.reset(px_meas, py_meas)
            return

        n_z = 2
        Zsig = self.Xsig_pred[0:2, :]

        z_pred = np.zeros(n_z, dtype=float)
        for i in range(Zsig.shape[1]):
            z_pred += self.weights[i] * Zsig[:, i]

        R = np.diag([self.std_pos ** 2, self.std_pos ** 2])
        self._do_update(Zsig, z_pred, R, np.array([px_meas, py_meas]), angle_row=None)

    # ------------------------------------------------------------
    # Update: position + heading (3D measurement)
    # ------------------------------------------------------------
    def update_position_heading(self, px_meas: float, py_meas: float, yaw_meas: float):
        if not self.initialized:
            self.reset(px_meas, py_meas, yaw_meas)
            return

        n_z = 3
        Zsig = np.zeros((n_z, self.Xsig_pred.shape[1]), dtype=float)
        Zsig[0, :] = self.Xsig_pred[0, :]
        Zsig[1, :] = self.Xsig_pred[1, :]
        Zsig[2, :] = self.Xsig_pred[3, :]  # yaw row

        z_pred = np.zeros(n_z, dtype=float)
        for i in range(Zsig.shape[1]):
            z_pred[0] += self.weights[i] * Zsig[0, i]
            z_pred[1] += self.weights[i] * Zsig[1, i]
        # circular mean for the angle component
        sin_sum = np.sum(self.weights * np.sin(Zsig[2, :]))
        cos_sum = np.sum(self.weights * np.cos(Zsig[2, :]))
        z_pred[2] = math.atan2(sin_sum, cos_sum)

        R = np.diag([self.std_pos ** 2, self.std_pos ** 2, self.std_yaw ** 2])
        self._do_update(
            Zsig, z_pred, R, np.array([px_meas, py_meas, _wrap_to_pi(yaw_meas)]), angle_row=2
        )

    def _do_update(self, Zsig, z_pred, R, z_meas, angle_row):
        n_z = Zsig.shape[0]
        n_sig = Zsig.shape[1]

        S = np.zeros((n_z, n_z), dtype=float)
        Tc = np.zeros((self.n_x, n_z), dtype=float)

        for i in range(n_sig):
            z_diff = Zsig[:, i] - z_pred
            if angle_row is not None:
                z_diff[angle_row] = _wrap_to_pi(z_diff[angle_row])
            S += self.weights[i] * np.outer(z_diff, z_diff)

            x_diff = self.Xsig_pred[:, i] - self.x
            x_diff[3] = _wrap_to_pi(x_diff[3])

            Tc += self.weights[i] * np.outer(x_diff, z_diff)

        S = S + R

        K = Tc @ np.linalg.inv(S)

        z_diff = z_meas - z_pred
        if angle_row is not None:
            z_diff[angle_row] = _wrap_to_pi(z_diff[angle_row])

        self.x = self.x + K @ z_diff
        self.x[3] = _wrap_to_pi(self.x[3])
        self.P = self.P - K @ S @ K.T

    # ------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------
    def get_position(self):
        return float(self.x[0]), float(self.x[1])

    def get_speed(self) -> float:
        return float(self.x[2])

    def get_heading(self) -> float:
        return float(self.x[3])

    def get_yaw_rate(self) -> float:
        yawd = float(self.x[4])
        return max(-self.max_yaw_rate, min(self.max_yaw_rate, yawd))
