#!/usr/bin/env python3
"""World-frame target estimate from detector measurements (pure, no ROS).

Each accepted detection (already transformed into the world frame, see
camera_frames.py) goes through:
  1. gating  - implausible range, or too far from the KF prediction;
  2. a constant-velocity linear KF on position -> position + velocity;
  3. heading - a jump-gated circular EMA of the detector's heading plus a
     yaw rate from its smoothed derivative, or optionally a CTRV UKF.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .geometry import angle_diff, smooth_angle, wrap_to_pi
    from .target_ctrv_ukf import TargetCTRVUKF
    from .target_kalman_filter import TargetKalmanFilter
except ImportError:
    from geometry import angle_diff, smooth_angle, wrap_to_pi
    from target_ctrv_ukf import TargetCTRVUKF
    from target_kalman_filter import TargetKalmanFilter


@dataclass
class Detection:
    """Last accepted detection, as the detector saw it (for control + logging)."""
    cam_x: float            # camera optical frame (m)
    cam_y: float
    cam_z: float
    distance: float         # straight-line camera -> target (m)
    world_yaw: float        # target heading in world ENU from its orientation (rad)
    stamp_sec: float        # frame capture time
    received_time: float    # node time the detection was accepted


class TargetEstimator:
    def __init__(
        self,
        *,
        heading_axis: int,
        # position gating
        min_distance: float,
        max_distance: float,
        max_position_jump: float,
        jump_per_m: float,
        max_consecutive_rejects: int,
        # position KF
        noise_per_m: float,
        use_filtered_position: bool,
        kf_coast_timeout_sec: float,
        kf_accel_noise_std: float,
        kf_measurement_noise_std: float,
        kf_initial_pos_std: float,
        kf_initial_vel_std: float,
        kf_max_target_speed: float,
        kf_enable_adaptive_q: bool,
        kf_adaptive_q_max_scale: float,
        kf_adaptive_q_decay: float,
        # heading
        heading_alpha: float,
        yaw_rate_tau_sec: float,
        yaw_rate_min_speed: float,
        max_yaw_rate: float,
        max_heading_jump: float,
        max_heading_distance: float,
        enable_heading_filter: bool,
        enable_heading_ukf: bool,
        ukf_coast_timeout_sec: float,
        ukf_std_a: float,
        ukf_std_yawdd: float,
        ukf_std_pos: float,
        ukf_std_yaw: float,
        ukf_max_yaw_rate: float,
    ):
        self.heading_axis = heading_axis
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_position_jump = max_position_jump
        self.jump_per_m = jump_per_m
        self.max_consecutive_rejects = max_consecutive_rejects
        self.noise_per_m = noise_per_m
        self.use_filtered_position = use_filtered_position
        self.kf_coast_timeout_sec = kf_coast_timeout_sec
        self.heading_alpha = heading_alpha
        self.yaw_rate_tau_sec = yaw_rate_tau_sec
        self.yaw_rate_min_speed = yaw_rate_min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_heading_jump = max_heading_jump
        self.max_heading_distance = max_heading_distance
        self.enable_heading_filter = enable_heading_filter
        self.enable_heading_ukf = enable_heading_ukf
        self.ukf_coast_timeout_sec = ukf_coast_timeout_sec

        self.kf = TargetKalmanFilter(
            accel_noise_std=kf_accel_noise_std,
            measurement_noise_std=kf_measurement_noise_std,
            initial_pos_std=kf_initial_pos_std,
            initial_vel_std=kf_initial_vel_std,
            max_speed=kf_max_target_speed,
            enable_adaptive_q=kf_enable_adaptive_q,
            adaptive_q_max_scale=kf_adaptive_q_max_scale,
            adaptive_q_decay=kf_adaptive_q_decay,
        )
        self.ukf = TargetCTRVUKF(
            std_a=ukf_std_a,
            std_yawdd=ukf_std_yawdd,
            std_pos=ukf_std_pos,
            std_yaw=ukf_std_yaw,
            max_yaw_rate=ukf_max_yaw_rate,
        )
        self._kf_last_update_time = None
        self._ukf_last_update_time = None
        self._reject_count = 0
        self._heading_reject_count = 0
        self._heading_valid = False
        self._filtered_heading = 0.0
        self._yaw_rate_lpf = 0.0
        self._prev_heading_stamp = None

        # ---- Estimate (world ENU) ----
        self.x, self.y, self.z = 0.0, 3.0, 0.0
        # Velocity (m/s) from the KF, used as PID feedforward. Zero until the
        # filter has seen at least two measurements.
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.heading = 0.0
        # Yaw rate (rad/s), only estimated by the UKF.
        self.yaw_rate = 0.0

    @property
    def reject_count(self) -> int:
        return self._reject_count

    def raw_heading(self, T_world_target: np.ndarray) -> float:
        """World yaw of the detector frame's heading axis."""
        axis = T_world_target[:3, self.heading_axis]
        return math.atan2(axis[1], axis[0])

    def update(self, T_world_target: np.ndarray, distance: float, stamp_sec: float) -> Optional[str]:
        """Fold in one detection. Returns None if accepted, else why it was rejected.

        stamp_sec is the frame's capture time (not reception time), so detection
        latency jitter doesn't show up in the velocity as spurious acceleration.
        """
        meas = T_world_target[:3, 3].copy()
        kf_fresh = (
            self._kf_last_update_time is not None
            and (stamp_sec - self._kf_last_update_time) <= self.kf_coast_timeout_sec
        )

        # ---- 1. Gate before touching any state ----
        reject_reason = None
        if not (self.min_distance <= distance <= self.max_distance):
            reject_reason = f"range {distance:.1f} m outside " \
                f"[{self.min_distance:.1f}, {self.max_distance:.1f}]"
        elif kf_fresh and self._reject_count < self.max_consecutive_rejects:
            dt_pred = max(0.0, stamp_sec - self._kf_last_update_time)
            predicted = np.array(self.kf.get_position()) + dt_pred * np.array(self.kf.get_velocity())
            jump = float(np.linalg.norm(meas - predicted))
            gate = self.max_position_jump + self.jump_per_m * distance
            if jump > gate:
                reject_reason = f"jump {jump:.1f} m from prediction > gate {gate:.1f} m"

        if reject_reason is not None:
            self._reject_count += 1
            return reject_reason
        # After max_consecutive_rejects drops in a row, accept and re-seed so
        # a filter that latched onto a bad value cannot lock the target out.
        reseed = self._reject_count >= self.max_consecutive_rejects
        self._reject_count = 0

        # ---- 2. Constant-velocity KF on the world position ----
        # The nonlinear camera->world transform already happened, so the
        # filter only sees a noisy xyz point. PnP error grows with range, so
        # the measurement noise does too.
        r_std = self.kf.r_std + self.noise_per_m * distance
        self.kf.R = np.eye(3, dtype=float) * r_std ** 2
        position_meas = tuple(float(v) for v in meas)

        if not kf_fresh or reseed:
            # First measurement, previous one too old to predict from, or the
            # gate kept rejecting -- start clean.
            self.kf.reset(position_meas)
        else:
            self.kf.predict(stamp_sec - self._kf_last_update_time)
            self.kf.update(position_meas)

        self._kf_last_update_time = stamp_sec
        self.vx, self.vy, self.vz = self.kf.get_velocity()
        if self.use_filtered_position:
            self.x, self.y, self.z = self.kf.get_position()
        else:
            self.x, self.y, self.z = position_meas

        # ---- 3. Heading ----
        raw_heading = wrap_to_pi(self.raw_heading(T_world_target))
        if not self.enable_heading_ukf:
            self._update_heading_ema(raw_heading, kf_fresh, reseed, stamp_sec)
        else:
            self._update_heading_ukf(raw_heading, distance, stamp_sec)
        return None

    def _update_heading_ema(self, raw_heading: float, kf_fresh: bool, reseed: bool, stamp_sec: float):
        """Raw heading, or with enable_heading_filter a jump-gated circular EMA
        (DOPE orientation flips by large angles on single frames), plus a yaw
        rate from the low-passed derivative of that heading."""
        prev_heading = self._filtered_heading
        restarted = False
        if not self.enable_heading_filter or not self._heading_valid or not kf_fresh or reseed:
            self._filtered_heading = raw_heading
            self._heading_valid = True
            self._heading_reject_count = 0
            restarted = True
        elif abs(angle_diff(raw_heading, self._filtered_heading)) > self.max_heading_jump:
            # Persistent disagreement means the filter is the one that is
            # wrong: snap to the measurement.
            self._heading_reject_count += 1
            if self._heading_reject_count > self.max_consecutive_rejects:
                self._filtered_heading = raw_heading
                self._heading_reject_count = 0
                restarted = True
        else:
            self._heading_reject_count = 0
            self._filtered_heading = smooth_angle(
                self._filtered_heading, raw_heading, self.heading_alpha
            )
        self.heading = self._filtered_heading

        # Yaw rate: low-passed derivative of the filtered heading (capture-time
        # dt). A car can't turn in place, so it fades to 0 below
        # yaw_rate_min_speed (and is exactly 0 when stopped), which keeps
        # heading noise on a parked car out of the controller's feedforward.
        if restarted or self._prev_heading_stamp is None:
            self._yaw_rate_lpf = 0.0
        else:
            dt = stamp_sec - self._prev_heading_stamp
            if dt > 1e-3:
                raw_rate = angle_diff(self._filtered_heading, prev_heading) / dt
                self._yaw_rate_lpf += dt / (self.yaw_rate_tau_sec + dt) * (raw_rate - self._yaw_rate_lpf)
        self._prev_heading_stamp = stamp_sec
        speed = math.hypot(self.vx, self.vy)
        if self.yaw_rate_min_speed > 0.0:
            fade = min(1.0, max(0.0, (speed - self.yaw_rate_min_speed) / self.yaw_rate_min_speed))
        else:
            fade = 1.0
        self.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self._yaw_rate_lpf * fade))
        self._ukf_last_update_time = None  # forces a clean reset() if re-enabled later

    def _update_heading_ukf(self, raw_heading: float, distance: float, stamp_sec: float):
        """CTRV UKF on position + heading (world frame). The process model is
        nonlinear (yaw inside sin/cos), hence a UKF rather than a linear KF."""
        if (
            self._ukf_last_update_time is None
            or (stamp_sec - self._ukf_last_update_time) > self.ukf_coast_timeout_sec
        ):
            self.ukf.reset(self.x, self.y, raw_heading)
        else:
            self.ukf.predict(stamp_sec - self._ukf_last_update_time)
            # A bad single-frame heading (far away, or a big jump) falls back
            # to a position-only update so it can't corrupt the filter.
            heading_trusted = True
            if self.enable_heading_filter:
                if distance > self.max_heading_distance:
                    heading_trusted = False
                elif abs(wrap_to_pi(raw_heading - self.ukf.get_heading())) > self.max_heading_jump:
                    heading_trusted = False
            if heading_trusted:
                self.ukf.update_position_heading(self.x, self.y, raw_heading)
            else:
                self.ukf.update_position(self.x, self.y)

        self._ukf_last_update_time = stamp_sec
        self._prev_heading_stamp = None  # clean yaw-rate restart if the EMA is used again
        self.heading = wrap_to_pi(self.ukf.get_heading())
        self.yaw_rate = self.ukf.get_yaw_rate()
