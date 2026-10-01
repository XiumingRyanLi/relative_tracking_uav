#!/usr/bin/env python3
"""World-frame target estimate from detector measurements (pure, no ROS).

Each accepted detection (already transformed into the world frame, see
camera_frames.py) goes through:
  1. gating  - implausible range, or too far from the KF prediction;
  2. a constant-velocity linear KF on position -> position + velocity;
  3. heading - a jump-gated circular EMA of the detector's heading plus a
     yaw rate from its smoothed derivative, or optionally a CTRV UKF.
With use_ctra (target_filter=ctra, the default) the CV KF is not used: a
heading-aided CTRA EKF (target_ctra_ekf.py) gates detections on its own
prediction (chi-square), and gives x, y, velocity, acceleration, heading and
yaw rate; see _update_ctra_pipeline.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .geometry import angle_diff, smooth_angle, wrap_to_pi
    from .target_ctra_ekf import TargetCTRAEKF
    from .target_ctrv_ukf import TargetCTRVUKF
    from .target_kalman_filter import TargetKalmanFilter
except ImportError:
    from geometry import angle_diff, smooth_angle, wrap_to_pi
    from target_ctra_ekf import TargetCTRAEKF
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
    # Time the target estimate was last updated, if not this frame's (a
    # steep-view detection that only steers the gimbal, see the node's
    # steep_skip_elevation_deg); predictions run from here.
    state_stamp: Optional[float] = None


class TargetEstimator:
    STARTUP_VOTES = 5
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
        heading_course_min_speed: float,
        heading_flip_threshold: float,
        heading_course_memory_sec: float,
        enable_heading_ukf: bool,
        use_ctra: bool = False,
        ctra_gate_nis: float = 25.0,
        ctra_reacquire_sec: float = 5.0,
        course_window_sec: float = 1.0,
        z_alpha: float = 0.3,
        ctra_s_wdot: float = 0.2,
        ctra_s_jerk: float = 0.5,
        ctra_heading_std: float = math.radians(4.0),
        ctra_estimate_accel: bool = False,
        ctra_speed_noise: float = 2.0,
        ctra_max_reverse_speed: float = 8.0,
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
        self.heading_course_min_speed = heading_course_min_speed
        self.heading_flip_threshold = heading_flip_threshold
        self.heading_course_memory_sec = heading_course_memory_sec
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
        self.use_ctra = use_ctra
        self.ctra_gate_nis = ctra_gate_nis
        self.ctra_reacquire_sec = ctra_reacquire_sec
        self.course_window_sec = course_window_sec
        self.z_alpha = z_alpha
        self._course_window = deque()   # (stamp, x, y) of accepted raw positions
        self.ctra = TargetCTRAEKF(
            s_wdot=ctra_s_wdot, s_jerk=ctra_s_jerk, heading_std=ctra_heading_std,
            max_speed=kf_max_target_speed, max_turn_rate=max_yaw_rate,
            estimate_accel=ctra_estimate_accel, s_speed=ctra_speed_noise,
            max_reverse_speed=ctra_max_reverse_speed,
        )
        self._ctra_last_update_time = None
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
        self.heading_flips = 0          # DOPE headings turned round by the course check
        self._startup_votes = None      # after a CTRA (re)start: did the next headings disagree?
        self._ctra_start_time = None    # last (re)start / turn-round of the CTRA filter
        self._reverse_pinned = 0        # consecutive updates at the reverse-speed limit
        self._course = None             # (direction of travel rad, stamp) at speed

        # ---- Estimate (world ENU) ----
        self.x, self.y, self.z = 0.0, 3.0, 0.0
        # Velocity (m/s) from the KF, used as PID feedforward. Zero until the
        # filter has seen at least two measurements.
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.heading = 0.0
        # World-frame acceleration (m/s^2); only estimated by the CTRA EKF.
        self.ax, self.ay = 0.0, 0.0
        # Yaw rate (rad/s), only estimated by the UKF.
        self.yaw_rate = 0.0

    @property
    def reject_count(self) -> int:
        return self._reject_count

    @property
    def last_update_stamp(self) -> Optional[float]:
        """Capture time of the last detection folded into the CTRA filter."""
        return self._ctra_last_update_time if self.use_ctra and self.ctra.initialized else None

    def raw_heading(self, T_world_target: np.ndarray) -> float:
        """World yaw of the detector frame's heading axis."""
        axis = T_world_target[:3, self.heading_axis]
        return math.atan2(axis[1], axis[0])

    def update(self, T_world_target: np.ndarray, distance: float, stamp_sec: float,
               use_heading: bool = True, pos_noise_scale: float = 1.0) -> Optional[str]:
        """Fold in one detection. Returns None if accepted, else why it was rejected.

        stamp_sec is the frame's capture time (not reception time), so detection
        latency jitter doesn't show up in the velocity as spurious acceleration.
        use_heading=False (CTRA filter): position only, the detector's heading
        is not used -- e.g. looking almost straight down, where DOPE's heading
        was 15-27 deg off and once flipped (A1 overpass runs, 2026-10-01).
        Ignored when the filter has to (re)start: it needs some heading.
        pos_noise_scale: multiplies the detection's position std (steep views:
        the A1 overpass estimates were ~3 m off above 80 deg).
        """
        meas = T_world_target[:3, 3].copy()
        if self.use_ctra:
            return self._update_ctra_pipeline(T_world_target, meas, distance, stamp_sec, use_heading,
                                              pos_noise_scale)
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
        # (Checked against the KF velocity, which doesn't depend on the
        # heading, so a flipped heading can't confirm itself.)
        raw_heading = self._resolve_heading_flip(raw_heading, stamp_sec, (self.vx, self.vy))
        if not self.enable_heading_ukf:
            self._update_heading_ema(raw_heading, kf_fresh, reseed, stamp_sec)
        else:
            self._update_heading_ukf(raw_heading, distance, stamp_sec)
        return None

    def _resolve_heading_flip(self, raw_heading, stamp_sec, course_velocity):
        """DOPE sometimes reads the car nose-to-tail (~180 deg off, mostly at
        long range). A moving car drives the way it points, so a heading
        more than heading_flip_threshold from the direction of travel is
        turned round. In runs 20260928_134234/145616/150220/142126 flipped
        readings were >= 127 deg off the course and correct ones <= 86 deg.

        The direction of travel is remembered for heading_course_memory_sec:
        a KF re-seed zeroes the velocity, and right after one the old gate
        let a flipped heading straight in (run 20260928_145616, t=91 s)."""
        vx, vy = course_velocity
        if math.hypot(vx, vy) >= self.heading_course_min_speed:
            self._course = (math.atan2(vy, vx), stamp_sec)
        if self._course is None or stamp_sec - self._course[1] > self.heading_course_memory_sec:
            return raw_heading
        if abs(angle_diff(raw_heading, self._course[0])) > self.heading_flip_threshold:
            self.heading_flips += 1
            return wrap_to_pi(raw_heading + math.pi)
        return raw_heading

    def _update_ctra_pipeline(self, T_world_target, meas, distance, stamp_sec, use_heading=True,
                              pos_noise_scale=1.0) -> Optional[str]:
        """target_filter=ctra: the CTRA EKF alone, no CV KF.

        1. gate: range, then a chi-square gate on the EKF's own predicted
           position and its uncertainty (NIS > ctra_gate_nis is rejected).
           It widens by itself after gaps and in corners, where a fixed
           distance from a straight-line prediction rejected good frames;
        Gaps up to ctra_reacquire_sec are predicted through (not restarted).
        2. direction of travel for the flip check from a straight-line fit
           through the last course_window_sec of accepted raw positions --
           independent of the heading, so a flipped heading can't confirm
           itself through the EKF's velocity;
        3. CTRA EKF update; z is a light EMA of the measured z.
        """
        if not (self.min_distance <= distance <= self.max_distance):
            self._reject_count += 1
            return f"range {distance:.1f} m outside [{self.min_distance:.1f}, {self.max_distance:.1f}]"
        ekf = self.ctra
        r_std = (self.kf.r_std + self.noise_per_m * distance) * max(pos_noise_scale, 1.0)
        last = self._ctra_last_update_time
        gap = stamp_sec - last if last is not None else math.inf
        # Short gaps (the car behind a tree): predict through them and gate
        # the new detection against the prediction -- its uncertainty has
        # grown with the gap, so the gate is wider by itself -- rather than
        # restarting the filter and re-learning speed and heading.
        fresh = ekf.initialized and 0.0 <= gap <= self.ctra_reacquire_sec
        if fresh and self._reject_count < self.max_consecutive_rejects:
            nis = ekf.position_nis(float(meas[0]), float(meas[1]), r_std, stamp_sec - last)
            if nis > self.ctra_gate_nis:
                self._reject_count += 1
                return f"NIS {nis:.1f} > gate {self.ctra_gate_nis:.1f} (EKF prediction)"
        # After max_consecutive_rejects drops in a row, accept and re-seed so
        # a filter that latched onto a bad value cannot lock the target out.
        reseed = self._reject_count >= self.max_consecutive_rejects
        self._reject_count = 0

        window = self._course_window
        if not fresh or reseed or gap > self.kf_coast_timeout_sec:
            window.clear()
        window.append((stamp_sec, float(meas[0]), float(meas[1])))
        while stamp_sec - window[0][0] > self.course_window_sec:
            window.popleft()
        course_velocity = (0.0, 0.0)
        if len(window) >= 4 and window[-1][0] - window[0][0] >= 0.4 * self.course_window_sec:
            w = np.array(window)
            t = w[:, 0] - w[:, 0].mean()
            denom = float(t @ t)
            course_velocity = (float(t @ w[:, 1]) / denom, float(t @ w[:, 2]) / denom)

        if use_heading or not (fresh and not reseed and ekf.initialized):
            raw_heading = wrap_to_pi(self.raw_heading(T_world_target))
            raw_heading = self._resolve_flip_ctra(raw_heading, stamp_sec, gap, reseed, course_velocity)
        else:
            raw_heading = None          # position-only update; flip check / start-up votes untouched
        self._update_ctra(meas, r_std, raw_heading, fresh and not reseed, stamp_sec)
        self.z = float(meas[2]) if not fresh else self.z + self.z_alpha * (float(meas[2]) - self.z)
        self.vz = 0.0
        if not self.use_filtered_position:
            self.x, self.y = float(meas[0]), float(meas[1])
        return None

    def _resolve_flip_ctra(self, raw_heading, stamp_sec, gap, reseed, course_velocity):
        """Nose-to-tail flip check for the CTRA/CTRV filter, by CONTINUITY.

        A DOPE flip is a sudden ~180 deg jump from one frame to the next; a
        reversing car keeps its heading (only the speed goes through zero and
        negative). So while the filter is running (last update <= 3 s ago)
        a flip is a heading more than heading_flip_threshold from the
        filter's own heading predicted to this frame. Checking against the
        direction of travel instead (the old check) turned every correct
        heading of a reversing car round (123 times in a 6 s reverse, the
        shot then flipped to the other side of the car).
        Only on a fresh start / after a longer gap, with no heading to be
        continuous with, fall back to the direction of travel -- assuming the
        car drives forward, as it almost always does."""
        ekf = self.ctra
        if ekf.initialized and not reseed and 0.0 <= gap <= self.heading_course_memory_sec:
            predicted = ekf.heading + ekf.turn_rate * gap
            disagrees = abs(angle_diff(raw_heading, predicted)) > self.heading_flip_threshold
            # Right after a (re)start the filter's heading is a single
            # detection's: if that one was the flip, the next ones all
            # "disagree". DOPE flips are rare (~3 %), so a majority of the
            # first STARTUP_VOTES disagreeing means the start was wrong:
            # turn the filter round instead of the measurements.
            if self._startup_votes is not None:
                self._startup_votes.append(disagrees)
                if len(self._startup_votes) >= self.STARTUP_VOTES:
                    if sum(self._startup_votes) > self.STARTUP_VOTES // 2:
                        ekf.turn_round()
                        self._ctra_start_time = stamp_sec     # let it settle before the speed guard
                        disagrees = False
                    self._startup_votes = None
            if disagrees:
                self.heading_flips += 1
                return wrap_to_pi(raw_heading + math.pi)
            return raw_heading
        self._startup_votes = []          # a (re)start follows: vote on its heading
        return self._resolve_heading_flip(raw_heading, stamp_sec, course_velocity)

    def _update_ctra(self, meas, pos_std, raw_heading, fresh, stamp_sec):
        """Heading-aided CTRA EKF -> x, y, velocity, acceleration, heading,
        yaw rate. A heading far off the prediction (a bad frame DOPE didn't
        flip cleanly) updates the position only; if that persists, the
        filter is the one that is wrong and restarts on the measurement."""
        ekf = self.ctra
        if not fresh or not ekf.initialized:
            # Carry the speed across a short loss (KF re-seed zeroes its own).
            recent = (self._ctra_last_update_time is not None
                      and stamp_sec - self._ctra_last_update_time <= self.heading_course_memory_sec)
            speed = ekf.speed if (recent and ekf.initialized) else 0.0
            ekf.reset(float(meas[0]), float(meas[1]), raw_heading, speed,
                      speed_std=3.0 if speed != 0.0 else None)
            self._heading_reject_count = 0
        else:
            ekf.predict(stamp_sec - self._ctra_last_update_time)
            psi = raw_heading
            if raw_heading is None:
                pass                                   # position only (use_heading=False)
            elif abs(angle_diff(raw_heading, ekf.heading)) > self.max_heading_jump:
                self._heading_reject_count += 1
                psi = None
            else:
                self._heading_reject_count = 0
            if self._heading_reject_count > self.max_consecutive_rejects:
                ekf.reset(float(meas[0]), float(meas[1]), raw_heading, ekf.speed, speed_std=3.0)
                self._heading_reject_count = 0
            else:
                # Position-only (use_heading=False): keep the heading as it was.
                ekf.update(float(meas[0]), float(meas[1]), pos_std, psi, hold_heading=raw_heading is None)
        # A filter pinned at the reverse-speed limit is a forward-driving car
        # with its heading 180 deg wrong: turn it round. Only once it has
        # settled (1 s after a (re)start or turn-round) and stays pinned for
        # 3 updates: right after a start the speed estimate overshoots, and
        # the guard undid a correct start-up vote. A new start-up vote
        # follows (earlier votes were against the old heading).
        if not fresh or self._ctra_start_time is None:
            self._ctra_start_time = stamp_sec
        # Only with a heading measurement: on position-only (steep view)
        # updates a pinned speed is position noise, not a flipped heading
        # (A1 overpass rep02 re-run: the guard turned a parked car round).
        pinned = raw_heading is not None and ekf.speed <= -(ekf.max_reverse_speed - 0.5)
        self._reverse_pinned = self._reverse_pinned + 1 if pinned else 0
        if self._reverse_pinned >= 3 and stamp_sec - self._ctra_start_time > 1.0:
            ekf.turn_round()
            self._reverse_pinned = 0
            self._ctra_start_time = stamp_sec
            self._startup_votes = []
        self._ctra_last_update_time = stamp_sec

        if self.use_filtered_position:
            self.x, self.y = ekf.x, ekf.y
        self.vx, self.vy = ekf.velocity()
        self.ax, self.ay = ekf.acceleration()
        self.heading = ekf.heading
        # Same low-speed fade as the EMA path: a car can't turn in place.
        speed = abs(ekf.speed)
        if self.yaw_rate_min_speed > 0.0:
            fade = min(1.0, max(0.0, (speed - self.yaw_rate_min_speed) / self.yaw_rate_min_speed))
        else:
            fade = 1.0
        self.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, ekf.turn_rate * fade))

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
