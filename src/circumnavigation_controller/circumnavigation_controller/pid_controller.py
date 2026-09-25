#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float


class PIDRelativeController:
    """Pure position-error PID -> MAVROS velocity setpoint (world ENU).

    Inputs are position/yaw errors plus the target's velocity as feedforward;
    output is vx, vy, vz, yaw_rate. Shaped so the command stays something the
    multicopter can actually follow:
      - dt is measured, not assumed;
      - the derivative is low-pass filtered and skipped on the first update
        after a reset, so a step in the setpoint (target estimate jump, shot
        change, tracking start) no longer gives a one-cycle kick of
        kd/dt * step;
      - horizontal speed is limited as a vector to speed_margin_xy + target
        speed (capped at max_speed_xy);
      - the change in commanded velocity is limited to max_accel_* per second,
        close to what ArduCopter can do, so the P/I terms don't build up
        commands the drone can't follow (overshoot, ringing);
      - the horizontal integral is frozen while the output is speed-limited,
        so it can't wind up during transients the drone can't follow anyway.
    """

    def __init__(
        self,
        nominal_dt: float = 0.15,
        speed_margin_xy: float = 8.0,
        max_speed_xy: float = 20.0,
        max_speed_z: float = 3.0,
        max_accel_xy: float = 5.0,
        max_accel_z: float = 2.0,
        derivative_tau: float = 0.3,
        kp_xy: float = 0.80,
        ki_xy: float = 0.02,
        max_integral_xy: float = 3.0,
    ):
        self.nominal_dt = nominal_dt
        self.min_dt = 0.02
        self.max_dt = 0.5

        self.kp_x, self.ki_x, self.kd_x = kp_xy, ki_xy, 0.30
        self.kp_y, self.ki_y, self.kd_y = kp_xy, ki_xy, 0.30
        self.kp_z, self.ki_z, self.kd_z = 0.60, 0.03, 0.10
        self.kp_yaw = 0.60

        self.speed_margin_xy = speed_margin_xy
        self.max_speed_xy = max_speed_xy
        self.max_speed_z = max_speed_z
        self.max_accel_xy = max_accel_xy
        self.max_accel_z = max_accel_z
        self.max_yaw_rate = 4.0
        self.derivative_tau = derivative_tau

        self.max_integral_xy = max_integral_xy
        self.max_integral_z = 2.0

        self.reset()

    def reset(self, current_velocity=(0.0, 0.0, 0.0)):
        """Clear integrators/derivative; current_velocity seeds the accel limit."""
        self.int_ex = 0.0
        self.int_ey = 0.0
        self.int_ez = 0.0
        self.prev_e: Optional[tuple] = None
        self.d_filt = (0.0, 0.0, 0.0)
        self.prev_time: Optional[float] = None
        self.prev_cmd = tuple(float(v) for v in current_velocity)
        self.speed_limited = False

    @staticmethod
    def clamp(value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    def update(
        self,
        ex: float,
        ey: float,
        ez: float,
        eyaw: float,
        now: float,
        ff_vx: float = 0.0,
        ff_vy: float = 0.0,
        ff_vz: float = 0.0,
    ) -> PIDCommand:
        """ff_*: world-frame target velocity (m/s), e.g. from the target KF,
        added so the drone matches the target's motion instead of only
        reacting to lag-induced position error."""
        if self.prev_time is None:
            dt = self.nominal_dt
        else:
            dt = min(max(now - self.prev_time, self.min_dt), self.max_dt)
        self.prev_time = now

        # ---- I (horizontal frozen while the last output was speed-limited) ----
        if not self.speed_limited:
            self.int_ex = self.clamp(self.int_ex + ex * dt, self.max_integral_xy)
            self.int_ey = self.clamp(self.int_ey + ey * dt, self.max_integral_xy)
        self.int_ez = self.clamp(self.int_ez + ez * dt, self.max_integral_z)

        # ---- D: low-pass filtered error rate, none on the first update ----
        if self.prev_e is None:
            raw_d = (0.0, 0.0, 0.0)
        else:
            raw_d = tuple((e - p) / dt for e, p in zip((ex, ey, ez), self.prev_e))
        alpha = dt / (self.derivative_tau + dt) if self.derivative_tau > 0 else 1.0
        self.d_filt = tuple(d + alpha * (r - d) for d, r in zip(self.d_filt, raw_d))
        self.prev_e = (ex, ey, ez)
        dex, dey, dez = self.d_filt

        vx = self.kp_x * ex + self.ki_x * self.int_ex + self.kd_x * dex + ff_vx
        vy = self.kp_y * ey + self.ki_y * self.int_ey + self.kd_y * dey + ff_vy
        vz = self.kp_z * ez + self.ki_z * self.int_ez + self.kd_z * dez + ff_vz
        yaw_rate = self.clamp(self.kp_yaw * eyaw, self.max_yaw_rate)

        # ---- speed limit (horizontal as a vector) ----
        speed_limit = min(self.max_speed_xy, self.speed_margin_xy + math.hypot(ff_vx, ff_vy))
        speed = math.hypot(vx, vy)
        self.speed_limited = speed > speed_limit
        if self.speed_limited:
            vx, vy = vx * speed_limit / speed, vy * speed_limit / speed
        vz = self.clamp(vz, self.max_speed_z)

        # ---- acceleration limit on the command ----
        pvx, pvy, pvz = self.prev_cmd
        dvx, dvy = vx - pvx, vy - pvy
        dv = math.hypot(dvx, dvy)
        max_dv = self.max_accel_xy * dt
        if dv > max_dv:
            vx, vy = pvx + dvx * max_dv / dv, pvy + dvy * max_dv / dv
        vz = pvz + self.clamp(vz - pvz, self.max_accel_z * dt)
        self.prev_cmd = (vx, vy, vz)

        return PIDCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
