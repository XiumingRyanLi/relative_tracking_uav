#!/usr/bin/env python3
"""Glue between the target estimate / shot planner and DroneMPC (no ROS).

Each control cycle:
  1. stage_times(now): the plant times the MPC stages stand for. The drone
     only reacts to a command ~0.8 s after it is sent, so the plan starts
     one delay ahead: t_k = now + delay + k * dt.
  2. The caller predicts the car (position, heading, position std) and asks
     the planner for the shot offset at those times.
  3. command(...): predicts the drone ahead by the delay with the commands
     already sent, builds the per-stage parameters (shot point, radial
     direction, band scales, uncertainty and yaw weights) and solves.
The returned command is recorded as sent.
"""
import math
from collections import deque

import numpy as np

try:
    from . import drone_mpc as dm
except ImportError:
    import drone_mpc as dm


class MpcTracker:
    def __init__(
        self,
        mpc,
        delays=(0.8, 0.7, 0.25),
        taus=(0.45, 0.3, 0.3),
        r_min: float = 10.0,
        band_radial: float = 5.0,
        band_angle_deg: float = 10.0,
        band_tangential_floor: float = 2.0,
        sigma0: float = 5.0,
        yaw_hold_radius: float = 5.0,
        brake_decel: float = 4.0,
        car_accel: float = 2.0,
        car_max_speed: float = 15.0,
        scenario_window_sec: float = 1.0,
    ):
        self.mpc = mpc
        self.delays = tuple(delays)
        self.taus = tuple(taus)
        self.r_min = r_min
        self.band_radial = band_radial
        self.band_angle = math.radians(band_angle_deg)
        self.band_tangential_floor = band_tangential_floor
        self.sigma0 = sigma0
        self.yaw_hold_radius = yaw_hold_radius
        self.brake_decel = brake_decel
        self.car_accel = car_accel
        self.car_max_speed = car_max_speed
        self.scenario_window_sec = scenario_window_sec
        self.sent = deque()        # (t_sent, [ux uy uz ur])
        self.last_info = None

    @property
    def delay(self) -> float:
        return max(self.delays)

    def stage_times(self, now: float):
        return now + self.delay + self.mpc.dt * np.arange(self.mpc.N + 1)

    def reset(self, now: float = None, current_command=None):
        """After a mode switch: forget the warm start; optionally seed the
        command history with what the drone is doing now."""
        self.mpc.reset()
        self.sent.clear()
        if now is not None and current_command is not None:
            self.sent.append((now, np.asarray(current_command, dtype=float)))

    def record(self, now: float, command):
        """Remember a command sent by someone else (e.g. the PID during search),
        so the delay prediction stays right when the MPC takes over."""
        self.sent.append((now, np.asarray(command, dtype=float)))
        self._trim(now)

    def _trim(self, now):
        while len(self.sent) > 1 and self.sent[1][0] <= now - self.delay - 0.5:
            self.sent.popleft()

    def scenario_shift(self, speed, tau):
        """How far a car at `speed` that starts braking hard (behind) or
        accelerating (ahead) now would be from the constant-speed prediction
        after tau seconds: (brake_back, accel_ahead) in metres."""
        b, a = max(self.brake_decel, 1e-3), max(self.car_accel, 1e-3)
        t_stop = speed / b
        brake = 0.5 * b * tau * tau if tau < t_stop else speed * tau - speed * speed / (2.0 * b)
        t_sat = max(0.0, self.car_max_speed - speed) / a
        accel = (0.5 * a * tau * tau if tau < t_sat
                 else 0.5 * a * t_sat * t_sat + (self.car_max_speed - speed) * (tau - t_sat))
        return brake, max(0.0, accel)

    def stage_parameters(self, car_xy, car_z, car_psi, sigma, offsets, z_ref, car_speed=0.0, scenarios=None):
        """(N+1, NP) parameters. car_xy (N+1, 2), car_z scalar or (N+1,),
        car_psi (N+1,), sigma (N+1,) car position std, offsets (N+1, 3) shot
        offset in the car frame (+x front, +y left), z_ref (N+1,), car_speed
        for the braking / accelerating scenarios; scenarios: optional
        (brake_xy, accel_xy) arrays from target_prediction.range_scenarios
        (brake while turning), else straight-line shifts along the heading."""
        n1 = self.mpc.N + 1
        P = np.tile(dm.DroneMPC.default_parameters(), (n1, 1))
        car_z = np.broadcast_to(np.asarray(car_z, dtype=float), (n1,))
        for k in range(n1):
            c, s = math.cos(car_psi[k]), math.sin(car_psi[k])
            ox, oy = offsets[k][0], offsets[k][1]
            rx, ry = c * ox - s * oy, s * ox + c * oy          # offset in world
            r = math.hypot(rx, ry)
            dx, dy = (rx / r, ry / r) if r > 0.5 else (c, s)    # overhead: along the car
            P[k, dm.P_REF_X] = car_xy[k][0] + rx
            P[k, dm.P_REF_Y] = car_xy[k][1] + ry
            P[k, dm.P_REF_Z] = z_ref[k]
            P[k, dm.P_CAR_X], P[k, dm.P_CAR_Y], P[k, dm.P_CAR_Z] = car_xy[k][0], car_xy[k][1], car_z[k]
            P[k, dm.P_DIR_X], P[k, dm.P_DIR_Y] = dx, dy
            P[k, dm.P_S_RAD] = self.band_radial
            P[k, dm.P_S_TAN] = max(r * self.band_angle, self.band_tangential_floor)
            P[k, dm.P_W_POS] = 1.0 / (1.0 + (sigma[k] / self.sigma0) ** 2)
            P[k, dm.P_W_YAW] = 1.0 if r >= self.yaw_hold_radius else 0.0
            P[k, dm.P_R_MIN] = self.r_min
            P[k, dm.P_TAU_XY], P[k, dm.P_TAU_Z], P[k, dm.P_TAU_R] = self.taus
            # Brake / accelerate scenarios, over the first scenario_window_sec
            # of the plan (the MPC re-plans every cycle, so it only has to be
            # safe for what it can't react to in time).
            if k * self.mpc.dt <= self.scenario_window_sec:
                if scenarios is not None:
                    (P[k, dm.P_BRK_X], P[k, dm.P_BRK_Y]), (P[k, dm.P_ACC_X], P[k, dm.P_ACC_Y]) = \
                        scenarios[0][k], scenarios[1][k]
                else:
                    back, ahead = self.scenario_shift(car_speed, self.delay + k * self.mpc.dt)
                    P[k, dm.P_BRK_X], P[k, dm.P_BRK_Y] = car_xy[k][0] - back * c, car_xy[k][1] - back * s
                    P[k, dm.P_ACC_X], P[k, dm.P_ACC_Y] = car_xy[k][0] + ahead * c, car_xy[k][1] + ahead * s
                P[k, dm.P_R_MIN_SCEN] = self.r_min
            else:
                P[k, dm.P_BRK_X], P[k, dm.P_BRK_Y] = car_xy[k][0], car_xy[k][1]
                P[k, dm.P_ACC_X], P[k, dm.P_ACC_Y] = car_xy[k][0], car_xy[k][1]
                P[k, dm.P_R_MIN_SCEN] = 0.0
        return P

    def command(self, now, x_meas, car_xy, car_z, car_psi, sigma, offsets, z_ref, car_speed=0.0,
                scenarios=None):
        """x_meas: [px py pz vx vy vz psi r] now. Returns ([ux uy uz ur], info)."""
        x0, _ = dm.propagate_delay(x_meas, list(self.sent), now, self.delays, self.taus)
        P = self.stage_parameters(car_xy, car_z, car_psi, sigma, offsets, z_ref, car_speed, scenarios)
        cmd, info = self.mpc.solve(x0, P)
        info["x0"] = x0
        info["params"] = P
        self.sent.append((now, cmd.copy()))
        self._trim(now)
        self.last_info = info
        return cmd, info
