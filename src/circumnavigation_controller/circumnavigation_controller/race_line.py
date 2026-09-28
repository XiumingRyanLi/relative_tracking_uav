#!/usr/bin/env python3
"""Racing line + speed profile + pure-pursuit driver for the target car.

Pure logic (numpy only, no ROS), used by race_driver.py. The line comes from
config/race_lines/<world>.csv (tools/build_race_lines.py): a closed loop of
x, y points ~1 m apart in the Gazebo world frame, minimum-curvature inside
the lane.

Speed profile, the usual lap-time-simulator recipe:
  1. corner limit   v <= sqrt(lat_accel / |curvature|), capped at max_speed
  2. forward pass   v[i+1] <= sqrt(v[i]^2 + 2 * accel * ds)   (traction)
  3. backward pass  v[i]   <= sqrt(v[i+1]^2 + 2 * brake * ds) (braking)
with accel/brake scaled down by the grip cornering already uses (friction
circle), so the car is flat out on straights, brakes before corners early
enough to reach the corner speed, and accelerates out as the corner opens.

Steering is pure pursuit on the line, turned into the yaw rate the car's
AckermannSteering plugin takes (/cmd_vel angular.z = actual speed *
curvature).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DriveCommand:
    speed: float        # m/s, /cmd_vel linear.x
    yaw_rate: float     # rad/s, /cmd_vel angular.z
    s: float            # arc length of the nearest line point, m
    target_speed: float  # profile speed at the car, m/s
    cross_track: float  # signed distance from the line (+ = left), m


class RaceLine:
    def __init__(
        self,
        points,
        max_speed: float = 15.0,
        min_speed: float = 3.0,
        lat_accel: float = 2.5,
        accel: float = 2.0,
        brake: float = 4.0,
        curvature_window: float = 3.0,
        speed_zones=(),
    ):
        self.pts = np.asarray(points, dtype=float)[:, :2]
        n = len(self.pts)
        seg = np.linalg.norm(np.roll(self.pts, -1, axis=0) - self.pts, axis=1)
        self.ds = seg                                    # i -> i+1
        self.s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        self.length = float(seg.sum())
        self.n = n
        self.curvature = self._curvature(curvature_window)
        limits = (max_speed, min_speed, lat_accel, accel, brake)
        # Profile without speed zones, for listing the track's corners.
        self.speed_zones = []
        self.natural_speed = self._speed_profile(*limits)
        self.speed_zones = [tuple(map(float, z)) for z in speed_zones]
        self.speed = (self._speed_profile(*limits) if self.speed_zones
                      else self.natural_speed)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(np.loadtxt(path, delimiter=',', comments='#'), **kwargs)

    def _curvature(self, window):
        """Signed curvature (+ = left turn) from the circle through the
        points `window` m behind and ahead, so 1 m sampling noise doesn't
        show up as sharp corners."""
        k = max(1, int(round(window / np.mean(self.ds))))
        a = np.roll(self.pts, k, axis=0)
        b = self.pts
        c = np.roll(self.pts, -k, axis=0)
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        prod = (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1)
                * np.linalg.norm(c - a, axis=1))
        return 2.0 * cross / np.maximum(prod, 1e-9)

    def _speed_profile(self, v_max, v_min, lat_accel, accel, brake):
        kappa = np.abs(self.curvature)
        v = np.minimum(v_max, np.sqrt(lat_accel / np.maximum(kappa, 1e-6)))
        v = np.maximum(v, v_min)
        # Speed zones: (s_start, s_end, max_speed), metres along the lap
        # (s_end < s_start wraps past the start/finish). Applied before the
        # passes, so the car brakes in time for a zone and accelerates out.
        for s0, s1, cap in self.speed_zones:
            inside = ((self.s >= s0) & (self.s <= s1) if s0 <= s1
                      else (self.s >= s0) | (self.s <= s1))
            v[inside] = np.minimum(v[inside], cap)
        n = self.n

        def grip_left(j):
            # Friction circle: only the grip cornering doesn't use is left
            # for accelerating/braking, so the car doesn't go full throttle
            # while still turning hard (the rear-drive car spins in Gazebo).
            use = v[j] ** 2 * kappa[j] / lat_accel
            return math.sqrt(max(0.0, 1.0 - use * use))

        # Closed loop: two laps of each pass so the limits wrap round the
        # start/finish. The floor on grip_left keeps the car from stalling at
        # a corner's speed limit (min_speed corners use more than lat_accel).
        for _ in range(2):
            for i in range(2 * n):
                j, k = i % n, (i + 1) % n
                a = accel * max(0.2, grip_left(j))
                v[k] = min(v[k], math.sqrt(v[j] ** 2 + 2.0 * a * self.ds[j]))
            for i in range(2 * n, 0, -1):
                j, k = i % n, (i - 1) % n
                a = brake * max(0.2, grip_left(j))
                v[k] = min(v[k], math.sqrt(v[j] ** 2 + 2.0 * a * self.ds[k]))
        return v

    def corners(self, window=15.0, max_frac=0.9):
        """Slowest point of each corner: (s, speed, radius) for local speed
        minima (+-window m) below max_frac of the top speed, ignoring speed
        zones."""
        k = max(1, int(round(window / np.mean(self.ds))))
        v = self.natural_speed
        out = []
        for i in range(self.n):
            near = v[np.arange(i - k, i + k + 1) % self.n]
            if v[i] < max_frac * v.max() and v[i] <= near.min() + 1e-9:
                if out and (self.s[i] - out[-1][0]) < window:
                    continue  # same corner (flat minimum)
                radius = 1.0 / max(abs(self.curvature[i]), 1e-6)
                out.append((float(self.s[i]), float(v[i]), float(radius)))
        return out

    def lap_time(self):
        """Lap time if the car followed the profile exactly, s."""
        v_mid = 0.5 * (self.speed + np.roll(self.speed, -1))
        return float(np.sum(self.ds / v_mid))

    def nearest(self, x, y, hint=None, window=30):
        """Index of the closest line point; searches +-window points round
        `hint` (the previous index) so crossing sections of track can't
        make the car jump to another part of the lap."""
        if hint is None:
            idx = np.arange(self.n)
        else:
            idx = np.arange(hint - window, hint + window + 1) % self.n
        d = np.hypot(self.pts[idx, 0] - x, self.pts[idx, 1] - y)
        return int(idx[np.argmin(d)])

    def index_ahead(self, i, dist):
        """Index of the point `dist` m further along the line from i."""
        steps = int(round(dist / np.mean(self.ds)))
        return (i + steps) % self.n


class RaceDriver:
    """Pure pursuit on a RaceLine, commanding the profile speed."""

    def __init__(
        self,
        line: RaceLine,
        lookahead_min: float = 8.0,
        lookahead_gain: float = 0.8,
        lookahead_max: float = 25.0,
        speed_lead_time: float = 0.6,
        min_turn_radius: float = 5.5,
    ):
        self.line = line
        # Lookahead (m) = lookahead_gain * speed, clamped. Long on purpose:
        # the audi_r8 steers slowly (plugin steer_p_gain 1 -> ~1 s steering
        # lag) and wobbles in yaw, and a short lookahead then weaves and
        # hits the wall in chicanes (seen in Gazebo at 5 m / 0.6 s).
        self.lookahead_min = lookahead_min
        self.lookahead_gain = lookahead_gain
        self.lookahead_max = lookahead_max
        # Command the speed the profile wants this far ahead: the car's
        # speed lags the command (jerk/accel limits, odometry delay), so
        # braking starts on time.
        self.speed_lead_time = speed_lead_time
        self.max_curvature = 1.0 / min_turn_radius
        self.min_steer_speed = 0.5
        self.index = None

    def reset(self, x, y):
        self.index = self.line.nearest(x, y)

    def step(self, x, y, yaw, v, v_lat=0.0) -> DriveCommand:
        """x, y, yaw: car pose; v, v_lat: forward and leftward velocity in
        the car frame (odometry twist)."""
        line = self.line
        if self.index is None:
            self.reset(x, y)
        self.index = line.nearest(x, y, hint=self.index)
        i = self.index

        v_now = max(v, 0.0)
        target_speed = float(line.speed[i])
        speed = float(line.speed[line.index_ahead(i, v_now * self.speed_lead_time)])

        # Steer from the direction of travel (course), not where the nose
        # points: the car fishtails at ~1 Hz (body yaw +-0.3 rad/s while its
        # path stays straight), and chasing the nose kept that going as a
        # constant weave on the straights.
        course = yaw + math.atan2(v_lat, v_now) if v_now > 1.0 else yaw

        # Pure pursuit: arc through the car and the point `ld` ahead.
        ld = min(self.lookahead_max,
                 max(self.lookahead_min, self.lookahead_gain * v_now))
        tx, ty = line.pts[line.index_ahead(i, ld)]
        dx, dy = tx - x, ty - y
        alpha = math.atan2(dy, dx) - course
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        dist = max(math.hypot(dx, dy), 1e-3)
        curvature = 2.0 * math.sin(alpha) / dist
        curvature = max(-self.max_curvature, min(self.max_curvature, curvature))

        # Signed cross-track error for logging (+ = car left of the line).
        j = (i + 1) % line.n
        tan = line.pts[j] - line.pts[i]
        tan = tan / max(np.linalg.norm(tan), 1e-9)
        ex, ey = x - line.pts[i, 0], y - line.pts[i, 1]
        cross_track = float(tan[0] * ey - tan[1] * ex)

        # AckermannSteering steers to radius = v / yaw_rate using its own
        # accel/jerk-limited speed, not the commanded one, so scale by the
        # car's actual speed (floored: at standstill any yaw rate would be
        # full lock).
        yaw_rate = max(v_now, self.min_steer_speed) * curvature
        return DriveCommand(speed, yaw_rate, float(line.s[i]),
                            target_speed, cross_track)
