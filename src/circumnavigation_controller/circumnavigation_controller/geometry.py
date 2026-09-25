#!/usr/bin/env python3
"""Small angle / frame helpers shared by the tracking modules (pure, no ROS)."""
import math

import numpy as np

# NED -> ENU rotation (swap x/y, flip z). FLU <-> FRD is diag(1, -1, -1).
R_ENU_NED = np.array([
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
])
R_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def smooth_angle(old_angle: float, new_angle: float, alpha: float) -> float:
    """Circular EMA step from old_angle towards new_angle."""
    return wrap_to_pi(old_angle + alpha * wrap_to_pi(new_angle - old_angle))


def body_to_world(x_body: float, y_body: float, heading: float):
    """Rotate a planar offset from a frame with yaw `heading` into the world."""
    c = math.cos(heading)
    s = math.sin(heading)
    return c * x_body - s * y_body, s * x_body + c * y_body


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def enu_to_lla(x, y, origin_lat_deg, origin_lon_deg):
    """Flat-earth ENU (m) -> lat/lon (deg) around an origin. Good to well
    under a metre within the few km of a sim runway."""
    R = 6378137.0
    lat = origin_lat_deg + math.degrees(y / R)
    lon = origin_lon_deg + math.degrees(x / (R * math.cos(math.radians(origin_lat_deg))))
    return lat, lon
