#!/usr/bin/env python3
"""Camera -> gimbal -> drone -> world transforms (pure, no ROS).

The detector reports the target in the OpenCV camera optical frame
(x right, y down, z forward). To get it into the MAVROS local/world ENU frame
it is chained through the camera mount, the gimbal attitude reported by the
FCU and the drone pose:

    world ENU <- (drone pose, per gimbal_attitude_frame) <- gimbal FRD
              <- camera optical <- target
"""
import math
from collections import deque

import numpy as np
import tf_transformations

try:
    from .geometry import R_ENU_NED, R_FLU_FRD
except ImportError:
    from geometry import R_ENU_NED, R_FLU_FRD

# GIMBAL_DEVICE_FLAGS bits (MAVLink common.xml)
GIMBAL_FLAG_YAW_IN_VEHICLE_FRAME = 32
GIMBAL_FLAG_YAW_IN_EARTH_FRAME = 64

GIMBAL_ATTITUDE_FRAMES = ("body", "horizon", "earth", "auto")


def pose_matrix(position, orientation) -> np.ndarray:
    """4x4 transform from a geometry_msgs position + quaternion."""
    T = tf_transformations.quaternion_matrix([
        orientation.x, orientation.y, orientation.z, orientation.w,
    ])
    T[0, 3] = position.x
    T[1, 3] = position.y
    T[2, 3] = position.z
    return T


def gimbal_camera_transform(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """gimbal FRD <- camera optical.

    The roll/pitch/yaw calibration is applied first, then the FRD -> optical
    axis swap (optical z = gimbal forward, x = right, y = down).
    """
    T_calib = tf_transformations.euler_matrix(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg), axes="sxyz",
    )
    T_axes = np.eye(4)
    T_axes[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    return T_calib @ T_axes


def resolve_gimbal_attitude_frame(mode: str, flags: int) -> str:
    """'auto' -> earth if the status flags say YAW_IN_EARTH_FRAME, else horizon."""
    if mode != "auto":
        return mode
    if flags & GIMBAL_FLAG_YAW_IN_EARTH_FRAME:
        return "earth"
    return "horizon"


def world_from_gimbal(T_frd_gimbal: np.ndarray, drone_pose, frame: str) -> np.ndarray:
    """world ENU <- gimbal FRD, placed at the drone position.

    drone_pose is (x, y, z, qx, qy, qz, qw) of the body FLU in world ENU.
    frame selects how the gimbal quaternion is interpreted:
      body    : relative to the drone body FRD (full drone attitude applied)
      horizon : roll/pitch relative to the horizon, yaw relative to the nose
                (only drone yaw applied)
      earth   : fully earth-referenced NED (no drone attitude applied)
    """
    dpx, dpy, dpz, dqx, dqy, dqz, dqw = drone_pose
    T = np.eye(4)
    if frame == "body":
        R_world_flu = tf_transformations.quaternion_matrix([dqx, dqy, dqz, dqw])[:3, :3]
        T[:3, :3] = R_world_flu @ R_FLU_FRD @ T_frd_gimbal[:3, :3]
    elif frame == "horizon":
        yaw = tf_transformations.euler_from_quaternion([dqx, dqy, dqz, dqw])[2]
        R_world_yaw = tf_transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0))[:3, :3]
        T[:3, :3] = R_world_yaw @ R_FLU_FRD @ T_frd_gimbal[:3, :3]
    elif frame == "earth":
        T[:3, :3] = R_ENU_NED @ T_frd_gimbal[:3, :3]
    else:
        raise ValueError(f"unknown gimbal attitude frame '{frame}'")
    T[0, 3] = dpx
    T[1, 3] = dpy
    T[2, 3] = dpz
    return T


class TimedHistory:
    """Rolling (time, value) buffer for looking up a pose at frame-capture time."""

    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self._buf = deque()

    def append(self, t_sec: float, value):
        self._buf.append((t_sec, value))
        while self._buf and (t_sec - self._buf[0][0]) > self.window_sec:
            self._buf.popleft()

    def closest(self, t_sec: float):
        """(value, |time gap|) of the entry nearest t_sec, or (None, None)."""
        if not self._buf:
            return None, None
        best = min(self._buf, key=lambda entry: abs(entry[0] - t_sec))
        return best[1], abs(best[0] - t_sec)
