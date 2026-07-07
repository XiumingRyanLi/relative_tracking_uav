#!/usr/bin/env python3
import math
from dataclasses import dataclass


@dataclass
class GimbalCommand:
    roll: float
    pitch: float
    yaw: float


class GimbalController:
    """Pure gimbal pointing controller.

    Computes roll/pitch/yaw commands so the camera points from the drone
    toward the target. Commands are relative to the drone body frame.
    """

    def __init__(self):
        self.max_pitch_up = math.radians(45.0)
        self.max_pitch_down = math.radians(-135.0)
        self.max_yaw = math.radians(160.0)

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(min(value, high), low)

    def update(
        self,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
        target_x: float,
        target_y: float,
        target_z: float,
    ) -> GimbalCommand:
        dx = target_x - drone_x
        dy = target_y - drone_y
        dz = target_z - drone_z

        desired_world_yaw = math.atan2(dy, dx)
        gimbal_yaw = self.wrap_to_pi(desired_world_yaw - drone_yaw)
        gimbal_yaw = self.clamp(gimbal_yaw, -self.max_yaw, self.max_yaw)

        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        horizontal_dist = max(horizontal_dist, 1e-6)

        # In this Gazebo gimbal, negative pitch points downward.
        # If target is below drone, dz is negative, so pitch becomes negative.
        gimbal_pitch = -math.atan2(dz, horizontal_dist)
        gimbal_pitch = self.clamp(gimbal_pitch, self.max_pitch_down, self.max_pitch_up)

        return GimbalCommand(roll=0.0, pitch=gimbal_pitch, yaw=gimbal_yaw)