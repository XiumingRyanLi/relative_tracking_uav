#!/usr/bin/env python3
import math
from dataclasses import dataclass


@dataclass
class GimbalCommand:
    roll: float
    pitch: float
    yaw: float


class GimbalController:
    """
    Pure gimbal controller.

    This class does not subscribe, publish, or call MAVROS services.
    It only computes gimbal commands.

    Main mode:
        image-space PD tracking using camera-frame target position.

    Fallback mode:
        world-frame pointing using drone pose and target local pose.
    """

    def __init__(self):
        # Gimbal command limits, radians.
        self.max_pitch_up = math.radians(30.0)
        self.max_pitch_down = math.radians(-80.0)
        self.max_yaw = math.radians(90.0)

        # Image-space PD gains.
        self.kp_yaw = 0.5
        self.kd_yaw = 0.08
        self.kp_pitch = 0.5
        self.kd_pitch = 0.08

        # Sign corrections.
        # Flip these if the gimbal moves the wrong way.
        self.yaw_sign = 1.0
        self.pitch_sign = -1.0

        # PD memory.
        self.prev_yaw_error = 0.0
        self.prev_pitch_error = 0.0
        self.prev_time = None

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

    def reset(self):
        self.prev_yaw_error = 0.0
        self.prev_pitch_error = 0.0
        self.prev_time = None

    def update_image_pd(
        self,
        cam_x: float,
        cam_y: float,
        cam_z: float,
        current_pitch: float,
        current_yaw: float,
        now: float,
    ) -> GimbalCommand:
        

        if cam_z <= 0.1:
            return GimbalCommand(
                roll=0.0,
                pitch=current_pitch,
                yaw=current_yaw,
            )

        # Angular image errors.
        yaw_error = math.atan2(cam_x, cam_z)
        pitch_error = math.atan2(cam_y, math.sqrt(cam_x * cam_x + cam_z * cam_z))

        if self.prev_time is None:
            dt = 0.1
            yaw_error_rate = 0.0
            pitch_error_rate = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            yaw_error_rate = self.wrap_to_pi(yaw_error - self.prev_yaw_error) / dt
            pitch_error_rate = (pitch_error - self.prev_pitch_error) / dt

        yaw_correction = self.kp_yaw * yaw_error + self.kd_yaw * yaw_error_rate
        pitch_correction = self.kp_pitch * pitch_error + self.kd_pitch * pitch_error_rate

        cmd_yaw = current_yaw + self.yaw_sign * yaw_correction
        cmd_pitch = current_pitch + self.pitch_sign * pitch_correction

        cmd_yaw = self.clamp(cmd_yaw, -self.max_yaw, self.max_yaw)
        cmd_pitch = self.clamp(cmd_pitch, self.max_pitch_down, self.max_pitch_up)

        self.prev_yaw_error = yaw_error
        self.prev_pitch_error = pitch_error
        self.prev_time = now

        return GimbalCommand(
            roll=0.0,
            pitch=cmd_pitch,
            yaw=cmd_yaw,
        )

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

        # Negative pitch points downward in your Gazebo/MAVROS setup.
        gimbal_pitch = -math.atan2(dz, horizontal_dist)
        gimbal_pitch = self.clamp(
            gimbal_pitch,
            self.max_pitch_down,
            self.max_pitch_up,
        )

        return GimbalCommand(
            roll=0.0,
            pitch=gimbal_pitch,
            yaw=gimbal_yaw,
        )