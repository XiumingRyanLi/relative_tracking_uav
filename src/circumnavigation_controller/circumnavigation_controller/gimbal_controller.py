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

    def __init__(
        self,
        kp_yaw: float = 0.4,
        kd_yaw: float = 0.0,
        yaw_deadband_deg: float = 0.5,
        max_yaw_step_deg: float = 8.0,
        max_pitch_up_deg: float = 30.0,
        max_pitch_down_deg: float = -135.0,
        max_yaw_deg: float = 160.0,
    ):
        # Gimbal command limits, radians. Match the mount (MNT1_PITCH_MIN/MAX,
        # MNT1_YAW_MIN/MAX in gazebo-iris-gimbal.parm): pitch below -90 deg
        # looks backwards under the drone, so the camera can follow the car
        # straight through an overhead pass without a 180 deg yaw flip.
        self.max_pitch_up = math.radians(max_pitch_up_deg)
        self.max_pitch_down = math.radians(max_pitch_down_deg)
        self.max_yaw = math.radians(max_yaw_deg)

        # Image-space PD gains. Yaw is softer than pitch: the drone body also
        # turns to face the target, so both loops correct the same bearing
        # error, and the D term mostly amplified detection noise (jitter).
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.kp_pitch = 0.5
        self.kd_pitch = 0.08
        # Yaw jitter guards: ignore errors within the deadband (target already
        # centred) and limit how far one command can move the yaw.
        self.yaw_deadband = math.radians(yaw_deadband_deg)
        self.max_yaw_step = math.radians(max_yaw_step_deg)

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

        yaw_correction = 0.0
        if abs(yaw_error) > self.yaw_deadband:
            yaw_correction = self.kp_yaw * yaw_error + self.kd_yaw * yaw_error_rate
            yaw_correction = self.clamp(yaw_correction, -self.max_yaw_step, self.max_yaw_step)
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

        # Gimbal yaw is relative to the nose and positive to the RIGHT (FRD),
        # world yaw is ENU (positive counter-clockwise): camera heading =
        # drone_yaw - gimbal_yaw.
        desired_world_yaw = math.atan2(dy, dx)
        gimbal_yaw = self.wrap_to_pi(drone_yaw - desired_world_yaw)
        gimbal_yaw = self.clamp(gimbal_yaw, -self.max_yaw, self.max_yaw)

        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        horizontal_dist = max(horizontal_dist, 1e-6)

        # Pitch is relative to the horizon, negative = down. dz < 0 when the
        # target is below the drone, so pitch = atan2(dz, horizontal) < 0.
        gimbal_pitch = math.atan2(dz, horizontal_dist)
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