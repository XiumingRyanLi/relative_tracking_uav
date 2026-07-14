#!/usr/bin/env python3
import math
from dataclasses import dataclass


@dataclass
class PIDCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    ex: float
    ey: float
    ez: float
    eyaw: float
    desired_x: float
    desired_y: float
    desired_z: float


class PIDRelativeController:
    """Pure relative-position PID controller.

    Inputs are drone state + target state + desired body-frame offset.
    Output is MAVROS velocity setpoint: vx, vy, vz, yaw_rate.
    """

    def __init__(self, dt: float = 0.15):
        self.dt = dt

        self.kp_x, self.ki_x, self.kd_x = 0.80, 0.02, 0.10
        self.kp_y, self.ki_y, self.kd_y = 0.80, 0.02, 0.10
        self.kp_z, self.ki_z, self.kd_z = 0.60, 0.03, 0.10
        self.kp_yaw, self.ki_yaw, self.kd_yaw = 0.80, 0.00, 0.00

        self.max_vel_xy = 10.0
        self.max_vel_z = 3.0
        self.max_yaw_rate = 4.0

        self.max_integral_xy = 3.0
        self.max_integral_z = 2.0
        self.max_integral_yaw = 2.0

        self.reset()

    def reset(self):
        self.int_ex = 0.0
        self.int_ey = 0.0
        self.int_ez = 0.0
        self.int_eyaw = 0.0

        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_ez = 0.0
        self.prev_eyaw = 0.0

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clamp(value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    @staticmethod
    def body_to_world(x_body: float, y_body: float, heading: float):
        c = math.cos(heading)
        s = math.sin(heading)
        x_world = c * x_body - s * y_body
        y_world = s * x_body + c * y_body
        return x_world, y_world

    def update(
        self,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
        target_x: float,
        target_y: float,
        target_z: float,
        target_heading: float,
        rel_x_body: float,
        rel_y_body: float,
        rel_z_body: float,
    ) -> PIDCommand:
        rel_x_w, rel_y_w = self.body_to_world(rel_x_body, rel_y_body, target_heading)

        desired_x = target_x + rel_x_w
        desired_y = target_y + rel_y_w
        desired_z = target_z + rel_z_body

        ex = desired_x - drone_x
        ey = desired_y - drone_y
        ez = desired_z - drone_z

        desired_yaw = math.atan2(target_y - drone_y, target_x - drone_x)
        eyaw = self.wrap_to_pi(desired_yaw - drone_yaw)

        self.int_ex = self.clamp(self.int_ex + ex * self.dt, self.max_integral_xy)
        self.int_ey = self.clamp(self.int_ey + ey * self.dt, self.max_integral_xy)
        self.int_ez = self.clamp(self.int_ez + ez * self.dt, self.max_integral_z)
        self.int_eyaw = self.clamp(self.int_eyaw + eyaw * self.dt, self.max_integral_yaw)

        dex = (ex - self.prev_ex) / self.dt
        dey = (ey - self.prev_ey) / self.dt
        dez = (ez - self.prev_ez) / self.dt
        deyaw = (eyaw - self.prev_eyaw) / self.dt

        vx = self.kp_x * ex + self.ki_x * self.int_ex + self.kd_x * dex
        vy = self.kp_y * ey + self.ki_y * self.int_ey + self.kd_y * dey
        vz = self.kp_z * ez + self.ki_z * self.int_ez + self.kd_z * dez
        yaw_rate = self.kp_yaw * eyaw + self.ki_yaw * self.int_eyaw + self.kd_yaw * deyaw

        vx = self.clamp(vx, self.max_vel_xy)
        vy = self.clamp(vy, self.max_vel_xy)
        vz = self.clamp(vz, self.max_vel_z)
        yaw_rate = self.clamp(yaw_rate, self.max_yaw_rate)

        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_ez = ez
        self.prev_eyaw = eyaw

        return PIDCommand(
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            ex=ex,
            ey=ey,
            ez=ez,
            eyaw=eyaw,
            desired_x=desired_x,
            desired_y=desired_y,
            desired_z=desired_z,
        )
    
    def update_from_error(self, ex, ey, ez, eyaw, desired_x, desired_y, desired_z):
        self.int_ex = self.clamp(self.int_ex + ex * self.dt, self.max_integral_xy)
        self.int_ey = self.clamp(self.int_ey + ey * self.dt, self.max_integral_xy)
        self.int_ez = self.clamp(self.int_ez + ez * self.dt, self.max_integral_z)
        self.int_eyaw = self.clamp(self.int_eyaw + eyaw * self.dt, self.max_integral_yaw)

        dex = (ex - self.prev_ex) / self.dt
        dey = (ey - self.prev_ey) / self.dt
        dez = (ez - self.prev_ez) / self.dt
        deyaw = (eyaw - self.prev_eyaw) / self.dt

        vx = self.kp_x * ex + self.ki_x * self.int_ex + self.kd_x * dex
        vy = self.kp_y * ey + self.ki_y * self.int_ey + self.kd_y * dey
        vz = self.kp_z * ez + self.ki_z * self.int_ez + self.kd_z * dez
        yaw_rate = self.kp_yaw * eyaw + self.ki_yaw * self.int_eyaw + self.kd_yaw * deyaw

        vx = self.clamp(vx, self.max_vel_xy)
        vy = self.clamp(vy, self.max_vel_xy)
        vz = self.clamp(vz, self.max_vel_z)
        yaw_rate = self.clamp(yaw_rate, self.max_yaw_rate)

        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_ez = ez
        self.prev_eyaw = eyaw

        return PIDCommand(
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            desired_x=desired_x,
            desired_y=desired_y,
            desired_z=desired_z,
            ex=ex,
            ey=ey,
            ez=ez,
            eyaw=eyaw,
        )