#!/usr/bin/env python3
"""Run logging: the relative_pid_<timestamp>.csv log plus ground-truth evaluation.

Nothing here feeds back into control. The node passes in what it did each
setpoint cycle; this adds the evaluation columns (Gazebo car truth, GPS,
detector-vs-truth errors) that scripts/plot_dope_evaluation.py reads by
column name, and prints the [chain] transform debug line.
"""
import csv
import math
from datetime import datetime

import numpy as np
import tf_transformations

try:
    from .camera_frames import world_from_gimbal
    from .geometry import enu_to_lla, wrap_deg, wrap_to_pi
except ImportError:
    from camera_frames import world_from_gimbal
    from geometry import enu_to_lla, wrap_deg, wrap_to_pi

CSV_HEADER = [
    "time", "stage",
    "drone_x", "drone_y", "drone_z",
    "target_x", "target_y", "target_z", "target_heading_deg",
    "desired_x", "desired_y", "desired_z",
    "ex", "ey", "ez", "eyaw_deg",
    "vx", "vy", "vz", "yaw_rate",
    "gimbal_roll", "gimbal_pitch", "gimbal_yaw",
    # ---- evaluation columns (see _eval_columns) ----
    "drone_yaw_deg", "drone_lat", "drone_lon", "drone_alt",
    "dope_dist_m", "dope_age_s", "dope_target_yaw_deg",
    "car_gt_x", "car_gt_y", "car_gt_z", "car_gt_yaw_deg", "car_gt_lat", "car_gt_lon",
    "gt_dist_m", "dope_dist_err_m", "dope_pos_err_m", "dope_yaw_err_deg",
    "dope_cam_x", "dope_cam_y", "dope_cam_z",
]


class RunLogger:
    def __init__(self, logger, world_origin_lat: float, world_origin_lon: float):
        self._log = logger
        self.world_origin_lat = world_origin_lat
        self.world_origin_lon = world_origin_lon
        # Evaluation inputs (not used for control).
        self.truth_car = None   # (x, y, z, yaw_rad) world ENU, from Gazebo
        self.gps = None         # (lat, lon, alt)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"relative_pid_{ts}.csv"
        self._file = open(self.filename, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)

    # ---- evaluation inputs ----
    def on_truth_car(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.truth_car = (float(p.x), float(p.y), float(p.z), float(yaw))

    def on_gps(self, msg):
        self.gps = (float(msg.latitude), float(msg.longitude), float(msg.altitude))

    # ---- CSV ----
    def log(
        self,
        now,
        stage,
        drone_xyz,
        desired_xyz,
        errors,             # (ex, ey, ez, eyaw_deg)
        command,            # (vx, vy, vz, yaw_rate)
        gimbal_rpy_deg,
        target,             # TargetEstimator
        detection,          # Detection or None
        drone_yaw_deg,      # None without a drone pose
    ):
        dx, dy, dz = drone_xyz
        self._writer.writerow([
            now, stage,
            dx, dy, dz,
            target.x, target.y, target.z, math.degrees(target.heading),
            *desired_xyz,
            *errors,
            *command,
            *gimbal_rpy_deg,
            *self._eval_columns(now, drone_xyz, target, detection, drone_yaw_deg),
        ])
        self._file.flush()

    def _eval_columns(self, now, drone_xyz, target, detection, drone_yaw_deg):
        """Ground-truth vs detector values for offline evaluation.

        drone_*      : drone yaw (ENU deg), GPS fix
        dope_dist_m  : detector's straight-line camera->car range
        dope_age_s   : time since that detection was accepted
        dope_target_yaw_deg : car heading in world ENU from the detector's orientation
        car_gt_*     : Gazebo truth for the car (world ENU, yaw ENU deg, lat/lon
                       via the world origin)
        gt_dist_m    : |car_gt - drone| (assumes world origin == local origin)
        dope_dist_err_m : dope_dist_m - gt_dist_m
        dope_pos_err_m  : |target estimate (world) - car_gt_xyz|
        dope_yaw_err_deg: wrap(dope_target_yaw - car_gt_yaw)
        dope_cam_*   : last accepted detection in the camera optical frame
        Empty fields mean that input was not available yet.
        """
        lat, lon, alt = self.gps if self.gps is not None else (None, None, None)
        have_dope = detection is not None
        dope_age = (now - detection.received_time) if have_dope else None
        dope_yaw_deg = math.degrees(detection.world_yaw) if have_dope else None
        cols = [
            drone_yaw_deg, lat, lon, alt,
            detection.distance if have_dope else None, dope_age, dope_yaw_deg,
        ]
        cam = [detection.cam_x, detection.cam_y, detection.cam_z] if have_dope else [None] * 3
        if self.truth_car is None:
            return cols + [None] * 10 + cam
        dx, dy, dz = drone_xyz
        cx, cy, cz, cyaw = self.truth_car
        clat, clon = enu_to_lla(cx, cy, self.world_origin_lat, self.world_origin_lon)
        gt_dist = (
            math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2 + (cz - dz) ** 2)
            if drone_yaw_deg is not None else None
        )
        dist_err = (detection.distance - gt_dist) if (have_dope and gt_dist is not None) else None
        pos_err = (
            math.sqrt((target.x - cx) ** 2 + (target.y - cy) ** 2 + (target.z - cz) ** 2)
            if have_dope else None
        )
        yaw_err = wrap_deg(dope_yaw_deg - math.degrees(cyaw)) if have_dope else None
        return cols + [cx, cy, cz, math.degrees(cyaw), clat, clon, gt_dist, dist_err, pos_err, yaw_err] + cam

    def close(self):
        self._file.close()
        return self.filename

    # ---- transform-chain debug ----
    def log_chain_debug(
        self,
        drone_pose,
        T_frd_gimbal,
        gimbal_flags,
        active_frame,
        T_gimbal_camera,
        T_world_camera,
        meas,
        p_cam,
        cmd_pitch,
        cmd_yaw,
    ):
        """Split the target error into frame-chain vs detector error.

        Needs Gazebo car truth. Prints the drone/gimbal attitude, the
        horizontal bearing error each gimbal frame convention would give
        (the right one is ~0), and where the car *should* appear in the
        camera frame according to the active chain vs where the detector
        saw it (they match when the chain is right).
        """
        if self.truth_car is None:
            return
        dpx, dpy, dpz, dqx, dqy, dqz, dqw = drone_pose
        cx, cy, cz, _ = self.truth_car
        d_rpy = [math.degrees(a) for a in tf_transformations.euler_from_quaternion([dqx, dqy, dqz, dqw])]
        g_rpy = [math.degrees(a) for a in tf_transformations.euler_from_matrix(T_frd_gimbal, axes="sxyz")]
        brg_true = math.atan2(cy - dpy, cx - dpx)
        brg_meas = math.atan2(meas[1] - dpy, meas[0] - dpx)

        p_cam_h = np.array([p_cam.x, p_cam.y, p_cam.z, 1.0])
        brg_err = {}
        for frame in ("body", "horizon", "earth"):
            p_w = world_from_gimbal(T_frd_gimbal, drone_pose, frame) @ T_gimbal_camera @ p_cam_h
            brg_err[frame] = math.degrees(
                wrap_to_pi(math.atan2(p_w[1] - dpy, p_w[0] - dpx) - brg_true)
            )
        p_exp = np.linalg.inv(T_world_camera) @ np.array([cx, cy, cz, 1.0])

        self._log.info(
            f"[chain] drone rpy=({d_rpy[0]:.1f},{d_rpy[1]:.1f},{d_rpy[2]:.1f}) "
            f"gimbal rpy=({g_rpy[0]:.1f},{g_rpy[1]:.1f},{g_rpy[2]:.1f}) "
            f"flags={gimbal_flags} "
            f"cmd p/y=({math.degrees(cmd_pitch):.1f},{math.degrees(cmd_yaw):.1f}) | "
            f"bearing meas={math.degrees(brg_meas):.1f} true={math.degrees(brg_true):.1f} | "
            f"bearing err body={brg_err['body']:.1f} horizon={brg_err['horizon']:.1f} "
            f"earth={brg_err['earth']:.1f} (active={active_frame}) | "
            f"p_cam dope=({p_cam.x:.2f},{p_cam.y:.2f},{p_cam.z:.2f}) "
            f"expected=({p_exp[0]:.2f},{p_exp[1]:.2f},{p_exp[2]:.2f})",
            throttle_duration_sec=1.0,
        )
