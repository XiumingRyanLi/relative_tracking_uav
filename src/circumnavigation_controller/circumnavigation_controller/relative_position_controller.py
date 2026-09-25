#!/usr/bin/env python3
"""Relative position controller node: keeps the drone at a cinematic shot
position relative to a visually detected target (car).

Module layout:
  controller_params.py  every ROS parameter (defaults + descriptions)
  camera_frames.py      camera -> gimbal -> drone -> world transforms
  target_estimator.py   gating, position/velocity KF, heading filter
  pid_controller.py     drone velocity PID            (control)
  gimbal_controller.py  gimbal image-space PD         (control)
  cinematic_planner.py  shot offset sequence          (control)
  flight_sequencer.py   GUIDED/arm/takeoff/RTL        (logistics)
  gimbal_interface.py   MAVROS gimbal manager I/O     (logistics)
  run_logger.py         CSV + ground-truth evaluation (logging)

This file wires them to ROS topics/timers and holds the two control loops:
_publish_setpoint (drone velocity) and _publish_gimbal_setpoint (gimbal).
"""
import json
import math

import numpy as np
if not hasattr(np, "float"):
    np.float = float  # tf_transformations still uses np.float

import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import GimbalDeviceAttitudeStatus, State
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, String

try:
    from .camera_frames import (
        GIMBAL_ATTITUDE_FRAMES, gimbal_camera_transform, pose_matrix,
        resolve_gimbal_attitude_frame, world_from_gimbal, TimedHistory,
    )
    from .cinematic_action_schema import validate_action
    from .cinematic_planner import CinematicPlanner
    from .controller_params import declare_parameters
    from .flight_sequencer import FlightSequencer
    from .geometry import body_to_world, quat_to_yaw, stamp_to_sec, wrap_to_pi
    from .gimbal_controller import GimbalController
    from .gimbal_interface import GimbalInterface
    from .pid_controller import PIDRelativeController
    from .run_logger import RunLogger
    from .target_estimator import Detection, TargetEstimator
except ImportError:
    from camera_frames import (
        GIMBAL_ATTITUDE_FRAMES, gimbal_camera_transform, pose_matrix,
        resolve_gimbal_attitude_frame, world_from_gimbal, TimedHistory,
    )
    from cinematic_action_schema import validate_action
    from cinematic_planner import CinematicPlanner
    from controller_params import declare_parameters
    from flight_sequencer import FlightSequencer
    from geometry import body_to_world, quat_to_yaw, stamp_to_sec, wrap_to_pi
    from gimbal_controller import GimbalController
    from gimbal_interface import GimbalInterface
    from pid_controller import PIDRelativeController
    from run_logger import RunLogger
    from target_estimator import Detection, TargetEstimator

# Standoff is bounded by the camera, not the controller: with the sim
# camera's 0.8 rad HFOV the visible width is 0.845 * range, so a side-on Audi
# (4.4 m) fills the whole frame at 5.2 m and half of it at 10.4 m. DOPE needs
# the full cuboid inside the image, so keep the straight-line range >= ~8 m
# (measured: full 8-corner detections from 8 m, nothing below 8 m at any
# input scale).
DEFAULT_SHOT_SEQUENCE = [
    # {
    #     "type": "hold_location",
    #     "location": "right",
    #     "radius": 18.0,
    #     "height": 4.0,
    #     "duration": 1000.0,
    # },

    # Other action types (see cinematic_action_schema.py), e.g.:
    {"type": "move_location", "from": "back", "to": "front",
     "radius": 20.0, "height": 4.0, "duration": 20.0},
    {"type": "overpass", "from": "front", "to": "back", "radius": 3.0,
     "start_height": 4.0, "peak_height": 10, "end_height": 4.0, "duration": 20.0},
    # {"type": "move_location", "from": "right", "to": "left", "via": "front",
    #  "radius": 3.0, "height": 3.0, "duration": 12.0},
]

SETPOINT_PERIOD_SEC = 0.15
GIMBAL_PERIOD_SEC = 0.10
ORCHESTRATE_PERIOD_SEC = 0.20
SAFETY_PERIOD_SEC = 1.00


class RelativePositionController(Node):
    def __init__(self):
        super().__init__("relative_position_controller")
        cfg = self.cfg = declare_parameters(self)

        self.gimbal_attitude_frame = str(cfg.gimbal_attitude_frame).lower()
        if self.gimbal_attitude_frame not in GIMBAL_ATTITUDE_FRAMES:
            raise ValueError(
                f"gimbal_attitude_frame must be one of {GIMBAL_ATTITUDE_FRAMES}, "
                f"got '{self.gimbal_attitude_frame}'"
            )
        self.T_gimbal_camera = gimbal_camera_transform(
            cfg.gimbal_camera_roll_deg, cfg.gimbal_camera_pitch_deg, cfg.gimbal_camera_yaw_deg,
        )

        # ---------------- Control ----------------
        self.pid = PIDRelativeController(
            nominal_dt=SETPOINT_PERIOD_SEC,
            speed_margin_xy=cfg.pid_speed_margin_xy,
            max_speed_xy=cfg.pid_max_speed_xy,
            max_speed_z=cfg.pid_max_speed_z,
            max_accel_xy=cfg.pid_max_accel_xy,
            max_accel_z=cfg.pid_max_accel_z,
            derivative_tau=cfg.pid_derivative_tau_sec,
            kp_xy=cfg.pid_kp_xy,
            ki_xy=cfg.pid_ki_xy,
            max_integral_xy=cfg.pid_max_integral_xy,
        )
        self.gimbal_ctrl = GimbalController(
            kp_yaw=cfg.gimbal_kp_yaw,
            kd_yaw=cfg.gimbal_kd_yaw,
            yaw_deadband_deg=cfg.gimbal_yaw_deadband_deg,
            max_yaw_step_deg=cfg.gimbal_max_yaw_step_deg,
        )
        self.cinematic_planner = CinematicPlanner()
        self.cinematic_planner.set_sequence(DEFAULT_SHOT_SEQUENCE)
        self.estimator = TargetEstimator(
            heading_axis={"x": 0, "y": 1, "z": 2}[str(cfg.target_heading_axis).lower()],
            min_distance=cfg.min_visual_target_distance,
            max_distance=cfg.max_visual_target_distance,
            max_position_jump=cfg.max_visual_position_jump,
            jump_per_m=cfg.visual_jump_per_m,
            max_consecutive_rejects=int(cfg.visual_max_consecutive_rejects),
            noise_per_m=cfg.visual_noise_per_m,
            use_filtered_position=bool(cfg.enable_visual_position_filter),
            kf_coast_timeout_sec=cfg.kf_coast_timeout_sec,
            kf_accel_noise_std=cfg.kf_accel_noise_std,
            kf_measurement_noise_std=cfg.kf_measurement_noise_std,
            kf_initial_pos_std=cfg.kf_initial_pos_std,
            kf_initial_vel_std=cfg.kf_initial_vel_std,
            kf_max_target_speed=cfg.kf_max_target_speed,
            kf_enable_adaptive_q=bool(cfg.kf_enable_adaptive_q),
            kf_adaptive_q_max_scale=cfg.kf_adaptive_q_max_scale,
            kf_adaptive_q_decay=cfg.kf_adaptive_q_decay,
            heading_alpha=cfg.visual_heading_alpha,
            yaw_rate_tau_sec=cfg.yaw_rate_tau_sec,
            yaw_rate_min_speed=cfg.yaw_rate_min_speed,
            max_yaw_rate=math.radians(cfg.max_yaw_rate_deg),
            max_heading_jump=math.radians(cfg.max_visual_heading_jump_deg),
            max_heading_distance=cfg.max_visual_heading_distance,
            enable_heading_filter=bool(cfg.enable_visual_heading_filter),
            enable_heading_ukf=bool(cfg.enable_heading_ukf),
            ukf_coast_timeout_sec=cfg.ukf_coast_timeout_sec,
            ukf_std_a=cfg.ukf_std_a,
            ukf_std_yawdd=cfg.ukf_std_yawdd,
            ukf_std_pos=cfg.ukf_std_pos,
            ukf_std_yaw=cfg.ukf_std_yaw,
            ukf_max_yaw_rate=cfg.ukf_max_yaw_rate,
        )

        # ---------------- Logistics + logging ----------------
        self.flight = FlightSequencer(
            self,
            takeoff_altitude=cfg.target_altitude,
            hover_delay_sec=cfg.hover_delay_sec,
            boundary_limit=cfg.boundary_limit,
            on_tracking_enabled=self.pid.reset,
        )
        self.gimbal = GimbalInterface(
            self,
            device_id=int(cfg.gimbal_device_id),
            command_period_sec=cfg.gimbal_command_period_sec,
            initial_pitch_rad=math.radians(cfg.gimbal_initial_pitch_deg),
            history_sec=cfg.pose_history_buffer_sec,
        )
        self.run_log = RunLogger(self.get_logger(), cfg.world_origin_lat_deg, cfg.world_origin_lon_deg)

        # ---------------- State ----------------
        self.drone_pose = PoseStamped()
        self.have_drone_pose = False
        self._drone_pose_history = TimedHistory(cfg.pose_history_buffer_sec)
        self.compass_heading_rad = 0.0
        self.have_compass = False

        self.target_received = False
        self.last_target_time = None
        self.detection = None           # last accepted Detection
        self._search_start_time = None

        # ---------------- Subscriptions ----------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        self.create_subscription(State, "/mavros/state", self._on_state, reliable_qos)
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self._on_drone_pose, best_effort_qos
        )
        self.create_subscription(
            Float64, "/mavros/global_position/compass_hdg", self._on_compass,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10),
        )
        self.create_subscription(
            GimbalDeviceAttitudeStatus, "/mavros/gimbal_control/device/attitude_status",
            self._on_gimbal_attitude_status, 10,
        )
        self.create_subscription(Odometry, "/aruco_target/visual_odom", self._on_visual_odom, 10)
        self.create_subscription(
            String, cfg.cinematic_command_topic, self._on_cinematic_command, reliable_qos
        )
        # Evaluation inputs (logged to CSV only). The Gazebo world origin and
        # the MAVROS local origin coincide when the drone spawns at the world
        # origin (iris_runway_new.sdf), so car truth and drone local xyz are
        # directly comparable.
        if cfg.truth_target_odom_topic:
            self.create_subscription(
                Odometry, cfg.truth_target_odom_topic, self.run_log.on_truth_car, 10
            )
        if cfg.gps_topic:
            self.create_subscription(
                NavSatFix, cfg.gps_topic, self.run_log.on_gps,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5),
            )

        # ---------------- Publishers ----------------
        self.vel_pub = self.create_publisher(TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)

        # ---------------- Timers ----------------
        self.create_timer(SETPOINT_PERIOD_SEC, self._publish_setpoint)
        self.create_timer(GIMBAL_PERIOD_SEC, self._publish_gimbal_setpoint)
        self.create_timer(ORCHESTRATE_PERIOD_SEC, self._orchestrate)
        self.create_timer(SAFETY_PERIOD_SEC, self._check_safety)

        self.get_logger().info(
            "RelativePositionController started. Vision target source: /aruco_target/visual_odom, "
            f"gimbal_attitude_frame={self.gimbal_attitude_frame}, "
            f"search_mode={cfg.enable_search_mode}, CSV: {self.run_log.filename}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _target_is_fresh(self, now: float) -> bool:
        return self.last_target_time is not None and (now - self.last_target_time) <= self.cfg.target_timeout_sec

    def _target_available(self, now: float) -> bool:
        return self.target_received and self._target_is_fresh(now)

    def _detection_fresh(self, now: float) -> bool:
        return self.detection is not None and (now - self.detection.received_time) <= self.cfg.target_timeout_sec

    def _drone_xyz(self):
        p = self.drone_pose.pose.position
        return p.x, p.y, p.z

    def _drone_yaw(self) -> float:
        """Drone yaw (ENU rad) for control: compass if available, else the pose."""
        if self.have_compass:
            return self.compass_heading_rad
        q = self.drone_pose.pose.orientation
        return quat_to_yaw(q.x, q.y, q.z, q.w)

    def _drone_yaw_deg(self):
        """Drone yaw (ENU deg) from the pose, for logging; None without a pose."""
        if not self.have_drone_pose:
            return None
        q = self.drone_pose.pose.orientation
        return math.degrees(tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2])

    def _enter_search_mode(self):
        if self._search_start_time is None:
            self._search_start_time = self._now()
            self.pid.reset()
            self.get_logger().warning(
                "No visual target available — entering SEARCH mode. Sweeping the gimbal."
            )

    def _exit_search_mode(self):
        if self._search_start_time is not None:
            self._search_start_time = None
            self.pid.reset()
            self.get_logger().info("Visual target acquired — switching to TRACK mode.")

    def _log_row(self, now, stage, drone_xyz, desired_xyz, errors, command):
        self.run_log.log(
            now, stage, drone_xyz, desired_xyz, errors, command,
            self.gimbal.attitude_rpy_deg(), self.estimator, self.detection, self._drone_yaw_deg(),
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def _on_state(self, msg: State):
        self.flight.on_state(msg, self._now())

    def _on_drone_pose(self, msg: PoseStamped):
        self.drone_pose = msg
        self.have_drone_pose = True
        t_sec = stamp_to_sec(msg.header.stamp) or self._now()
        p, q = msg.pose.position, msg.pose.orientation
        self._drone_pose_history.append(t_sec, (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        self.flight.on_altitude(p.z, self._now())

    def _on_compass(self, msg: Float64):
        # compass: 0=N, 90=E  ->  ROS ENU yaw: 0=E, 90=N
        self.compass_heading_rad = wrap_to_pi(math.radians(90.0 - float(msg.data)))
        self.have_compass = True

    def _on_gimbal_attitude_status(self, msg):
        header = getattr(msg, "header", None)
        t_sec = (stamp_to_sec(header.stamp) if header is not None else 0.0) or self._now()
        self.gimbal.on_attitude_status(msg, t_sec)

    def _on_cinematic_command(self, msg: String):
        try:
            raw_actions = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().error(f"Cinematic command: invalid JSON ({exc}); ignoring.")
            return
        if not isinstance(raw_actions, list) or not raw_actions:
            self.get_logger().error(
                "Cinematic command: expected a non-empty JSON list of actions; ignoring."
            )
            return

        # Planner-derived defaults (used when a field is missing/invalid), so
        # a bare hold_location falls back to the current default shot.
        planner = self.cinematic_planner
        dynamic_defaults = {
            "radius": planner.default_radius,
            "height": planner.default_height,
            "start_height": planner.default_height,
            "peak_height": planner.default_height + 3.0,
            "end_height": planner.default_height,
        }
        validated = []
        for i, raw in enumerate(raw_actions):
            action = validate_action(raw, i, dynamic_defaults, warn=self.get_logger().warn)
            if action is not None:
                validated.append(action)
        if not validated:
            self.get_logger().error(
                "Cinematic command: no valid actions after validation; sequence not applied."
            )
            return
        planner.set_sequence(validated)
        self.get_logger().info(f"Cinematic command: applied new sequence with {len(validated)} action(s).")

    def _poses_at(self, stamp_sec: float):
        """Gimbal attitude and drone pose at frame-capture time.

        Detection latency + queueing mean "now" is not the capture time; the
        gimbal in particular may have moved on. A match further off than the
        buffer window is most likely a clock-domain mismatch (sim vs wall
        time), so the live values are used instead.
        """
        max_gap = self.cfg.pose_history_buffer_sec

        T_frd_gimbal, gimbal_gap = self.gimbal.history.closest(stamp_sec)
        if T_frd_gimbal is None or gimbal_gap > max_gap:
            if T_frd_gimbal is not None:
                self.get_logger().warning(
                    f"Gimbal history match {gimbal_gap:.2f}s from capture time -- "
                    f"ignoring history, using live value. Check clock domains "
                    f"(use_sim_time) across detector/controller/mavros.",
                    throttle_duration_sec=2.0,
                )
            T_frd_gimbal = self.gimbal.T_frd_gimbal
        elif gimbal_gap > 0.10:
            self.get_logger().warning(
                f"Gimbal attitude history match is {gimbal_gap * 1000:.0f} ms "
                f"from frame capture time -- detection latency is significant.",
                throttle_duration_sec=2.0,
            )

        drone_pose, drone_gap = self._drone_pose_history.closest(stamp_sec)
        if drone_pose is None or drone_gap > max_gap:
            if drone_pose is not None:
                self.get_logger().warning(
                    f"Drone pose history match {drone_gap:.2f}s from capture time -- "
                    f"ignoring history, using live value.",
                    throttle_duration_sec=2.0,
                )
            p, q = self.drone_pose.pose.position, self.drone_pose.pose.orientation
            drone_pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        return T_frd_gimbal, drone_pose

    def _on_visual_odom(self, msg: Odometry):
        """Detection in the camera optical frame (x right, y down, z forward)
        -> world frame -> target estimator."""
        if not self.have_drone_pose:
            return
        if not self.gimbal.have_attitude:
            self.get_logger().warning(
                "No gimbal attitude yet, cannot transform visual target to local frame.",
                throttle_duration_sec=2.0,
            )
            return

        p_cam = msg.pose.pose.position
        T_camera_target = pose_matrix(p_cam, msg.pose.pose.orientation)
        stamp_sec = stamp_to_sec(msg.header.stamp) or self._now()

        T_frd_gimbal, drone_pose = self._poses_at(stamp_sec)
        gimbal_frame = resolve_gimbal_attitude_frame(self.gimbal_attitude_frame, self.gimbal.flags)
        T_world_camera = world_from_gimbal(T_frd_gimbal, drone_pose, gimbal_frame) @ self.T_gimbal_camera
        T_world_target = T_world_camera @ T_camera_target

        distance = math.sqrt(p_cam.x ** 2 + p_cam.y ** 2 + p_cam.z ** 2)
        reject_reason = self.estimator.update(T_world_target, distance, stamp_sec)
        if reject_reason is not None:
            self.get_logger().warning(
                f"Rejected visual target ({reject_reason}), "
                f"{self.estimator.reject_count} in a row.",
                throttle_duration_sec=1.0,
            )
            return

        now = self._now()
        self.detection = Detection(
            cam_x=float(p_cam.x), cam_y=float(p_cam.y), cam_z=float(p_cam.z),
            distance=float(distance),
            world_yaw=float(self.estimator.raw_heading(T_world_target)),
            stamp_sec=float(stamp_sec),
            received_time=now,
        )
        if self.cfg.debug_transform_chain:
            self.run_log.log_chain_debug(
                drone_pose, T_frd_gimbal, self.gimbal.flags, gimbal_frame,
                self.T_gimbal_camera, T_world_camera, T_world_target[:3, 3], p_cam,
                self.gimbal.current_pitch, self.gimbal.current_yaw,
            )
        self.target_received = True
        self.last_target_time = now
        self._exit_search_mode()

        self.get_logger().info(
            f"Visual target: camera_dist={distance:.2f} m, "
            f"camera_optical=({p_cam.x:.2f}, {p_cam.y:.2f}, {p_cam.z:.2f}), "
            f"heading={math.degrees(self.estimator.heading):.1f} deg, "
            f"yaw_rate={math.degrees(self.estimator.yaw_rate):.1f} deg/s",
            throttle_duration_sec=0.5,
        )

    # ------------------------------------------------------------------
    # Logistics timers
    # ------------------------------------------------------------------
    def _orchestrate(self):
        self.flight.tick(self._now(), self.have_drone_pose)

    def _check_safety(self):
        x, y, _ = self._drone_xyz()
        self.flight.check_safety(self._now(), x, y)

    # ------------------------------------------------------------------
    # Control loops
    # ------------------------------------------------------------------
    def _publish_setpoint(self):
        """Drone velocity: hold the cinematic shot offset relative to the target."""
        if self.flight.rtl_initiated:
            return

        now = self._now()
        drone_xyz = self._drone_xyz()
        drone_x, drone_y, drone_z = drone_xyz
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        if not self.flight.tracking_enabled:
            self._log_row(now, "stage1", drone_xyz, drone_xyz, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
            return

        if not self._target_available(now):
            if not self.cfg.enable_search_mode:
                self.get_logger().warning(
                    "Tracking enabled but no visual target received, holding.",
                    throttle_duration_sec=5.0,
                )
                self.vel_pub.publish(msg)   # zero velocity
            return

        self._exit_search_mode()
        target = self.estimator

        # Desired position = target position + shot offset rotated by the
        # target heading (predicted ahead by its filter lag + detection
        # latency). The shot height is flown above the ground, not above
        # target.z: the car is on the ground, and target.z is a noisy estimate
        # of its box centre (~0.6 m up) that the altitude would otherwise chase.
        shot_offset = self.cinematic_planner.update(now)
        shot_heading = target.heading + target.yaw_rate * self.cfg.heading_prediction_sec
        rel_x_world, rel_y_world = body_to_world(shot_offset.x, shot_offset.y, shot_heading)
        desired_x = target.x + rel_x_world
        desired_y = target.y + rel_y_world
        desired_z = self.cfg.shot_ground_z + shot_offset.z

        ex = desired_x - drone_x
        ey = desired_y - drone_y
        ez = desired_z - drone_z
        # Face the target.
        desired_yaw = math.atan2(target.y - drone_y, target.x - drone_x)
        eyaw = wrap_to_pi(desired_yaw - self._drone_yaw())

        self.get_logger().info(
            f"SHOT offset=({shot_offset.x:.2f}, {shot_offset.y:.2f}, {shot_offset.z:.2f}), "
            f"desired=({desired_x:.2f}, {desired_y:.2f}, {desired_z:.2f}), "
            f"target=({target.x:.2f}, {target.y:.2f}, {target.z:.2f}), "
            f"target_vel=({target.vx:.2f}, {target.vy:.2f}, {target.vz:.2f}), "
            f"heading={math.degrees(target.heading):.1f}",
            throttle_duration_sec=0.5,
        )

        # Feedforward = velocity of the shot point (no vertical: fixed
        # altitude): the car's velocity plus, in a turn, yaw rate x offset --
        # the shot point swings around the car.
        ff_vx, ff_vy = 0.0, 0.0
        if self.cfg.enable_velocity_feedforward:
            ff_vx, ff_vy = target.vx, target.vy
        if self.cfg.enable_shot_rotation_feedforward:
            ff_vx -= target.yaw_rate * rel_y_world
            ff_vy += target.yaw_rate * rel_x_world
        cmd = self.pid.update(ex, ey, ez, eyaw, now, ff_vx=ff_vx, ff_vy=ff_vy)

        msg.twist.linear.x = cmd.vx
        msg.twist.linear.y = cmd.vy
        msg.twist.linear.z = cmd.vz
        msg.twist.angular.z = cmd.yaw_rate
        self.vel_pub.publish(msg)

        self._log_row(
            now, "stage2", drone_xyz,
            (desired_x, desired_y, desired_z),
            (ex, ey, ez, math.degrees(eyaw)),
            (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate),
        )

    def _publish_gimbal_setpoint(self):
        """Gimbal: keep the detection centred; sweep or aim at the estimate otherwise."""
        now = self._now()
        if not self.have_drone_pose:
            self.gimbal.send(0.0, 0.0, now)
            return

        # 1. Target visible: image-space PD on the camera-frame detection.
        if self._detection_fresh(now):
            d = self.detection
            cmd = self.gimbal_ctrl.update_image_pd(
                cam_x=d.cam_x, cam_y=d.cam_y, cam_z=d.cam_z,
                current_pitch=self.gimbal.current_pitch,
                current_yaw=self.gimbal.current_yaw,
                now=now,
            )
            self.gimbal.send(cmd.pitch, cmd.yaw, now)
            return

        # 2. No target: search sweep.
        if not self.target_received or not self._target_is_fresh(now):
            if self.cfg.enable_search_mode and self.flight.tracking_enabled:
                self._search_sweep(now)
            return

        # 3. Fallback: point at the target estimate.
        drone_x, drone_y, drone_z = self._drone_xyz()
        cmd = self.gimbal_ctrl.update(
            drone_x=drone_x, drone_y=drone_y, drone_z=drone_z, drone_yaw=self._drone_yaw(),
            target_x=self.estimator.x, target_y=self.estimator.y, target_z=self.estimator.z,
        )
        self.gimbal.send(cmd.pitch, cmd.yaw, now)

    def _search_sweep(self, now: float):
        self._enter_search_mode()
        cfg = self.cfg
        t = now - self._search_start_time
        phase = 2.0 * math.pi * (t / max(1.0, cfg.search_gimbal_period_sec))
        yaw_cmd = math.radians(cfg.search_gimbal_yaw_amp_deg) * math.sin(phase)
        pitch_cmd = (
            math.radians(cfg.search_gimbal_pitch_center_deg)
            + math.radians(cfg.search_gimbal_pitch_amp_deg) * math.sin(0.5 * phase)
        )
        self.gimbal.send(pitch_cmd, yaw_cmd, now)

    def destroy_node(self):
        if hasattr(self, "run_log"):
            self.get_logger().info(f"CSV saved: {self.run_log.close()}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RelativePositionController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
