#!/usr/bin/env python3
import math
import csv
from datetime import datetime

try:
    from .cinematic_planner import CinematicPlanner
except ImportError:
    from cinematic_planner import CinematicPlanner
    
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_msgs.msg import Float64, Bool
try:
    from .pid_controller import PIDRelativeController
    from .gimbal_controller import GimbalController
except ImportError:
    from pid_controller import PIDRelativeController
    from gimbal_controller import GimbalController

import numpy as np
if not hasattr(np, "float"):
    np.float = float

import tf_transformations
from mavros_msgs.srv import GimbalManagerConfigure, GimbalManagerPitchyaw
from mavros_msgs.msg import GimbalDeviceAttitudeStatus

class RelativePositionController(Node):
    def __init__(self):
        super().__init__("relative_position_controller")

        # ---------------- QoS ----------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )



        # ---------------- Parameters ----------------
        self.declare_parameter("target_odom_topic", "/aruco_target/odom")
        self.declare_parameter("target_altitude", 3.0)
        self.declare_parameter("rel_x_body", -2.0)
        self.declare_parameter("rel_y_body", 0.0)
        self.declare_parameter("rel_z_body", 3.0)
        self.declare_parameter("target_timeout_sec", 5.0)
        self.declare_parameter("hover_delay_sec", 4.0)
        self.declare_parameter("boundary_limit", 40.0)

        self.target_odom_topic = self.get_parameter("target_odom_topic").value
        self.target_altitude = float(self.get_parameter("target_altitude").value)
        self.rel_x_body = float(self.get_parameter("rel_x_body").value)
        self.rel_y_body = float(self.get_parameter("rel_y_body").value)
        self.rel_z_body = float(self.get_parameter("rel_z_body").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.hover_delay_sec = float(self.get_parameter("hover_delay_sec").value)
        self.boundary_limit = float(self.get_parameter("boundary_limit").value)

        # Gimbal/camera calibration. MAVROS gimbal feedback gives the
        # base_link_frd -> gimbal attitude. This fixed transform maps the
        # gimbal frame into the camera/OpenCV frame used by the ArUCo detector.
        self.declare_parameter("gimbal_camera_roll_deg", 0.0)
        self.declare_parameter("gimbal_camera_pitch_deg", 0.0)
        self.declare_parameter("gimbal_camera_yaw_deg", 0.0)
        self.T_gimbal_camera = tf_transformations.euler_matrix(
            math.radians(float(self.get_parameter("gimbal_camera_roll_deg").value)),
            math.radians(float(self.get_parameter("gimbal_camera_pitch_deg").value)),
            math.radians(float(self.get_parameter("gimbal_camera_yaw_deg").value)),
            axes="sxyz",
        )

        # Visual-heading filtering. ArUCo orientation is noisy at long range,
        # so update target heading only when the visual estimate is plausible.
        self.declare_parameter("visual_heading_alpha", 0.25)
        self.declare_parameter("max_visual_heading_jump_deg", 60.0)
        self.declare_parameter("max_visual_heading_distance", 5.0)
        self.declare_parameter("enable_visual_heading_filter", True)
        self.visual_heading_alpha = float(self.get_parameter("visual_heading_alpha").value)
        self.max_visual_heading_jump = math.radians(
            float(self.get_parameter("max_visual_heading_jump_deg").value)
        )
        self.max_visual_heading_distance = float(
            self.get_parameter("max_visual_heading_distance").value
        )
        self.enable_visual_heading_filter = bool(
            self.get_parameter("enable_visual_heading_filter").value
        )
        self.visual_heading_valid = False
        self.filtered_target_heading = 0.0

        # Vision-only target position. /aruco_target/visual_odom contains the
        # camera-frame target position from solvePnP/tvec. When enabled, the
        # controller transforms that position into the MAVROS local/world frame
        # and uses it as target_x/y/z. External target odom can still be kept as
        # ground truth for debugging.
        self.declare_parameter("use_visual_target_position", True)
        self.declare_parameter("use_external_target_position", False)
        self.declare_parameter("enable_visual_position_filter", True)
        self.declare_parameter("visual_position_alpha", 0.45)
        self.declare_parameter("max_visual_position_jump", 3.0)
        self.declare_parameter("min_visual_target_distance", 0.20)
        self.use_visual_target_position = bool(
            self.get_parameter("use_visual_target_position").value
        )
        self.use_external_target_position = bool(
            self.get_parameter("use_external_target_position").value
        )
        self.enable_visual_position_filter = bool(
            self.get_parameter("enable_visual_position_filter").value
        )
        self.visual_position_alpha = float(
            self.get_parameter("visual_position_alpha").value
        )
        self.max_visual_position_jump = float(
            self.get_parameter("max_visual_position_jump").value
        )
        self.min_visual_target_distance = float(
            self.get_parameter("min_visual_target_distance").value
        )
        self.visual_position_valid = False
        self.filtered_target_position = np.zeros(3, dtype=float)

        # Search/acquisition behaviour. Before a visual target is received,
        # do not hold at the initial zero target. Instead, slowly explore and
        # sweep the gimbal until /aruco_target/visual_odom arrives.
        self.declare_parameter("enable_search_mode", True)
        self.declare_parameter("search_forward_speed", 0.25)
        self.declare_parameter("search_yaw_rate_deg", 10.0)
        self.declare_parameter("search_altitude_kp", 0.5)
        self.declare_parameter("search_max_vertical_speed", 0.4)
        self.declare_parameter("search_gimbal_pitch_center_deg", -35.0)
        self.declare_parameter("search_gimbal_pitch_amp_deg", 15.0)
        self.declare_parameter("search_gimbal_yaw_amp_deg", 75.0)
        self.declare_parameter("search_gimbal_period_sec", 8.0)

        self.enable_search_mode = bool(self.get_parameter("enable_search_mode").value)
        self.search_forward_speed = float(self.get_parameter("search_forward_speed").value)
        self.search_yaw_rate = math.radians(
            float(self.get_parameter("search_yaw_rate_deg").value)
        )
        self.search_altitude_kp = float(self.get_parameter("search_altitude_kp").value)
        self.search_max_vertical_speed = float(
            self.get_parameter("search_max_vertical_speed").value
        )
        self.search_gimbal_pitch_center = math.radians(
            float(self.get_parameter("search_gimbal_pitch_center_deg").value)
        )
        self.search_gimbal_pitch_amp = math.radians(
            float(self.get_parameter("search_gimbal_pitch_amp_deg").value)
        )
        self.search_gimbal_yaw_amp = math.radians(
            float(self.get_parameter("search_gimbal_yaw_amp_deg").value)
        )
        self.search_gimbal_period_sec = max(
            1.0, float(self.get_parameter("search_gimbal_period_sec").value)
        )
        

        # Configure gimbal parameters and gimbal controller parameters.
        # The service expects degrees; GimbalController returns radians.
        self.declare_parameter("gimbal_device_id", 0)
        self.declare_parameter("gimbal_command_period_sec", 0.2)
        self.declare_parameter("gimbal_pitch_sign", -1.0)
        self.declare_parameter("gimbal_yaw_sign", -1.0)
        self.gimbal_device_id = int(self.get_parameter("gimbal_device_id").value)
        self.gimbal_command_period_sec = float(
            self.get_parameter("gimbal_command_period_sec").value
        )
        self.gimbal_pitch_sign = float(self.get_parameter("gimbal_pitch_sign").value)
        self.gimbal_yaw_sign = float(self.get_parameter("gimbal_yaw_sign").value)

        self._gimbal_configured = False
        self._gimbal_config_in_progress = False
        self._last_gimbal_cmd_time = 0.0

        self.current_gimbal_pitch = 0.0
        self.current_gimbal_yaw = 0.0

        # Compass heading
        self.compass_heading_rad = 0.0
        self.have_compass = False

        # Gimbal status and orientation
        self.have_gimbal_attitude = False
        self.T_frd_gimbal = np.eye(4)
        self.gimbal_feedback_flags = 0

        self.T_gimbal_camera = np.eye(4)
        self.T_gimbal_camera[:3, :3] = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        # ---------------- Pure controllers ----------------
        self.pid = PIDRelativeController(dt=0.15)
        self.gimbal = GimbalController()

        # ---------------- Subscriptions ----------------
        self.state_sub = self.create_subscription(
            State, "/mavros/state", self._on_state, reliable_qos
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self._on_drone_pose, best_effort_qos
        )
        # self.target_odom_sub = self.create_subscription(
        #     Odometry, self.target_odom_topic, self._on_target_odom, 10
        # )

        self.visual_odom_sub = self.create_subscription(
            Odometry, "/aruco_target/visual_odom", self._on_visual_odom, 10
        )

        self.target_found_sub = self.create_subscription(
            Bool, "/aruco_target/found", self._on_target_found, 10
        )

        self.compass_subscription = self.create_subscription(
            Float64,
            "/mavros/global_position/compass_hdg",
            self.compass_callback,
            mavros_qos
        )


        self.gimbal_attitude_sub = self.create_subscription(
            GimbalDeviceAttitudeStatus,
            "/mavros/gimbal_control/device/attitude_status",
            self._on_gimbal_attitude_status,
            10
        )

        # ---------------- Publishers ----------------
        self.vel_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )

        self.gimbal_config_client = self.create_client(
            GimbalManagerConfigure,
            "/mavros/gimbal_control/manager/configure"
        )

        self.gimbal_pitchyaw_client = self.create_client(
            GimbalManagerPitchyaw,
            "/mavros/gimbal_control/manager/pitchyaw"
        )

        # ---------------- MAVROS service clients ----------------
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")

        # ---------------- Drone state ----------------
        self.fcu_state = State()
        self.drone_pose = PoseStamped()
        self.have_drone_pose = False

        # ---------------- Target state ----------------
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_heading = 0.0
        self.target_received = False
        self.last_target_time = None

        # Latest visual target measurement in camera optical frame
        self._search_start_time = None
        self._last_search_log_time = 0.0
        self._last_target_found_msg = None
        self.visual_cam_x = 0.0
        self.visual_cam_y = 0.0
        self.visual_cam_z = 0.0
        self.last_visual_camera_time = None

        # ---------------- Cinmatic planner ----------------
        self.cinematic_planner = CinematicPlanner()

        self.cinematic_planner.set_sequence([
            # {
            #     "type": "hold_location",
            #     "location": "right",
            #     "radius": 3.0,
            #     "height": 3.0,
            #     "duration": 20.0,
            # },
            {
                "type": "move_location",
                "from": "back",
                "to": "front",
                "radius": 3.0,
                "height": 3.0,
                "duration": 10.0,
            },
            {
                "type": "overpass",
                "from": "front",
                "to": "back",
                "radius": 3.0,
                "start_height": 3.0,
                "peak_height": 3.5,
                "end_height": 3.0,
                "duration": 15.0,
            },
            {
                "type": "hold_location",
                "location": "right",
                "radius": 3.0,
                "height": 3.0,
                "duration": 5.0,
            },
            {
                "type": "move_location",
                "from": "right",
                "to": "left",
                "via": "front",
                "radius": 3.0,
                "height": 3.0,
                "duration": 12.0,
            },


        ]) 


        # ---------------- Flight sequence flags ----------------
        self._guided_requested = False
        self._guided_confirmed = False
        self._arm_requested = False
        self._armed_confirmed = False
        self._tko_requested = False
        self._tko_reached = False
        self._tracking_enabled = False
        self._rtl_initiated = False
        self._armed_time = None
        self._takeoff_complete_time = None
        self._armed_wait_logged = False

        # ---------------- Timers ----------------
        self.setpoint_timer = self.create_timer(0.15, self._publish_setpoint)
        self.gimbal_timer = self.create_timer(0.10, self._publish_gimbal_setpoint)
        self.orchestrator_timer = self.create_timer(0.20, self._orchestrate)
        self.safety_timer = self.create_timer(1.00, self._check_safety)

        # ---------------- Logging ----------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"relative_pid_{ts}.csv"
        self.csv_file = open(self.csv_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time", "stage",
            "drone_x", "drone_y", "drone_z",
            "target_x", "target_y", "target_z", "target_heading_deg",
            "desired_x", "desired_y", "desired_z",
            "ex", "ey", "ez", "eyaw_deg",
            "vx", "vy", "vz", "yaw_rate",
            "gimbal_roll", "gimbal_pitch", "gimbal_yaw",
        ])

        self.get_logger().info(
            "RelativePositionController started. "
            f"Vision target source: /aruco_target/visual_odom, "
            f"external/debug odom topic: {self.target_odom_topic}, "
            f"search_mode={self.enable_search_mode}"
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _target_is_fresh(self, now: float) -> bool:
        return self.last_target_time is not None and (now - self.last_target_time) <= self.target_timeout_sec

    def _target_available(self, now: float) -> bool:
        return self.target_received and self._target_is_fresh(now)

    def _enter_search_mode(self):
        if self._search_start_time is None:
            self._search_start_time = self._now()
            self.pid.reset()
            self.get_logger().warn(
                "No visual target available — entering SEARCH mode. "
                "Drone will move slowly and sweep the gimbal."
            )

    def _exit_search_mode(self):
        if self._search_start_time is not None:
            self._search_start_time = None
            self.pid.reset()
            self.get_logger().info("Visual target acquired — switching to TRACK mode.")

    def _get_drone_yaw(self) -> float:
        if self.have_compass:
            return self.compass_heading_rad

        q = self.drone_pose.pose.orientation
        return self._quat_to_yaw(q.x, q.y, q.z, q.w)

    def _visual_camera_target_fresh(self, now: float) -> bool:
        return (
            self.last_visual_camera_time is not None
            and (now - self.last_visual_camera_time) <= self.target_timeout_sec
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_state(self, msg: State):
        self.fcu_state = msg

        if msg.mode == "GUIDED" and not self._guided_confirmed:
            self._guided_confirmed = True
            self.get_logger().info("GUIDED confirmed.")

        if msg.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self._now()
            self.get_logger().info("Armed confirmed.")

    def _on_drone_pose(self, msg: PoseStamped):
        self.drone_pose = msg
        self.have_drone_pose = True

        alt = msg.pose.position.z
        if self._armed_confirmed and not self._tko_reached and alt >= self.target_altitude - 0.5:
            self._tko_reached = True
            self._takeoff_complete_time = self._now()
            self.get_logger().info(f"Takeoff complete at {alt:.2f} m.")


    def _on_visual_odom(self, msg: Odometry):
        if not self.have_drone_pose:
            return

        if not self.have_gimbal_attitude:
            self.get_logger().warn(
                "No gimbal attitude yet, cannot transform visual target to local frame.",
                throttle_duration_sec=2.0,
            )
            return

        # --------------------------------------------------
        # 1. Camera optical -> target
        # /aruco_target/visual_odom is in OpenCV camera optical frame:
        # x = right, y = down, z = forward
        # --------------------------------------------------
        p_vis = msg.pose.pose.position
        q_vis = msg.pose.pose.orientation

        T_camera_target = tf_transformations.quaternion_matrix([
            q_vis.x,
            q_vis.y,
            q_vis.z,
            q_vis.w,
        ])

        T_camera_target[0, 3] = p_vis.x
        T_camera_target[1, 3] = p_vis.y
        T_camera_target[2, 3] = p_vis.z


        # --------------------------------------------------
        # 2. Local/world -> drone body FLU
        # --------------------------------------------------
        p_drone = self.drone_pose.pose.position
        q_drone = self.drone_pose.pose.orientation

        T_world_body_flu = tf_transformations.quaternion_matrix([
            q_drone.x,
            q_drone.y,
            q_drone.z,
            q_drone.w,
        ])

        T_world_body_flu[0, 3] = p_drone.x
        T_world_body_flu[1, 3] = p_drone.y
        T_world_body_flu[2, 3] = p_drone.z

        # --------------------------------------------------
        # 3. Body FRD -> body FLU
        # This lets us chain world -> FLU -> FRD
        # --------------------------------------------------
        T_flu_frd = np.eye(4)
        T_flu_frd[:3, :3] = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])

        # --------------------------------------------------
        # 4. Full transform:
        # world -> body FLU -> body FRD -> gimbal FRD -> camera optical -> target
        # --------------------------------------------------
        T_world_target = (
            T_world_body_flu
            @ T_flu_frd
            @ self.T_frd_gimbal
            @ self.T_gimbal_camera
            @ T_camera_target
        )

        # --------------------------------------------------
        # 5. Update target position
        # --------------------------------------------------
        self.visual_cam_x = float(p_vis.x)
        self.visual_cam_y = float(p_vis.y)
        self.visual_cam_z = float(p_vis.z)
        self.last_visual_camera_time = self._now()

        self.target_x = float(T_world_target[0, 3])
        self.target_y = float(T_world_target[1, 3])
        self.target_z = float(T_world_target[2, 3])

        # --------------------------------------------------
        # 6. Update target heading
        # --------------------------------------------------
        q_world_target = tf_transformations.quaternion_from_matrix(T_world_target)

        self.target_heading = self._wrap_to_pi(
            self._quat_to_yaw(
                q_world_target[0],
                q_world_target[1],
                q_world_target[2],
                q_world_target[3],
            )
        )

        self.target_received = True
        self.last_target_time = self._now()

        if hasattr(self, "_exit_search_mode"):
            self._exit_search_mode()

        visual_distance = math.sqrt(
            p_vis.x * p_vis.x +
            p_vis.y * p_vis.y +
            p_vis.z * p_vis.z
        )

        self.get_logger().info(
            f"Visual target: camera_dist={visual_distance:.2f} m, "
            f"camera_optical=({p_vis.x:.2f}, {p_vis.y:.2f}, {p_vis.z:.2f}), ",
            # f"local=({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), "
            # f"heading={math.degrees(self.target_heading):.2f} deg",
            throttle_duration_sec=0.5,
        )

    def _on_target_found(self, msg: Bool):
        self._last_target_found_msg = bool(msg.data)
        if not msg.data:
            return

        # self.get_logger().info(
        #     "ArUco marker detected in image; waiting for visual odometry pose.",
        #     throttle_duration_sec=2.0,
        # )

    def compass_callback(self, msg: Float64):
        heading_deg = float(msg.data)

        # compass: 0=N, 90=E
        # ROS ENU yaw: 0=E, 90=N
        self.compass_heading_rad = self._wrap_to_pi(
            math.radians(90.0 - heading_deg)
        )

        self.have_compass = True
        # self.get_logger().info(
        #     f"Compass heading: {heading_deg:.2f} deg, "
        #     f"ENU yaw: {math.degrees(self.compass_heading_rad):.2f} deg",
        #     throttle_duration_sec=1.0
        # )

    def _on_gimbal_attitude_status(self, msg):
        q = msg.q

        self.T_frd_gimbal = tf_transformations.quaternion_matrix([
            q.x,
            q.y,
            q.z,
            q.w,
        ])
        self.gimbal_feedback_flags = msg.flags
        self.have_gimbal_attitude = True

        roll, pitch, yaw = tf_transformations.euler_from_quaternion([
            q.x,
            q.y,
            q.z,
            q.w,
        ])

        # self.get_logger().info(
        #     f"Gimbal feedback: roll={math.degrees(roll):.2f} deg, "
        #     f"pitch={math.degrees(pitch):.2f} deg, "
        #     f"yaw={math.degrees(yaw):.2f} deg, flags={msg.flags}",
        #     throttle_duration_sec=2.0,
        # )

    def _transform_to_matrix(self, tf_msg):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        T = tf_transformations.quaternion_matrix([
            q.x, q.y, q.z, q.w
        ])

        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        return T
    

    def _wrap_to_pi(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _angle_diff(self, a, b):
        return self._wrap_to_pi(a - b)

    def _smooth_angle(self, old_angle, new_angle, alpha):
        diff = self._wrap_to_pi(new_angle - old_angle)
        return self._wrap_to_pi(old_angle + alpha * diff)

    def _body_to_world(self, x_body, y_body, heading):
        c = math.cos(heading)
        s = math.sin(heading)

        x_world = c * x_body - s * y_body
        y_world = s * x_body + c * y_body

        return x_world, y_world
    # ------------------------------------------------------------------
    # Flight orchestration
    # ------------------------------------------------------------------
    def _orchestrate(self):
        if not self.fcu_state.connected:
            return

        if not self._guided_confirmed:
            if not self._guided_requested:
                self._request_guided()
            return

        if not self.have_drone_pose:
            return

        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            return

        if self._armed_time is None:
            return

        if not self._tko_requested:
            elapsed = self._now() - self._armed_time
            if elapsed < 5.0:
                if not self._armed_wait_logged:
                    self.get_logger().info("Armed — waiting 5 s before takeoff...")
                    self._armed_wait_logged = True
                return
            self._request_takeoff()
            return

        if self._tko_reached and not self._tracking_enabled and self._takeoff_complete_time is not None:
            hover_elapsed = self._now() - self._takeoff_complete_time
            if hover_elapsed >= self.hover_delay_sec:
                self.pid.reset()
                self._tracking_enabled = True
                self.get_logger().info("Tracking enabled.")

    def _request_guided(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SetMode service not ready.")
            return
        self._guided_requested = True
        req = SetMode.Request()
        req.custom_mode = "GUIDED"
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(self._on_set_mode_done)
        self.get_logger().info("Requesting GUIDED...")

    def _on_set_mode_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"SetMode error: {e}")
            self._guided_requested = False
            return
        if not getattr(res, "mode_sent", False):
            self.get_logger().error("GUIDED rejected by FCU.")
            self._guided_requested = False

    def _request_arm(self):
        if not self.arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Arming service not ready.")
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        fut = self.arming_client.call_async(req)
        fut.add_done_callback(self._on_arm_done)
        self.get_logger().info("Requesting ARM...")

    def _on_arm_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Arm error: {e}")
            self._arm_requested = False
            return
        if not getattr(res, "success", False):
            self.get_logger().error("Arm rejected by FCU.")
            self._arm_requested = False

    def _request_takeoff(self):
        if not self.takeoff_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Takeoff service not ready.")
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self.target_altitude)
        fut = self.takeoff_client.call_async(req)
        fut.add_done_callback(self._on_takeoff_done)
        self.get_logger().info(f"Requesting takeoff to {self.target_altitude:.1f} m...")

    def _on_takeoff_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Takeoff error: {e}")
            self._tko_requested = False
            return
        if not getattr(res, "success", False):
            self.get_logger().error("Takeoff rejected by FCU.")
            self._tko_requested = False

    def _configure_gimbal(self):
        if self._gimbal_configured or self._gimbal_config_in_progress:
            return

        if not self.gimbal_config_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                "Gimbal configure service not ready.",
                throttle_duration_sec=2.0
            )
            return

        req = GimbalManagerConfigure.Request()
        req.sysid_primary = -2
        req.compid_primary = -2
        req.sysid_secondary = 0
        req.compid_secondary = 0
        req.gimbal_device_id = self.gimbal_device_id

        self._gimbal_config_in_progress = True
        fut = self.gimbal_config_client.call_async(req)
        fut.add_done_callback(self._on_gimbal_config_done)

    
    def _on_gimbal_config_done(self, fut):
        self._gimbal_config_in_progress = False
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"Gimbal configure failed: {e}")
            return

        if res.success:
            self._gimbal_configured = True
            self.get_logger().info("Gimbal manager configured.")
        else:
            self.get_logger().warn(f"Gimbal configure rejected, result={res.result}")

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def _check_safety(self):
        if self._rtl_initiated or not self._tko_reached or self._takeoff_complete_time is None:
            return

        if self._now() - self._takeoff_complete_time >= 120.0:
            self._initiate_rtl("120 s timer expired")
            return

        x = self.drone_pose.pose.position.x
        y = self.drone_pose.pose.position.y
        if abs(x) > self.boundary_limit or abs(y) > self.boundary_limit:
            self._initiate_rtl(f"boundary violation ({x:.1f}, {y:.1f})")

    def _initiate_rtl(self, reason: str):
        if self._rtl_initiated:
            return
        self._rtl_initiated = True
        self._tracking_enabled = False
        self.get_logger().warn(f"SAFETY RTL: {reason}")

        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("SetMode service not ready for RTL.")
            return

        req = SetMode.Request()
        req.custom_mode = "RTL"
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_rtl_done(f, reason))

    def _on_rtl_done(self, fut, reason):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"RTL error: {e}")
            return
        if getattr(res, "mode_sent", False):
            self.get_logger().info(f"RTL accepted ({reason})")
        else:
            self.get_logger().error(f"RTL rejected ({reason})")

    # ------------------------------------------------------------------
    # Control loops
    # ------------------------------------------------------------------
    def _publish_setpoint(self):
        if self._rtl_initiated:
            return

        now = self._now()
        drone_x = self.drone_pose.pose.position.x
        drone_y = self.drone_pose.pose.position.y
        drone_z = self.drone_pose.pose.position.z

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        if not self._tracking_enabled:
            self._log_csv(
                "stage1", drone_x, drone_y, drone_z,
                drone_x, drone_y, drone_z,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            )
            return

        # if not self._target_available(now):
        #     if self.enable_search_mode:
        #         self._publish_search_setpoint(msg, drone_x, drone_y, drone_z)
        #     else:
        #         self.get_logger().warn(
        #             "Tracking enabled but no visual target received — holding.",
        #             throttle_duration_sec=5.0,
        #         )
        #         self._publish_zero(msg)
        #     return

        self._exit_search_mode()
        drone_yaw = self._get_drone_yaw()

        # --------------------------------------------------
        # Use target position from visual ArUco odom when enabled.
        # target_position + R(target_heading) * body_offset
        # --------------------------------------------------

        shot_offset = self.cinematic_planner.update(now)

        rel_x_world, rel_y_world = self._body_to_world(
            shot_offset.x,
            shot_offset.y,
            self.target_heading,
        )

        desired_x = self.target_x + rel_x_world
        desired_y = self.target_y + rel_y_world
        desired_z = self.target_z + shot_offset.z

        # rel_x_world, rel_y_world = self._body_to_world(
        #     self.rel_x_body,
        #     self.rel_y_body,
        #     self.target_heading,
        # )

        # desired_x = self.target_x + rel_x_world
        # desired_y = self.target_y + rel_y_world
        # desired_z = self.target_z + self.rel_z_body

        
        ex = desired_x - drone_x
        ey = desired_y - drone_y
        ez = desired_z - drone_z

        desired_yaw = math.atan2(
            self.target_y - drone_y,
            self.target_x - drone_x,
        )
        eyaw = self._wrap_to_pi(desired_yaw - drone_yaw)
    

        self.get_logger().info(
        f"SHOT offset=({shot_offset.x:.2f}, {shot_offset.y:.2f}, {shot_offset.z:.2f}), "
        f"desired=({desired_x:.2f}, {desired_y:.2f}, {desired_z:.2f}), "
        f"target=({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), "
        f"heading={math.degrees(self.target_heading):.1f}",
        throttle_duration_sec=0.5,
)

        cmd = self.pid.update_from_error(
            ex=ex,
            ey=ey,
            ez=ez,
            eyaw=eyaw,
            desired_x=desired_x,
            desired_y=desired_y,
            desired_z=desired_z,
        )

        msg.twist.linear.x = cmd.vx
        msg.twist.linear.y = cmd.vy
        msg.twist.linear.z = cmd.vz
        msg.twist.angular.z = cmd.yaw_rate
        self.vel_pub.publish(msg)

        self._log_csv(
            "stage2", drone_x, drone_y, drone_z,
            cmd.desired_x, cmd.desired_y, cmd.desired_z,
            cmd.ex, cmd.ey, cmd.ez, math.degrees(cmd.eyaw),
            cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate,
            math.degrees(self.target_heading),
            rel_x_world,
            rel_y_world,
        )

    def _publish_gimbal_setpoint(self):
        if not self.have_drone_pose:
            self._send_gimbal_pitchyaw(
                pitch_rad=0.0,
                yaw_rad=0.0,
            )
            return

        now = self._now()

        # 1. If visual target is currently visible, use image-space PD.
        if self._visual_camera_target_fresh(now):
            cmd = self.gimbal.update_image_pd(
                cam_x=self.visual_cam_x,
                cam_y=self.visual_cam_y,
                cam_z=self.visual_cam_z,
                current_pitch=self.current_gimbal_pitch,
                current_yaw=self.current_gimbal_yaw,
                now=now,
            )

            self._send_gimbal_pitchyaw(
                pitch_rad=cmd.pitch,
                yaw_rad=cmd.yaw,
            )
            return

        # 2. If no visual target, search.
        if not self.target_received or not self._target_is_fresh(now):
            if self.enable_search_mode and self._tracking_enabled:
                self._publish_search_gimbal_setpoint()
            return

        # 3. Fallback: point at last known local target pose.
        drone_x = self.drone_pose.pose.position.x
        drone_y = self.drone_pose.pose.position.y
        drone_z = self.drone_pose.pose.position.z
        drone_yaw = self._get_drone_yaw()

        cmd = self.gimbal.update(
            drone_x=drone_x,
            drone_y=drone_y,
            drone_z=drone_z,
            drone_yaw=drone_yaw,
            target_x=self.target_x,
            target_y=self.target_y,
            target_z=self.target_z,
        )

        self._send_gimbal_pitchyaw(
            pitch_rad=cmd.pitch,
            yaw_rad=cmd.yaw,
        )

    
    def _publish_search_setpoint(self, msg: TwistStamped, drone_x: float, drone_y: float, drone_z: float):
        self._enter_search_mode()

        drone_yaw = self._get_drone_yaw()
        vx = self.search_forward_speed * math.cos(drone_yaw)
        vy = self.search_forward_speed * math.sin(drone_yaw)

        alt_error = self.target_altitude - drone_z
        vz = self.search_altitude_kp * alt_error
        vz = max(-self.search_max_vertical_speed, min(self.search_max_vertical_speed, vz))

        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = self.search_yaw_rate
        self.vel_pub.publish(msg)

        now = self._now()
        if now - self._last_search_log_time > 2.0:
            self._last_search_log_time = now
            # self.get_logger().info(
            #     f"SEARCH: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, "
            #     f"yaw_rate={math.degrees(self.search_yaw_rate):.1f} deg/s, "
            #     f"alt={drone_z:.2f} m"
            # )

        self._log_csv(
            "search", drone_x, drone_y, drone_z,
            drone_x, drone_y, self.target_altitude,
            0.0, 0.0, alt_error, 0.0,
            vx, vy, vz, self.search_yaw_rate,
            None, math.degrees(self.current_gimbal_pitch), math.degrees(self.current_gimbal_yaw),
        )

    def _publish_search_gimbal_setpoint(self):
        self._enter_search_mode()
        t = self._now() - self._search_start_time
        phase = 2.0 * math.pi * (t / self.search_gimbal_period_sec)

        yaw_cmd = self.search_gimbal_yaw_amp * math.sin(phase)
        pitch_cmd = (
            self.search_gimbal_pitch_center
            + self.search_gimbal_pitch_amp * math.sin(0.5 * phase)
        )

        self._send_gimbal_pitchyaw(
            pitch_rad=pitch_cmd,
            yaw_rad=yaw_cmd,
        )

    def _send_gimbal_pitchyaw(self, pitch_rad, yaw_rad):
        if not self._gimbal_configured:
            self._configure_gimbal()
            return

        now = self._now()

        # Limit service calls to avoid spamming COMMAND_LONG.
        if now - self._last_gimbal_cmd_time < self.gimbal_command_period_sec:
            return

        if not self.gimbal_pitchyaw_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                "Gimbal pitchyaw service not ready.",
                throttle_duration_sec=2.0
            )
            return

        req = GimbalManagerPitchyaw.Request()

        req.pitch = float(math.degrees(pitch_rad))
        req.yaw = float(math.degrees(yaw_rad))
        req.pitch_rate = 0.0
        req.yaw_rate = 0.0
        req.flags = 0
        req.gimbal_device_id = self.gimbal_device_id

        self.current_gimbal_pitch = pitch_rad
        self.current_gimbal_yaw = yaw_rad
        self._last_gimbal_cmd_time = now

        self.gimbal_pitchyaw_client.call_async(req)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_zero(self, msg: TwistStamped):
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.z = 0.0
        self.vel_pub.publish(msg)

    def _log_csv(
        self,
        stage,
        dx, dy, dz,
        des_x, des_y, des_z,
        ex, ey, ez, eyaw_deg,
        vx, vy, vz, yaw_rate,
        g_roll=None, g_pitch=None, g_yaw=None,
    ):
        self.csv_writer.writerow([
            self._now(), stage,
            dx, dy, dz,
            self.target_x, self.target_y, self.target_z,
            math.degrees(self.target_heading),
            des_x, des_y, des_z,
            ex, ey, ez, eyaw_deg,
            vx, vy, vz, yaw_rate,
            g_roll, g_pitch, g_yaw,
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()
            self.get_logger().info(f"CSV saved: {self.csv_filename}")
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