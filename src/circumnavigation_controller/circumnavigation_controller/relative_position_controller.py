#!/usr/bin/env python3
import math
import csv
from datetime import datetime

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_msgs.msg import Float64
import tf2_ros

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
        self.declare_parameter("target_timeout_sec", 1.0)
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

        # Get the gimbal frame parameters
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        
        self.declare_parameter("gimbal_base_frame", "base_link_frd")
        self.declare_parameter("gimbal_frame", "gimbal_0")

        self.gimbal_base_frame = self.get_parameter("gimbal_base_frame").value
        self.gimbal_frame = self.get_parameter("gimbal_frame").value

        # Configure gimbal parameters and gimbal controller parameters
        self._gimbal_configured = False
        self._last_gimbal_cmd_time = 0.0

        self.current_gimbal_pitch = 0.0
        self.current_gimbal_yaw = 0.0

        # Compass heading
        self.compass_heading_rad = 0.0
        self.have_compass = False

        # Gimbal status and orientation
        self.have_gimbal_attitude = False
        self.T_frd_gimbal = np.eye(4)

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
        self.target_odom_sub = self.create_subscription(
            Odometry, self.target_odom_topic, self._on_target_odom, 10
        )

        self.visual_odom_sub = self.create_subscription(
            Odometry, "/aruco_target/visual_odom", self._on_visual_odom, 10
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
            f"RelativePositionController started. Target odom topic: {self.target_odom_topic}"
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

    def _on_target_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.target_x = p.x
        self.target_y = p.y
        self.target_z = p.z

        # self.target_heading = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        self.target_received = True
        self.last_target_time = self._now()

       

    def _on_visual_odom(self, msg):
        if not self.have_drone_pose:
            return

        if not self.have_gimbal_attitude:
            self.get_logger().warn(
                "No MAVROS gimbal attitude feedback received yet.",
                throttle_duration_sec=2.0
            )
            return

        

        # ArUco detector gives camera/gimbal-relative target orientation.
        q_vis = msg.pose.pose.orientation
        T_camera_target = tf_transformations.quaternion_matrix([
            q_vis.x,
            q_vis.y,
            q_vis.z,
            q_vis.w,
        ])

        # Drone pose from MAVROS local position.
        q_drone = self.drone_pose.pose.orientation
        T_world_body_flu = tf_transformations.quaternion_matrix([
            q_drone.x,
            q_drone.y,
            q_drone.z,
            q_drone.w,
        ])

        p_drone = self.drone_pose.pose.position
        T_world_body_flu[0, 3] = p_drone.x
        T_world_body_flu[1, 3] = p_drone.y
        T_world_body_flu[2, 3] = p_drone.z

        # MAVROS gimbal TF uses base_link_frd.
        # Convert body FLU -> body FRD.
        T_flu_frd = np.eye(4)
        T_flu_frd[:3, :3] = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])

        # Live gimbal transform from MAVROS TF:
        T_frd_gimbal = self.T_frd_gimbal

        # Approximate optical camera correction.
        # If gimbal_0 already behaves like your camera optical frame, set this to identity.
        T_gimbal_camera = np.eye(4)

        # Final chain:
        # world -> drone_body_flu -> drone_body_frd -> gimbal -> camera -> aruco_target
        T_world_target = (
            T_world_body_flu
            @ T_flu_frd
            @ T_frd_gimbal
            @ T_gimbal_camera
            @ T_camera_target
        )

        q_world_target = tf_transformations.quaternion_from_matrix(T_world_target)

        visual_heading_world = self._quat_to_yaw(
            q_world_target[0],
            q_world_target[1],
            q_world_target[2],
            q_world_target[3],
        )

        self.target_heading = self._wrap_to_pi(visual_heading_world)

        self.get_logger().info(
            f"Visual heading using MAVROS gimbal TF: "
            f"{math.degrees(self.target_heading):.2f} deg",
            throttle_duration_sec=0.5
        )


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

        self.have_gimbal_attitude = True

        roll, pitch, yaw = tf_transformations.euler_from_quaternion([
            q.x,
            q.y,
            q.z,
            q.w,
        ])

        # self.get_logger().info(
        #     f"Gimbal attitude feedback: "
        #     f"roll={math.degrees(roll):.2f}, "
        #     f"pitch={math.degrees(pitch):.2f}, "
        #     f"yaw={math.degrees(yaw):.2f}, "
        #     f"flags={msg.flags}",
        #     throttle_duration_sec=1.0
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
        if self._gimbal_configured:
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
        req.gimbal_device_id = 0

        fut = self.gimbal_config_client.call_async(req)
        fut.add_done_callback(self._on_gimbal_config_done)

    
    def _on_gimbal_config_done(self, fut):
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

        if not self.target_received:
            self.get_logger().warn(
                "Tracking enabled but no target odom received — holding.",
                throttle_duration_sec=5.0,
            )
            self._publish_zero(msg)
            return

        if not self._target_is_fresh(now):
            self.get_logger().warn(
                "Target odom stale — holding.",
                throttle_duration_sec=2.0,
            )
            self._publish_zero(msg)
            return

        if self.have_compass:
            drone_yaw = self.compass_heading_rad
        else:
            q = self.drone_pose.pose.orientation
            drone_yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        # --------------------------------------------------
        # Use target position from /aruco_target/odom,
        # and target_heading from /aruco_target/visual_odom.
        #
        # The desired relative point is:
        # target_position + R(target_heading) * body_offset
        # --------------------------------------------------
        rel_x_world, rel_y_world = self._body_to_world(
            self.rel_x_body,
            self.rel_y_body,
            self.target_heading,
        )

        desired_x = self.target_x + rel_x_world
        desired_y = self.target_y + rel_y_world
        desired_z = self.target_z + self.rel_z_body

        
        ex = desired_x - drone_x
        ey = desired_y - drone_y
        ez = desired_z - drone_z

        desired_yaw = math.atan2(
            self.target_y - drone_y,
            self.target_x - drone_x,
        )
        eyaw = self._wrap_to_pi(desired_yaw - drone_yaw)
        # eyaw = 0

        # self.get_logger().info(f"desired_x: {desired_x:.2f}, desired_y: {desired_y:.2f}, desired_z: {desired_z:.2f}")
        # self.get_logger().info(f"ex: {ex:.2f}, ey: {ey:.2f}, ez: {ez:.2f}, eyaw: {math.degrees(eyaw):.2f} deg")
        # self.get_logger().info(f"desired_yaw: {math.degrees(desired_yaw):.2f} deg")
        # self.get_logger().info(f"drone_yaw: {math.degrees(drone_yaw):.2f} deg")
        # self.get_logger().info(f"eyaw: {math.degrees(eyaw):.2f} deg")

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
        if not self.target_received or not self.have_drone_pose:
            return

        now = self._now()
        if not self._target_is_fresh(now):
            return

        drone_x = self.drone_pose.pose.position.x
        drone_y = self.drone_pose.pose.position.y
        drone_z = self.drone_pose.pose.position.z
        q = self.drone_pose.pose.orientation
        drone_yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        cmd = self.gimbal.update(
            drone_x=drone_x,
            drone_y=drone_y,
            drone_z=drone_z,
            drone_yaw=drone_yaw,
            target_x=self.target_x,
            target_y=self.target_y,
            target_z=self.target_z,
        )

        # MAVROS GimbalManagerPitchyaw supports pitch/yaw, not roll.
        self._send_gimbal_pitchyaw(
            pitch_rad=-cmd.pitch,
            yaw_rad=-cmd.yaw,
        )

    
    def _send_gimbal_pitchyaw(self, pitch_rad, yaw_rad):
        if not self._gimbal_configured:
            self._configure_gimbal()
            return

        now = self._now()

        # Limit service calls to around 5 Hz.
        if now - self._last_gimbal_cmd_time < 0.2:
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
        req.gimbal_device_id = 0

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