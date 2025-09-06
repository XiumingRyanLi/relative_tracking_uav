#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data  # handy preset



class CircumnavigationController(Node):
    def __init__(self):
        super().__init__('circumnavigation_controller')


        # For /mavros/state (usually fine as RELIABLE)
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # For /mavros/local_position/pose (match SensorData: BEST_EFFORT, VOLATILE)
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state_sub = self.create_subscription( State, '/mavros/state', self._on_state, state_qos)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose, pose_qos)

        self.err_sub = self.create_subscription( Float64, '/yaw_error', self._on_person_error, 10)
        self.tracking_enable_sub = self.create_subscription(Bool, '/tracking_enable', self._on_tracking_enable, 10)

        # ---- Publisher: always stream setpoints (>=2 Hz for GUIDED pre-arm) ----
        self.vel_pub = self.create_publisher( TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.setpoint_timer = self.create_timer(0.05, self._publish_setpoint)  # 20 Hz

        # ---- Services ----
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # ---- Internal state ----
        self.state = State()
        self.pose = PoseStamped()
        self.person_err = 0.0

        self.target_altitude = 4.0
        self.yaw_kp = 0.4
        self.max_yaw_rate = 1.0
        self._last_yaw_rate = 0.0

        # Idempotent step guards
        self._guided_requested = False
        self._guided_confirmed = False

        self._arm_requested = False
        self._armed_confirmed = False

        self._tko_requested = False
        self._tko_reached = False

        self._tracking_enabled = False
        self._tracking_manual_override = False  # Manual enable/disable via topic
        self._armed_time = None  # when FCU confirmed armed

        # Small non-blocking orchestration tick
        self.orchestrator = self.create_timer(0.2, self._orchestrate)  # 5 Hz

        self.get_logger().info('Circumnavigation Controller (callbacks) started')

    # ------------------------------------------------------------------
    # Topic callbacks
    # ------------------------------------------------------------------
    def _on_state(self, msg: State):
        self.state = msg

        # Confirm GUIDED
        if self.state.mode == 'GUIDED' and not self._guided_confirmed:
            self._guided_confirmed = True
            self.get_logger().info('GUIDED confirmed by FCU.')

        # Confirm ARMED
        if self.state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9  # store current time
            self.get_logger().info('Armed confirmed by FCU.')


    def _on_pose(self, msg: PoseStamped):
        self.pose = msg
        alt = msg.pose.position.z
        if self._armed_confirmed and not self._tko_reached and alt > (self.target_altitude - 0.5):
            self._tko_reached = True
            self.get_logger().info(f'Takeoff complete at {alt:.2f} m')
            self._enable_tracking()

    def _on_person_error(self, msg: Float64):
        self.person_err = float(msg.data)

    def _on_tracking_enable(self, msg: Bool):
        """Handle manual tracking enable/disable commands"""
        self._tracking_manual_override = msg.data
        if msg.data:
            self._enable_tracking_manual()
        else:
            self._disable_tracking_manual()
        self.get_logger().info(f'Manual tracking override: {"ENABLED" if msg.data else "DISABLED"}')

    # ------------------------------------------------------------------
    # Orchestration (non-blocking)
    # ------------------------------------------------------------------
    def _orchestrate(self):
        if not self.state.connected:
            return

        # 1) Ensure GUIDED
        if not self._guided_confirmed:
            if not self._guided_requested:
                self._request_guided()
            return

        # 2) Ensure ARMED
        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            return

        # 3) Wait 5 seconds after arming before takeoff
        if self._armed_confirmed and self._armed_time is not None and not self._tko_requested:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._armed_time
            if elapsed < 5.0:
                if not hasattr(self, '_armed_wait_logged') or not self._armed_wait_logged:
                    self.get_logger().info("Armed. Waiting 5s before takeoff...")
                    self._armed_wait_logged = True
                return
            else:
                self._request_takeoff()
                self._armed_wait_logged = False  # reset for next cycle
                return


    # ------------------------------------------------------------------
    # Service requests (async) + done-callbacks
    # ------------------------------------------------------------------
    def _request_guided(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('SetMode service not ready yet.')
            return
        self._guided_requested = True
        req = SetMode.Request()
        req.custom_mode = 'GUIDED'
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(self._on_set_mode_done)
        self.get_logger().info('Requesting GUIDED...')

    def _on_set_mode_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'SetMode exception: {e}')
            self._guided_requested = False  # allow retry
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info('GUIDED command accepted (awaiting FCU report).')
        else:
            self.get_logger().error('GUIDED command rejected by FCU.')
            self._guided_requested = False  # allow retry on next orchestrate tick

    def _request_arm(self):
        if not self.arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Arming service not ready yet.')
            return
        self._arm_requested = True
        req = CommandBool.Request()
        req.value = True
        fut = self.arming_client.call_async(req)
        fut.add_done_callback(self._on_arm_done)
        self.get_logger().info('Requesting ARM...')

    def _on_arm_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'Arming exception: {e}')
            self._arm_requested = False  # allow retry
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Arm accepted (awaiting FCU armed=true).')
        else:
            self.get_logger().error(f'Arm rejected by FCU (result={getattr(res, "result", None)}).')
            self._arm_requested = False  # allow retry

    def _request_takeoff(self):
        if not self.takeoff_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Takeoff service not ready yet.')
            return
        self._tko_requested = True
        req = CommandTOL.Request()
        req.altitude = float(self.target_altitude)
        fut = self.takeoff_client.call_async(req)
        fut.add_done_callback(self._on_takeoff_done)
        self.get_logger().info(f'Requesting takeoff to {self.target_altitude:.1f} m...')

    def _on_takeoff_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'Takeoff exception: {e}')
            self._tko_requested = False  # allow retry
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Takeoff command accepted (monitoring altitude).')
        else:
            self.get_logger().error('Takeoff rejected by FCU.')
            self._tko_requested = False  # allow retry

    # ------------------------------------------------------------------
    # Control / setpoint streaming
    # ------------------------------------------------------------------
    def _enable_tracking(self):
        if not self._tracking_enabled:
            self._tracking_enabled = True
            self.get_logger().info('Tracking enabled (automatic - takeoff complete).')

    def _enable_tracking_manual(self):
        """Enable tracking manually via topic command"""
        if not self._tracking_enabled:
            self._tracking_enabled = True
            self.get_logger().info('Tracking enabled (manual override).')

    def _disable_tracking_manual(self):
        """Disable tracking manually via topic command"""
        if self._tracking_enabled:
            self._tracking_enabled = False
            self.get_logger().info('Tracking disabled (manual override).')

    def _publish_setpoint(self):
        """
        Always stream velocity setpoints.
        - Before armed/takeoff: stream zeros (satisfies GUIDED + pre-arm conditions).
        - After tracking enabled: yaw-rate = -kp * error (saturated).
        """

        # Always create and publish a message (even if zeros)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        print(f"is tracking enabled? {self._tracking_enabled}")
        print(f"armed_confirmed: {self._armed_confirmed}, tko_reached: {self._tko_reached}")
        print(f"current altitude: {self.pose.pose.position.z:.2f}m, target: {self.target_altitude}m")
        
        yaw_rate = 0.0
        if self._tracking_enabled:
            yaw_rate = -self.yaw_kp * self.person_err
            yaw_rate = max(min(yaw_rate, self.max_yaw_rate), -self.max_yaw_rate)
            print(f'Person error: {self.person_err:.3f} rad, Yaw rate: {yaw_rate:.3f} rad/s')

            msg.twist.angular.z = float(yaw_rate)
            self.vel_pub.publish(msg)




def main(args=None):
    rclpy.init(args=args)
    node = CircumnavigationController()
    # Multi-threaded executor keeps service callbacks responsive.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
