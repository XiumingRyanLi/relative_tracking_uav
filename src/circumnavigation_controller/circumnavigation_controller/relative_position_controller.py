#!/usr/bin/env python3
import rclpy
import math
import csv
# import os
# import numpy as np
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, MessageInterval

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# from rclpy.qos import qos_profile_sensor_data

# from .implimented_controllers import MD_Controller, MDV_Controller, DKR_Controller, KRV_Controller



class RelativePositionController(Node):
    def __init__(self):
        super().__init__('relative_position_controller')

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        pose_qos = QoSProfile( reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.state_sub = self.create_subscription( State, '/mavros/state', self._on_state, state_qos)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose, pose_qos)
        self.global_pos_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self._on_global_position, pose_qos)

        self.compass_sub = self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self._on_compass, pose_qos)


        # self.err_sub = self.create_subscription( Float64, '/yaw_error', self._on_person_error, 10)
        # self.bearing_sub = self.create_subscription(Float64, '/bearing', self._on_bearing, 10)
        # self.tracking_enable_sub = self.create_subscription(Bool, '/tracking_enable', self._on_tracking_enable, 10)

        self.pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.setpoint_timer = self.create_timer(0.15, self._publish_setpoint)

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.message_interval_client = self.create_client(MessageInterval, '/mavros/set_message_interval')

        self.state = State()
        self.pose = PoseStamped()
        self.global_pos = NavSatFix()
        self.compass_hdg = 0.0

        self.have_local_pose = False
        self.have_global_pose = False

        self.target_altitude = 3.0
        
        self.target_x = 0.0
        self.target_y = 5.0
        self.target_z = 0.0
        self.target_heading = 0.0   # radians, dummy for now

        self.range_r = 10.0
        self.azimuth_offset = math.radians(50.0)
        self.elevation_offset = math.radians(20.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"relative_pose_control_data_{timestamp}.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'drone_x', 'drone_y', 'drone_z',
            'target_x', 'target_y', 'target_z',
            'desired_x', 'desired_y', 'desired_z',
            'target_heading', 'desired_yaw'
        ])
        self.get_logger().info(f'CSV logging initialized: {self.csv_filename}')

        self._guided_requested = False
        self._guided_confirmed = False

        self._arm_requested = False
        self._armed_confirmed = False

        self._tko_requested = False
        self._tko_reached = False

        self._tracking_enabled = False
        # self._tracking_manual_override = False
        self._armed_time = None
        self._takeoff_complete_time = None
        self._rtl_initiated = False

        self.boundary_limit = 25.0  # 15m from center in any direction
        
        self.orchestrator = self.create_timer(0.2, self._orchestrate)
        self.safety_timer = self.create_timer(1.0, self._check_safety_conditions)


        # Set MAVROS message intervals
        self._set_message_intervals()

        self.get_logger().info('Relative Position Controller started')

    def _set_message_intervals(self):
        """Set MAVROS message intervals for global position and compass to 30Hz"""
        self._set_single_message_interval(32, 30.0, "Global Position") 
        self._set_single_message_interval(74, 30.0, "Compass Heading")

    def _set_single_message_interval(self, message_id, rate, description):
        """Set a single message interval"""
        if not self.message_interval_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'MessageInterval service not ready for {description}')
            return
            
        req = MessageInterval.Request()
        req.message_id = message_id
        req.message_rate = rate
        
        fut = self.message_interval_client.call_async(req)
        fut.add_done_callback(lambda f, desc=description, mid=message_id, r=rate: self._on_message_interval_done(f, desc, mid, r))
        
        self.get_logger().info(f'Setting {description} (ID: {message_id}) to {rate}Hz...')

    def _on_message_interval_done(self, fut, description, message_id, rate):
        """Handle message interval service response"""
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'{description} interval setting exception: {e}')
            return

        if getattr(res, 'success', False):
            self.get_logger().info(f'{description} interval set to {rate}Hz successfully')
        else:
            self.get_logger().warn(f'{description} interval setting failed (ID: {message_id}, Rate: {rate}Hz)')

    def _on_state(self, msg: State):
        self.state = msg

        if self.state.mode == 'GUIDED' and not self._guided_confirmed:
            self._guided_confirmed = True
            self.get_logger().info('GUIDED confirmed by FCU.')

        if self.state.armed and not self._armed_confirmed:
            self._armed_confirmed = True
            self._armed_time = self.get_clock().now().nanoseconds / 1e9
            
            
            self.get_logger().info('Armed confirmed by FCU.')


    def _on_pose(self, msg: PoseStamped):
        self.pose = msg
        alt = msg.pose.position.z
        
        if self._armed_confirmed and not self._tko_reached and alt > (self.target_altitude - 0.5):
            self._tko_reached = True
            self._takeoff_complete_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f'Takeoff complete at {alt:.2f} m - Starting 60s safety timer')
            self._enable_tracking()

    def _on_global_position(self, msg: NavSatFix):
        self.global_pos = msg


    def _on_compass(self, msg: Float64):
        self.compass_hdg = float(msg.data)

    def _orchestrate(self):
        if not self.state.connected:
            return

        if not self._guided_confirmed:
            if not self._guided_requested:
                self._request_guided()
            return

        if not self._armed_confirmed:
            if not self._arm_requested:
                self._request_arm()
            return

        if self._armed_confirmed and self._armed_time is not None and not self._tko_requested:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._armed_time
            if elapsed < 5.0:
                if not hasattr(self, '_armed_wait_logged') or not self._armed_wait_logged:
                    self.get_logger().info("Armed. Waiting 5s before takeoff...")
                    self._armed_wait_logged = True
                return
            else:
                self._request_takeoff()
                self._armed_wait_logged = False
                return

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
            self._guided_requested = False
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info('GUIDED command accepted (awaiting FCU report).')
        else:
            self.get_logger().error('GUIDED command rejected by FCU.')
            self._guided_requested = False

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
            self._arm_requested = False
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Arm accepted (awaiting FCU armed=true).')
        else:
            self.get_logger().error(f'Arm rejected by FCU (result={getattr(res, "result", None)}).')
            self._arm_requested = False

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
            self._tko_requested = False
            return

        if getattr(res, 'success', False):
            self.get_logger().info('Takeoff command accepted (monitoring altitude).')
        else:
            self.get_logger().error('Takeoff rejected by FCU.')
            self._tko_requested = False

    def _enable_tracking(self):
        if not self._tracking_enabled:
            self._tracking_enabled = True
            self.get_logger().info('Tracking enabled (automatic - takeoff complete).')

    def _check_safety_conditions(self):
        """Check safety conditions and initiate RTL if necessary"""
        if self._rtl_initiated or not self._tko_reached:
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check 120-second timer after takeoff
        if self._takeoff_complete_time is not None:
            elapsed_since_takeoff = current_time - self._takeoff_complete_time
            if elapsed_since_takeoff >= 120.0:
                self.get_logger().warn('120 seconds elapsed since takeoff - Initiating RTL')
                self._initiate_rtl('120-second timer expired')
                return
        
        # Check boundary conditions (30x30m square)
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        if abs(x) > self.boundary_limit or abs(y) > self.boundary_limit:
            self.get_logger().warn(f'Boundary violation: position ({x:.1f}, {y:.1f}) - Initiating RTL')
            self._initiate_rtl(f'boundary violation at ({x:.1f}, {y:.1f})')
            return

    def _initiate_rtl(self, reason):
        """Initiate return to land mode and disable tracking"""
        if self._rtl_initiated:
            return
            
        self._rtl_initiated = True
        self._tracking_enabled = False
        
        self.get_logger().warn(f'SAFETY: Initiating RTL due to {reason}')
        
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('SetMode service not available for RTL!')
            return
            
        req = SetMode.Request()
        req.custom_mode = 'RTL'
        fut = self.set_mode_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_rtl_done(f, reason))

    def _on_rtl_done(self, fut, reason):
        """Handle RTL mode change response"""
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f'RTL exception: {e}')
            return

        if getattr(res, 'mode_sent', False):
            self.get_logger().info(f'RTL command accepted (reason: {reason})')
        else:
            self.get_logger().error(f'RTL command rejected by FCU (reason: {reason})')


    def _publish_setpoint(self):

        # Don't send position commands if RTL has been initiated
        if self._rtl_initiated:
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.csv_writer.writerow([
                current_time,
                self.pose.pose.position.x,
                self.pose.pose.position.y,
                self.pose.pose.position.z,
                self.target_x,
                self.target_y,
                self.target_z,
                self.pose.pose.position.x,
                self.pose.pose.position.y,
                self.pose.pose.position.z,
                self.target_heading,
                0.0,
            ])
            self.csv_file.flush()
            return

        # Before takeoff is complete, hold current XY and command takeoff altitude only
        if not self._tko_reached:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.position.x = self.pose.pose.position.x
            msg.pose.position.y = self.pose.pose.position.y
            msg.pose.position.z = self.target_altitude

            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0

            self.pos_pub.publish(msg)

            current_time = self.get_clock().now().nanoseconds / 1e9
            self.csv_writer.writerow([
                current_time,
                self.pose.pose.position.x,
                self.pose.pose.position.y,
                self.pose.pose.position.z,
                self.target_x,
                self.target_y,
                self.target_z,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                self.target_heading,
                0.0,
            ])
            self.csv_file.flush()
            return

        psi = self.target_heading + self.azimuth_offset
        r_xy = self.range_r * math.cos(self.elevation_offset)
        dz = self.range_r * math.sin(self.elevation_offset)

        dx = r_xy * math.cos(psi)
        dy = r_xy * math.sin(psi)

        desired_x = self.target_x + dx
        desired_y = self.target_y + dy
        desired_z = self.target_z + dz

        desired_yaw = math.atan2(self.target_y - desired_y, self.target_x - desired_x)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.position.x = desired_x
        msg.pose.position.y = desired_y
        msg.pose.position.z = desired_z

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(desired_yaw / 2.0)
        msg.pose.orientation.w = math.cos(desired_yaw / 2.0)

        self.pos_pub.publish(msg)

        current_time = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([
            current_time,
            self.pose.pose.position.x,
            self.pose.pose.position.y,
            self.pose.pose.position.z,
            self.target_x,
            self.target_y,
            self.target_z,
            desired_x,
            desired_y,
            desired_z,
            self.target_heading,
            desired_yaw,
        ])
        self.csv_file.flush()


    def destroy_node(self):
        """Clean up CSV file when node is destroyed"""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            self.get_logger().info(f'CSV file closed: {self.csv_filename}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RelativePositionController()
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
