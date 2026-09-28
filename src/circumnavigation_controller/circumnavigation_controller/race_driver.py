#!/usr/bin/env python3
"""Drive the target car round a race-track world like a racing driver.

Follows config/race_lines/<world>.csv (tools/build_race_lines.py) with pure
pursuit, flat out on the straights, braking into corners and accelerating
out of them (speed profile: see race_line.py). Publishes /cmd_vel for the
audi_r8 AckermannSteering plugin and reads the car's ground-truth pose from
/landing_vehicle/odometry (both bridged by sim_launch.py).

By default it waits for the drone: it starts once /mavros/local_position/pose
is above start_altitude, plus start_delay (takeoff + hover before the
controller starts tracking). Set wait_for_drone:=false to go immediately.

    ros2 run circumnavigation_controller race_driver --ros-args \\
        -p world:=iris_silverstone -p max_speed:=12.0 -p laps:=2

or `race:=true` on sim_launch.py.
"""
import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from .race_line import RaceDriver, RaceLine

PACKAGE_NAME = 'circumnavigation_controller'


class RaceDriverNode(Node):
    def __init__(self):
        super().__init__('race_driver')
        p = self.declare_parameter
        world = p('world', 'iris_monza').value
        line_file = p('race_line_file', '').value or os.path.join(
            get_package_share_directory(PACKAGE_NAME),
            'config', 'race_lines', world + '.csv')
        # Defaults: 15 m/s is the drone's guided speed limit (WP_SPD). The
        # rest were tuned in Gazebo on all three tracks: the rear-drive
        # audi_r8 spins if it corners or powers out of corners much harder
        # (lat_accel 3-4 / accel 2.5 hit the wall), and pure pursuit
        # overshoots lat_accel a little in the chicanes.
        self.line = RaceLine.from_csv(
            line_file,
            max_speed=p('max_speed', 15.0).value,
            min_speed=p('min_speed', 3.0).value,
            lat_accel=p('lat_accel', 2.5).value,
            accel=p('accel', 2.0).value,
            brake=p('brake', 4.0).value,
            speed_zones=self._zones(p('speed_zones', [0.0]).value),
        )
        self.driver = RaceDriver(self.line)
        self.laps = p('laps', 0).value  # 0 = keep racing
        self.wait_for_drone = p('wait_for_drone', True).value
        self.start_altitude = p('start_altitude', 2.5).value
        self.start_delay = p('start_delay', 6.0).value
        odom_topic = p('odom_topic', '/landing_vehicle/odometry').value
        cmd_topic = p('cmd_topic', '/cmd_vel').value

        # Control runs on each odometry message (20 Hz sim time) and all
        # timing uses its stamps, so it behaves the same at any Gazebo
        # real-time factor.
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        if self.wait_for_drone:
            self.create_subscription(
                PoseStamped, '/mavros/local_position/pose', self._on_drone_pose,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST, depth=10))

        self.sim_time = None       # latest odometry stamp, s
        self.airborne_time = None  # sim time the drone passed start_altitude
        self.racing = not self.wait_for_drone
        self.finished = False
        self.lap = 0
        self.lap_start = None
        self.prev_s = None
        self.progress = 0.0

        self.get_logger().info(
            f'{os.path.basename(line_file)}: {self.line.length:.0f} m lap, '
            f'{self.line.speed.min():.1f}-{self.line.speed.max():.1f} m/s, '
            f'~{self.line.lap_time():.0f} s per lap at the profile speed; '
            + ('waiting for the drone to take off' if self.wait_for_drone
               else 'racing now'))
        for z in self.line.speed_zones:
            self.get_logger().info(
                f'speed zone: s {z[0]:.0f}-{z[1]:.0f} m capped at {z[2]:.1f} m/s')
        self.get_logger().info(
            'corners (s = metres from the start line, for speed_zones):\n'
            + '\n'.join(f'  s {s:6.0f} m  {v:4.1f} m/s  radius {r:5.1f} m'
                         for s, v, r in self.line.corners()))

    def _zones(self, flat):
        """speed_zones parameter: flat list [s_start, s_end, max_speed, ...]
        (ROS parameters can't be nested lists). [0.0] = none."""
        flat = [float(x) for x in flat]
        if len(flat) < 3:
            return []
        if len(flat) % 3:
            raise ValueError('speed_zones needs triples: s_start, s_end, max_speed')
        return [flat[i:i + 3] for i in range(0, len(flat), 3)]

    def _on_drone_pose(self, msg: PoseStamped):
        if (self.airborne_time is None and self.sim_time is not None
                and msg.pose.position.z >= self.start_altitude):
            self.airborne_time = self.sim_time
            self.get_logger().info(
                f'drone at {msg.pose.position.z:.1f} m, racing in '
                f'{self.start_delay:.0f} s')

    def _on_odom(self, msg: Odometry):
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.racing:
            if (self.airborne_time is None
                    or self.sim_time - self.airborne_time < self.start_delay):
                return
            self.racing = True
            self.get_logger().info('lights out')
        if self.finished:
            self.cmd_pub.publish(Twist())
            return

        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        lin = msg.twist.twist.linear  # car frame: x forward, y left
        cmd = self.driver.step(msg.pose.pose.position.x,
                               msg.pose.pose.position.y, yaw, lin.x, lin.y)
        self._count_laps(cmd.s)
        out = Twist()
        if not self.finished:
            out.linear.x = cmd.speed
            out.angular.z = cmd.yaw_rate
        self.cmd_pub.publish(out)

    def _count_laps(self, s):
        # Distance driven along the line (wrapping at the start/finish), so
        # a start just behind or past the first line point can't count as
        # a lap.
        now = self.sim_time
        length = self.line.length
        if self.prev_s is None:
            self.lap_start = now
        else:
            self.progress += (s - self.prev_s + 0.5 * length) % length - 0.5 * length
        self.prev_s = s
        if self.progress >= (self.lap + 1) * length:
            self.lap += 1
            lap_time = now - self.lap_start
            self.get_logger().info(
                f'lap {self.lap}: {lap_time:.1f} s '
                f'(avg {length / lap_time:.1f} m/s)')
            self.lap_start = now
            if self.laps and self.lap >= self.laps:
                self.finished = True
                self.get_logger().info(f'{self.laps} lap(s) done, stopping')

    def stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = RaceDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
