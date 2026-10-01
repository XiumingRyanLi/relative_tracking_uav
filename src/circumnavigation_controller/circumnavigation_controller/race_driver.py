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

mode:=launch (acceleration test, A4) ignores the racing line: it holds the
car's starting heading and line and runs a straight-line speed profile --
accelerate at launch_accel to launch_speed, hold it for launch_hold_sec,
brake at launch_brake to a stop. Use the audi_r8_launch car (iris_flat_exp):
the stock rear-drive audi_r8 can't pull much more than ~5 m/s^2.
launch_path:=sine weaves along the starting line instead: lateral offset
sine_amplitude * sin(2 pi s / wavelength), wavelength = launch_speed *
sine_period (so at launch_speed one weave takes sine_period s and the peak
lateral acceleration, amplitude * (2 pi / period)^2, is the same at every
speed: 3 m / 7.7 s -> 2.0 m/s^2). The amplitude ramps in over the first
quarter wavelength so the car starts straight.

For scripts/run_experiments.py it logs `EXPERIMENT_EVENT car_start` when the
car sets off and `EXPERIMENT_EVENT car_done` after the last lap / the stop.
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
        self.mode = p('mode', 'race').value
        if self.mode not in ('race', 'launch'):
            raise ValueError(f"mode must be 'race' or 'launch', got {self.mode!r}")
        self.launch_speed = p('launch_speed', 13.0).value
        self.launch_accel = p('launch_accel', 8.0).value
        self.launch_hold_sec = p('launch_hold_sec', 10.0).value
        self.launch_brake = p('launch_brake', 6.0).value
        self.launch_path = p('launch_path', 'straight').value
        if self.launch_path not in ('straight', 'sine'):
            raise ValueError(f"launch_path must be 'straight' or 'sine', got {self.launch_path!r}")
        self.sine_amplitude = p('sine_amplitude', 3.0).value
        self.sine_period = p('sine_period', 7.7).value
        self._launch = None   # launch mode: dict(phase, t0, v_cmd, x0, y0, heading, t_phase)
        line_file = p('race_line_file', '').value or os.path.join(
            get_package_share_directory(PACKAGE_NAME),
            'config', 'race_lines', world + '.csv')
        # Defaults: 15 m/s is the drone's guided speed limit (WP_SPD). The
        # rest were tuned in Gazebo on all three tracks: the rear-drive
        # audi_r8 spins if it corners or powers out of corners much harder
        # (lat_accel 3-4 / accel 2.5 hit the wall), and pure pursuit
        # overshoots lat_accel a little in the chicanes.
        line_kwargs = dict(
            max_speed=p('max_speed', 15.0).value,
            min_speed=p('min_speed', 3.0).value,
            lat_accel=p('lat_accel', 2.5).value,
            accel=p('accel', 2.0).value,
            brake=p('brake', 4.0).value,
            speed_zones=self._zones(p('speed_zones', [0.0]).value),
        )
        self.line = self.driver = None
        if self.mode == 'race':
            self.line = RaceLine.from_csv(line_file, **line_kwargs)
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

        if self.mode == 'launch':
            self.get_logger().info(
                f'launch profile: 0 -> {self.launch_speed:.1f} m/s at {self.launch_accel:.1f} m/s^2, '
                f'hold {self.launch_hold_sec:.0f} s, brake at {self.launch_brake:.1f} m/s^2, '
                + (f'sine path {self.sine_amplitude:.0f} m x {self.launch_speed * self.sine_period:.0f} m; '
                   if self.launch_path == 'sine' else 'straight; ')
                + ('waiting for the drone to take off' if self.wait_for_drone else 'starting now'))
            return
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
            self.get_logger().info('EXPERIMENT_EVENT car_start')
        if self.finished:
            self.cmd_pub.publish(Twist())
            return

        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        lin = msg.twist.twist.linear  # car frame: x forward, y left
        if self.mode == 'launch':
            self.cmd_pub.publish(self._launch_step(msg.pose.pose.position, yaw, lin.x))
            return
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
                self.get_logger().info('EXPERIMENT_EVENT car_done')

    def _launch_step(self, pos, yaw, speed):
        """Launch mode: one control step (Twist) of the straight-line
        accelerate / hold / brake profile. The speed command ramps at the
        profile's rates (the plugin's own limit is +-8 m/s^2); steering holds
        the starting line: heading error plus a cross-track pull."""
        now = self.sim_time
        st = self._launch
        if st is None:
            st = self._launch = dict(phase='accel', t_phase=now, v_cmd=0.0, last=now,
                                     x0=pos.x, y0=pos.y, heading=yaw)
            self.get_logger().info(f'launch: accelerating to {self.launch_speed:.1f} m/s')
        dt = max(0.0, min(0.2, now - st['last']))
        st['last'] = now
        if st['phase'] == 'accel':
            st['v_cmd'] = min(self.launch_speed, st['v_cmd'] + self.launch_accel * dt)
            if speed >= self.launch_speed - 0.3:
                self.get_logger().info(
                    f'launch: at {speed:.1f} m/s after {now - st["t_phase"]:.1f} s, holding')
                st['phase'], st['t_phase'] = 'hold', now
                self.get_logger().info('EXPERIMENT_EVENT car_at_speed')
        elif st['phase'] == 'hold':
            st['v_cmd'] = self.launch_speed
            if now - st['t_phase'] >= self.launch_hold_sec:
                st['phase'], st['t_phase'] = 'brake', now
                self.get_logger().info('launch: braking')
                self.get_logger().info('EXPERIMENT_EVENT car_brake')
        elif st['phase'] == 'brake':
            st['v_cmd'] = max(0.0, st['v_cmd'] - self.launch_brake * dt)
            if st['v_cmd'] <= 0.0 and abs(speed) < 0.3:
                st['phase'] = 'done'
                self.finished = True
                self.get_logger().info('launch: stopped')
                self.get_logger().info('EXPERIMENT_EVENT car_done')
        out = Twist()
        if st['phase'] == 'done':
            return out
        h = st['heading']
        dx, dy = pos.x - st['x0'], pos.y - st['y0']
        along = dx * math.cos(h) + dy * math.sin(h)
        # Signed distance left of the starting line.
        lateral = -dx * math.sin(h) + dy * math.cos(h)
        off, slope, curv = self._path(along)
        want = h + math.atan(slope) - math.atan(0.15 * (lateral - off))
        err = math.atan2(math.sin(want - yaw), math.cos(want - yaw))
        out.linear.x = st['v_cmd']
        # Path curvature feedforward + heading correction.
        out.angular.z = max(-0.8, min(0.8, speed * curv + 1.5 * err)) if speed > 0.5 else 0.0
        return out

    def _path(self, s):
        """Launch path at distance s along the starting line: (offset left
        of the line, d offset / ds, curvature)."""
        if self.launch_path != 'sine' or s <= 0.0:
            return 0.0, 0.0, 0.0
        lam = max(self.launch_speed * self.sine_period, 1.0)

        def y(u):
            r = min(max(u / (0.25 * lam), 0.0), 1.0)
            ramp = r * r * (3.0 - 2.0 * r)           # amplitude ramps in over a quarter wavelength
            return self.sine_amplitude * ramp * math.sin(2.0 * math.pi * u / lam)
        e = 0.5
        y0, yp, ym = y(s), y(s + e), y(s - e)
        d1 = (yp - ym) / (2.0 * e)
        d2 = (yp - 2.0 * y0 + ym) / (e * e)
        return y0, d1, d2 / (1.0 + d1 * d1) ** 1.5

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
