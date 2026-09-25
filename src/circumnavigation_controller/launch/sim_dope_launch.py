"""
Sim launch with the DOPE detector: Gazebo + ros_gz_bridge + ArduPilot SITL +
MAVROS + dope_detector + relative_position_controller + cinematic_gui.

This is sim_launch.py with `detector:=dope` baked in, so it launches the
exact same stack, but with the DOPE (Audi R8) perception node in place of
the AprilTag/ArUco detector. The Audi's drive and odometry bridge lives in
sim_launch.py, so it is available from both launches.

Usage:
    ros2 launch circumnavigation_controller sim_dope_launch.py

Drive the car (with this launch running):
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \\
        "{linear: {x: 2.0}, angular: {z: 0.0}}"
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Ground-truth car odometry (Gazebo world frame) for comparing against DOPE:
    ros2 topic echo /landing_vehicle/odometry

DOPE weights, object name and cuboid dimensions are set in sim_launch.py
(DOPE_WEIGHTS / DOPE_OBJECT / DOPE_CUBOID_DIMENSIONS_CM).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

PACKAGE_NAME = 'circumnavigation_controller'


def generate_launch_description():
    sim_launch_path = os.path.join(
        get_package_share_directory(PACKAGE_NAME), 'launch', 'sim_launch.py'
    )

    # Full sim stack, DOPE detector selected.
    sim_with_dope = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={'detector': 'dope'}.items(),
    )

    return LaunchDescription([
        sim_with_dope,
    ])
