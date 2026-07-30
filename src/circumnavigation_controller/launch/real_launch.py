"""
Real hardware launch: camera driver + MAVROS + perception/control nodes.

No Gazebo, no ros_gz_bridge - those only exist to fake a camera topic and a
simulated clock/odometry. On real hardware:
  - a real camera driver node publishes /camera/image_raw directly
  - MAVROS talks to the flight controller over its actual link (serial /
    telemetry radio / onboard companion-computer UART), not a SITL UDP port
  - ArduPilot itself runs on the flight controller - nothing to launch for it here

Adjust CAMERA_* and FCU_URL for your actual setup before running.

Usage:
    ros2 launch circumnavigation_controller real_launch.py
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

PACKAGE_NAME = 'circumnavigation_controller'

# Give the flight controller time to boot / finish its init sequence before
# MAVROS tries to connect. Real FCs are usually faster than SITL, but a
# power-on-reset can still take a few seconds before the link is usable.
FC_STARTUP_DELAY_SEC = 8.0

# --- Adjust for your hardware ---
# Example uses usb_cam; swap for whatever driver matches your actual camera
# (v4l2_camera, gscam, an Isaac ROS camera node, a gimbal SDK node, etc.)
CAMERA_PACKAGE = 'usb_cam'
CAMERA_EXECUTABLE = 'usb_cam_node_exe'
CAMERA_PARAMS = {
    'video_device': '/dev/video0',
    'image_width': 1280,
    'image_height': 720,
    'framerate': 30.0,
}

# Serial connection to the flight controller (change to match your wiring),
# e.g. a direct FTDI/telem cable, or a radio telemetry link.
FCU_URL = '/dev/ttyACM0:57600'


def generate_launch_description():
    camera_driver = Node(
        package=CAMERA_PACKAGE,
        executable=CAMERA_EXECUTABLE,
        parameters=[CAMERA_PARAMS],
        remappings=[
            ('image_raw', '/camera/image_raw'),
        ],
        prefix=['gnome-terminal --title="Camera Driver" --'],
        output='screen'
    )

    mavros = Node(
        package='mavros',
        executable='mavros_node',
        parameters=[{'fcu_url': FCU_URL}],
        prefix=['gnome-terminal --title="MAVROS" --'],
        output='screen'
    )

    aruco_detector = Node(
        package=PACKAGE_NAME,
        executable='aruco_detector',
        prefix=['gnome-terminal --title="Aruco Detector" --'],
        output='screen'
    )

    relative_position_controller = Node(
        package=PACKAGE_NAME,
        executable='relative_position_controller',
        prefix=['gnome-terminal --title="Relative Position Controller" --'],
        output='screen'
    )

    cinematic_gui = Node(
        package=PACKAGE_NAME,
        executable='cinematic_gui',
        prefix=['gnome-terminal --title="Cinematic GUI" --'],
        output='screen'
    )

    delayed_mavlink_stack = TimerAction(
        period=FC_STARTUP_DELAY_SEC,
        actions=[
            mavros,
            aruco_detector,
            relative_position_controller,
            cinematic_gui,
        ]
    )

    return LaunchDescription([
        camera_driver,
        delayed_mavlink_stack,
    ])