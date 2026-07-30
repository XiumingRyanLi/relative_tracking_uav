"""
Sim launch: Gazebo + ros_gz_bridge + ArduPilot SITL + MAVROS + perception/control nodes.

Assumes you've already got:
  - ardupilot_gazebo built at $HOME/ardupilot_gazebo
  - ardupilot checked out at $HOME/ardupilot
  - the world file at <this_pkg>/worlds/iris_runway_new.sdf (adjust path below)

Usage:
    ros2 launch circumnavigation_controller sim_launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node

PACKAGE_NAME = 'circumnavigation_controller'

# --- Adjust these paths for your machine ---
HOME = os.environ['HOME']
ARDUPILOT_GAZEBO_PLUGIN_PATH = os.path.join(HOME, 'ardupilot_gazebo', 'build')
ARDUPILOT_DIR = os.path.join(HOME, 'ardupilot')
GIMBAL_PARM_FILE = os.path.join(HOME, 'ardupilot_gazebo', 'config', 'gazebo-iris-gimbal.parm')
WORLDS_DIR = os.path.join(HOME, 'ardupilot_gazebo', 'worlds')
WORLD_FILE = 'iris_runway_new.sdf'  # relative - gz must be run from WORLDS_DIR so
                                    # relative <uri> includes in the sdf resolve
WORLD_NAME = 'iris_runway_new'  # must match whatever <world name="..."> is in the sdf

# ArduPilot SITL takes a while to boot, connect to Gazebo over JSON, and start
# accepting the MAVLink UDP connection. If MAVROS starts too soon it'll just
# retry until ArduPilot is up, but a real failure looks the same as "still
# booting" from the launch system's point of view - so give it a head start.
# Bump this up if you see MAVROS repeatedly failing to connect (e.g. FCU
# heartbeat never appears in `ros2 topic echo /mavros/state`).
ARDUPILOT_STARTUP_DELAY_SEC = 25.0


def generate_launch_description():
    # Make sure gz can find the ardupilot gazebo plugins
    set_gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=ARDUPILOT_GAZEBO_PLUGIN_PATH + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    )

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', WORLD_FILE, '-v', '-r'],
        cwd=WORLDS_DIR,
        prefix=['gnome-terminal --title="Gazebo" --'],
        output='screen'
    )

    # Camera + clock bridge
    CAMERA_TOPIC = (
        f'/world/{WORLD_NAME}/model/iris_with_gimbal/model/gimbal/'
        'link/pitch_link/sensor/camera/image'
    )
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'{CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[
            (CAMERA_TOPIC, '/camera/image_raw'),
        ],
        prefix=['gnome-terminal --title="GZ Bridge" --'],
        output='screen'
    )

    ardupilot_sitl = ExecuteProcess(
        cmd=[
            'sim_vehicle.py',
            '-D',
            '-v', 'ArduCopter',
            '-f', 'JSON',
            '--add-param-file', GIMBAL_PARM_FILE,
            '--console', '--map',
            '--out=udp:127.0.0.1:14555',
        ],
        cwd=ARDUPILOT_DIR,
        prefix=['gnome-terminal --title="ArduPilot SITL" --'],
        output='screen'
    )

    mavros = Node(
        package='mavros',
        executable='mavros_node',
        parameters=[{'fcu_url': 'udp://:14555@'}],
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

    # MAVROS (and anything that depends on a live FCU connection) waits until
    # ArduPilot has had a chance to boot. aruco_detector doesn't strictly need
    # to wait, but keeping everything downstream together avoids the
    # controller/gui coming up against a MAVROS that isn't connected yet.
    delayed_mavlink_stack = TimerAction(
        period=ARDUPILOT_STARTUP_DELAY_SEC,
        actions=[
            mavros,
            aruco_detector,
            relative_position_controller,
            cinematic_gui,
        ]
    )

    return LaunchDescription([
        set_gz_plugin_path,
        gazebo,
        gz_bridge,
        ardupilot_sitl,
        delayed_mavlink_stack,
    ])