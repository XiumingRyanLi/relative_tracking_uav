"""
Sim launch: Gazebo + ros_gz_bridge + ArduPilot SITL + MAVROS + perception/control nodes.

Assumes you've already got:
  - ardupilot_gazebo built at $HOME/ardupilot_gazebo
  - ardupilot checked out at $HOME/ardupilot
  - the world file at <this_pkg>/worlds/iris_runway_new.sdf (adjust path below)

Usage:
    ros2 launch circumnavigation_controller sim_launch.py
    ros2 launch circumnavigation_controller sim_launch.py detector:=dope

`detector` picks the perception node feeding /aruco_target/visual_odom:
  aruco (default) - AprilTag/ArUco solvePnP detector
  dope            - NVIDIA DOPE network (see DOPE_WEIGHTS below)
"""

import os
import signal
import subprocess
import time
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction, RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

PACKAGE_NAME = 'circumnavigation_controller'

# --- Adjust these paths for your machine ---
HOME = os.environ['HOME']
ARDUPILOT_GAZEBO_PLUGIN_PATH = os.path.join(HOME, 'ardupilot_gazebo', 'build')
ARDUPILOT_DIR = os.path.join(HOME, 'ardupilot')
SIM_VEHICLE_SCRIPT = os.path.join(ARDUPILOT_DIR, 'Tools', 'autotest', 'sim_vehicle.py')
COPTER_PARM_FILE = os.path.join(ARDUPILOT_DIR, 'Tools', 'autotest', 'default_params', 'copter.parm')
GAZEBO_IRIS_PARM_FILE = os.path.join(ARDUPILOT_DIR, 'Tools', 'autotest', 'default_params', 'gazebo-iris.parm')
GIMBAL_PARM_FILE = os.path.join(HOME, 'ardupilot_gazebo', 'config', 'gazebo-iris-gimbal.parm')
# Attitude-rate / gyro-filter tuning that removes the roll/pitch twitch seen
# as vibration in the gimbal camera. Ships with this package (config/); loaded
# last so it overrides the ArduPilot defaults above.
ATTITUDE_TUNING_PARM_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME), 'config', 'attitude_tuning.parm'
)
# Gimbal startup pose (Neutral mode, pitch -5 deg) so the camera can see the
# car before the controller's first detection. Ships with this package (config/).
GIMBAL_STARTUP_PARM_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME), 'config', 'gimbal_startup.parm'
)
# Guided-mode speed/acceleration limits (WP_SPD 15 m/s, WP_ACC 5 m/s^2) so the
# drone can keep up with the shot point while the car turns. Ships with this
# package (config/); must match pid_max_speed_xy / pid_max_accel_xy.
GUIDED_LIMITS_PARM_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME), 'config', 'guided_limits.parm'
)
WORLDS_DIR = os.path.join(HOME, 'ardupilot_gazebo', 'worlds')
MODELS_DIR = os.path.join(HOME, 'ardupilot_gazebo', 'models')
WORLD_FILE = 'iris_runway_new.sdf'  # relative - gz must be run from WORLDS_DIR so
                                    # relative <uri> includes in the sdf resolve
WORLD_NAME = 'iris_runway_new'  # must match whatever <world name="..."> is in the sdf

# relative_position_controller.py writes its relative_pid_<timestamp>.csv
# log using a bare relative filename - i.e. wherever the node's cwd happens
# to be. Pin that to a known directory (instead of leaving it to whatever
# directory `ros2 launch` happened to be run from) so the debug-plot step
# below can reliably find the CSV afterwards, and so your logs don't end up
# scattered wherever you happened to launch from.
WORKSPACE_ROOT = os.path.join(HOME, 'Desktop', 'ryan', 'relative_tracking_uav')
# This repo's own models (iris_with_dope_gimbal, gimbal_small_3d_dope). Put
# last on GZ_SIM_RESOURCE_PATH so it never shadows the ardupilot_gazebo copies.
REPO_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'models')
LOG_DIR = os.path.join(WORKSPACE_ROOT, 'logs')

# DOPE checkpoint used by `detector:=dope`. Kept outside the repo (201 MB);
# a plain PyTorch state_dict trained with DDP ("module." prefix stripped on
# load). Object + cuboid dimensions below must match what it was trained on.
DOPE_WEIGHTS = os.path.join(
    HOME, 'Desktop', 'ryan', 'Deep_Object_Pose', 'train', 'output', 'weights_droneview_v2', 'net_epoch_0725.pth'
)
DOPE_OBJECT = 'Audi'
DOPE_CUBOID_DIMENSIONS_CM = [203.82, 123.95, 441.46]  # from DOPE config_pose.yaml

# Plot shown automatically when relative_position_controller exits: scores the
# DOPE estimate against the Gazebo ground truth logged in the controller CSV
# (range / position / heading error, tracks). It picks the newest CSV in
# LOG_DIR by itself. Run it by hand on any CSV with:
#   python3 scripts/plot_dope_evaluation.py logs/relative_pid_<ts>.csv
PLOT_SCRIPT = os.path.join(WORKSPACE_ROOT, 'scripts', 'plot_dope_evaluation.py')

# ArduPilot SITL takes a while to boot, connect to Gazebo over JSON, and start
# accepting the MAVLink UDP connection. If MAVROS starts too soon it'll just
# retry until ArduPilot is up, but a real failure looks the same as "still
# booting" from the launch system's point of view - so give it a head start.
#
# 25s was too short in practice: EKF3 tilt/yaw alignment and the GPS 1 fix
# (the SITL GPS backend takes time to go from "not found" -> "detected" ->
# "configuring" -> good 3D fix) weren't done yet, so COMMAND_ACK:
# COMPONENT_ARM_DISARM kept coming back FAILED ("Accels inconsistent",
# "GPS 1: Bad fix") when the controller tried to arm too early.
#
# Bump this further if you still see arm failures in the ArduPilot console -
# EKF/GPS settle time varies with machine load, and there isn't a fixed-time
# guarantee here (see note below on a more robust alternative).
ARDUPILOT_STARTUP_DELAY_SEC = 30.0

# SITL is started this long after Gazebo. With lock_step the ArduPilot plugin
# steps physics only when SITL sends a new servo frame. If SITL connects
# before Gazebo has finished initialising its renderer/camera (~3-5 s), the
# first exchange stalls, SITL re-sends the same frame ("No JSON sensor message
# received, resending servos"), the plugin drops it as a "Duplicate input
# frame", and both wait on each other forever (2026-09-25 11:45 session).
GAZEBO_STARTUP_DELAY_SEC = 10.0

# Every process this launch starts runs inside its own gnome-terminal (see the
# prefix= arguments below). gnome-terminal hands the command to its server and
# returns immediately, so from the launch system's point of view each node
# "exited" straight away and Ctrl+C has nothing left to kill: Gazebo, SITL,
# MAVProxy, MAVROS and the ROS nodes all keep running in their windows. This
# shutdown hook finds them by command line and terminates them (SIGINT first
# so they shut down cleanly, then SIGKILL for anything still alive).
# The plot script deliberately isn't in this list -- it's what you want to
# stay open after a run.
SIM_PROCESS_PATTERNS = [
    'gz sim',                                  # Gazebo wrapper (Ruby `gz`)
    'gz-sim-main',                             # Gazebo server (what `gz sim` execs)
    'gz-sim-gui-client',                       # Gazebo GUI window
    'sim_vehicle.py',                          # ArduPilot SITL wrapper
    'arducopter',                              # SITL binary
    'mavproxy',                                # MAVProxy console/map
    'mavros_node',
    'parameter_bridge',                        # ros_gz_bridge(s)
    'circumnavigation_controller/lib/circumnavigation_controller/',  # all our nodes
]


def _kill_sim_processes(event=None, context=None):
    """OnShutdown callback: terminate every sim process this launch spawned."""
    def pids(pattern):
        try:
            out = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout
        except OSError:
            return []
        me = {os.getpid(), os.getppid()}
        return [int(x) for x in out.split() if int(x) not in me]

    def signal_all(sig):
        found = set()
        for pat in SIM_PROCESS_PATTERNS:
            for pid in pids(pat):
                found.add(pid)
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        return found

    first = signal_all(signal.SIGINT)
    deadline = time.time() + 4.0
    while time.time() < deadline and signal_all(0):
        time.sleep(0.25)
    left = signal_all(signal.SIGKILL)
    print(f'[sim_launch] shutdown: sent SIGINT to {len(first)} sim process(es), '
          f'force-killed {len(left)} that were still running.')
    return None


def generate_launch_description():
    detector_arg = DeclareLaunchArgument(
        'detector', default_value='aruco', choices=['aruco', 'dope'],
        description='Perception node publishing /aruco_target/visual_odom: aruco or dope',
    )

    # relative_position_controller's cwd (see LOG_DIR above) needs to exist
    # before the node tries to open its CSV log there.
    os.makedirs(LOG_DIR, exist_ok=True)

    # Make sure gz can find the ardupilot gazebo plugins
    set_gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=ARDUPILOT_GAZEBO_PLUGIN_PATH + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
    )

    # Make sure gz can find custom models referenced by model:// URIs in the
    # world file (e.g. model://camera_calibrate_wall) - this is a different
    # env var from GZ_SIM_SYSTEM_PLUGIN_PATH above (that one's for plugins,
    # this one's for model/world assets). Adjust MODELS_DIR at the top if
    # your models live somewhere other than ardupilot_gazebo/models.
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=MODELS_DIR + ':' + WORLDS_DIR + ':' + REPO_MODELS_DIR + ':'
        + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', WORLD_FILE, '-v', '-r'],
        cwd=WORLDS_DIR,
        prefix=['gnome-terminal --title="Gazebo" --'],
        output='screen'
    )

    # Camera + clock bridge
    # NOTE: iris_runway_new.sdf includes model://iris_with_dope_gimbal (this
    # repo's models/) under the name iris_with_gimbal: the Iris with a real
    # 3-axis gimbal that ArduPilot drives on SERVO9/10/11. Its camera is on the
    # gimbal's pitch_link, 1280x720, 0.8 rad HFOV -- matching the detector
    # intrinsics below. (The old model://iris_with_gimbal resolved to
    # ardupilot_gazebo/worlds/iris_with_gimbal, a fixed body camera on
    # `camera_link`, so gimbal commands moved nothing.) Check with
    # `gz topic -l` if you change the model.
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

    # Bridge for driving the target car and reading its true pose.
    # DiffDrive in models/audi_r8/model.sdf listens on the bare gz topic
    # /cmd_vel; OdometryPublisher publishes /model/<world model name>/odometry
    # and the world names the car "LandingVehicle". The odometry is what the
    # controller logs as ground truth (car_gt_* columns in its CSV).
    car_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='car_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/LandingVehicle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        remappings=[
            ('/model/LandingVehicle/odometry', '/landing_vehicle/odometry'),
        ],
        output='screen',
    )

    ardupilot_sitl = ExecuteProcess(
        # 'JSON' is not a frame name (sim_vehicle.py rejects it); the frame is
        # gazebo-iris with the JSON physics backend selected via --model. On
        # current ArduPilot master the SITL binary looks up per-frame default
        # params by the --model name, so with --model JSON the gazebo-iris
        # defaults are NOT applied automatically and must be passed explicitly
        # (otherwise FRAME_CLASS stays 0 -> "Frame: UNSUPPORTED", cannot arm).
        cmd=[
            SIM_VEHICLE_SCRIPT,
            '-v', 'ArduCopter',
            '-f', 'gazebo-iris',
            '--model', 'JSON',
            '--add-param-file', COPTER_PARM_FILE,
            '--add-param-file', GAZEBO_IRIS_PARM_FILE,
            '--add-param-file', GIMBAL_PARM_FILE,
            '--add-param-file', GIMBAL_STARTUP_PARM_FILE,
            '--add-param-file', ATTITUDE_TUNING_PARM_FILE,
            '--add-param-file', GUIDED_LIMITS_PARM_FILE,
            '--console', '--map',
            '--out=udp:127.0.0.1:14555',
            # MAVProxy re-requests "all streams at --streamrate Hz" every 15 s
            # (default 4), which also resets any SET_MESSAGE_INTERVAL. At 4 Hz
            # /mavros/local_position/pose arrived only every ~0.3 s and the
            # controller flew on a stale position. 20 Hz for everything.
            '--mavproxy-args=--streamrate=20',
        ],
        cwd=ARDUPILOT_DIR,
        prefix=['gnome-terminal --title="ArduPilot SITL" --'],
        output='screen'
    )

    mavros = Node(
        package='mavros',
        executable='mavros_node',
        # distance_sensor: SITL streams its downward rangefinder as sensor id 0
        # and the plugin logs "DS: no mapping for sensor id: 0 ..." on every
        # message. This MAVROS build ignores per-plugin YAML config for the
        # mapping (verified: even the stock apm_config.yaml leaves it empty),
        # and nothing here consumes the rangefinder, so the plugin is disabled.
        parameters=[{'fcu_url': 'udp://:14555@', 'plugin_denylist': ['distance_sensor']}],
        prefix=['gnome-terminal --title="MAVROS" --'],
        output='screen'
    )

    # Camera intrinsics must match the sensor in the SDF (see CAMERA_TOPIC note):
    # 1280x720 with horizontal_fov 0.8 rad. Without these the detector resizes
    # 16:9 frames to 640x480 and builds its camera matrix from a 0.87 rad FOV,
    # so tag poses come out wrong.
    aruco_detector = Node(
        package=PACKAGE_NAME,
        executable='aruco_detector',
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('detector'), 'aruco')),
        parameters=[{
            'imgsz_width': 1280,
            'imgsz_height': 720,
            'camera_fov_horizontal': 0.8,
        }],
        prefix=['gnome-terminal --title="Aruco Detector" --'],
        output='screen'
    )

    # Same camera intrinsics as above. Publishes the same /aruco_target/*
    # topics with the same frame conventions, so the controller is unchanged.
    dope_detector = Node(
        package=PACKAGE_NAME,
        executable='dope_detector',
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('detector'), 'dope')),
        parameters=[{
            'imgsz_width': 1280,
            'imgsz_height': 720,
            'camera_fov_horizontal': 0.8,
            'weights_path': DOPE_WEIGHTS,
            'object_name': DOPE_OBJECT,
            'cuboid_dimensions_cm': DOPE_CUBOID_DIMENSIONS_CM,
            'processing_rate': 20.0,
        }],
        prefix=['gnome-terminal --title="DOPE Detector" --'],
        output='screen'
    )

    relative_position_controller = Node(
        package=PACKAGE_NAME,
        executable='relative_position_controller',
        cwd=LOG_DIR,
        # DOPE's object frame has the car's nose along +z; the ArUco marker
        # frame's heading is its x axis.
        parameters=[{
            'target_heading_axis': PythonExpression(
                ["'z' if '", LaunchConfiguration('detector'), "' == 'dope' else 'x'"]
            ),
            # ArduPilot's servo mount (MNT1_TYPE 1) reports the body-frame
            # servo angles (commanded earth-frame angle minus the drone's
            # tilt), even though its flags (44) claim horizon roll/pitch. Run
            # 20260924_140321 confirmed it: reported pitch = cmd + drone pitch
            # (r=0.94), and 'horizon'/'auto' left a vertical error equal to the
            # drone pitch (slope -0.97, up to 31 deg). So use 'body'.
            'gimbal_attitude_frame': 'body',
        }],
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
    delayed_sitl = TimerAction(period=GAZEBO_STARTUP_DELAY_SEC, actions=[ardupilot_sitl])

    delayed_mavlink_stack = TimerAction(
        period=GAZEBO_STARTUP_DELAY_SEC + ARDUPILOT_STARTUP_DELAY_SEC,
        actions=[
            mavros,
            aruco_detector,
            dope_detector,
            relative_position_controller,
            cinematic_gui,
        ]
    )

    # Automatically plot the tracking debug graphs as soon as
    # relative_position_controller exits (Ctrl+C in its terminal, or a
    # crash) - reads whatever relative_pid_<timestamp>.csv it just finished
    # writing in LOG_DIR and saves/shows the debug figure, no manual step
    # needed after a test run.
    plot_on_controller_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=relative_position_controller,
            on_exit=[
                ExecuteProcess(
                    cmd=['python3', PLOT_SCRIPT],
                    cwd=LOG_DIR,
                    prefix=['gnome-terminal --title="Tracking Debug Plot" --'],
                    output='screen'
                )
            ]
        )
    )

    # Ctrl+C on this launch (or any shutdown) tears down all the terminals'
    # processes -- see SIM_PROCESS_PATTERNS / _kill_sim_processes above.
    kill_all_on_shutdown = RegisterEventHandler(
        OnShutdown(on_shutdown=_kill_sim_processes)
    )

    return LaunchDescription([
        detector_arg,
        kill_all_on_shutdown,
        set_gz_plugin_path,
        set_gz_resource_path,
        gazebo,
        gz_bridge,
        car_bridge,
        delayed_sitl,
        delayed_mavlink_stack,
        plot_on_controller_exit,
    ])
