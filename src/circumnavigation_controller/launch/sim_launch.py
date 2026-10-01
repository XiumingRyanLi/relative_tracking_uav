"""
Sim launch: Gazebo + ros_gz_bridge + ArduPilot SITL + MAVROS + perception/control nodes.

Assumes you've already got:
  - ardupilot_gazebo built at $HOME/ardupilot_gazebo
  - ardupilot checked out at $HOME/ardupilot
  - the world file at ~/ardupilot_gazebo/worlds/<world>.sdf (copies in <repo>/worlds)

Usage:
    ros2 launch circumnavigation_controller sim_launch.py
    ros2 launch circumnavigation_controller sim_launch.py detector:=dope
    ros2 launch circumnavigation_controller sim_launch.py world:=iris_monza
    ros2 launch circumnavigation_controller sim_launch.py controller:=pid
    ros2 launch circumnavigation_controller sim_launch.py world:=iris_silverstone race:=true

`world` is the world file name without .sdf (default iris_runway_new). The
file's <world name="..."> must equal it, since gz topic names contain it:
  iris_runway_new    - airfield runway
  iris_monza         - Monza race track (models/race_track_monza)
  iris_silverstone   - Silverstone race track (models/race_track_silverstone)
  iris_oschersleben  - Oschersleben race track (models/race_track_oschersleben)
  iris_<track>_trees - the same tracks lined with trees that can block the
                       drone's view of the car (tools/build_tree_worlds.py)

`race:=true` (track worlds only) starts race_driver, which drives the car
round the world's racing line (config/race_lines/<world>.csv) once the drone
has taken off: flat out on straights, braking into corners.

`detector` picks the perception node feeding /aruco_target/visual_odom:
  aruco (default) - AprilTag/ArUco solvePnP detector
  dope            - NVIDIA DOPE network (see DOPE_WEIGHTS below)

Experiments (scripts/run_experiments.py starts this file with these):
  headless:=true     no windows: no gnome-terminals, Gazebo server only (-s;
                     the camera still renders on the GPU through the desktop's
                     display), SITL without console/map, no GUI, no plot. All
                     output on this launch's stdout. The machine-wide kill on
                     shutdown is skipped: the runner kills its own process group.
  shot_sequence:=<json>   controller shot sequence (else DEFAULT_SHOT_SEQUENCE)
  run_name:=<name>        CSV name <log_dir>/<name>.csv
  log_dir:=<dir>          controller cwd / CSV directory (default logs/)
  sitl_dir:=<dir>         SITL --use-dir (eeprom.bin, dataflash logs)
  experiment_duration:=<s>  controller LANDs this long after the shot starts
  shot_start_speed:=<m/s>   shot waits for the car to exceed this speed
  car_mode:=race|launch, max_speed, laps, race_line_file, start_delay,
  launch_speed, launch_accel, launch_hold, launch_brake,
  launch_path (straight|sine), sine_amplitude, sine_period  -> race_driver
"""

import os
import signal
import subprocess
import time
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

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
# World files live in WORLDS_DIR as <world>.sdf (see the `world` launch arg);
# gz must be run from WORLDS_DIR so relative <uri> includes in the sdf resolve.
DEFAULT_WORLD = 'iris_runway_new'

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

# DOPE checkpoint used by `detector:=dope`: a plain PyTorch state_dict trained
# with DDP ("module." prefix stripped on load). Object + cuboid dimensions below
# must match what it was trained on. Local copies live in weights/dope/ (201 MB
# each, not committed; see weights/dope/README.md).
#   v3 epoch 850 (2026-09-25): + overhead / close-range training views; on the
#       held-out test set 98% overhead / 65% close / 99% standard detection.
#   v2 epoch 725: previous model (93% / 56% / 99%); to go back, use
#       'audi_droneview_v2_epoch0725.pth'.
DOPE_WEIGHTS = os.path.join(WORKSPACE_ROOT, 'weights', 'dope', 'audi_droneview_v3_epoch0850.pth')
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


def _term(headless, title):
    """gnome-terminal window for one process (none when headless)."""
    return None if headless else [f'gnome-terminal --title="{title}" --']


def _launch_setup(context):
    cfg = {k: v for k, v in context.launch_configurations.items()}
    headless = cfg['headless'] == 'true'
    world = cfg['world']
    detector = cfg['detector']
    log_dir = os.path.expanduser(cfg['log_dir']) or LOG_DIR
    # relative_position_controller's cwd needs to exist before the node
    # tries to open its CSV log there.
    os.makedirs(log_dir, exist_ok=True)

    # Every ROS node runs on the Gazebo /clock (bridged by gz_bridge). The
    # camera images carry sim-time stamps; with MAVROS and the controller on
    # wall time the controller's capture-time pose/gimbal lookup was ~1.8e9 s
    # off, so it fell back to the live values ("history match ... from
    # capture time" warnings) and mis-placed detections while the drone
    # turned or the gimbal moved.
    SIM_TIME = {'use_sim_time': cfg['use_sim_time'] == 'true'}

    # Gazebo. Headless: server only (-s). The camera sensor still renders
    # with ogre2 on the GPU through the desktop session's display (DISPLAY is
    # kept); fully headless EGL/software rendering ran at ~0.5x real time.
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world + '.sdf', '-v', '-r'] + (['-s'] if headless else []),
        cwd=WORLDS_DIR,
        prefix=_term(headless, 'Gazebo'),
        output='screen'
    )

    # Camera + clock bridge
    # NOTE: the worlds include model://iris_with_dope_gimbal (this
    # repo's models/) under the name iris_with_gimbal: the Iris with a real
    # 3-axis gimbal that ArduPilot drives on SERVO9/10/11. Its camera is on the
    # gimbal's pitch_link, 1280x720, 0.8 rad HFOV -- matching the detector
    # intrinsics below. (The old model://iris_with_gimbal resolved to
    # ardupilot_gazebo/worlds/iris_with_gimbal, a fixed body camera on
    # `camera_link`, so gimbal commands moved nothing.) Check with
    # `gz topic -l` if you change the model.
    camera_topic = ('/world/' + world + '/model/iris_with_gimbal/model/gimbal/'
                    'link/pitch_link/sensor/camera/image')
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            camera_topic + '@sensor_msgs/msg/Image[gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[(camera_topic, '/camera/image_raw')],
        prefix=_term(headless, 'GZ Bridge'),
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
        remappings=[('/model/LandingVehicle/odometry', '/landing_vehicle/odometry')],
        output='screen',
    )

    actions = [gazebo, gz_bridge, car_bridge]

    # Races the car round the track (race:=true). Waits by itself until the
    # drone has taken off and hovered (wait_for_drone), so it can start with
    # Gazebo. Experiment overrides are passed only when given.
    if cfg['race'] == 'true':
        driver_params = {'world': world, 'mode': cfg['car_mode']}
        for arg, name, cast in (('max_speed', 'max_speed', float), ('laps', 'laps', int),
                                ('race_line_file', 'race_line_file', str),
                                ('start_delay', 'start_delay', float),
                                ('launch_speed', 'launch_speed', float),
                                ('launch_accel', 'launch_accel', float),
                                ('launch_hold', 'launch_hold_sec', float),
                                ('launch_brake', 'launch_brake', float),
                                ('launch_path', 'launch_path', str),
                                ('sine_amplitude', 'sine_amplitude', float),
                                ('sine_period', 'sine_period', float)):
            if cfg[arg] != '':
                driver_params[name] = cast(cfg[arg])
        actions.append(Node(
            package=PACKAGE_NAME,
            executable='race_driver',
            parameters=[driver_params, SIM_TIME],
            prefix=_term(headless, 'Race Driver'),
            output='screen'
        ))

    # Parallel experiment runs: SITL instance N moves all its ports by 10 N
    # (MAVLink 5760, Gazebo JSON 9002 -- the world's drone model must listen
    # there, see scripts/run_experiments.py), and the MAVROS link moves with it.
    instance = int(cfg['instance'] or 0)
    mavlink_port = 14555 + 10 * instance
    sitl_cmd = [
        # 'JSON' is not a frame name (sim_vehicle.py rejects it); the frame is
        # gazebo-iris with the JSON physics backend selected via --model. On
        # current ArduPilot master the SITL binary looks up per-frame default
        # params by the --model name, so with --model JSON the gazebo-iris
        # defaults are NOT applied automatically and must be passed explicitly
        # (otherwise FRAME_CLASS stays 0 -> "Frame: UNSUPPORTED", cannot arm).
        SIM_VEHICLE_SCRIPT,
        '-v', 'ArduCopter',
        '-f', 'gazebo-iris',
        '--model', 'JSON',
        '-I', str(instance),
        '--add-param-file', COPTER_PARM_FILE,
        '--add-param-file', GAZEBO_IRIS_PARM_FILE,
        '--add-param-file', GIMBAL_PARM_FILE,
        '--add-param-file', GIMBAL_STARTUP_PARM_FILE,
        '--add-param-file', ATTITUDE_TUNING_PARM_FILE,
        '--add-param-file', GUIDED_LIMITS_PARM_FILE,
    ]
    if not headless:
        sitl_cmd += ['--console', '--map']
    if cfg['sitl_rebuild'] != 'true':
        sitl_cmd += ['--no-rebuild']
    if cfg['sitl_dir']:
        sitl_dir = os.path.expanduser(cfg['sitl_dir'])
        os.makedirs(sitl_dir, exist_ok=True)
        sitl_cmd += ['--use-dir=' + sitl_dir]
    sitl_cmd += [
        '--out=udp:127.0.0.1:' + str(mavlink_port),
        # MAVProxy re-requests "all streams at --streamrate Hz" every 15 s
        # (default 4), which also resets any SET_MESSAGE_INTERVAL. At 4 Hz
        # /mavros/local_position/pose arrived only every ~0.3 s and the
        # controller flew on a stale position. 20 Hz for everything.
        '--mavproxy-args=--streamrate=20',
    ]
    sitl_env = None
    if headless:
        # Without DISPLAY (or tmux/screen) run_in_terminal_window.sh runs the
        # arducopter binary in the background instead of in an xterm, logging
        # to $TMPDIR/ArduCopter.log; MAVProxy runs in this process.
        sitl_env = {k: v for k, v in os.environ.items()
                    if k not in ('DISPLAY', 'WAYLAND_DISPLAY', 'TMUX', 'STY', 'ZELLIJ')}
        if cfg['sitl_dir']:
            sitl_env['TMPDIR'] = os.path.expanduser(cfg['sitl_dir'])
    ardupilot_sitl = ExecuteProcess(
        cmd=sitl_cmd,
        cwd=ARDUPILOT_DIR,
        env=sitl_env,
        prefix=_term(headless, 'ArduPilot SITL'),
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
        parameters=[{'fcu_url': f'udp://:{mavlink_port}@', 'plugin_denylist': ['distance_sensor']}, SIM_TIME],
        prefix=_term(headless, 'MAVROS'),
        output='screen'
    )

    stack = [mavros]
    # Camera intrinsics must match the sensor in the SDF (see CAMERA_TOPIC note):
    # 1280x720 with horizontal_fov 0.8 rad. Without these the detector resizes
    # 16:9 frames to 640x480 and builds its camera matrix from a 0.87 rad FOV,
    # so tag poses come out wrong.
    if detector == 'aruco':
        stack.append(Node(
            package=PACKAGE_NAME,
            executable='aruco_detector',
            parameters=[{
                'imgsz_width': 1280,
                'imgsz_height': 720,
                'camera_fov_horizontal': 0.8,
            }, SIM_TIME],
            prefix=_term(headless, 'Aruco Detector'),
            output='screen'
        ))
    else:
        # Same camera intrinsics as above. Publishes the same /aruco_target/*
        # topics with the same frame conventions, so the controller is unchanged.
        stack.append(Node(
            package=PACKAGE_NAME,
            executable='dope_detector',
            parameters=[{
                'imgsz_width': 1280,
                'imgsz_height': 720,
                'camera_fov_horizontal': 0.8,
                'weights_path': DOPE_WEIGHTS,
                'object_name': DOPE_OBJECT,
                'cuboid_dimensions_cm': DOPE_CUBOID_DIMENSIONS_CM,
                'processing_rate': 20.0,
            }, SIM_TIME],
            prefix=_term(headless, 'DOPE Detector'),
            output='screen'
        ))

    controller_params = {
        # DOPE's object frame has the car's nose along +z; the ArUco marker
        # frame's heading is its x axis.
        'target_heading_axis': 'z' if detector == 'dope' else 'x',
        # ArduPilot's servo mount (MNT1_TYPE 1) reports the body-frame
        # servo angles (commanded earth-frame angle minus the drone's
        # tilt), even though its flags (44) claim horizon roll/pitch. Run
        # 20260924_140321 confirmed it: reported pitch = cmd + drone pitch
        # (r=0.94), and 'horizon'/'auto' left a vertical error equal to the
        # drone pitch (slope -0.97, up to 31 deg). So use 'body'.
        'gimbal_attitude_frame': 'body',
        'controller': cfg['controller'],
        # Experiments. Strings go through ParameterValue(str) so a JSON shot
        # sequence is not parsed as YAML.
        'shot_sequence': ParameterValue(cfg['shot_sequence'], value_type=str),
        'run_name': ParameterValue(cfg['run_name'], value_type=str),
        'experiment_duration_sec': float(cfg['experiment_duration'] or 0.0),
        'shot_start_speed': float(cfg['shot_start_speed'] or 0.0),
    }
    relative_position_controller = Node(
        package=PACKAGE_NAME,
        executable='relative_position_controller',
        cwd=log_dir,
        parameters=[controller_params, SIM_TIME],
        # drone_mpc.py loads acados from here (built once, see README).
        additional_env={'ACADOS_SOURCE_DIR': os.path.join(HOME, 'acados')},
        prefix=_term(headless, 'Relative Position Controller'),
        output='screen'
    )
    stack.append(relative_position_controller)
    if not headless:
        stack.append(Node(
            package=PACKAGE_NAME,
            executable='cinematic_gui',
            parameters=[SIM_TIME],
            prefix=_term(headless, 'Cinematic GUI'),
            output='screen'
        ))

    # MAVROS (and anything that depends on a live FCU connection) waits until
    # ArduPilot has had a chance to boot. aruco_detector doesn't strictly need
    # to wait, but keeping everything downstream together avoids the
    # controller/gui coming up against a MAVROS that isn't connected yet.
    actions.append(TimerAction(period=GAZEBO_STARTUP_DELAY_SEC, actions=[ardupilot_sitl]))
    actions.append(TimerAction(period=GAZEBO_STARTUP_DELAY_SEC + ARDUPILOT_STARTUP_DELAY_SEC,
                               actions=stack))

    if not headless:
        # Automatically plot the tracking debug graphs as soon as
        # relative_position_controller exits (Ctrl+C in its terminal, or a
        # crash) - reads whatever relative_pid_<timestamp>.csv it just finished
        # writing in LOG_DIR and saves/shows the debug figure, no manual step
        # needed after a test run.
        actions.append(RegisterEventHandler(
            OnProcessExit(
                target_action=relative_position_controller,
                on_exit=[
                    ExecuteProcess(
                        cmd=['python3', PLOT_SCRIPT],
                        cwd=log_dir,
                        prefix=_term(headless, 'Tracking Debug Plot'),
                        output='screen'
                    )
                ]
            )
        ))
        # Ctrl+C on this launch (or any shutdown) tears down all the terminals'
        # processes -- see SIM_PROCESS_PATTERNS / _kill_sim_processes above.
        # Headless runs are real children of this launch (no terminals), so
        # the runner stops them by process group instead of this machine-wide
        # kill.
        actions.append(RegisterEventHandler(OnShutdown(on_shutdown=_kill_sim_processes)))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'detector', default_value='aruco', choices=['aruco', 'dope'],
            description='Perception node publishing /aruco_target/visual_odom: aruco or dope'),
        DeclareLaunchArgument(
            'world', default_value=DEFAULT_WORLD,
            description='World in ~/ardupilot_gazebo/worlds, without .sdf; its '
                        '<world name> must match (e.g. iris_runway_new, iris_monza)'),
        DeclareLaunchArgument(
            'race', default_value='false', choices=['true', 'false'],
            description='Drive the car with race_driver (car_mode race needs '
                        'config/race_lines/<world>.csv, i.e. a race-track world)'),
        DeclareLaunchArgument(
            'controller', default_value='mpc', choices=['mpc', 'pid'],
            description='Drone tracking controller: mpc (acados, ~/acados) or pid'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true', choices=['true', 'false'],
            description='Run the ROS nodes on the Gazebo /clock'),
        DeclareLaunchArgument(
            'headless', default_value='false', choices=['true', 'false'],
            description='No windows (Gazebo server only, still GPU rendering); for scripts/run_experiments.py'),
        DeclareLaunchArgument('shot_sequence', default_value='',
                              description='Controller shot sequence, JSON list (empty: DEFAULT_SHOT_SEQUENCE)'),
        DeclareLaunchArgument('run_name', default_value='', description='CSV log name (without .csv)'),
        DeclareLaunchArgument('log_dir', default_value=LOG_DIR, description='Directory for the controller CSV'),
        DeclareLaunchArgument('sitl_dir', default_value='', description='SITL --use-dir (eeprom, logs)'),
        DeclareLaunchArgument('instance', default_value='0',
                              description='SITL instance (ports + 10 N) for parallel runs; the world must '
                                          'carry a drone model listening on 9002 + 10 N'),
        DeclareLaunchArgument('sitl_rebuild', default_value='true', choices=['true', 'false'],
                              description='Let sim_vehicle.py rebuild ArduCopter first'),
        DeclareLaunchArgument('experiment_duration', default_value='0',
                              description='LAND this long after the shot sequence starts (s); 0 = off'),
        DeclareLaunchArgument('shot_start_speed', default_value='0',
                              description='Shot sequence waits for the car to exceed this speed (m/s)'),
        DeclareLaunchArgument('car_mode', default_value='race', choices=['race', 'launch'],
                              description='race_driver mode: race (racing line) or launch (straight-line profile)'),
        DeclareLaunchArgument('max_speed', default_value='', description='race_driver max_speed (m/s)'),
        DeclareLaunchArgument('laps', default_value='', description='race_driver laps (0 = keep racing)'),
        DeclareLaunchArgument('race_line_file', default_value='', description='race_driver racing line CSV'),
        DeclareLaunchArgument('start_delay', default_value='', description='race_driver start_delay (s)'),
        DeclareLaunchArgument('launch_speed', default_value='', description='launch mode target speed (m/s)'),
        DeclareLaunchArgument('launch_accel', default_value='', description='launch mode acceleration (m/s^2)'),
        DeclareLaunchArgument('launch_hold', default_value='', description='launch mode hold time (s)'),
        DeclareLaunchArgument('launch_brake', default_value='', description='launch mode braking (m/s^2)'),
        DeclareLaunchArgument('launch_path', default_value='',
                              description='launch mode path: straight or sine (weave along the start line)'),
        DeclareLaunchArgument('sine_amplitude', default_value='', description='sine path amplitude (m)'),
        DeclareLaunchArgument('sine_period', default_value='',
                              description='sine path: seconds per weave at launch_speed'),
    ]

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

    return LaunchDescription(args + [
        set_gz_plugin_path,
        set_gz_resource_path,
        OpaqueFunction(function=_launch_setup),
    ])
