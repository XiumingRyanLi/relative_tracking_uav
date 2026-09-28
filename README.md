# Ardupilot Quadcopter Simulation for Circumnavigation

This repository serves as a basis for ArduPilot-based quadcopter projects. It uses ArduPilot Software-In-The-Loop (SITL) simulation to simulate the physical flight controller in Gazebo and a custom ROS 2 node as the high-level controller. This allows algorithms developed with ROS 2 to be directly applied on ArduPilot-based quadcopter hardware with a sufficient companion computer. MAVROS interfaces with the flight controller using MAVLink messages.

The main application is **cinematic car tracking**: a drone with a 3-axis gimbal camera detects a car with DOPE (or an ArUco marker), estimates its pose in the world frame and holds a shot position relative to it (e.g. 18 m to the car's right, 4 m up).

## Installation

### Required Setup

1. ROS 2 (developed on Humble; currently run on ROS 2 "lyrical" with its vendored Gazebo)
2. Gazebo Harmonic (or newer, Gazebo Sim 10 is in use)
3. ArduPilot
4. ardupilot_gazebo
5. MAVROS

### ROS2

Install ROS2 Humble following the instructions below. Make sure you install `ros-humble-desktop` only and not `ros-humble-ros-base`.

https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

### Gazebo Harmonic

Follow the installation instructions for Gazebo Harmonic:

https://gazebosim.org/docs/harmonic/ros_installation/

Additionally, install these dependencies:

```bash
sudo apt update
sudo pip3 install transforms3d
sudo apt install ros-humble-tf-transformations
sudo apt remove 'ros-humble-ros-gz-*'
sudo apt install ros-humble-ros-gzharmonic-*
pip install pandas
```

### ArduPilot Local Installation

Follow the instructions here:

https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux

### ArduPilot Gazebo Plugin Installation

Follow the Harmonic instructions to download and install:

https://github.com/ArduPilot/ardupilot_gazebo

### MAVROS Installation

Follow these steps to install MAVROS. Ensure you install `ros-humble-mavros`:

https://github.com/mavlink/mavros/blob/ros2/mavros/README.md#installation

### Simulation assets (one-time)

`sim_launch.py` runs the world from `~/ardupilot_gazebo/worlds/iris_runway_new.sdf`, which includes
`model://iris_with_dope_gimbal` (the Iris with a real 3-axis gimbal). That model and its camera
gimbal live in this repo's `models/`; link them into ardupilot_gazebo so a manual `gz sim` finds
them too (the launch file also adds `models/` to `GZ_SIM_RESOURCE_PATH`):

```bash
ln -s $PWD/models/iris_with_dope_gimbal ~/ardupilot_gazebo/models/
ln -s $PWD/models/gimbal_small_3d_dope  ~/ardupilot_gazebo/models/
```

Keep the repo copy `worlds/iris_runway_new.sdf` and `~/ardupilot_gazebo/worlds/iris_runway_new.sdf`
in sync (the launch uses the latter).

> Gotcha: `model://` URIs resolve to the world file's own folder first, so a folder called
> `iris_with_gimbal` in `~/ardupilot_gazebo/worlds/` (a fixed body camera, no gimbal joints)
> silently shadows the stock model. That is why the gimbal model has a unique name. If the camera or
> gimbal behaves oddly, check the topic names with `gz topic -l`: the gimbal camera is
> `.../model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image`.

## Running

### Everything at once (recommended)

```bash
colcon build --packages-select circumnavigation_controller
source install/setup.bash
ros2 launch circumnavigation_controller sim_launch.py detector:=dope   # or detector:=aruco
```

The launch opens one terminal per process:

| t | Started |
|---|---|
| 0 s | Gazebo, camera + clock bridge, car bridge (`/cmd_vel`, `/landing_vehicle/odometry` ground truth) |
| 10 s | ArduPilot SITL (`sim_vehicle.py`, MAVProxy with `--streamrate=20`) |
| 40 s | MAVROS, detector (DOPE or ArUco), `relative_position_controller`, cinematic GUI |

All ROS nodes run on the Gazebo `/clock` (`use_sim_time:=true`, the default). The camera images carry sim-time stamps, so the controller can only look up the drone pose and gimbal angle at frame-capture time when it, MAVROS and the detector share that clock; on wall time it warned `history match ... from capture time` on every frame and used the live values.

- SITL waits 10 s for Gazebo (`GAZEBO_STARTUP_DELAY_SEC`): if it connects before Gazebo has set up
  its renderer, the lock-step link deadlocks ("No JSON sensor message received, resending servos" /
  "Duplicate input frame" and an empty Gazebo window).
- The controller then does GUIDED, arm, takeoff to 3 m, hovers 4 s and starts tracking.
- **Ctrl+C** in the launch terminal closes every sim process (Gazebo server and GUI included).
- When the controller exits, `scripts/plot_dope_evaluation.py` plots the run automatically.
- Rebuild after changing any launch or Python file (the workspace is not `--symlink-install`).

ArduPilot parameter files loaded by the launch (in order, later wins):

| File | Purpose |
|---|---|
| `copter.parm`, `gazebo-iris.parm` (ArduPilot) | frame defaults |
| `~/ardupilot_gazebo/config/gazebo-iris-gimbal.parm` | servo gimbal on SERVO9-11 |
| `config/gimbal_startup.parm` | gimbal boots in Neutral mode at pitch -5 deg so the camera sees the car (default RC-targeting points it at -45 deg) |
| `config/attitude_tuning.parm` | softer rate loop to reduce camera vibration |
| `config/guided_limits.parm` | `WP_SPD 15`, `WP_ACC 5` so the drone can keep up in turns |

If a parameter change seems to have no effect, SITL is using a value saved in
`~/ardupilot/eeprom.bin`: run `sim_vehicle.py` once with `-w`. Check with `param show <NAME>` in
the MAVProxy console.

### Running the pieces manually

```bash
# Gazebo
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
cd ~/ardupilot_gazebo/worlds && gz sim iris_runway_new.sdf -v -r

# Camera + clock bridge (gimbal camera, 1280x720, 0.8 rad HFOV, 30 Hz)
ros2 run ros_gz_bridge parameter_bridge \
  /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  --ros-args -r /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image:=/camera/image_raw

# ArduPilot SITL (wait until Gazebo shows the world)
cd ~/ardupilot && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON \
  --add-param-file=$HOME/ardupilot_gazebo/config/gazebo-iris-gimbal.parm \
  --console --map --out=udp:127.0.0.1:14555 --mavproxy-args=--streamrate=20

# MAVROS
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14555@

# Detector (one of)
ros2 run circumnavigation_controller dope_detector
ros2 run circumnavigation_controller aruco_detector

# Controller (DOPE: heading is the object z axis)
ros2 run circumnavigation_controller relative_position_controller --ros-args -p target_heading_axis:=z

# GUI (sends shot sequences on /cinematic_command)
ros2 run circumnavigation_controller cinematic_gui
```

The Gazebo camera only reaches ROS (`/camera/image_raw`) through the bridge above; without it the
detector and any ROS image viewer stay black.
The GStreamer stream on UDP 5600 only starts after
`gz topic -t <camera image topic>/enable_streaming -m gz.msgs.Boolean -p "data: 1"`.

### Controlling the car

The Audi (`models/audi_r8`, also in `~/ardupilot_gazebo/models/audi_r8`, keep both in sync) drives like
a real car: Ackermann steering (rear-wheel drive, steered front wheels, wheelbase 2.65 m, track 1.64 m,
max wheel angle 0.5 rad, ~5.5 m minimum turning radius) with acceleration-limited speed
(accelerate <= 3 m/s², brake <= 6 m/s², jerk <= 15 m/s³). `linear.x` is the target speed and
`angular.z` the target yaw rate, which becomes a steering angle, so the car only turns while moving
and can't stop or start instantly.

```bash
gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 5.0}, angular: {z: 0.0}"   # drive straight
gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 5.0}, angular: {z: 0.3}"   # turn (R = v / w ~ 17 m)
gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.0}, angular: {z: 0.0}"   # brake to a stop
```

Measured in Gazebo: 0 -> 8 m/s in ~2.7 s, 8 m/s with 0.4 rad/s commanded turns at ~21-22 deg/s, stops
from 8 m/s in ~2 s, and a turn command at standstill does nothing.

### Racing the car round a track

`race_driver` drives the car round a track world like a racing driver: flat out on the straights
(15 m/s, the drone's guided speed limit), braking into corners and accelerating out of them. It
follows a minimum-curvature racing line with pure pursuit and waits until the drone has taken off
and hovered before it starts.

```bash
ros2 launch circumnavigation_controller sim_launch.py world:=iris_silverstone race:=true
# or on its own (e.g. with the sim already running), without waiting for the drone:
ros2 run circumnavigation_controller race_driver --ros-args -p world:=iris_monza -p wait_for_drone:=false
```

| World | Lap | Lap time in Gazebo | Corner speeds |
|---|---|---|---|
| `iris_oschersleben` | 1.2 km | ~122 s | 6-9 m/s (slowest 5.7) |
| `iris_silverstone` | 1.9 km | ~177 s | 6-9 m/s (slowest 3.5, the hairpin) |
| `iris_monza` | 2.2 km | ~176 s | 6-9 m/s (slowest 4.7) |

Parameters: `max_speed` (15 m/s), `lat_accel` (2.5 m/s², cornering), `accel` (2.0 m/s²),
`brake` (4.0 m/s²), `min_speed` (3 m/s), `laps` (0 = keep going, else stop after N), `wait_for_drone`
(true), `start_altitude` (2.5 m), `start_delay` (6 s after reaching it). Lap times are logged.

`speed_zones` caps the speed on stretches of the lap, as flat triples `[s_start, s_end, max_speed, ...]`
(metres along the lap from the start line, decimals required; `s_end < s_start` wraps past the line).
The car brakes into each zone in time and accelerates out. At startup `race_driver` logs every corner
with its `s`, speed and radius to pick zones from:

```bash
ros2 run circumnavigation_controller race_driver --ros-args -p world:=iris_oschersleben \
  -p speed_zones:="[280.0, 330.0, 4.0, 1050.0, 1100.0, 4.5]"
```

- The speed profile is the usual lap-time-simulator recipe: corner speed `sqrt(lat_accel /
  curvature)`, then forward (acceleration) and backward (braking) passes, with a friction circle so
  the car doesn't accelerate hard while still cornering.
- Don't raise `lat_accel` or `accel` much: the rear-drive Audi spins out in Gazebo (power oversteer
  on corner exit) and its steering is slow (AckermannSteering `steer_p_gain` defaults to 1, a ~1 s
  steering lag). Both were tuned in Gazebo on all three tracks, down from 3-4 m/s², which crashed.
- Pure pursuit steers from the car's direction of travel (yaw plus slip from the odometry's lateral
  velocity), not its nose: the car fishtails at ~1 Hz after any steering input, and chasing the nose
  kept that going as a constant weave on the straights (yaw rate ±12 deg/s, now ~2 deg/s).
- Racing lines live in `config/race_lines/<world>.csv`, made by `tools/build_race_lines.py` from each
  world's track mesh and pose (run it with `/usr/bin/python3`; `--plot` saves a picture). Re-run it
  and rebuild after moving or rescaling a track in its world file.

### Controlling the gimbal

Pitch/yaw are degrees; pitch 0 = horizon, negative = down (limits -135..45), yaw positive = right of
the nose (±160). Stop `relative_position_controller` first, it re-commands the gimbal every 0.2 s.

```bash
ros2 service call /mavros/gimbal_control/manager/configure mavros_msgs/srv/GimbalManagerConfigure \
  "{sysid_primary: -2, compid_primary: -2, sysid_secondary: 0, compid_secondary: 0, gimbal_device_id: 0}"
ros2 service call /mavros/gimbal_control/manager/pitchyaw mavros_msgs/srv/GimbalManagerPitchyaw \
  "{pitch: -30.0, yaw: 0.0, pitch_rate: 0.0, yaw_rate: 0.0, flags: 0, gimbal_device_id: 0}"
```

## Controller

`relative_position_controller` holds the shot position relative to the detected car:

```
camera image -> detector -> car pose in the camera frame
  -> camera -> gimbal -> drone -> world transform
  -> target estimator: gating, constant-velocity KF (position, velocity),
     heading EMA + yaw rate
  -> shot point = car position + shot offset rotated by the (predicted) car heading,
     kept >= 10 m from the car by the cinematic planner
  -> PID with feedforward = car velocity + yaw rate x shot offset (only while detections are fresh)
  -> /mavros/setpoint_velocity/cmd_vel  (body yaw faces the car, held when within 5 m of it)
gimbal: image-space controller keeps the car centred (5 Hz pitch/yaw commands, one correction per
        detection); after 0.5 s without a detection it points at the car's estimated position
```

### Cinematic shots

`cinematic_planner.py` turns the GUI's shot sequence (`/cinematic_command`) into an offset from the car
(x forward, y left, z up). Every offset is kept at least `MIN_TARGET_RANGE` = 10 m from the car's box
centre (`_enforce_min_range`): closer than that the 4.4 m car no longer fits in the 46 x 26 deg camera
view and DOPE loses it. Holds/moves are pushed outwards along the same bearing at the same height;
an overpass climbs instead, so its path stays continuous and it can fly straight over the car.

### Code layout (`src/circumnavigation_controller/circumnavigation_controller/`)

| Module | Part | Contents |
|---|---|---|
| `relative_position_controller.py` | node | topics, timers, the drone and gimbal control loops |
| `controller_params.py` | config | every ROS parameter with its default and description (`ros2 param describe`) |
| `target_estimator.py` | control | detection gating, position/velocity KF, heading filter and yaw rate |
| `camera_frames.py` | control | camera -> gimbal -> drone -> world transforms, pose history |
| `pid_controller.py` | control | velocity PID (filtered D, speed and acceleration limits) |
| `gimbal_controller.py` | control | gimbal image-space controller |
| `cinematic_planner.py` | control | shot offset sequence (hold, move, orbit, overpass, push/pull), 10 m minimum distance to the car |
| `flight_sequencer.py` | logistics | GUIDED -> arm -> takeoff -> hover -> tracking, safety RTL |
| `gimbal_interface.py` | logistics | MAVROS gimbal manager commands and attitude feedback |
| `run_logger.py` | logging | CSV log, ground-truth evaluation columns, `[chain]` transform debug line |
| `race_driver.py`, `race_line.py` | sim | race-track car driver node; racing line, speed profile and pure pursuit |
| `geometry.py` | shared | angle and frame helpers |
| `target_kalman_filter.py`, `target_ctrv_ukf.py` | filters | CV Kalman filter; CTRV UKF (off by default, see below) |

### Tuning (current defaults and why)

| Parameter | Value | Notes |
|---|---|---|
| `gimbal_attitude_frame` | `body` | ArduPilot's servo mount reports body-frame angles despite its flags; `horizon` added the drone tilt twice (target estimate off by up to 30 deg) |
| `shot_ground_z` | 0.0 | the shot height is flown above the ground, not above the noisy car height estimate |
| `pid_kp_xy` / `pid_ki_xy` | 0.6 / 0.02 | the drone answers a velocity command ~1.2 s late (sim); 0.8 overshot ~2.5 m on arrival, `ki` 0.15 made turns oscillate |
| `pid_speed_margin_xy` / `pid_max_speed_xy` | 8 / 15 m/s | horizontal speed <= car speed + 8, total |
| `pid_max_accel_xy` | 5 m/s² | matches `WP_ACC` |
| `heading_prediction_sec` | 0.35 s | aim with the heading predicted over filter lag + detection latency |
| `enable_shot_rotation_feedforward` | true | adds yaw rate x shot offset; removed most of the sideways lag in turns (4 m -> 1.2 m at 4 deg/s) |
| `yaw_rate_tau_sec` / `yaw_rate_min_speed` | 1.0 s / 1.0 m/s | yaw rate = smoothed heading derivative, 0 below 1 m/s, full from 2 m/s (0.5 let heading noise on a parked car through) |
| `max_rotation_feedforward` | 5 m/s | cap on yaw rate x shot offset; noise had pushed the drone sideways at 5+ m/s |
| `feedforward_timeout_sec` | 0.5 s | a detection older than this starts COAST: shot heading frozen (no rotation feedforward or heading prediction), drone holds its current height, shot sequence paused until re-acquisition |
| `coast_full_speed_sec` / `coast_taper_sec` / `coast_max_distance` | 2 s / 1 s / 20 m | while coasting, the target (drone setpoint and gimbal aim) is carried on at its last velocity for 2 s, slowed to a stop over 1 s and never moved more than 20 m; after `target_timeout_sec` (10 s) it searches. The CSV `stage` column reads `coast` |
| `shot_transition_speed` / `shot_transition_min_sec` | 5 m/s / 2 s | a new GUI sequence first flies around the car (bearing, radius and height blended, the short way) from the current shot point to the sequence's start point; jumping there had the drone cut past the car at 8 m and lose it (run 20260928_120132) |
| `move_location` (shot) | — | arcs round the car at a constant radius and height, on the side that passes `via` (any location; the short way if `via` is one of the ends). It used to be a straight line except for back<->front, which cut the corner to 0.7 x radius |
| `enable_search_mode` / `search_altitude` | true / 20 m | SEARCH after `target_timeout_sec`: climb to 20 m where the drone is, face the last estimate and sweep the gimbal around it (±75 deg yaw, ±15 deg pitch). False hovers in place instead. CSV `stage` = `search` (or `hold`) |
| `gimbal_kp_yaw`, `gimbal_kd_yaw`, `gimbal_yaw_deadband_deg`, `gimbal_max_yaw_step_deg` | 0.4, 0, 0.5, 8 | less yaw jitter without big overshoot when the body turns fast |
| `gimbal_max_pitch_down_deg` / `gimbal_max_yaw_deg` | -135 / ±160 deg | the mount's real range (was -80 / ±90 in code, so the camera could never look straight down); below -90 it looks backwards, so an overhead pass needs no 180 deg yaw flip |
| `yaw_hold_radius` | 5 m | the drone holds its body yaw when nearly above the car, where the bearing flips 180 deg |
| `gimbal_detection_timeout_sec` | 0.5 s | each detection steers the gimbal once; after 0.5 s without one the gimbal points at the car's estimated position instead (re-applying the old error drove it to its pitch limit, e.g. into the sky) |
| `heading_flip_threshold_deg` / `heading_course_min_speed` / `heading_course_memory_sec` | 110 deg / 3 m/s / 3 s | DOPE sometimes reads the car nose-to-tail at long range; a heading more than 110 deg from the direction of travel is turned round 180 deg (flips were >= 127 deg off, correct readings <= 86 deg, 4 race runs). The direction of travel is remembered for 3 s so a KF re-seed can't let a flip in (run 20260928_145616 lost the car that way) |
| (latency) | — | the target estimate is carried on from the frame's capture stamp, not its arrival: it trailed the car by 0.7-0.8 m at 11 m/s. Falls back to arrival time if the stamp is in another clock domain |
| `target_filter` | ctra | heading-aided CTRA EKF (`target_ctra_ekf.py`: x, y, heading, speed, turn rate, along-track accel; measures DOPE position + flip-resolved heading) for the target position, velocity, acceleration, heading and yaw rate. The CV KF is not used: detections are gated by a chi-square test on the EKF's own prediction (`ctra_gate_nis` 25; the textbook 13.8 rejected 10 % of good frames because DOPE's drifting bias makes the EKF overconfident), the flip check's direction of travel comes from a line fit through the last 1 s of raw positions (independent of the heading), and z is an EMA of the measured z. `cv` = the previous KF + fixed jump gate + heading EMA. `scripts/benchmark_target_filter.py` (race-line truth, DOPE-like noise, held-out tracks): median error 15 % lower now, 28-34 % lower 1-2 s ahead, 38 % on velocity, 28 % on heading, and 1.7 % of 5-15 m outliers accepted vs 6.8 %. Tuning: `ctra_turn_rate_noise` 0.2, `ctra_jerk_noise` 0.5, `ctra_heading_std_deg` 4 |
| `enable_heading_ukf` | false | the CTRV UKF flips its heading by 180 deg in turns with DOPE-level noise (crash bug fixed, filter still not usable) |

Things that were tried and made it worse: `PSC_NE_JERK 20` (removed damping; the drone oscillated
±6 m around a parked car) and a speed-scaled heading filter (barely helped).

Current behaviour (car parked, run `20260925_173510`): holds 0.4 m from the shot point, flies an
overpass straight over the car (88 deg look-down) while losing the detection only 4.5 % of the time,
1.1 % over the whole run.

Known limitations:
- Directly overhead the gimbal is near gimbal lock: its yaw corrections barely move the car in the
  image and swing through large angles (seen -92 -> +157 deg). Harmless while the pitch keeps the
  car in view, but a pass slightly beside the car is more robust (DOPE is also weakest straight down).
- After leaving the 5 m yaw-hold zone the body turns ~180 deg to face the car again (up to ~47 deg/s)
  and the car sits 12-20 deg off-centre for ~5 s while the gimbal unwinds.
- Sudden car speed changes give a 10-25 m transient (drone acceleration and command delay).
- While a shot transitions (e.g. back -> front) the shot point's own motion is not fed forward yet
  (5-8 m lag during the move).
- On a parked car the shot point can wander ~±2 m from DOPE's viewing-angle-dependent heading bias.
- The planner's default shot (`default_radius` / `default_height` 3 m) is inside the 10 m limit; it is
  pushed out automatically, but setting defaults that already satisfy it avoids the correction.

## Logging and evaluation

Each controller run writes `logs/relative_pid_<timestamp>.csv` (control state, target estimate,
detector output and Gazebo ground truth). Plot a run:

```bash
python3 scripts/plot_dope_evaluation.py                              # newest CSV in logs/
python3 scripts/plot_dope_evaluation.py logs/relative_pid_<ts>.csv   # a specific run
```

Useful extra sources:
- `[chain]` lines in the controller output (`debug_transform_chain`): bearing error for each
  gimbal-frame convention and where the car should appear in the camera vs where the detector saw it.
- ArduPilot dataflash logs in `~/ardupilot/logs/*.BIN` (`GUIP` guided command vs `PSCN/PSCE` target
  and actual velocity) to measure the command -> response delay.
- The sim runs at ~0.8x real time while the controller and CSV use wall-clock time, so per-second
  values in the CSV read ~20% low relative to simulation time.

## DOPE model and training data

- Weights used by the sim: `DOPE_WEIGHTS` in `sim_launch.py`, now
  `weights/dope/audi_droneview_v3_epoch0850.pth` (the previous
  `audi_droneview_v2_epoch0725.pth` is kept next to it). The `.pth` files are local only
  (not committed); see `weights/dope/README.md` for provenance and test results.
- Training data (BlenderProc, `Deep_Object_Pose/data_generation/blenderproc_data_gen`, locally
  patched with `--near/--far` true-distance sampling and `--elev_min/--elev_max` drone viewpoints).
  Each dataset folder has a `GENERATION_COMMAND.txt`.

| Dataset | Frames | Viewpoints | Purpose |
|---|---|---|---|
| `~/data/AudiDroneView_v1` | 300 | 5-75 deg, 7-45 m, whole car in frame | first drone-view set |
| `~/data/AudiDroneView_v2` | 3000 | 3-75 deg, 7-45 m, whole car in frame | v2 model (with v1) |
| `~/data/AudiDroneView_v3_overhead` | 1200 | 55-89 deg, 6-20 m, mild truncation | failure case: nearly overhead |
| `~/data/AudiDroneView_v3_close` | 1200 | 20-89 deg, 4-10 m, car partly out of frame | failure case: close range |
| `~/data/AudiDroneView_test_v3` | 3 x 150 | overhead / close / standard | held-out test set (not trained on) |

The v3 sets target the detection losses in runs `20260925_131648` / `_131943`: every lost frame had
the car partly outside the image, at 4-9 m range or 59-89 deg look-down. The v3 model was fine-tuned
from v2 epoch 750 to 850 on all four sets (`~/data/AudiDroneView_all_v3`, 5700 frames, ~60 s/epoch):

```bash
cd ~/Desktop/ryan/Deep_Object_Pose/train
~/miniconda3/envs/ryan_6dof/bin/torchrun --nproc_per_node=1 train.py --data ~/data/AudiDroneView_all_v3 \
  --object Audi --net_path output/weights_droneview_v2/final_net_epoch_0750.pth --epochs 850 \
  --outf output/weights_droneview_v3 --batchsize 32 --imagesize 448 --lr 0.0001 --workers 16 --save_every 25
```

Held-out test set `~/data/AudiDroneView_test_v3` (150 frames each, never trained on):

| Test set | v2 epoch 725 | v3 epoch 850 |
|---|---|---|
| overhead (55-89 deg, 6-20 m) | 93% detected | 98% |
| close (20-89 deg, 4-10 m) | 56% | 65% |
| standard (3-75 deg, 7-45 m) | 99% | 99% |

Remaining close-range misses have at most 3 of the 8 car corners in the image (a limit of the
cuboid-keypoint method), so keep shots >= 10 m from the car. Compare models with:

```bash
source /opt/ros/lyrical/setup.bash
python3 scripts/eval_dope_weights.py --weights weights/dope/*.pth \
  --data ~/data/AudiDroneView_test_v3/{overhead,close,standard}
```

Don't render or train while flying the sim: it saturates the GPU/CPU, DOPE stops detecting and the
tracking falls apart.
