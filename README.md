# Ardupilot Quadcopter Simulation for Circumnavigation

This repository serves as a basis for ArduPilot-based quadcopter projects. It uses ArduPilot Software-In-The-Loop (SITL) simulation to simulate the physical flight controller in gazebo and a custom ROS2 node as the high-level controller. This allows algorithms developed with ROS2 to be directly applied on ArduPilot-based quadcopter hardware with a sufficent companion computer. MAVROS to interface with the flight controller using MAVLink messages. 

## Installation

### Required Setup

1. ROS2 Humble
2. Gazebo Harmonic
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



## Running

Each application will need to be run in separate terminal windows.

### Run Gazebo Simulation

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/{PATH_TO_INSTALL}/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
cd worlds
gz sim iris_runway.sdf -v -r

gz sim iris_runway_new.sdf -v -r

```

### Run Gazebo Camera Bridge

```bash
ros2 run ros_gz_bridge parameter_bridge /world/iris_runway/model/iris_with_gimbal/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image --ros-args -r /world/iris_runway/model/iris_with_gimbal/link/camera_link/sensor/camera/image:=/camera/image_raw
```

Final bridge
```bash
ros2 run ros_gz_bridge parameter_bridge \
/world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
/model/LandingVehicle/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
/gimbal/cmd_roll@std_msgs/msg/Float64]gz.msgs.Double \
/gimbal/cmd_pitch@std_msgs/msg/Float64]gz.msgs.Double \
/gimbal/cmd_yaw@std_msgs/msg/Float64]gz.msgs.Double \
--ros-args \
-r /world/iris_runway_new/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image:=/camera/image_raw \
-r /model/LandingVehicle/odometry:=/aruco_target/odom
```

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /world/iris_runway/model/iris_with_gimbal/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  --ros-args -r /world/iris_runway/model/iris_with_gimbal/link/camera_link/sensor/camera/image:=/camera/image_raw
```


### Run ArduPilot

```bash
cd ~/ardupilot && sim_vehicle.py -v ArduCopter --console --map -w --out=udp:127.0.0.1:14555 -f gazebo-iris --model JSON

cd ~/ardupilot && sim_vehicle.py -D -v ArduCopter -f JSON \
--add-param-file=$HOME/ardupilot_gazebo/config/gazebo-iris-gimbal.parm \
--console --map \
--out=udp:127.0.0.1:14555
```

### Run MAVROS

```bash
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14555@
```

### Run YOLO Perception

```bash
source install/setup.bash
ros2 run circumnavigation_controller bearing_measurement_generation
```

### Run Circumnavigation Controller

```bash
source install/setup.bash
ros2 run circumnavigation_controller controller

ros2 run circumnavigation_controller relative_position_controller
```

### Controlling the Rover
```bash
gz topic -t "/cmd_rover_vel" -m gz.msgs.Twist -p "linear: {x: 0.6}, angular: {z: 1.0}"

```