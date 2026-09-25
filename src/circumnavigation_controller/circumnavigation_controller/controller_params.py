#!/usr/bin/env python3
"""ROS parameters of relative_position_controller: name, default, description.

declare_parameters(node) declares them all and returns their values as a
namespace (cfg.target_altitude, ...), so the node reads each one in one place.
"""
from types import SimpleNamespace

from rcl_interfaces.msg import ParameterDescriptor

PARAMETERS = [
    # ---- Flight sequence / safety ----
    ("target_altitude", 3.0, "Takeoff altitude (m)."),
    ("hover_delay_sec", 4.0, "Hover time after takeoff before tracking starts (s)."),
    ("boundary_limit", 1000.0, "RTL when |x| or |y| of the drone exceeds this (m)."),
    ("target_timeout_sec", 10.0,
     "A target (and its camera-frame detection) counts as fresh for this long (s)."),

    # ---- Drone position loop ----
    ("shot_ground_z", 0.0,
     "Ground height in the local frame (m). The shot height is flown above this, "
     "not above the noisy target z estimate."),
    ("pid_speed_margin_xy", 8.0,
     "Horizontal speed limit = target speed + this margin (m/s)."),
    ("pid_max_speed_xy", 15.0,
     "Absolute horizontal speed limit (m/s); keep <= WP_SPD in config/guided_limits.parm."),
    ("pid_max_speed_z", 3.0, "Vertical speed limit (m/s)."),
    ("pid_max_accel_xy", 5.0,
     "Max change of the horizontal velocity command (m/s^2); match WP_ACC in config/guided_limits.parm."),
    ("pid_max_accel_z", 2.0, "Max change of the vertical velocity command (m/s^2)."),
    ("pid_derivative_tau_sec", 0.3, "Low-pass time constant of the PID derivative term (s)."),
    ("pid_kp_xy", 0.60,
     "Horizontal position gain (1/s). The drone answers a velocity command ~1.2 s late "
     "(sim); 0.8 overshot ~2.5 m arriving at the shot point. Lower = less overshoot, slower catch-up."),
    ("pid_ki_xy", 0.02,
     "Horizontal integral gain. 0.15 was tried: it did not remove the lag in turns and "
     "made a delayed loop more prone to oscillate."),
    ("pid_max_integral_xy", 3.0, "Horizontal integral clamp (m*s)."),

    # ---- Camera -> world transform ----
    ("gimbal_camera_roll_deg", 0.0, "Camera mount calibration on the gimbal (deg)."),
    ("gimbal_camera_pitch_deg", 0.0, "Camera mount calibration on the gimbal (deg)."),
    ("gimbal_camera_yaw_deg", 0.0, "Camera mount calibration on the gimbal (deg)."),
    ("gimbal_attitude_frame", "body",
     "Frame of the reported gimbal attitude: body (full drone attitude applied), "
     "horizon (only drone yaw), earth (none), auto (from the status flags). "
     "ArduPilot's servo mount reports body-frame angles."),
    ("pose_history_buffer_sec", 2.0,
     "How much drone pose / gimbal attitude history is kept for capture-time lookup (s)."),
    ("debug_transform_chain", True,
     "Log the [chain] frame-debug line (needs Gazebo car truth) once a second."),

    # ---- Target position: gating + constant-velocity KF ----
    ("enable_visual_position_filter", True,
     "Control on the KF position (True) or the raw detection (False; KF still gives velocity)."),
    ("min_visual_target_distance", 0.20, "Reject detections closer than this (m)."),
    ("max_visual_target_distance", 80.0, "Reject detections further than this (m)."),
    ("max_visual_position_jump", 3.0,
     "Reject a detection further than this + visual_jump_per_m * range from the KF prediction (m)."),
    ("visual_jump_per_m", 0.10, "Range-proportional part of the jump gate (m per m)."),
    ("visual_max_consecutive_rejects", 5,
     "After this many rejects in a row, accept the next detection and re-seed the filters."),
    ("visual_noise_per_m", 0.02, "KF measurement std grows by this per metre of range."),
    ("kf_accel_noise_std", 1.5, "KF white-noise acceleration std (m/s^2)."),
    ("kf_measurement_noise_std", 0.15, "KF measurement std at zero range (m)."),
    ("kf_initial_pos_std", 1.0, "KF initial position std (m)."),
    ("kf_initial_vel_std", 3.0, "KF initial velocity std (m/s)."),
    ("kf_max_target_speed", 15.0, "Clamp on the KF velocity estimate (m/s)."),
    ("kf_coast_timeout_sec", 1.0, "Re-seed the KF if the previous detection is older than this (s)."),
    ("kf_enable_adaptive_q", True, "Scale KF process noise up when innovations are large."),
    ("kf_adaptive_q_max_scale", 12.0, "Max adaptive process-noise scale."),
    ("kf_adaptive_q_decay", 0.5, "Adaptive process-noise decay."),
    ("enable_velocity_feedforward", True, "Add the KF target velocity to the PID output."),

    # ---- Target heading ----
    ("target_heading_axis", "x",
     "Axis of the detector's target frame that is the heading: x (ArUco), z (DOPE)."),
    ("enable_visual_heading_filter", True, "Jump-gate and smooth the detector heading."),
    ("visual_heading_alpha", 0.25, "Heading EMA factor per detection."),
    ("yaw_rate_tau_sec", 1.0, "Low-pass time constant of the yaw rate taken from the filtered heading (s)."),
    ("yaw_rate_min_speed", 1.0,
     "Yaw rate is 0 below this target speed (m/s) and fades in up to twice it: a car can't turn "
     "in place, and at 0.5 position noise on a parked car was enough to pass a fake yaw rate."),
    ("max_yaw_rate_deg", 30.0, "Clamp on the estimated target yaw rate (deg/s)."),
    ("heading_prediction_sec", 0.35,
     "Aim the shot with the heading predicted this far ahead (heading filter lag + detection "
     "latency), so in a turn the shot point isn't placed behind the car's real heading (s)."),
    ("enable_shot_rotation_feedforward", True,
     "Add yaw rate x shot offset to the velocity feedforward: in a turn the shot point swings "
     "around the car, which the car velocity alone doesn't include (the steady offset in turns)."),
    ("max_rotation_feedforward", 5.0,
     "Cap on the yaw rate x shot offset feedforward (m/s). A sharp turn with a 20 m shot needs "
     "~5 m/s; 3 cost 3 m of tracking there. Noise had pushed 5+ m/s on a parked car."),
    ("feedforward_timeout_sec", 0.5,
     "No velocity / yaw-rate feedforward (and no heading prediction) once the last detection is "
     "older than this: the estimate's motion is frozen at its last value during a loss."),
    ("max_visual_heading_jump_deg", 60.0, "Heading jumps larger than this are gated (deg)."),
    ("max_visual_heading_distance", 60.0, "UKF only: ignore headings beyond this range (m)."),
    ("enable_heading_ukf", False,
     "Estimate heading + yaw rate with the CTRV UKF instead of the EMA. Not recommended: with "
     "DOPE-level noise it flipped to the opposite heading in turns (offline test 2026-09-25)."),
    ("ukf_std_a", 1.5, "UKF longitudinal acceleration noise std."),
    ("ukf_std_yawdd", 0.5, "UKF yaw acceleration noise std."),
    ("ukf_std_pos", 0.15, "UKF position measurement std."),
    ("ukf_std_yaw", 0.20, "UKF heading measurement std."),
    ("ukf_coast_timeout_sec", 1.0, "Re-seed the UKF after a gap longer than this (s)."),
    ("ukf_max_yaw_rate", 3.0, "UKF yaw-rate clamp (rad/s)."),

    # ---- Gimbal ----
    ("gimbal_device_id", 0, "MAVLink gimbal device id."),
    ("gimbal_command_period_sec", 0.2, "Min time between gimbal pitch/yaw commands (s)."),
    ("gimbal_kp_yaw", 0.4,
     "Gimbal yaw gain on the image bearing error (per command). 0.3 reduced jitter but "
     "overshot ~15 deg when the drone body turned fast (tracking start); 0.5 was jittery."),
    ("gimbal_kd_yaw", 0.0, "Gimbal yaw derivative gain; 0.08 mostly amplified detection noise."),
    ("gimbal_yaw_deadband_deg", 0.5, "Leave the gimbal yaw alone while the target is within this of centre (deg)."),
    ("gimbal_max_yaw_step_deg", 8.0,
     "Max gimbal yaw change per command (deg); 3 made it unwind too slowly after fast body turns."),
    ("gimbal_max_pitch_down_deg", -135.0,
     "Lowest gimbal pitch command (deg); -90 is straight down, below that looks backwards. "
     "Mount limit MNT1_PITCH_MIN is -135 (was -80 in code, so it could never look straight down)."),
    ("gimbal_max_pitch_up_deg", 30.0, "Highest gimbal pitch command (deg); mount allows 45."),
    ("gimbal_max_yaw_deg", 160.0, "Gimbal yaw command limit (deg, +- about the nose); mount limit 160 (was 90)."),
    ("yaw_hold_radius", 5.0,
     "Hold the drone's body yaw while it is within this horizontal distance (m) of the target: "
     "overhead the bearing to the car flips 180 deg, which spun the drone (and the camera with it)."),
    ("gimbal_detection_timeout_sec", 0.5,
     "A detection older than this no longer steers the gimbal; it then points at the target "
     "estimate instead of re-applying the old image error (which made it drift to its limit)."),
    ("gimbal_initial_pitch_deg", -5.0,
     "Gimbal pitch at boot; keep equal to MNT1_NEUTRAL_Y in config/gimbal_startup.parm (deg)."),

    # ---- Search (gimbal sweep while no target) ----
    ("enable_search_mode", False, "Sweep the gimbal while tracking has no target."),
    ("search_gimbal_pitch_center_deg", -35.0, "Search sweep pitch centre (deg)."),
    ("search_gimbal_pitch_amp_deg", 15.0, "Search sweep pitch amplitude (deg)."),
    ("search_gimbal_yaw_amp_deg", 75.0, "Search sweep yaw amplitude (deg)."),
    ("search_gimbal_period_sec", 8.0, "Search sweep period (s)."),

    # ---- Cinematic commands ----
    ("cinematic_command_topic", "/cinematic_command", "JSON shot sequences from the GUI."),

    # ---- Evaluation inputs (logged only) ----
    ("truth_target_odom_topic", "/landing_vehicle/odometry",
     "Gazebo ground-truth car odometry (world ENU); empty to disable."),
    ("gps_topic", "/mavros/global_position/global", "Drone GPS fix; empty to disable."),
    ("world_origin_lat_deg", -35.363262, "Gazebo world origin latitude (iris_runway_new.sdf)."),
    ("world_origin_lon_deg", 149.165237, "Gazebo world origin longitude."),
]


def declare_parameters(node) -> SimpleNamespace:
    cfg = SimpleNamespace()
    for name, default, description in PARAMETERS:
        node.declare_parameter(name, default, ParameterDescriptor(description=description))
        setattr(cfg, name, node.get_parameter(name).value)
    return cfg
