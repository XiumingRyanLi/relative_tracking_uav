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
     "target_filter=cv only: reject a detection further than this + visual_jump_per_m * range "
     "from the KF prediction (m). ctra gates with ctra_gate_nis instead."),
    ("visual_jump_per_m", 0.10, "target_filter=cv only: range-proportional part of the jump gate (m per m)."),
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

    # ---- Target loss: coast on the last estimate ----
    # Past feedforward_timeout_sec without a detection the drone is COASTING:
    # the target is predicted on from its last position at its last velocity,
    # the shot heading is frozen, the height held and the shot sequence paused.
    # After target_timeout_sec it hovers.
    ("coast_full_speed_sec", 2.0,
     "Predict the lost target on at its last velocity for this long after the last detection (s). "
     "Longer helps on straights, but a constant-velocity guess runs off the track in corners."),
    ("coast_taper_sec", 1.0,
     "Then slow the prediction to a stop over this long (s); it stays there until target_timeout_sec."),
    ("coast_max_distance", 20.0,
     "Cap on how far the prediction moves the target from its last estimate (m)."),
    ("max_visual_heading_jump_deg", 60.0, "Heading jumps larger than this are gated (deg)."),
    ("heading_course_min_speed", 3.0,
     "Above this target speed (m/s) the detector heading is checked against the direction of travel."),
    ("heading_flip_threshold_deg", 110.0,
     "A detector heading further than this from the direction of travel is a nose-to-tail flip "
     "and is turned round 180 deg (flips measured >= 127 deg off, correct readings <= 86 deg)."),
    ("heading_course_memory_sec", 3.0,
     "After a KF re-seed (velocity zeroed) the previous velocity is used for that check if it is "
     "at most this old (s)."),
    ("max_visual_heading_distance", 60.0, "UKF only: ignore headings beyond this range (m)."),
    ("target_filter", "ctra",
     "ctra: heading-aided CTRA EKF for position, velocity, acceleration, heading and yaw rate "
     "(the KF still gates and gives z). cv: the constant-velocity KF + heading EMA as before. "
     "Offline (scripts/benchmark_target_filter.py, held-out tracks) ctra was 13 % better now, "
     "29-31 % at 1-2 s ahead, 40 % on velocity and 31 % on heading."),
    ("ctra_turn_rate_noise", 0.2, "CTRA: white-noise turn-rate derivative std (rad/s^2)."),
    ("ctra_jerk_noise", 0.5, "CTRA: white-noise jerk std (m/s^3)."),
    ("ctra_heading_std_deg", 4.0, "CTRA: detector heading measurement std (deg); DOPE measured ~3.6."),
    ("ctra_gate_nis", 25.0,
     "CTRA: reject a detection whose position NIS against the EKF prediction exceeds this "
     "(chi-square, 2 DOF). Above the textbook 13.8 because DOPE's drifting bias makes the EKF "
     "overconfident: 13.8 rejected 10 % of good frames, 25 about 1 % while letting 1-2 % of "
     "5-15 m outliers through (the cv jump gate: 5-7 %)."),
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
    ("enable_search_mode", True,
     "Once the target is lost for target_timeout_sec: climb to search_altitude where the drone "
     "is and sweep the gimbal around the last estimate. False: hover in place."),
    ("search_altitude", 20.0,
     "Search height above shot_ground_z (m): a wider view of the ground, and DOPE still "
     "detects the car out to ~45 m."),
    ("search_gimbal_pitch_center_deg", -35.0,
     "Search sweep pitch centre (deg) when there has been no target yet; afterwards the sweep "
     "is centred on the last target estimate."),
    ("search_gimbal_pitch_amp_deg", 15.0, "Search sweep pitch amplitude (deg)."),
    ("search_gimbal_yaw_amp_deg", 75.0, "Search sweep yaw amplitude (deg)."),
    ("search_gimbal_period_sec", 8.0, "Search sweep period (s)."),

    # ---- Cinematic commands ----
    ("cinematic_command_topic", "/cinematic_command", "JSON shot sequences from the GUI."),
    ("shot_transition_speed", 5.0,
     "A new sequence first flies around the car from the current shot point to its start "
     "point at this speed relative to the car (m/s); jumping there cut past the car."),
    ("shot_transition_min_sec", 2.0, "Shortest such transition (s)."),

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
