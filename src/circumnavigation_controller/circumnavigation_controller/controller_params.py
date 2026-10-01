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
    ("target_timeout_sec", 5.0,
     "A target counts as fresh for this long after its last detection (s); after that SEARCH "
     "starts. Just past the 4 s coast prediction: at 10 s (run 20260929_094556) the search "
     "began 130 m from a car that had driven on."),

    # ---- Controller choice ----
    ("controller", "mpc",
     "mpc: acados MPC (drone_mpc.py / mpc_tracker.py) flies tracking and coasting; the PID keeps "
     "search/hold and takes over if the MPC is unavailable or a solve fails. pid: the PID only. "
     "Offline closed loop (scripts/mpc_closed_loop_sim.py, Silverstone): error to the shot point "
     "5.2 / 7.4 m median / p90 vs 25.5 / 41.1 m, in the +-5 m / +-10 deg band 39 % vs 2 %."),

    # ---- MPC (controller=mpc) ----
    ("mpc_horizon_sec", 2.0, "MPC prediction horizon (s). Changing it regenerates the solver."),
    ("mpc_steps", 20, "MPC steps over the horizon. Changing it regenerates the solver."),
    ("mpc_build_dir", "~/.ros/drone_mpc", "Where the generated acados solver is kept (reused across runs)."),
    ("mpc_w_radial", 80.0,
     "Weight on the distance to the car vs the shot radius, divided by mpc_band_radial. Higher "
     "than mpc_w_tangential = keep the distance (and the car in DOPE's range) first, let the angle "
     "round the car catch up later. Offline, right shot, car below the drone's top speed: "
     "90/10 held the distance 92 % of the time (30/30: 67 %), never too close (7 %), angle slips "
     "back within 10 deg in a median 3.5 s; behind shot in band 36 vs 39 %."),
    ("mpc_w_tangential", 20.0,
     "Weight on the angle round the car (as the chord r x |unit(drone - car) - unit(shot - car)|), "
     "divided by radius x mpc_band_angle_deg. 80/20 since 2026-10-01 (offline sweep: within noise "
     "of 90/10 on band time, angle held slightly better at 11 m/s)."),
    ("mpc_band_radial", 5.0, "Radial band half-width: distance to the car within the shot radius +- this (m)."),
    ("mpc_band_angle_deg", 10.0,
     "Angular band half-width around the car (deg); also the yaw error that costs w_yaw."),
    ("mpc_band_tangential_floor", 2.0,
     "Tangential band never narrower than this (m), so the angle term fades out for overhead shots."),
    ("mpc_w_height", 5.0, "Weight on the height error (per m)."),
    ("mpc_w_yaw", 10.0, "Weight on facing the car (a yaw error of mpc_band_angle_deg costs this)."),
    ("mpc_r_velocity_change", 1.0,
     "Weight on horizontal command changes (per m/s^2). Was 0.1: with a parked car the command still "
     "changed at 3.5-4 m/s^2 (chasing detection noise, A1 runs); 1.0 cut that ~40 % offline with the "
     "same or slightly better tracking."),
    ("mpc_r_vertical_change", 1.0, "Weight on vertical command changes."),
    ("mpc_r_yaw_rate_change", 10.0, "Weight on yaw-rate command changes (per rad/s^2); was 2."),
    ("mpc_terminal_factor", 2.0, "Terminal-stage multiplier on the tracking weights."),
    ("mpc_sigma0", 5.0,
     "Position weight at stage k = 1 / (1 + (sigma_k / sigma0)^2), sigma_k the predicted car "
     "position std: far, uncertain stages only loosely steer the plan (m)."),
    ("mpc_min_range", 10.0, "Soft minimum 3D range to the car (m): DOPE needs the whole car in view."),
    ("mpc_scenario_window_sec", 1.0,
     "Over this much of the plan (s, after the delay) the >= mpc_min_range constraint also holds "
     "against a car that brakes hard / accelerates from now: the filter sees braking ~1 s late. "
     "< 0 switches the scenarios off."),
    ("mpc_car_brake_decel", 4.0, "Braking scenario deceleration (m/s^2); race_driver brakes at 4."),
    ("mpc_car_accel", 2.0, "Accelerating scenario (m/s^2, up to kf_max_target_speed); race_driver: 2."),
    ("mpc_yaw_rate_max", 1.5, "Yaw-rate command limit (rad/s)."),
    ("mpc_yaw_accel_max", 3.0, "Yaw-rate command change limit (rad/s^2)."),
    ("mpc_delay_xy_sec", 0.8, "Command -> response transport delay, horizontal (fitted on the race logs)."),
    ("mpc_delay_z_sec", 0.7, "Transport delay, vertical (s)."),
    ("mpc_delay_yaw_sec", 0.25, "Transport delay, yaw rate (s)."),
    ("mpc_tau_xy_sec", 0.45, "First-order lag of the horizontal velocity after the delay (s)."),
    ("mpc_tau_z_sec", 0.3, "Lag of the vertical velocity (s)."),
    ("mpc_tau_yaw_sec", 0.3, "Lag of the yaw rate (s)."),
    ("mpc_turn_fade_sec", 1.5,
     "Car over the MPC horizon keeps turning, its turn rate fading with this time constant (s); "
     "0 = straight line. The car position alone was a bit more robust straight (filter "
     "benchmark), but the shot point swings round the car in a turn and the MPC has to see it: "
     "closed loop the 'behind' shot went 31 -> 40 % in band (Silverstone), 16 -> 27 % "
     "(Oschersleben), yaw error halved; the 'right' shot 16 -> 13 m median error."),
    ("mpc_min_altitude", 2.0,
     "Soft altitude floor above shot_ground_z (m) for the MPC and for the height held while "
     "coasting. Run 20260929_141000 had no floor: the MPC traded height for (3D) range and "
     "flew into the ground."),
    ("mpc_max_range", 45.0,
     "Soft maximum HORIZONTAL range to the car (m; 0 = off). When the shot point runs away faster than the "
     "drone can fly (18 m right of a car in a long left turn needs ~2x its speed) the MPC gives "
     "up shot accuracy to keep the car detectable (DOPE to ~80 m) and cuts inside. Run "
     "20260929_124410 lost the car at 80 m chasing a right-side shot round a left turn."),
    ("mpc_max_range_margin", 15.0,
     "The maximum range follows the shot: max(mpc_max_range, shot radius + this margin) (m), "
     "so a 45 m shot gets 60 m instead of being pulled in to 45 m."),

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
    # After target_timeout_sec it searches (or hovers, enable_search_mode false).
    ("coast_full_speed_sec", 3.0,
     "Predict the lost target on at full speed for this long after the last detection (s): long "
     "enough to pick the car up again after a tree or a building hid it."),
    ("coast_taper_sec", 1.0,
     "Then slow the prediction to a stop over this long (s); it stays there until target_timeout_sec."),
    ("coast_max_distance", 40.0,
     "Cap on how far the prediction moves the target from its last estimate (m)."),
    ("coast_turn_fade_sec", 1.5,
     "CTRA filter: while coasting the predicted car keeps turning, its turn rate fading with this "
     "time constant (s); 0 = straight line. (The old coast froze the heading: in run "
     "20260929_094556 it stayed at 84 deg while the car went round a hairpin.)"),
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
    ("ctra_jerk_noise", 0.5, "CTRA: white-noise jerk std (m/s^3); only with ctra_estimate_accel."),
    ("ctra_estimate_accel", False,
     "Estimate along-track acceleration (CTRA). Off (CTRV): DOPE shows braking ~1 s late, and an "
     "estimated acceleration carried the straight's +2 m/s^2 into braking zones, pushing the speed "
     "estimate up while the car braked (offline +1.6 m/s vs +0.7 m/s; 10 % worse p90 at 1-2 s)."),
    ("ctra_speed_noise", 2.0, "CTRV: speed random-walk std (m/s^2)."),
    ("ctra_max_reverse_speed", 8.0,
     "Fastest the car can reverse (m/s; the drive plugin's limit). The filter's speed is signed "
     "down to -this; pinned there it must be a forward car with a flipped heading, and is turned "
     "round."),
    ("ctra_heading_std_deg", 4.0, "CTRA: detector heading measurement std (deg); DOPE measured ~3.6."),
    ("ctra_reacquire_sec", 5.0,
     "CTRA: after a detection gap up to this long (s) the EKF predicts through it and gates the "
     "new detection against the prediction (wider after longer gaps) instead of restarting."),
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
    ("heading_max_elevation_deg", 60.0,
     "Detections seen more steeply below than this update the car's position only, not its "
     "heading: from overhead DOPE's heading was 15-27 deg off (once flipped) and swung the shot "
     "frame 3-7 m sideways in the A1 overpasses (deg; 90 = always use the heading)."),
    ("steep_skip_elevation_deg", 70.0,
     "Detections seen more steeply than this only steer the gimbal (and keep the target seen); the "
     "estimate predicts through them: DOPE's position was ~3 m off above 80 deg (deg; 90 = off)."),
    ("steep_pos_noise_scale", 3.0,
     "Position std multiplier for detections seen more steeply than heading_max_elevation_deg: the "
     "A1 overpass position estimates were ~3 m off above 80 deg (0.7 m in normal views)."),
    ("gimbal_nadir_band_deg", 25.0,
     "Within this of straight down the gimbal yaw is held and pitch alone follows the car: yaw only "
     "spins the image there, and the image loop wound it up +-170 deg in the A1 overpasses (deg)."),
    ("gimbal_max_yaw_deg", 160.0, "Gimbal yaw command limit (deg, +- about the nose); mount limit 160 (was 90)."),
    ("yaw_hold_radius", 5.0,
     "Hold the drone's body yaw while it is within this horizontal distance (m) of the target: "
     "overhead the bearing to the car flips 180 deg, which spun the drone (and the camera with it)."),
    ("gimbal_detection_timeout_sec", 0.5,
     "A detection older than this no longer steers the gimbal; it then points at the target "
     "estimate instead of re-applying the old image error (which made it drift to its limit)."),
    ("coast_scan_enable", True,
     "While the car is hidden (coasting), sweep the gimbal yaw around its prediction once the "
     "prediction is too uncertain for the camera's view."),
    ("coast_scan_sigma", 2.5,
     "Sweep to cover this many position std of the prediction. Offline (closed loop, FOV + "
     "gimbal modelled) the car came back up to 30-72 deg off the aim after 3-4.5 s blackouts; "
     "1.5 sigma at 40 deg/s did not help, 2.5 sigma at 90 deg/s cut losses 7 -> 2 (Oschersleben)."),
    ("coast_scan_half_fov_deg", 23.0, "Camera half field of view, horizontal (deg; 0.8 rad HFOV)."),
    ("coast_scan_max_deg", 50.0, "Largest sweep half-width (deg)."),
    ("coast_scan_rate_deg", 90.0,
     "Gimbal slew between scan views (stare) / sweep speed (sweep), deg/s. Run 20260929_142724 "
     "showed the gimbal moving 100-190 deg/s, so 90 is within reach."),
    ("coast_scan_mode", "stare",
     "stare: stop-and-stare -- hold the prediction, then +-step, +-2 step ... each for "
     "coast_scan_dwell_sec. sweep: continuous back-and-forth. Lost, DOPE tries one of its 9 input "
     "scales per frame, so a view must be held ~0.6 s for the right one to come round; the sweep "
     "of run 20260929_142724 crossed the car at ~176 deg/s (3-4 frames) without a detection."),
    ("coast_scan_step_deg", 30.0, "Stare: angle between scan views (deg; the camera sees 46 deg)."),
    ("coast_scan_dwell_sec", 0.7,
     "Stare: hold each view this long (s): >= DOPE's scale cycle, 9 frames at 15 Hz = 0.6 s."),
    ("gimbal_initial_pitch_deg", -5.0,
     "Gimbal pitch at boot; keep equal to MNT1_NEUTRAL_Y in config/gimbal_startup.parm (deg)."),

    # ---- Search (gimbal sweep while no target) ----
    ("enable_search_mode", True,
     "Once the target is lost for target_timeout_sec: climb to search_altitude where the drone "
     "is and sweep the gimbal around the last estimate. False: hover in place."),
    ("search_altitude", 20.0,
     "Search height above shot_ground_z (m): a wider view of the ground, and DOPE still "
     "detects the car out to ~45 m."),
    ("search_timeout_sec", 10.0,
     "Give up after searching this long (s) and end the flight with search_timeout_action; "
     "0 = search until stopped."),
    ("search_timeout_action", "LAND",
     "ArduPilot mode when the search times out: LAND (land where it is) or RTL (fly home, land)."),
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

    # ---- Experiments (scripts/run_experiments.py) ----
    ("shot_sequence", "",
     "Shot sequence as a JSON list of actions (cinematic_action_schema.py); empty = "
     "DEFAULT_SHOT_SEQUENCE in relative_position_controller.py."),
    ("run_name", "",
     "CSV log name: <run_name>.csv; empty = relative_pid_<timestamp>.csv."),
    ("experiment_duration_sec", 0.0,
     "End the flight (LAND) this long after the shot sequence started (s, sim time); 0 = off."),
    ("shot_start_speed", 0.0,
     "Hold the shot sequence at its start until the car's ground-truth speed exceeds this (m/s), "
     "e.g. to start an overpass/orbit when a car launches; 0 = start with tracking."),

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
