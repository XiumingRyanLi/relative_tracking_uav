#!/usr/bin/env python3
"""
DOPE (Deep Object Pose Estimation) detector node.

Drop-in replacement for aruco_detector: consumes the bridged Gazebo camera
image and publishes the target pose in exactly the same format the
relative_position_controller already consumes from the ArUco node:

    /aruco_target/found        std_msgs/Bool        every processed frame
    /aruco_target/visual_odom  nav_msgs/Odometry    on a hit only

Odometry conventions (identical to aruco_detector.aruco_target_odometry):
  - header.stamp     = capture stamp of the image the pose came from
  - header.frame_id  = camera_frame (OpenCV optical: x right, y down, z fwd)
  - child_frame_id   = target_child_frame (default "aruco_target_link")
  - pose             = target in camera optical frame, metres
  - twist            = zero (controller estimates velocity itself)

Additionally publishes the straight-line camera->target distance on
/dope_target/distance (std_msgs/Float32) for logging/plotting.

The network is NVIDIA DOPE (VGG19 backbone, 6 belief/affinity stages). The
.pth checkpoint is a plain state_dict; weights trained with DDP carry a
"module." prefix on every key, which is stripped automatically on load.
Metric scale comes entirely from the cuboid dimensions parameter -- these
MUST match the object the network was trained on (same role as tag size in
the ArUco node).
"""
import math
import os
import time
from types import SimpleNamespace
from collections import OrderedDict

import cv2
import numpy as np
if not hasattr(np, "float"):
    np.float = float
import torch
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry

from .dope import (
    Cuboid3d,
    CuboidLineIndexes,
    CuboidPNPSolver,
    DopeNetwork,
    ObjectDetector,
)

CM_TO_M = 0.01


class DopeDetectorNode(Node):
    def __init__(self):
        super().__init__("dope_detection_node")

        # ---- MODEL PARAMETERS ----
        self.declare_parameter(
            "weights_path",
            os.path.join(
                os.environ.get("HOME", ""),
                "Desktop/ryan/Deep_Object_Pose/train/output/weights_droneview_v2/net_epoch_0725.pth",
            ),
        )
        self.declare_parameter("object_name", "Audi")
        # Cuboid dimensions in cm (x=width, y=height, z=depth/length), from
        # Deep_Object_Pose/config/config_pose.yaml "dimensions" for the
        # trained object. Sets the metric scale of the PnP solution.
        self.declare_parameter("cuboid_dimensions_cm", [203.82, 123.95, 441.46])
        # DOPE detection thresholds (config_pose.yaml defaults)
        self.declare_parameter("thresh_angle", 0.5)
        self.declare_parameter("thresh_map", 0.01)
        self.declare_parameter("sigma", 3)
        self.declare_parameter("thresh_points", 0.1)
        # ---- ADAPTIVE INPUT SCALE ----
        # DOPE is NOT scale invariant: this network was trained with the
        # car 28-32 m away, i.e. spanning ~70-110 px of the input. Measured
        # on sim frames, it only detects when the car is rendered at about
        # that size, so instead of a fixed downscale the node resizes the
        # frame so the target's expected length is object_px_target px.
        # While tracking, the scale follows the last measured distance;
        # when lost, it sweeps scale_pyramid one entry per frame.
        # Scales are fractions of the full imgsz frame.
        self.declare_parameter("scale_pyramid", [0.16, 0.2, 0.24, 0.28, 0.34, 0.4, 0.48, 0.556, 0.7])
        self.declare_parameter("object_px_target", 90.0)
        self.declare_parameter("track_timeout_sec", 1.0)
        # ---- TRACK ROBUSTNESS ----
        # On a miss while tracking, immediately retry the next candidate
        # scale within the same frame (up to this many inferences/frame).
        self.declare_parameter("max_tries_per_frame", 2)
        # While tracking, accept cuboids with some corners missing (stock
        # DOPE requires all 8) if they agree with the current track:
        # distance within track_gate_ratio of the last one and image centre
        # within track_gate_frac (fraction of frame width) of the last one.
        self.declare_parameter("min_corners_tracking", 4)
        # Minimum corners for a NEW detection (not tracking). Measured on sim
        # frames: 6- and 7-corner cuboids are as accurate as full ones when
        # their reprojection error is low, so they are accepted subject to
        # max_reproj_frac below.
        self.declare_parameter("min_corners_acquire", 6)
        # Reject any candidate (even 8/8) whose mean reprojection error of
        # the detected 2D points against the PnP solution exceeds this
        # fraction of the projected cuboid width. Measured: correct poses
        # are <= 3%, wrong ones >= 4.5%.
        self.declare_parameter("max_reproj_frac", 0.04)
        self.declare_parameter("track_gate_ratio", 0.25)
        self.declare_parameter("track_gate_frac", 0.15)
        # Keep /aruco_target/found true for this long after the last hit so
        # a single dropped frame doesn't flicker the flag. No odom is
        # published on such frames -- only the flag is held.
        self.declare_parameter("found_hold_sec", 0.5)
        # Views within this many degrees of head-on / tail-on are geometric-
        # ally degenerate for a cuboid (front and rear faces overlap), and
        # the PnP range is unreliable there (measured: 8 m rear view -> 13 m).
        # Such detections are still published but flagged in log + overlay.
        self.declare_parameter("degenerate_view_deg", 20.0)
        # ---- OPTIONAL GROUND TRUTH (sim only) ----
        # Straight-line drone->car distance from the Gazebo car odometry
        # (bridged by sim_dope_launch) and the MAVROS local pose, printed
        # next to the DOPE range. Assumes the Gazebo world origin and the
        # MAVROS local origin coincide (true when the drone spawns at the
        # world origin). Set either topic to "" to disable.
        self.declare_parameter("truth_target_odom_topic", "/landing_vehicle/odometry")
        self.declare_parameter("truth_drone_pose_topic", "/mavros/local_position/pose")

        self.weights_path = os.path.expanduser(self.get_parameter("weights_path").value)
        self.object_name = self.get_parameter("object_name").value
        self.cuboid_dimensions_cm = [float(v) for v in self.get_parameter("cuboid_dimensions_cm").value]
        self.scale_pyramid = sorted(float(v) for v in self.get_parameter("scale_pyramid").value)
        self.object_px_target = float(self.get_parameter("object_px_target").value)
        self.track_timeout_sec = float(self.get_parameter("track_timeout_sec").value)
        self.max_tries_per_frame = max(1, int(self.get_parameter("max_tries_per_frame").value))
        self.min_corners_tracking = int(self.get_parameter("min_corners_tracking").value)
        self.min_corners_acquire = int(self.get_parameter("min_corners_acquire").value)
        self.max_reproj_frac = float(self.get_parameter("max_reproj_frac").value)
        self.track_gate_ratio = float(self.get_parameter("track_gate_ratio").value)
        self.track_gate_frac = float(self.get_parameter("track_gate_frac").value)
        self.found_hold_sec = float(self.get_parameter("found_hold_sec").value)
        self.degenerate_view_deg = float(self.get_parameter("degenerate_view_deg").value)
        self._truth_car = None
        self._truth_drone = None
        self._last_centre_frac = None   # last detection's image centre, fraction of frame
        self._last_orientation_text = ""
        self.object_length_m = max(self.cuboid_dimensions_cm) * CM_TO_M
        self._last_detection_time = None
        self._last_distance_m = None
        self._last_ok_scale = None      # scale of the most recent detection
        self._track_miss = 0            # consecutive misses while tracking
        self._sweep_idx = 0
        self._current_scale = self.scale_pyramid[-1]

        self.config_detect = SimpleNamespace(
            mask_edges=1,
            mask_faces=1,
            vertex=1,
            threshold=0.5,
            softmax=1000,
            thresh_angle=float(self.get_parameter("thresh_angle").value),
            thresh_map=float(self.get_parameter("thresh_map").value),
            sigma=int(self.get_parameter("sigma").value),
            thresh_points=float(self.get_parameter("thresh_points").value),
        )

        # ---- CAMERA PARAMETERS (same as aruco_detector) ----
        self.declare_parameter("image_topic", "/camera/image_raw")
        # "best_available" (default), "reliable" or "best_effort".
        # best_available adopts whatever reliability the discovered image
        # publisher offers, so it can never be QoS-incompatible with the
        # Gazebo bridge or a webcam driver. Measured on this machine
        # (Fast DDS, no SHM segments, 212 KB default UDP receive buffer):
        # a 750 KB best-effort image loses UDP fragments between processes
        # and NO sample ever arrives, while reliable delivers every frame.
        # If the publisher is reliable, best_available becomes reliable
        # too. With KEEP_LAST depth=1 there is still no backlog -- only
        # fragment retransmission changes.
        self.declare_parameter("image_reliability", "best_available")
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 480)
        self.declare_parameter("camera_fov_horizontal", 0.87)  # radians, MUST match SDF
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("processing_rate", 20.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.image_reliability = str(self.get_parameter("image_reliability").value).lower()
        self._image_width = int(self.get_parameter("imgsz_width").value)
        self._image_height = int(self.get_parameter("imgsz_height").value)
        self._camera_fov_horizontal = float(self.get_parameter("camera_fov_horizontal").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self._processing_rate = float(self.get_parameter("processing_rate").value)

        self._camera_fov_vertical = 2 * np.arctan(
            np.tan(self._camera_fov_horizontal / 2) / (self._image_width / self._image_height)
        )
        fx = self._image_width / (2 * np.tan(self._camera_fov_horizontal / 2))
        fy = self._image_height / (2 * np.tan(self._camera_fov_vertical / 2))
        self._camera_matrix = np.array([
            [fx, 0, self._image_width / 2],
            [0, fy, self._image_height / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # ---- OUTPUT PARAMETERS ----
        # Defaults keep the controller unchanged: it subscribes to the
        # /aruco_target/* topics regardless of which detector produced them.
        self.declare_parameter("found_topic", "/aruco_target/found")
        self.declare_parameter("odom_topic", "/aruco_target/visual_odom")
        self.declare_parameter("distance_topic", "/dope_target/distance")
        self.declare_parameter("target_child_frame", "aruco_target_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("enable_debug_publish", True)
        self.declare_parameter("show_debug_window", True)
        # The network input changes size every time the adaptive scale
        # changes; the debug view is always rendered at this fixed width
        # so the window doesn't resize.
        self.declare_parameter("debug_display_width", 960)

        found_topic = self.get_parameter("found_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        distance_topic = self.get_parameter("distance_topic").value
        self.target_child_frame = self.get_parameter("target_child_frame").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.enable_debug_publish = bool(self.get_parameter("enable_debug_publish").value)
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.debug_display_width = int(self.get_parameter("debug_display_width").value)

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._found_publisher = self.create_publisher(Bool, found_topic, 10)
        self._odom_publisher = self.create_publisher(Odometry, odom_topic, 10)
        self._distance_publisher = self.create_publisher(Float32, distance_topic, 10)
        self._debug_image_publisher = self.create_publisher(Image, "/dope_detector/image", 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- NETWORK ----
        if not torch.cuda.is_available():
            # The vendored DOPE detector hard-codes .cuda() on the input tensor.
            raise RuntimeError("DOPE detector requires a CUDA-capable GPU (torch.cuda.is_available() is False)")
        self.net = self._load_network(self.weights_path)
        self.pnp_solver = CuboidPNPSolver(
            self.object_name,
            cuboid3d=Cuboid3d(self.cuboid_dimensions_cm),
            dist_coeffs=self._dist_coeffs,
        )
        self._warm_up()

        if self.show_debug_window:
            cv2.namedWindow("DOPE Detection", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE SOURCE ----
        # Same latest-frame-only scheme as aruco_detector: a depth=1
        # subscription stores the newest frame; a timer at processing_rate
        # consumes it. Inference never works through a backlog of stale
        # frames. Reliability is a parameter (see image_reliability above).
        self._img_msg = None
        self._last_frame_time = None
        self._fps = 0.0
        img_qos = QoSProfile(
            reliability={
                "best_effort": ReliabilityPolicy.BEST_EFFORT,
                "reliable": ReliabilityPolicy.RELIABLE,
            }.get(self.image_reliability, ReliabilityPolicy.BEST_AVAILABLE),
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_subscription = self.create_subscription(
            Image, self.image_topic, self._image_store_callback, img_qos
        )
        self._process_timer = self.create_timer(
            1.0 / self._processing_rate, self._process_timer_callback
        )

        truth_odom = self.get_parameter("truth_target_odom_topic").value
        truth_pose = self.get_parameter("truth_drone_pose_topic").value
        if truth_odom and truth_pose:
            self.create_subscription(Odometry, truth_odom, self._on_truth_car, 10)
            self.create_subscription(PoseStamped, truth_pose, self._on_truth_drone, 10)

        self.get_logger().info(
            f"DOPE detector ready: object={self.object_name} dims_cm={self.cuboid_dimensions_cm} "
            f"image={self.image_topic} [{self.image_reliability}] ({self._image_width}x{self._image_height}, "
            f"hfov={self._camera_fov_horizontal:.3f} rad) -> {odom_topic} @ {self._processing_rate:.0f} Hz; "
            f"scale pyramid {self.scale_pyramid}, target {self.object_px_target:.0f} px"
        )

    # ---- MODEL LOADING ----
    def _load_network(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DOPE weights not found: {path}")
        t0 = time.time()
        state_dict = torch.load(path, map_location="cpu")
        # DDP-trained checkpoints prefix every key with "module."
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = OrderedDict(
                (k[len("module."):] if k.startswith("module.") else k, v)
                for k, v in state_dict.items()
            )
        net = DopeNetwork()
        net.load_state_dict(state_dict)
        net = net.cuda().eval()
        self.get_logger().info(
            f"Loaded DOPE weights {path} ({len(state_dict)} tensors) in {time.time() - t0:.1f}s "
            f"on {torch.cuda.get_device_name(0)}"
        )
        return net

    def _warm_up(self):
        """Run one dummy inference so the first real frame doesn't pay for
        cuDNN autotuning / lazy CUDA initialisation."""
        h, w = self._network_input_size(self.scale_pyramid[-1])
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.pnp_solver.set_camera_intrinsic_matrix(self._scaled_camera_matrix(h))
        t0 = time.time()
        with torch.inference_mode():
            ObjectDetector.detect_object_in_image(self.net, self.pnp_solver, dummy, self.config_detect)
        torch.cuda.synchronize()
        self.get_logger().info(f"Warm-up inference at {w}x{h}: {(time.time() - t0) * 1000:.0f} ms")

    def _network_input_size(self, scale):
        scale = min(1.0, scale)
        return int(round(self._image_height * scale)), int(round(self._image_width * scale))

    def _choose_scale(self, now):
        """Pick the input scale for this frame (see ADAPTIVE INPUT SCALE)."""
        tracking = (
            self._last_detection_time is not None
            and self._last_distance_m is not None
            and (now - self._last_detection_time) < self.track_timeout_sec
        )
        if tracking:
            # The scale that last worked is the best bet. The scale implied
            # by the measured distance says which way to step if the object
            # is drifting out of the trained size band; the two are only
            # used on consecutive misses, so a working lock is never
            # abandoned for a computed scale that happens to fail.
            fx_full = self._camera_matrix[0, 0]
            wanted = self.object_px_target * self._last_distance_m / (fx_full * self.object_length_m)
            wanted = min(self.scale_pyramid, key=lambda v: abs(v - wanted))
            i_ok = self.scale_pyramid.index(self._last_ok_scale)
            step = 1 if wanted > self._last_ok_scale else -1
            candidates = [self._last_ok_scale]
            if wanted != self._last_ok_scale:
                candidates.append(wanted)
            for j in (i_ok + step, i_ok - step):
                if 0 <= j < len(self.scale_pyramid) and self.scale_pyramid[j] not in candidates:
                    candidates.append(self.scale_pyramid[j])
            scale = candidates[self._track_miss % len(candidates)]
        else:
            self._sweep_idx = (self._sweep_idx + 1) % len(self.scale_pyramid)
            scale = self.scale_pyramid[self._sweep_idx]
        self._current_scale = scale
        return scale, tracking

    def _scaled_camera_matrix(self, input_height):
        scale = input_height / self._image_height
        K = self._camera_matrix.copy()
        K[:2] *= scale
        return K

    # ---- CALLBACKS ----
    def _image_store_callback(self, msg):
        self._img_msg = msg

    def _process_timer_callback(self):
        msg = self._img_msg
        if msg is None:
            return
        self._img_msg = None
        try:
            self.process_frame(msg)
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"DOPE frame processing failed: {e}")

    def process_frame(self, msg):
        # Use the image's own capture stamp (see aruco_detector for why).
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            stamp = msg.header.stamp
        else:
            stamp = self.get_clock().now().to_msg()

        # DOPE is trained on RGB (ImageNet normalisation), not BGR.
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        # Match the intrinsics resolution, then apply DOPE's downscale.
        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(frame, (self._image_width, self._image_height), interpolation=cv2.INTER_LINEAR)
        base = frame
        now = time.monotonic()
        scale, tracking = self._choose_scale(now)
        infer_ms = 0.0
        tries = 0
        results = []
        while True:
            in_h, in_w = self._network_input_size(scale)
            if (in_h, in_w) != base.shape[:2]:
                frame = cv2.resize(base, (in_w, in_h), interpolation=cv2.INTER_AREA)
            else:
                frame = base
            self.pnp_solver.set_camera_intrinsic_matrix(self._scaled_camera_matrix(in_h))

            t0 = time.perf_counter()
            with torch.inference_mode():
                raw, _ = ObjectDetector.detect_object_in_image(
                    self.net, self.pnp_solver, frame, self.config_detect
                )
            infer_ms += (time.perf_counter() - t0) * 1000.0
            tries += 1

            results = self._filter_results(raw, tracking, in_w, in_h)
            if results or not tracking or tries >= self.max_tries_per_frame:
                break
            # Miss while tracking: retry the next candidate scale right now.
            self._track_miss += 1
            scale, tracking = self._choose_scale(now)

        found = len(results) > 0
        coasting = (
            not found
            and self._last_detection_time is not None
            and (now - self._last_detection_time) < self.found_hold_sec
        )
        self._found_publisher.publish(Bool(data=found or coasting))

        best = None
        if found:
            # Several cuboids can be assembled from one belief map set;
            # take the nearest one (largest in the image, most reliable).
            best = min(results, key=lambda r: float(np.linalg.norm(r["location"])))
            t_cam_target = np.array(best["location"], dtype=float) * CM_TO_M
            q_cam_target = [float(v) for v in best["quaternion"]]  # xyzw
            distance = float(np.linalg.norm(t_cam_target))
            self._last_detection_time = time.monotonic()
            self._last_distance_m = distance
            self._last_ok_scale = scale
            self._track_miss = 0
            pts = np.asarray(best["projected_points"], dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 9:
                self._last_centre_frac = (float(pts[8, 0]) / in_w, float(pts[8, 1]) / in_h)

            ori = self._describe_orientation(q_cam_target)
            self._last_orientation_text = ori["text"]
            truth_txt = self._truth_text()

            odom = self._target_odometry(stamp, t_cam_target, q_cam_target)
            self._odom_publisher.publish(odom)
            self._distance_publisher.publish(Float32(data=distance))
            if self.publish_tf:
                self._tf_broadcaster.sendTransform(self._target_transform(stamp, t_cam_target, q_cam_target))

            self.get_logger().info(
                f"DOPE {self.object_name}: distance={distance:.2f} m "
                f"camera_position=({t_cam_target[0]:.2f}, {t_cam_target[1]:.2f}, {t_cam_target[2]:.2f}) "
                f"{truth_txt}"
                f"orientation {ori['text']} "
                f"candidates={len(results)} corners={int(best.get('n_corners', 8))}/8 "
                f"reproj={100*best.get('reproj_frac', 0):.1f}% "
                f"scale={scale:.2f} ({in_w}x{in_h}) tries={tries} inference={infer_ms:.1f} ms",
                throttle_duration_sec=0.5,
            )
            if ori["degenerate"]:
                self.get_logger().warning(
                    f"Near head-on/tail-on view (nose {ori['nose_deg']:+.0f} deg): "
                    f"DOPE range {distance:.1f} m is unreliable from this angle.",
                    throttle_duration_sec=2.0,
                )
        else:
            self._track_miss += 1
            self.get_logger().info(
                f"DOPE {self.object_name}: no detection ({'tracking' if tracking else 'sweeping'} "
                f"scale={scale:.2f} {in_w}x{in_h}, tries={tries}, inference={infer_ms:.1f} ms"
                f"{', coasting' if coasting else ''})",
                throttle_duration_sec=2.0,
            )

        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_frame_time = now

        if self.show_debug_window or self.enable_debug_publish:
            self._debug_output(frame, results, best, infer_ms)

    @staticmethod
    def _reproj_frac(r):
        """Mean distance between the detected 2D points and the PnP
        reprojection, as a fraction of the projected cuboid width."""
        pts = np.asarray(r["projected_points"], dtype=float)
        raw = r.get("raw_points")
        if pts.ndim != 2 or pts.shape[0] < 9 or raw is None:
            return None
        errs = [
            float(np.hypot(*(np.asarray(raw[i], dtype=float) - pts[i])))
            for i in range(9) if raw[i] is not None
        ]
        width = float(np.ptp(pts[:8, 0]))
        if not errs or width < 1.0:
            return None
        return float(np.mean(errs)) / width

    def _filter_results(self, raw, tracking, in_w, in_h):
        """Keep candidates that are geometrically self-consistent (low
        reprojection error) and have enough corners: min_corners_acquire
        for a new detection, min_corners_tracking while tracking, where
        partial cuboids must also agree with the current track (distance
        within track_gate_ratio, image centre within track_gate_frac)."""
        out = []
        for r in raw:
            if r["location"] is None or r["quaternion"] is None:
                continue
            frac = self._reproj_frac(r)
            if frac is None or frac > self.max_reproj_frac:
                continue
            r["reproj_frac"] = frac
            n = int(r.get("n_corners", 8))
            if n >= self.min_corners_acquire:
                out.append(r)
                continue
            if (
                not tracking
                or n < self.min_corners_tracking
                or self._last_distance_m is None
                or self._last_centre_frac is None
            ):
                continue
            dist = float(np.linalg.norm(r["location"])) * CM_TO_M
            if abs(dist - self._last_distance_m) > self.track_gate_ratio * self._last_distance_m:
                continue
            pts = np.asarray(r["projected_points"], dtype=float)
            cx, cy = float(pts[8, 0]) / in_w, float(pts[8, 1]) / in_h
            if math.hypot(cx - self._last_centre_frac[0], cy - self._last_centre_frac[1]) > self.track_gate_frac:
                continue
            out.append(r)
        return out

    def _draw_body_axes(self, vis, r, kx, ky, axis_len_cm=150.0):
        """Draw the object's body axes at the cuboid centre, projected with
        the current (network-input) intrinsics and scaled to the display.
        Object frame (training mesh): x = width, y = up, z = nose.
        Roll is about the nose axis, pitch about the width axis, yaw
        about the up axis."""
        try:
            q = [float(v) for v in r["quaternion"]]
            R = tf_transformations.quaternion_matrix(q)[:3, :3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = np.asarray(r["location"], dtype=np.float64).reshape(3, 1)
            K = np.asarray(self.pnp_solver._camera_intrinsic_matrix, dtype=np.float64)
            # Object frame = training mesh frame: +x width, +y up, +z nose.
            pts3 = np.array([[0, 0, 0], [axis_len_cm, 0, 0], [0, axis_len_cm, 0], [0, 0, axis_len_cm]], dtype=np.float64)
            pts2, _ = cv2.projectPoints(pts3, rvec, tvec, K, np.zeros((4, 1)))
            pts2 = pts2.reshape(-1, 2) * np.array([kx, ky])
            o = tuple(int(round(v)) for v in pts2[0])
            axes = [
                (pts2[1], (0, 0, 255),   "x width (pitch axis)"),
                (pts2[2], (0, 255, 0),   "y up (yaw axis)"),
                (pts2[3], (255, 128, 0), "z nose (roll axis)"),
            ]
            # Arrows only; the colour key is in the status line under the banner.
            for end, colour, _label in axes:
                e = tuple(int(round(v)) for v in end)
                cv2.arrowedLine(vis, o, e, colour, 2, cv2.LINE_AA, tipLength=0.15)
        except Exception as e:  # never let a drawing problem kill the frame
            self.get_logger().debug(f"axis overlay failed: {e}")

    # ---- ORIENTATION / TRUTH HELPERS ----
    def _describe_orientation(self, q_xyzw):
        """Human-readable target orientation in the camera optical frame.

        DOPE's object frame is the training mesh frame (Audi_R8 OBJ):
        x = width, y = height, z = length, with the car's nose along +z
        (verified on sim frames with known car headings). 'nose' is the
        heading of the nose vector in the camera's horizontal (x-z) plane:
        0 = pointing away from the camera, +/-180 = pointing at the
        camera, +90 = pointing to the camera's right."""
        x, y, z, w = q_xyzw
        R = tf_transformations.quaternion_matrix([x, y, z, w])[:3, :3]
        nose = R @ np.array([0.0, 0.0, 1.0])
        nose_deg = math.degrees(math.atan2(nose[0], nose[2]))
        roll, pitch, yaw = (math.degrees(a) for a in tf_transformations.euler_from_quaternion([x, y, z, w]))
        toward = abs(nose_deg) > 90.0
        degenerate = min(abs(nose_deg), 180.0 - abs(nose_deg)) < self.degenerate_view_deg
        text = (
            f"q=({x:+.3f},{y:+.3f},{z:+.3f},{w:+.3f}) "
            f"rpy=({roll:+.0f},{pitch:+.0f},{yaw:+.0f})deg "
            f"nose={nose_deg:+.0f}deg({'toward' if toward else 'away'}"
            f"{', DEGENERATE' if degenerate else ''})"
        )
        return {"text": text, "nose_deg": nose_deg, "rpy": (roll, pitch, yaw), "degenerate": degenerate}

    def _on_truth_car(self, msg):
        p = msg.pose.pose.position
        self._truth_car = np.array([p.x, p.y, p.z])

    def _on_truth_drone(self, msg):
        p = msg.pose.position
        self._truth_drone = np.array([p.x, p.y, p.z])

    def _truth_text(self):
        if self._truth_car is None or self._truth_drone is None:
            return ""
        return f"truth~{float(np.linalg.norm(self._truth_car - self._truth_drone)):.1f} m (drone-car, approx) "

    # ---- MESSAGE BUILDERS ----
    def _target_odometry(self, stamp, t, q):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.camera_frame
        odom.child_frame_id = self.target_child_frame
        odom.pose.pose.position.x = float(t[0])
        odom.pose.pose.position.y = float(t[1])
        odom.pose.pose.position.z = float(t[2])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        # Velocity is not estimated by the detector.
        return odom

    def _target_transform(self, stamp, t, q):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.camera_frame
        tf_msg.child_frame_id = self.target_child_frame
        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        return tf_msg

    # ---- DEBUG ----
    def _draw_cuboid(self, vis, pts, colour, thickness):
        for i, j in CuboidLineIndexes:
            p1 = tuple(int(round(v)) for v in pts[i])
            p2 = tuple(int(round(v)) for v in pts[j])
            cv2.line(vis, p1, p2, colour, thickness)
        # Front face (first four edges) drawn heavier so the car's facing
        # direction is visible.
        for i, j in CuboidLineIndexes[:4]:
            p1 = tuple(int(round(v)) for v in pts[i])
            p2 = tuple(int(round(v)) for v in pts[j])
            cv2.line(vis, p1, p2, colour, thickness + 1)
        c = tuple(int(round(v)) for v in pts[8])
        cv2.circle(vis, c, 4, colour, -1)
        return c

    def _debug_output(self, frame_rgb, results, best, infer_ms):
        """Overlay every DOPE candidate on the frame. The selected (nearest)
        candidate is green and fully labelled; the others are orange with a
        short label. Shown in the 'DOPE Detection' window and published on
        /dope_detector/image."""
        in_h, in_w = frame_rgb.shape[:2]
        # Fixed canvas: width from the parameter, height from the camera's
        # aspect ratio (not the network input's, whose rounding differs per
        # scale) so the window never changes size.
        disp_w = self.debug_display_width
        disp_h = int(round(disp_w * self._image_height / self._image_width))
        kx, ky = disp_w / in_w, disp_h / in_h   # network-input px -> display px
        vis = cv2.resize(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (disp_w, disp_h),
            interpolation=cv2.INTER_LINEAR if kx > 1 else cv2.INTER_AREA,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        GREEN, ORANGE, RED, WHITE = (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255)

        for idx, r in enumerate(results):
            pts = np.asarray(r["projected_points"], dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 9:
                continue
            pts = pts * np.array([kx, ky])
            is_best = r is best
            colour = GREEN if is_best else ORANGE
            centre = self._draw_cuboid(vis, pts, colour, 2 if is_best else 1)
            loc_m = np.asarray(r["location"], dtype=float) * CM_TO_M
            dist = float(np.linalg.norm(loc_m))
            n_c = int(r.get("n_corners", 8))
            partial = f"  {n_c}/8" if n_c < 8 else ""
            if is_best:
                label = (
                    f"{self.object_name}  {dist:.1f} m  "
                    f"xyz=({loc_m[0]:+.1f}, {loc_m[1]:+.1f}, {loc_m[2]:+.1f}){partial}"
                )
            else:
                label = f"#{idx} {self.object_name} {dist:.1f} m"
            # Label just above the cuboid's top-most projected corner.
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            x = int(np.clip(pts[:8, 0].min(), 0, max(0, vis.shape[1] - tw - 6)))
            y = int(np.clip(pts[:8, 1].min() - 8, th + 30, vis.shape[0] - 1))
            cv2.rectangle(vis, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (x + 2, y - 2), font, 0.5, colour, 1, cv2.LINE_AA)

        # Body axes of the chosen detection, projected from its pose.
        if best is not None:
            self._draw_body_axes(vis, best, kx, ky)

        # Status banner (top-left)
        since = (time.monotonic() - self._last_detection_time) if self._last_detection_time is not None else None
        if best is not None:
            status = f"TRACKING {self.object_name}: {len(results)} candidate(s)"
            status_colour = GREEN
        elif since is not None and since < self.found_hold_sec:
            status = f"COASTING ({since:.1f} s since last hit)"
            status_colour = ORANGE
        else:
            status = "NO DETECTION"
            status_colour = RED
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(vis, status, (8, 18), font, 0.6, status_colour, 2, cv2.LINE_AA)
        if best is not None:
            ori = self._describe_orientation([float(v) for v in best["quaternion"]])
            r_, p_, y_ = ori["rpy"]
            line = (f"nose {ori['nose_deg']:+.0f} deg ({'toward' if abs(ori['nose_deg']) > 90 else 'away'})"
                    f"   cam-frame rpy {r_:+.0f}/{p_:+.0f}/{y_:+.0f} deg"
                    f"   axes: red=x width(pitch) green=y up(yaw) blue=z nose(roll)"
                    + ("   DEGENERATE VIEW" if ori["degenerate"] else ""))
            truth = self._truth_text()
            if truth:
                line += f"   {truth.strip()}"
            cv2.rectangle(vis, (0, 26), (vis.shape[1], 48), (0, 0, 0), -1)
            cv2.putText(vis, line, (8, 42), font, 0.5, ORANGE if ori["degenerate"] else WHITE, 1, cv2.LINE_AA)

        # Timing (bottom-left)
        mode = "track" if (best is not None or self._last_detection_time is not None
                           and time.monotonic() - self._last_detection_time < self.track_timeout_sec) else "sweep"
        timing = (f"inference {infer_ms:.0f} ms   {self._fps:.1f} fps   "
                  f"input {in_w}x{in_h} (scale {self._current_scale:.2f}, {mode})")
        cv2.putText(vis, timing, (8, vis.shape[0] - 8), font, 0.5, WHITE, 1, cv2.LINE_AA)

        if self.show_debug_window:
            cv2.imshow("DOPE Detection", vis)
            cv2.waitKey(1)
        if self.enable_debug_publish:
            self._debug_image_publisher.publish(self._ndarray_to_imgmsg(vis, "bgr8"))

    def _ndarray_to_imgmsg(self, arr, encoding):
        """Build a sensor_msgs/Image directly. cv_bridge.cv2_to_imgmsg is
        broken against OpenCV 5 (CV type constant mismatch -> KeyError);
        imgmsg_to_cv2 still works, so only the outgoing path is hand-rolled."""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(arr.strides[0])
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    def destroy_node(self):
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DopeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
