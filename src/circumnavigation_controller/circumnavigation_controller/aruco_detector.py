#!/usr/bin/env python3
import os
import sys
import signal
from datetime import datetime

import cv2
import numpy as np
if not hasattr(np, "float"):
    np.float = float
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations
from pupil_apriltags import Detector as AprilTagDetector

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

# Get the workspace root directory
def get_workspace_root():
    """Find the workspace root by looking for colcon workspace structure"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'src')) and \
           os.path.exists(os.path.join(current_dir, 'build')) and \
           os.path.exists(os.path.join(current_dir, 'install')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


class ArUCoNode(Node):
    def __init__(self):
        super().__init__("ArUCo_detection_node")

        # ---- PARAMETERS ----
        # Publish the image topic from the camera after processing
        self.declare_parameter("enable_debug_publish", False)
        self.enable_debug_publish = (
            self.get_parameter("enable_debug_publish").get_parameter_value().bool_value
        )

        # Choose between camera image topic or direct webcam source
        self.declare_parameter("image_source", "topic")
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        # Webcam index number
        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        # SHow the camera viewport
        self.declare_parameter("show_debug_window", True)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        # Save individual frames from vision
        self.declare_parameter("save_frames", False)
        self.save_frames = (
            self.get_parameter("save_frames").get_parameter_value().bool_value
        )

        # Create video of vision throughout running of program
        self.declare_parameter("create_video", False)
        self.create_video = (
            self.get_parameter("create_video").get_parameter_value().bool_value
        )

        # Declare video FPS
        self.declare_parameter("video_fps", 30.0)
        self.video_fps = float(
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        # Decalre video/frames output directory location
        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # Log parameter values for debugging
        self.get_logger().info(f"Video recording parameters: save_frames={self.save_frames}, create_video={self.create_video}, video_fps={self.video_fps}")
        self.get_logger().info(f"Output directory: '{self.output_dir}' (empty means workspace root)")

        # Decouples detection processing from frame arrival rate -- the
        # subscription/webcam capture always stores the latest frame
        # only, and this timer decides how often we actually run
        # detection on whatever's newest. If processing ever takes
        # longer than 1/processing_rate, we just skip straight to the
        # next-newest frame instead of working through a backlog.
        self.declare_parameter("processing_rate", 20.0)
        self._processing_rate = float(
            self.get_parameter("processing_rate").get_parameter_value().double_value
        )

        # ---- CAMERA PARAMETERS ----
        # Must match the IR input size you exported (default IRIS is 640 x 480)
        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(self.get_parameter("imgsz_width").get_parameter_value().integer_value)
        
        self.declare_parameter("imgsz_height", 480)
        self._image_height = int(self.get_parameter("imgsz_height").get_parameter_value().integer_value)

        # 114.6 deg (the old value) is fisheye-territory and makes any
        # tag at a normal shot distance unresolvably small in pixels --
        # see the analysis behind this change: at ~4.2m slant distance
        # (typical SHOT offset) with a 0.335m tag, 114.6 deg gave ~16px
        # across an 8x8-module AprilTag, i.e. ~2px/module -- undecodable
        # by any detector. ~50 deg gets that to a comfortable ~55px
        # (~7px/module) at the same distance while still keeping a
        # reasonably wide working field of view rather than tuning all
        # the way down to the bare-minimum ~28 deg. Re-tune this against
        # your actual planned shot distances/tag sizes -- this is a
        # starting point, not a final calibrated value, and it MUST also
        # match whatever FOV your camera sensor is actually configured
        # with in the SDF, or this camera_matrix will be wrong.
        self._camera_fov_horizontal = 0.87  # radians (≈50°) – tune for your camera
        self._camera_fov_vertical = 2 * np.arctan(np.tan(self._camera_fov_horizontal / 2) / (self._image_width/self._image_height))

        # Generate the camera matrix
        fx = self._image_width / (2 * np.tan(self._camera_fov_horizontal / 2))
        fy = self._image_height / (2 * np.tan(self._camera_fov_vertical / 2))
        self._camera_matrix = np.array([
            [fx, 0, self._image_width/2],
            [0, fy, self._image_height/2 ],
            [0, 0, 1],
        ], dtype=np.float64)

        self._dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

        # Frame used by the visual odometry output. This should match the
        # camera/OpenCV frame assumed by the controller.
        self.declare_parameter("camera_frame", "camera_link")
        self.camera_frame = self.get_parameter("camera_frame").value

        # Reject very small detections before pose estimation. Long-range
        # tiny markers give unstable rvec/yaw even when tvec is usable.
        self.declare_parameter("min_marker_area_px", 50.0)
        self.min_marker_area_px = float(self.get_parameter("min_marker_area_px").value)

        # ---- TAG PARAMETERS ----
        # NOTE: switched from ArUco (DICT_5X5_50) to AprilTag 36h11 for
        # Thursday's test. The IDs/sizes/offsets below are carried over
        # unchanged from the ArUco setup -- update them to match whatever
        # AprilTag 36h11 tags you actually print/place. AprilTag 36h11 ID
        # space goes well beyond 50, so 0/27/35 are still valid IDs, but
        # they must be the *AprilTag* graphics with those IDs, not the
        # old ArUco ones -- the bit patterns are different families.
        # NOTE: tag id 0 scaled 5x in model.sdf (0.067m -> 0.335m) --
        # update this if you scale it differently or scale a different
        # tag. This MUST match the actual real-world size of the black
        # tag pattern in the sim, or solvePnP's distance/pose output
        # will be off by exactly that scale factor.
        # Tag id 0: model.sdf applies <scale>6 6 1</scale> to april_tag.dae,
        # whose black tag pattern is 0.067m at base scale -- so the real
        # world size is 0.067 * 6 = 0.402m. Confirmed directly from
        # model.sdf, not a guess -- update this if you change the scale
        # or swap which tag is mounted.
        self._TAG_SIZES = {
            35: 0.455,
            27: 0.067,
            0 : 0.067
        }

        self._TAG_POSITIONS = {
            35: [0.0, 0.3100, 0.0], #x, y, z
            27: [0.0, 0.0000, 0.0],
            0 : [0.0, 0.0000, 0.0]
        }

        self._object_points = {}
        for tag_id, size in self._TAG_SIZES.items():
            half = size / 2.0
            self._object_points[tag_id] = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)

        # ---- PUBLISHERS ----
        self._bridge = CvBridge()
        self._webcam_publisher = self.create_publisher(Image, "/image", 10)
        self._aruco_target_found_publisher = self.create_publisher(Bool, "/aruco_target/found", 10)
        self._aruco_target_odom_publisher = self.create_publisher(Odometry, "/aruco_target/visual_odom", 10)
        self.debug_image_pub = self.create_publisher(Image, "/aruco_detector/image", 10)

        # ---- TF2 ----
        self._tf_cam_to_tag_broadcaster = tf2_ros.TransformBroadcaster(self)
        self._tf_tag_to_aruco_target_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- APRILTAG DETECTOR ----
        # Dedicated apriltag C library via pupil_apriltags, instead of
        # OpenCV's generic aruco-module AprilTag support. Detection only
        # -- pose is still done with our own solvePnP below (not this
        # library's built-in pose estimator), because that estimator
        # assumes one uniform tag size and our tags aren't uniform
        # (0.455m vs 0.067m). Everything downstream of detection is
        # unchanged.
        self.detector = AprilTagDetector(
            families="tag36h11",
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.75,
            debug=0,
        )

        # ---- DIAGNOSTICS ----
        # Logs one row per processing cycle -- hit AND miss -- so we can
        # actually see the pattern behind intermittent detection instead
        # of guessing from log lines that only fire on success. Look at
        # raw_detections/best_decision_margin especially: if
        # raw_detections is 0 on most misses, pupil_apriltags isn't even
        # finding a candidate quad (points to occlusion/angle/blur/
        # contrast, not a threshold tuning problem). If raw_detections
        # is >0 but accepted=False, it's finding something and our own
        # min_marker_area_px filter is rejecting it (a tuning fix, much
        # easier to solve). If best_decision_margin is consistently low
        # even when accepted, the tag is right at the edge of
        # decodability (matches the pixel-resolution analysis from
        # earlier -- distance/FOV/tag-size need more margin, not just
        # "enough" margin).
        diag_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_dir = self.output_dir if self.output_dir else (get_workspace_root() or os.getcwd())
        os.makedirs(diag_dir, exist_ok=True)
        self._diag_csv_path = os.path.join(diag_dir, f"detection_diagnostics_{diag_timestamp}.csv")
        self._diag_file = open(self._diag_csv_path, "w")
        self._diag_file.write(
            "wall_time,raw_detections,best_tag_id,best_decision_margin,"
            "best_area_px,min_marker_area_px,accepted,solvepnp_success,distance_m\n"
        )
        self.get_logger().info(f"Per-frame detection diagnostics: {self._diag_csv_path}")

        # ---- INITIALISATION ----
        self.frame_count = 0
        self.saved_frames = []
        
        if self.save_frames or self.create_video:
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_dir:
                self.frames_dir = os.path.join(self.output_dir, f"frames_{timestamp}")
            else:
                # Use workspace root or current directory
                workspace_root = get_workspace_root()
                base_dir = workspace_root if workspace_root else os.getcwd()
                self.frames_dir = os.path.join(base_dir, f"frames_{timestamp}")
            
            os.makedirs(self.frames_dir, exist_ok=True)
            self.get_logger().info(f"Frame saving ENABLED - Directory: {self.frames_dir}")
            self.get_logger().info(f"Video creation settings - save_frames: {self.save_frames}, create_video: {self.create_video}, fps: {self.video_fps}")
            
            # Video output filename
            self.video_filename = os.path.join(
                os.path.dirname(self.frames_dir), 
                f"yolo_detection_video_{timestamp}.mp4"
            )
            self.get_logger().info(f"Video will be saved as: {self.video_filename}")
        else:
            self.get_logger().info("Frame saving DISABLED - no video will be created")

        if self.show_debug_window:
            cv2.namedWindow("Detected Markers", cv2.WINDOW_AUTOSIZE)

        # ---- IMAGE SOURCE SETUP ----
        # Always hold at most one frame -- the newest one received/captured.
        # The processing timer below consumes it and clears it, so a slow
        # processing cycle skips straight to whatever's newest next time
        # instead of working through a backlog of stale frames.
        self._img_msg = None

        # BEST_EFFORT + depth=1: never queue frames waiting to be
        # processed. If detection falls behind the camera's publish
        # rate, older frames are simply dropped instead of piling up --
        # that backlog (with the previous default RELIABLE/depth=10
        # subscription) is what was causing "barely detects" and
        # effectively slow updates: we were working through increasingly
        # stale queued frames rather than the live feed.
        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self._image_store_callback, _img_qos
            )
            self.get_logger().info(
                "ArUCoImageNode started in TOPIC mode, waiting for MAVROS altitude and image topic..."
            )
        else:
            # Try different backends for camera access
            self.cap = None
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends_to_try:
                try:
                    self.cap = cv2.VideoCapture(self.webcam_index, backend)
                    if self.cap.isOpened():
                        self.get_logger().info(f"Successfully opened camera {self.webcam_index} with backend {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    self.get_logger().warning(f"Failed to open camera with backend {backend}: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            if self.cap is None or not self.cap.isOpened():
                self.get_logger().error(
                    f"Could not open webcam at index {self.webcam_index}. "
                    f"Make sure your user is in the 'video' group: sudo usermod -a -G video $USER"
                )
            else:
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._image_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._image_height)
                
                # Log actual camera properties
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                self.get_logger().info(
                    f"ArUCoImageNode started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )
            self.timer = self.create_timer(
                1.0 / 30.0, self._webcam_store_callback
            )  # capture rate -- separate from processing_rate below

        # Fires at processing_rate, independent of how fast frames are
        # arriving/being captured -- always processes whatever is
        # currently the newest stored frame, then clears it.
        self._process_timer = self.create_timer(
            1.0 / self._processing_rate, self._process_timer_callback
        )

    # ---- CALLBACK IMPLEMENTATIONS ----
    def _image_store_callback(self, msg):
        """Topic mode: just store the latest frame. Actual detection
        happens on the processing timer, decoupled from arrival rate."""
        self._img_msg = msg

    def _webcam_store_callback(self):
        """Webcam mode: grab one frame and store it as the latest. Actual
        detection happens on the processing timer, decoupled from
        capture rate."""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self._image_width, self._image_height), interpolation=cv2.INTER_NEAREST
                )
                msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self._img_msg = msg
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")

    def _process_timer_callback(self):
        """Fires at processing_rate. Consumes whichever frame is
        currently newest (from either source mode) and clears it, so a
        slow processing cycle skips straight to the next-newest frame
        instead of working through a backlog."""
        if self._img_msg is None:
            return  # nothing new since the last cycle
        msg = self._img_msg
        self._img_msg = None
        self.process_frame(msg)

    def process_frame(self, msg):
        """Process image and detect tag, calculate pose and publish tf_transform"""
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Prefer the incoming image's own timestamp -- that's the true
        # capture time from the camera driver/sim, i.e. the instant the
        # pixels we're about to run detection on were actually taken.
        # Detection + solvePnP can take a few to tens of ms, and by the
        # time this odom message is published/received downstream the
        # gimbal may already have moved on to a new attitude. Publishing
        # the *capture* stamp (not "now") lets the controller look up
        # the gimbal/drone pose that was true at capture time instead of
        # whatever the latest live gimbal reading happens to be.
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            stamp = msg.header.stamp
        else:
            # Webcam-sourced Image messages have no stamp set upstream --
            # fall back to reception time (still taken before processing,
            # so it's the best available approximation of capture time).
            stamp = self.get_clock().now().to_msg()

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(frame, (self._image_width, self._image_height), interpolation=cv2.INTER_LINEAR)

        # Inference (AprilTag detection via pupil_apriltags)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale before inference step
        detections = self.detector.detect(gray_frame)

        # Diagnostics: capture the best RAW detection (any tag ID, any
        # size) before our own filtering, so a miss can be traced back
        # to "pupil_apriltags found nothing" vs "found something but our
        # filters rejected it."
        diag = {
            "raw_detections": len(detections),
            "best_tag_id": "",
            "best_decision_margin": "",
            "best_area_px": "",
            "accepted": False,
            "solvepnp_success": False,
            "distance_m": "",
        }
        if detections:
            best_raw = max(detections, key=lambda d: d.decision_margin)
            diag["best_tag_id"] = int(best_raw.tag_id)
            diag["best_decision_margin"] = round(float(best_raw.decision_margin), 2)
            diag["best_area_px"] = round(float(cv2.contourArea(best_raw.corners.astype(np.float32))), 1)

        # Keep only recognised markers and choose the largest recognised one.
        valid_markers = []
        for d in detections:
            tag_id = int(d.tag_id)
            if tag_id not in self._object_points:
                continue

            area = cv2.contourArea(d.corners.astype(np.float32))
            if area < self.min_marker_area_px:
                continue

            valid_markers.append((area, d, tag_id))

        diag["accepted"] = len(valid_markers) > 0

        aruco_target_found = len(valid_markers) > 0
        self._aruco_target_found_publisher.publish(Bool(data=aruco_target_found))

        # If a recognised tag is visible, execute pose calculations.
        if aruco_target_found:
            area, best, tag_id = max(valid_markers, key=lambda item: item[0])

            # pupil_apriltags returns corners in a different order than
            # our object_points [top-left, top-right, bottom-right,
            # bottom-left] convention -- reorder to match.
            image_points = best.corners[[1, 0, 3, 2]].astype(np.float32)
            object_points = self._object_points[tag_id]

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            self.get_logger().info(
                f"ArUCo tag {tag_id} detected. area={area:.1f}, "
                f"rvec={rvec.flatten()}, tvec={tvec.flatten()}",
                throttle_duration_sec=0.5,
            )

            if success:
                diag["solvepnp_success"] = True

                # Draw pose axes for debugging
                if self.show_debug_window:
                    cv2.drawFrameAxes(
                        frame,
                        self._camera_matrix,
                        self._dist_coeffs,
                        rvec,
                        tvec,
                        self._TAG_SIZES[tag_id] * 0.5
                    )

                # Broadcast target position relative to the camera frame.
                cam_to_tag_tf_msg = self.cam_to_tag_transformstamped(stamp, tag_id, rvec, tvec)
                tag_to_aruco_target_tf_msg = self.tag_to_aruco_target_transformstamped(stamp, tag_id, self._TAG_POSITIONS)
                if cam_to_tag_tf_msg is not None and tag_to_aruco_target_tf_msg is not None:
                    self._tf_cam_to_tag_broadcaster.sendTransform(cam_to_tag_tf_msg)
                    self._tf_tag_to_aruco_target_broadcaster.sendTransform(tag_to_aruco_target_tf_msg)

                    aruco_target_odom_msg = self.aruco_target_odometry(stamp, tag_id, rvec, tvec, self._TAG_POSITIONS)
                    if aruco_target_odom_msg is not None:
                        self._aruco_target_odom_publisher.publish(aruco_target_odom_msg)

                        p = aruco_target_odom_msg.pose.pose.position
                        target_distance = float(np.sqrt(p.x * p.x + p.y * p.y + p.z * p.z))
                        diag["distance_m"] = round(target_distance, 2)
                        self.get_logger().info(
                            f"ArUco target distance={target_distance:.2f} m, "
                            f"camera_position=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})",
                            throttle_duration_sec=0.5,
                        )

        # Write one diagnostic row per cycle regardless of hit/miss --
        # this is what lets us see the actual failure pattern instead of
        # guessing from log lines that only fire on success.
        self._diag_file.write(
            f"{self.get_clock().now().nanoseconds / 1e9:.3f},"
            f"{diag['raw_detections']},{diag['best_tag_id']},{diag['best_decision_margin']},"
            f"{diag['best_area_px']},{self.min_marker_area_px},{diag['accepted']},"
            f"{diag['solvepnp_success']},{diag['distance_m']}\n"
        )
        self._diag_file.flush()

        # Show the output image after AprilTag detection (if debug window enabled)
        if self.show_debug_window:
            for d in detections:
                pts = d.corners.astype(np.int32)
                colour = (0, 255, 0) if int(d.tag_id) in self._object_points else (0, 165, 255)
                cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
                cv2.putText(
                    frame,
                    str(d.tag_id),
                    (int(d.center[0]), int(d.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow('Detected Markers', frame)
            cv2.waitKey(1)

        # Save frame if enabled
        if self.save_frames or self.create_video:
            if hasattr(self, 'frames_dir'):
                frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, frame)
                if success:
                    self.saved_frames.append(frame_filename)
                    self.frame_count += 1
                    
                    # Log progress every 100 frames
                    if self.frame_count % 100 == 0:
                        self.get_logger().info(f"Saved {self.frame_count} frames so far...")
                else:
                    self.get_logger().warning(f"Failed to save frame {self.frame_count}")
            else:
                self.get_logger().warning("Frame saving enabled but frames_dir not initialized")

        if self.enable_debug_publish:
            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self._webcam_publisher.publish(msg)

    # ---- HELPER FUNCTIONS ---- 
    def cam_to_tag_transformstamped(self, stamp, tag_id, rvec, tvec):
        # From ArUCo tag, find the camera --> tag transform (4x4)
        try:
            t_cam_to_tag = tvec.reshape(3) # position
            R_cam_to_tag, _ = cv2.Rodrigues(rvec)
            T_cam_to_tag = np.eye(4)
            T_cam_to_tag[:3, :3] = R_cam_to_tag
            q_cam_to_tag = tf_transformations.quaternion_from_matrix(T_cam_to_tag)
        
        except Exception as e:
            self.get_logger().error(f"cam_to_tag transform failed: {e}")
            return None

        # Header for pose
        tf_cam_to_tag = TransformStamped()
        tf_cam_to_tag.header.stamp = stamp
        tf_cam_to_tag.header.frame_id = self.camera_frame
        tf_cam_to_tag.child_frame_id = f"tag{tag_id}_link" # keeps tag_id positions in sync with cam detection
        tf_cam_to_tag.transform.translation.x = t_cam_to_tag[0]
        tf_cam_to_tag.transform.translation.y = t_cam_to_tag[1]
        tf_cam_to_tag.transform.translation.z = t_cam_to_tag[2]
        tf_cam_to_tag.transform.rotation.x = q_cam_to_tag[0]
        tf_cam_to_tag.transform.rotation.y = q_cam_to_tag[1]
        tf_cam_to_tag.transform.rotation.z = q_cam_to_tag[2]
        tf_cam_to_tag.transform.rotation.w = q_cam_to_tag[3]

        return tf_cam_to_tag
    
    def tag_to_aruco_target_transformstamped(self, stamp, tag_id, tag_positions):
        # Broadcast the tag_to_aruco_target tf transform
        t_tag_to_aruco_target = np.array(tag_positions[tag_id])
        q_tag_to_aruco_target = tf_transformations.quaternion_from_euler(0.0, 0.0, -1.570796326) # Turns out all the tags were 90deg off...

        # Header for pose
        tf_tag_to_aruco_target = TransformStamped()
        tf_tag_to_aruco_target.header.stamp = stamp
        tf_tag_to_aruco_target.header.frame_id = f"tag{tag_id}_link" # keeps tag_id positions in sync with cam detection
        tf_tag_to_aruco_target.child_frame_id = "aruco_target_link"
        tf_tag_to_aruco_target.transform.translation.x = t_tag_to_aruco_target[0]
        tf_tag_to_aruco_target.transform.translation.y = t_tag_to_aruco_target[1]
        tf_tag_to_aruco_target.transform.translation.z = t_tag_to_aruco_target[2]
        tf_tag_to_aruco_target.transform.rotation.x = q_tag_to_aruco_target[0]
        tf_tag_to_aruco_target.transform.rotation.y = q_tag_to_aruco_target[1]
        tf_tag_to_aruco_target.transform.rotation.z = q_tag_to_aruco_target[2]
        tf_tag_to_aruco_target.transform.rotation.w = q_tag_to_aruco_target[3]

        return tf_tag_to_aruco_target
    
    def aruco_target_odometry(self, stamp, tag_id, rvec, tvec, tag_positions):
        """
        Publish the final aruco_target_link pose relative to self.camera_frame as Odometry.

        This composes:
            camera frame -> tag<ID>_link
            tag<ID>_link -> aruco_target_link

        The resulting visual odometry message is therefore in self.camera_frame.
        The controller can either:
          1. use this directly for image-relative control, or
          2. transform it into map/local frame before doing world-frame PID.
        """
        try:
            # camera -> tag transform
            t_cam_to_tag = tvec.reshape(3)
            R_cam_to_tag, _ = cv2.Rodrigues(rvec)

            T_cam_to_tag = np.eye(4)
            T_cam_to_tag[:3, :3] = R_cam_to_tag
            T_cam_to_tag[:3, 3] = t_cam_to_tag

            # tag -> aruco target transform
            t_tag_to_aruco_target = np.array(tag_positions[tag_id], dtype=float)
            q_tag_to_aruco_target = tf_transformations.quaternion_from_euler(
                0.0, 0.0, -1.570796326
            )
            T_tag_to_aruco_target = tf_transformations.quaternion_matrix(
                q_tag_to_aruco_target
            )
            T_tag_to_aruco_target[:3, 3] = t_tag_to_aruco_target

            # camera -> aruco target
            T_cam_to_aruco_target = T_cam_to_tag @ T_tag_to_aruco_target

            q_cam_to_aruco_target = tf_transformations.quaternion_from_matrix(
                T_cam_to_aruco_target
            )
            t_cam_to_aruco_target = T_cam_to_aruco_target[:3, 3]

        except Exception as e:
            self.get_logger().error(f"aruco target odometry transform failed: {e}")
            return None

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.camera_frame
        odom.child_frame_id = "aruco_target_link"

        odom.pose.pose.position.x = float(t_cam_to_aruco_target[0])
        odom.pose.pose.position.y = float(t_cam_to_aruco_target[1])
        odom.pose.pose.position.z = float(t_cam_to_aruco_target[2])

        odom.pose.pose.orientation.x = float(q_cam_to_aruco_target[0])
        odom.pose.pose.orientation.y = float(q_cam_to_aruco_target[1])
        odom.pose.pose.orientation.z = float(q_cam_to_aruco_target[2])
        odom.pose.pose.orientation.w = float(q_cam_to_aruco_target[3])

        # Velocity is not estimated by the detector at this stage.
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        return odom

    def create_video_from_frames(self):
        """Create video from saved frames"""
        if not (self.save_frames or self.create_video) or not self.saved_frames:
            self.get_logger().info(f"Video creation skipped. save_frames={self.save_frames}, create_video={self.create_video}, frames_count={len(self.saved_frames) if hasattr(self, 'saved_frames') else 0}")
            return
            
        try:
            duration_seconds = len(self.saved_frames) / self.video_fps
            self.get_logger().info(f"Creating video from {len(self.saved_frames)} frames (estimated duration: {duration_seconds:.1f}s at {self.video_fps}fps)...")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(self.saved_frames[0])
            if first_frame is None:
                self.get_logger().error("Could not read first frame for video creation")
                return
                
            height, width, layers = first_frame.shape
            self.get_logger().info(f"Video dimensions: {width}x{height}")
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.video_filename, 
                fourcc, 
                self.video_fps, 
                (width, height)
            )
            
            if not video_writer.isOpened():
                self.get_logger().error("Failed to open video writer")
                return
            
            # Write all frames to video
            frames_written = 0
            for i, frame_path in enumerate(self.saved_frames):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                    
                    # Progress update every 100 frames
                    if (i + 1) % 100 == 0:
                        self.get_logger().info(f"Writing frame {i + 1}/{len(self.saved_frames)} to video...")
                else:
                    self.get_logger().warning(f"Could not read frame: {frame_path}")
            
            video_writer.release()
            self.get_logger().info(f"Video created successfully: {self.video_filename}")
            self.get_logger().info(f"Final video stats: {frames_written} frames written, duration: {frames_written/self.video_fps:.1f}s")
            
            # Optionally clean up frame files
            if not self.save_frames:  # Only delete frames if we don't want to keep them
                self.get_logger().info("Cleaning up temporary frame files...")
                for frame_path in self.saved_frames:
                    try:
                        os.remove(frame_path)
                    except OSError as e:
                        self.get_logger().warning(f"Could not remove frame {frame_path}: {e}")
                        
                # Remove frames directory if empty
                try:
                    os.rmdir(self.frames_dir)
                except OSError:
                    pass  # Directory not empty or other error
                    
        except Exception as e:
            self.get_logger().error(f"Error creating video: {e}")

# ---- MAIN ----
def main(args=None):
    rclpy.init(args=args)
    node = ArUCoNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down...")

    finally:
        if hasattr(node, "_diag_file") and node._diag_file is not None:
            try:
                node._diag_file.close()
                node.get_logger().info(f"Diagnostics saved to: {node._diag_csv_path}")
            except Exception:
                pass

        if hasattr(node, "cap") and node.cap is not None:
            node.cap.release()

        if getattr(node, "show_debug_window", True):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()