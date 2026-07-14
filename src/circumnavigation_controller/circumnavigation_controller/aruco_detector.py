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

import rclpy
from rclpy.node import Node
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

        # ---- CAMERA PARAMETERS ----
        # Must match the IR input size you exported (default IRIS is 640 x 480)
        self.declare_parameter("imgsz_width", 640)
        self._image_width = int(self.get_parameter("imgsz_width").get_parameter_value().integer_value)
        
        self.declare_parameter("imgsz_height", 480)
        self._image_height = int(self.get_parameter("imgsz_height").get_parameter_value().integer_value)

        self._camera_fov_horizontal = 2.0  # radians (≈114.6°) – tune for your camera
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
        self._TAG_SIZES = {
                35: 0.455,
                27: 0.067,
                0 : 0.067
            }

        self._TAG_POSITIONS = {
            35: [0.0, 0.3100, 0.0], #x, y, z
            27: [0.0, 0.0000, 0.0],
            0 : [0.0, 0.6200, 0.0]
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

        # ---- OPENCV ----
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        # self.aruco_params.adaptiveThreshWinSizeMin = 3
        # self.aruco_params.adaptiveThreshWinSizeMax = 23
        # self.aruco_params.adaptiveThreshWinSizeStep = 10
        self._aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

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

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self.image_callback, 10
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
                1.0 / 30.0, self.webcam_timer_callback
            )  # 30 Hz

    # ---- CALLBACK IMPLEMENTATIONS ----
    def webcam_timer_callback(self):
        """Read image from webcam and publish to /image topic"""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self._image_width, self._image_height), interpolation=cv2.INTER_NEAREST
                )
                msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")

                self.image_callback(msg)
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")

    def image_callback(self, msg):
        """Process image and detect tag, calculate pose and publish tf_transform"""
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = self.get_clock().now().to_msg() # Timestamp before image processing, because thats when the image was taken

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self._image_height or frame.shape[1] != self._image_width:
            frame = cv2.resize(frame, (self._image_width, self._image_height), interpolation=cv2.INTER_LINEAR)

        # Inference (ArUCo detection via OpenCV)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale before inference step
        corners, ids, _ = self.detector.detectMarkers(gray_frame)

        # Keep only recognised markers and choose the largest recognised one.
        # Previously, the code first checked that *some* recognised marker existed,
        # then selected the largest marker from all detections. If an unknown marker
        # was larger, object_points[tag_id] could fail or produce the wrong pose.
        valid_markers = []
        if ids is not None:
            for i, tag_id_arr in enumerate(ids):
                tag_id = int(tag_id_arr[0])
                if tag_id not in self._object_points:
                    continue

                area = cv2.contourArea(corners[i][0].astype(np.float32))
                if area < self.min_marker_area_px:
                    continue

                valid_markers.append((area, i, tag_id))

        aruco_target_found = len(valid_markers) > 0
        self._aruco_target_found_publisher.publish(Bool(data=aruco_target_found))

        # If a recognised tag is visible, execute pose calculations.
        if aruco_target_found:
            area, idx, tag_id = max(valid_markers, key=lambda item: item[0])

            image_points = corners[idx][0].astype(np.float32)
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
                        self.get_logger().info(
                            f"ArUco target distance={target_distance:.2f} m, "
                            f"camera_position=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})",
                            throttle_duration_sec=0.5,
                        )

        # Show the output image after ArUCo detection (if debug window enabled)
        if self.show_debug_window:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
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