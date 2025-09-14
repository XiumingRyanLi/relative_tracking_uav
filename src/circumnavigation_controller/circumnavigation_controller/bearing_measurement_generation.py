#!/usr/bin/env python3
import os

# ---- Ultralytics/OpenVINO runtime safety (set BEFORE importing ultralytics) ----
os.environ["AUTOINSTALL"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "0"
os.environ["OV_CPU_THREADS_NUM"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
from ultralytics import YOLO  
from ultralytics import utils as yutils
yutils.ONLINE = False
# -------------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

import numpy as np
import tf_transformations
import logging
import warnings
from datetime import datetime
import glob
import signal
import sys

# Silence YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")

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


class YoloImageNode(Node):
    def __init__(self):
        super().__init__("yolo_image_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("enable_debug_publish", False)
        self.enable_debug_publish = (
            self.get_parameter("enable_debug_publish").get_parameter_value().bool_value
        )

        self.declare_parameter("image_source", "topic")  # 'topic' or 'webcam'
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        self.declare_parameter("show_debug_window", True)
        self.show_debug_window = (
            self.get_parameter("show_debug_window").get_parameter_value().bool_value
        )

        # Frame saving and video creation parameters
        self.declare_parameter("save_frames", False)
        self.save_frames = (
            self.get_parameter("save_frames").get_parameter_value().bool_value
        )

        self.declare_parameter("create_video", True)
        self.create_video = (
            self.get_parameter("create_video").get_parameter_value().bool_value
        )

        self.declare_parameter("video_fps", 30.0)
        self.video_fps = float(
            self.get_parameter("video_fps").get_parameter_value().double_value
        )

        self.declare_parameter("output_dir", "")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # Log parameter values for debugging
        self.get_logger().info(f"Video recording parameters: save_frames={self.save_frames}, create_video={self.create_video}, video_fps={self.video_fps}")
        self.get_logger().info(f"Output directory: '{self.output_dir}' (empty means workspace root)")

        # Path to exported OpenVINO model (folder OR .xml)
        # Get workspace root and construct relative path
        workspace_root = get_workspace_root()
        default_model_path = os.path.join(workspace_root, "yolo11n_openvino_model") if workspace_root else "yolo11n_openvino_model"
        
        self.declare_parameter(
            "model_path",
            default_model_path,
        )
        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )

        # Must match the IR input size you exported (default OpenVINO export is 640)
        self.declare_parameter("imgsz", 256)
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        # ----------------------------------------------------------

        # MAVROS compass heading subscription (BEST_EFFORT)
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.compass_subscription = self.create_subscription(
            Float64, "/mavros/global_position/compass_hdg", self.compass_callback, mavros_qos
        )

        # Publishers
        self.bearing_publisher = self.create_publisher(Float64, "/bearing", 10)
        self.error_publisher = self.create_publisher(Float64, "/yaw_error", 10)

        self.bridge = CvBridge()

        # Load OpenVINO model via Ultralytics
        try:
            self.model = YOLO(self.model_path)  # CPU by default on Pi
        except Exception as e:
            self.get_logger().error(f"Failed to load model at {self.model_path}: {e}")
            raise

        # Initialize frame saving
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
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)

        # State
        self.current_yaw = 0.0
        self.image_width = self.imgsz
        self.image_height = self.imgsz
        self.camera_fov_horizontal = 0.8  # radians (≈114.6°) – tune for your camera

        self.webcam_publisher = self.create_publisher(Image, "/image", 10)

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self.image_callback, 10
            )
            self.get_logger().info(
                "YoloImageNode started in TOPIC mode, waiting for MAVROS compass heading and image topic..."
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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.imgsz)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgsz)
                
                # Log actual camera properties
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                self.get_logger().info(
                    f"YoloImageNode started in WEBCAM mode. Camera properties: "
                    f"FPS={actual_fps}, Width={actual_width}, Height={actual_height}"
                )
            self.timer = self.create_timer(
                1.0 / 30.0, self.webcam_timer_callback
            )  # 30 Hz

    def webcam_timer_callback(self):
        """Read image from webcam and publish to /image topic"""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST
                )
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

                self.image_callback(msg)
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")

    def compass_callback(self, msg):
        """Get current quadcopter orientation from MAVROS compass heading"""
        # Convert degrees to radians
        self.current_yaw = np.radians(msg.data)

    def calculate_person_error(self, person_center_x):
        """Error in radians between person and frame center"""
        image_center_x = self.image_width / 2
        pixel_offset = person_center_x - image_center_x
        angle_per_pixel = self.camera_fov_horizontal / self.image_width
        return pixel_offset * angle_per_pixel

    def image_callback(self, msg):
        """Process image and detect person, calculate bearing"""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Ensure inference size matches IR (handles topic frames of any size)
        if frame.shape[0] != self.imgsz or frame.shape[1] != self.imgsz:
            frame = cv2.resize(
                frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST
            )

        # Inference (OpenVINO IR)
        results = self.model(frame, imgsz=self.imgsz, verbose=False, device="cpu")
        annotated_frame = results[0].plot()

        # Use inference frame size for geometry (boxes are in this scale)
        self.image_height, self.image_width = annotated_frame.shape[:2]

        # Heading overlay
        heading_degrees = np.degrees(self.current_yaw)
        cv2.putText(
            annotated_frame,
            f"Heading: {heading_degrees:.1f}\u00b0",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Detect 'person' (COCO class 0)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id == 0 and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    error_radians = self.calculate_person_error(cx)
                    bearing = self.current_yaw + error_radians

                    self.bearing_publisher.publish(Float64(data=float(bearing)))
                    self.error_publisher.publish(Float64(data=float(error_radians)))

                    bearing_degrees = np.degrees(bearing)
                    error_degrees = np.degrees(error_radians)
                    #self.get_logger().info(
                    #    f"Person detected! Bearing: {bearing_degrees:.1f}\u00b0, Error: {error_degrees:.1f}\u00b0"
                    #)

                    cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        f"Bearing: {bearing_degrees:.1f}\u00b0",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Error: {error_degrees:.1f}\u00b0",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    center_x = int(self.image_width / 2)
                    cv2.line(
                        annotated_frame,
                        (center_x, 0),
                        (center_x, self.image_height),
                        (255, 255, 255),
                        2,
                    )
                    break  # one person is enough

        if self.show_debug_window:
            cv2.imshow("RealSense", annotated_frame)
            cv2.waitKey(1)

        # Save frame if enabled
        if self.save_frames or self.create_video:
            if hasattr(self, 'frames_dir'):
                frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, annotated_frame)
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
            msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.webcam_publisher.publish(msg)

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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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


def main(args=None):
    rclpy.init(args=args)
    node = YoloImageNode()
    
    # Global variable to track if we're already shutting down
    shutdown_in_progress = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            node.get_logger().warn("Second interrupt received! Force terminating without video creation...")
            sys.exit(1)
        else:
            shutdown_in_progress = True
            node.get_logger().info("Interrupt received, creating video before shutdown (press Ctrl+C again to force quit)...")
            raise KeyboardInterrupt()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        # Create video from saved frames before cleanup
        if hasattr(node, 'create_video_from_frames') and not shutdown_in_progress:
            node.create_video_from_frames()
        elif hasattr(node, 'create_video_from_frames'):
            try:
                # Give it a chance even if shutdown is in progress
                node.get_logger().info("Creating video during shutdown...")
                node.create_video_from_frames()
            except Exception as e:
                node.get_logger().error(f"Failed to create video during shutdown: {e}")
            
        if hasattr(node, "cap") and node.cap is not None:
            node.cap.release()
        node.destroy_node()
        if getattr(node, "show_debug_window", True):
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
