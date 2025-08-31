#!/usr/bin/env python3
import os

# ---- Ultralytics/OpenVINO runtime safety (set BEFORE importing ultralytics) ----
os.environ["AUTOINSTALL"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "0"
os.environ["OV_CPU_THREADS_NUM"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
from ultralytics.yolo import utils as yutils
yutils.ONLINE = False
# -------------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import tf_transformations
import logging
import warnings

# Silence YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")


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

        # Path to exported OpenVINO model (folder OR .xml)
        self.declare_parameter(
            "model_path",
            "/home/case/circumnavigation_ws/yolo11n_openvino_model",
        )
        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )

        # Must match the IR input size you exported (default OpenVINO export is 640)
        self.declare_parameter("imgsz", 256)
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        # ----------------------------------------------------------

        # MAVROS pose subscription (BEST_EFFORT)
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pose_subscription = self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, mavros_qos
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

        if self.show_debug_window:
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)

        # State
        self.current_yaw = 0.0
        self.image_width = self.imgsz
        self.image_height = self.imgsz
        self.camera_fov_horizontal = 2.0  # radians (≈114.6°) – tune for your camera

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/image", self.image_callback, 10
            )
            self.get_logger().info(
                "YoloImageNode started in TOPIC mode, waiting for MAVROS pose and image topic..."
            )
        else:
            self.webcam_publisher = self.create_publisher(Image, "/image", 10)
            # Use V4L2 for better throughput; request imgsz x imgsz @ 30fps
            self.cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.imgsz)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgsz)

            if not self.cap.isOpened():
                self.get_logger().error(
                    f"Could not open webcam at index {self.webcam_index}"
                )
            else:
                self.get_logger().info(
                    "YoloImageNode started in WEBCAM mode, publishing webcam frames to /image..."
                )
            self.timer = self.create_timer(
                1.0 / 30.0, self.webcam_timer_callback
            )  # 30 Hz

    def webcam_timer_callback(self):
        """Read image from webcam and publish to /image topic"""
        if hasattr(self, "cap") and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to IR input size (square) – must match export
                frame = cv2.resize(
                    frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST
                )
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                if self.enable_debug_publish:
                    self.webcam_publisher.publish(msg)
                # Process inline
                self.image_callback(msg)
            else:
                self.get_logger().warning("Failed to read frame from webcam.")
        else:
            self.get_logger().warning("Webcam not opened.")

    def pose_callback(self, msg):
        """Get current quadcopter orientation from MAVROS"""
        q = msg.pose.orientation
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

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
                    bearing = self.current_yaw - error_radians

                    self.bearing_publisher.publish(Float64(data=float(bearing)))
                    self.error_publisher.publish(Float64(data=float(error_radians)))

                    bearing_degrees = np.degrees(bearing)
                    error_degrees = np.degrees(error_radians)
                    self.get_logger().info(
                        f"Person detected! Bearing: {bearing_degrees:.1f}\u00b0, Error: {error_degrees:.1f}\u00b0"
                    )

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


def main(args=None):
    rclpy.init(args=args)
    node = YoloImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down...")
    finally:
        if hasattr(node, "cap"):
            node.cap.release()
        node.destroy_node()
        if getattr(node, "show_debug_window", True):
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
