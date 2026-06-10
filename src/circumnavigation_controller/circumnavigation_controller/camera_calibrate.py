#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraCalibrate(Node):
    def __init__(self):
        super().__init__("camera_calibration_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("image_source", "topic")  # 'topic' or 'webcam'
        self.image_source = (
            self.get_parameter("image_source").get_parameter_value().string_value
        )

        self.declare_parameter("webcam_index", 0)
        self.webcam_index = int(
            self.get_parameter("webcam_index").get_parameter_value().integer_value
        )

        # Must match the IR input size you exported (default OpenVINO export is 640)
        self.declare_parameter("imgsz", 256)
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        # ----------------------------------------------------------

        self.bridge = CvBridge()
        cv2.namedWindow("Calibration Camera", cv2.WINDOW_AUTOSIZE)

        # State
        self.image_width = self.imgsz
        self.image_height = self.imgsz
        self.camera_fov_horizontal = 0.8  # radians (≈114.6°) – tune for your camera

        self.webcam_publisher = self.create_publisher(Image, "/image", 10)

        self.objpoints = []  # 3D points in real world
        self.imgpoints = []  # 2D points in image plane
        self.frames_collected = 0
        self.num_frames = 20  # how many checkerboard images to use

        # Define checkerboard
        self.CHECKERBOARD = (4, 4)
        self.square_size = 0.24  # meters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Precompute object points (3D coordinates)
        self.objp = np.zeros((self.CHECKERBOARD[0]*self.CHECKERBOARD[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1,2)
        self.objp *= self.square_size

        # Image source
        if self.image_source == "topic":
            self.image_subscription = self.create_subscription(
                Image, "/camera/image_raw", self.image_callback, 10
            )
            self.get_logger().info(
                "Camera Calibration Node started in TOPIC mode, waiting for image topic..."
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
                    f"ArUCoImageNode started in WEBCAM mode. Camera properties: "
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

    def image_callback(self, msg):
        """Process image and calculate camera coefficients"""
        # Code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray_frame,
            self.CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray_frame, corners, (11,11), (-1,-1), self.criteria)

            self.objpoints.append(self.objp)       # same 3D points for all frames
            self.imgpoints.append(corners2)        # detected 2D corners

            self.frames_collected += 1
            print(f"Collected frame {self.frames_collected}/{self.num_frames}")

            # Draw corners for visualization
            cv2.drawChessboardCorners(frame, self.CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration Camera', frame)
            cv2.waitKey(5000)

        # Once enough frames collected, calibrate
        if self.frames_collected >= self.num_frames:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, gray_frame.shape[::-1], None, None
            )
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)
            print("rvecs:\n", rvecs)
            print("tvecs:\n", tvecs)
            cv2.destroyAllWindows()

            # Stop further calibration
            self.frames_collected = 0
            self.objpoints = []
            self.imgpoints = []
 

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrate()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally: 
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
