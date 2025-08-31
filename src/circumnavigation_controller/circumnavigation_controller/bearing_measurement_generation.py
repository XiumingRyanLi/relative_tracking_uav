import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import tf_transformations
import logging
import os

# Silence YOLO logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ['YOLO_VERBOSE'] = 'False'

class YoloImageNode(Node):
    def __init__(self):
        super().__init__('yolo_image_node')
        self.declare_parameter('enable_debug_publish', False)
        self.enable_debug_publish = self.get_parameter('enable_debug_publish').get_parameter_value().bool_value

        # Parameter to select image source: 'topic' or 'webcam'
        self.declare_parameter('image_source', 'topic')
        self.image_source = self.get_parameter('image_source').get_parameter_value().string_value

        # If using webcam, declare webcam index parameter
        self.declare_parameter('webcam_index', 0)
        self.webcam_index = self.get_parameter('webcam_index').get_parameter_value().integer_value

        self.declare_parameter('show_debug_window', True)
        self.show_debug_window = self.get_parameter('show_debug_window').get_parameter_value().bool_value
        # MAVROS pose subscription for orientation with correct QoS
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            mavros_qos)

        # Bearing publisher
        self.bearing_publisher = self.create_publisher(Float64, '/bearing', 10)
        # Error publisher for yaw controller (person position error from center)
        self.error_publisher = self.create_publisher(Float64, '/yaw_error', 10)

        self.bridge = CvBridge()
        import warnings
        warnings.filterwarnings("ignore")
        # Load YOLO11n NCNN model from absolute path provided by user
        self.ncnn_model = YOLO('/home/case/circumnavigation_ws/yolo11n_openvino_model')
        if self.show_debug_window:
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        # Current quadcopter yaw (heading) in radians
        self.current_yaw = 0.0
        # Camera parameters
        self.image_width = 320
        self.image_height = 320
        self.camera_fov_horizontal = 2.0  # Camera FOV in radians

        # If using topic, subscribe to image topic
        if self.image_source == 'topic':
            self.image_subscription = self.create_subscription(
                Image,
                '/image',
                self.image_callback,
                10)
            self.get_logger().info('YoloImageNode started in TOPIC mode, waiting for MAVROS pose and image topic...')
        else:
            # If using webcam, set up timer to read from webcam and publish to /image
            self.webcam_publisher = self.create_publisher(Image, '/image', 10)
            self.cap = cv2.VideoCapture(self.webcam_index)
            if not self.cap.isOpened():
                self.get_logger().error(f'Could not open webcam at index {self.webcam_index}')
            else:
                self.get_logger().info(f'YoloImageNode started in WEBCAM mode, publishing webcam frames to /image...')
            self.timer = self.create_timer(1.0/30.0, self.webcam_timer_callback)  # 30 FPS

    def webcam_timer_callback(self):
        """Read image from webcam and publish to /image topic"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize to expected size
                frame = cv2.resize(frame, (self.image_width, self.image_height))
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                if self.enable_debug_publish:
                    self.webcam_publisher.publish(msg)
                # Directly call image_callback for processing
                self.image_callback(msg)
            else:
                self.get_logger().warn('Failed to read frame from webcam.')
        else:
            self.get_logger().warn('Webcam not opened.')
        
    def pose_callback(self, msg):
        """Get current quadcopter orientation from MAVROS"""
        # Debug: Log that callback was triggered
        print("test")
        
        # Extract quaternion
        q = msg.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        
        # Convert to Euler angles (roll, pitch, yaw)
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(quaternion)
        
        # Print current heading in degrees
        heading_degrees = np.degrees(self.current_yaw)
        self.get_logger().info(f'Current heading: {heading_degrees:.1f}° from North')
        

    def calculate_person_error(self, person_center_x):
        """Calculate error in radians between person position and frame center"""
        # Calculate pixel offset from center
        image_center_x = self.image_width / 2
        pixel_offset = person_center_x - image_center_x
        
        # Convert pixel offset to angular error
        angle_per_pixel = self.camera_fov_horizontal / self.image_width
        error_radians = pixel_offset * angle_per_pixel
        
        return error_radians

    def image_callback(self, msg):
        """Process image and detect person, calculate bearing"""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Run YOLO inference with NCNN model
        results = self.ncnn_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Store image dimensions
        self.image_height, self.image_width = frame.shape[:2]

        # Always draw current heading on image
        heading_degrees = np.degrees(self.current_yaw)
        cv2.putText(annotated_frame, f'Heading: {heading_degrees:.1f}°', 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Look for person detections (class 0 in COCO dataset)
        person_detected = False
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a person (class 0) with good confidence
                    if class_id == 0 and confidence > 0.5:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center of person
                        person_center_x = (x1 + x2) / 2
                        person_center_y = (y1 + y2) / 2
                        
                        # Calculate bearing to person and error from center
                        error_radians = self.calculate_person_error(person_center_x)
                        bearing = self.current_yaw - error_radians
                        
                        # Publish bearing
                        bearing_msg = Float64()
                        bearing_msg.data = float(bearing)
                        self.bearing_publisher.publish(bearing_msg)
                        
                        # Publish error for yaw controller
                        error_msg = Float64()
                        error_msg.data = float(error_radians)
                        self.error_publisher.publish(error_msg)
                        
                        # Log the bearing and error
                        bearing_degrees = np.degrees(bearing)
                        error_degrees = np.degrees(error_radians)
                        self.get_logger().info(f'Person detected! Bearing: {bearing_degrees:.1f}° from North, Error: {error_degrees:.1f}° from center')
                        
                        # Draw additional info on image
                        cv2.circle(annotated_frame, (int(person_center_x), int(person_center_y)), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, f'Bearing: {bearing_degrees:.1f}°',  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'Error: {error_degrees:.1f}°', 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Draw center line for reference
                        center_x = int(self.image_width / 2)
                        cv2.line(annotated_frame, (center_x, 0), (center_x, self.image_height), (255, 255, 255), 2)
                        
                        person_detected = True
                        break
        
        if self.show_debug_window:
            cv2.imshow('RealSense', annotated_frame)
            cv2.waitKey(1)
            

def main(args=None):
    rclpy.init(args=args)
    node = YoloImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down...')
    finally:
        if hasattr(node, 'cap'):
            node.cap.release()
        node.destroy_node()
        if getattr(node, 'show_debug_window', True):
            cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
