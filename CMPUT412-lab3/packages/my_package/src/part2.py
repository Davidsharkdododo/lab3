#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped

# Assuming DTROS and NodeType are defined elsewhere in your project (e.g., from Duckietown)
from duckietown.dtros import DTROS, NodeType

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        # Camera calibration parameters (intrinsic matrix and distortion coefficients)
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                     -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                    dtype=np.float32)
        # Homography matrix from the YAML file (assumed to map image coordinates to ground coordinates in meters)
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # Color detection parameters in HSV format
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, 216]), np.array([179, 55, 255]))
        }
        
        # Initialize CvBridge and subscribe to the camera feed
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Publishers for processed images
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=1)
        self.rate = rospy.Rate(3)
        self.controller_type = rospy.get_param("~controller_type", "PID")
        self.Kp = rospy.get_param("~Kp", 1)
        self.Ki = rospy.get_param("~Ki", 0.75)
        self.Kd = rospy.get_param("~Kd", 0.25)
        self.base_speed = rospy.get_param("~base_speed", 0.5)
        # Scale factor to convert meter error to a pixel-like error
        # self.distance_error_scale = rospy.get_param("~distance_error_scale", 100.0)
        
        # Variables for PD/PID control.
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()

    def undistort_image(self, image):
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def compute_distance_homography(self, u, v):
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        # Normalize to convert from homogeneous coordinates
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        distance = np.sqrt(X**2 + Y**2)
        return distance

    def detect_lane_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        output = image.copy()
        image_center = image.shape[1] / 2.0
        
        # Dictionary to store detection for each color: {color: (u, distance)}
        detections = {}
        
        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Choose the largest contour assuming it corresponds to the lane marking.
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:  # Threshold to filter noise
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Use the bottom-center of the bounding box as the point of interest
                    u = x + w / 2.0
                    v = y + h
                    distance = self.compute_distance_homography(u, v)
                    
                    cv2.putText(output, f"{color}: {distance:.2f}m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    detections[color] = (u, distance)
                    
        error = None
        # If both white and yellow markers are detected
        if "white" in detections and "yellow" in detections:
            white_u, white_distance = detections["white"]
            yellow_u, yellow_distance = detections["yellow"]
            # error = (white_u + yellow_u)/2.0 -image_center
            error =  yellow_distance - white_distance
            
        return output, detections, error
    
    def calculate_p_control(self, error, dt):
        return self.Kp * error

    def calculate_pd_control(self, error, dt):
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Kd * derivative
        self.prev_error = error
        return output

    def calculate_pid_control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def get_control_output(self, error, dt):
        ctrl_type = self.controller_type.upper()
        if ctrl_type == "P":
            return self.calculate_p_control(error, dt)
        elif ctrl_type == "PD":
            return self.calculate_pd_control(error, dt)
        elif ctrl_type == "PID":
            return self.calculate_pid_control(error, dt)
        else:
            rospy.logwarn("Unknown controller type '%s'. Using P controller.", self.controller_type)
            return self.calculate_p_control(error, dt)
        
    def publish_cmd(self, control_output):
        # Differential drive: adjust left/right speeds based on control output.
        control_output = control_output / 10.0  # Scale control output appropriately.
        left_speed = self.base_speed * (1 - control_output)
        right_speed = self.base_speed * (1 + control_output)
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
        rospy.loginfo("Published wheel cmd: left=%.3f, right=%.3f", left_speed, right_speed)
    
    def callback(self, msg):
        current_time = rospy.Time.now()
        if self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
        else:
            dt = 0.1  # default dt value for the first callback
        self.last_time = current_time
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        color_image, detections, error = self.detect_lane_color(preprocessed)
        lane_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        self.pub_lane.publish(lane_msg)
        if error is not None:
            rospy.loginfo("Control error: %.3f", error)
            control_output = self.get_control_output(error, dt)
            rospy.loginfo("Control output: %.3f", control_output)

            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
