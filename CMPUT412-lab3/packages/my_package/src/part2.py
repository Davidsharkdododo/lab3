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
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Get control parameters from ROS parameters.
        self.controller_type = "P"
        self.Kp = 0.5
        self.Ki = 0.025
        self.Kd = 0.05
        self.base_speed = 0.4

        # self.controller_type = rospy.get_param("~controller_type", "P")
        # self.Kp = rospy.get_param("~Kp", 0.8)
        # self.Ki = rospy.get_param("~Ki", 0.025)
        # self.Kd = rospy.get_param("~Kd", 0.05)
        # self.base_speed = rospy.get_param("~base_speed", 1.0)
        
        # Initialize control variables.
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters.s
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                    [0.0, 322.6864063251382, 215.88480909087127],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                    -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                    dtype=np.float32)
        
        # Homography matrix (maps image coordinates to ground coordinates).
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # Color detection parameters (HSV ranges).
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, 216]), np.array([179, 55, 255]))
        }
        
        # Initialize CvBridge.
        self.bridge = CvBridge()
        
        # Create publishers.
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        
        # Now that all variables are initialized, subscribe to the camera feed.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(2)
        

    def undistort_image(self, image):
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return undistorted

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def detect_lane_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        output = image.copy()
        image_center = image.shape[1] / 2.0
        detections = {}  # Will store detection for each color as (u, distance)

        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            nearest_point = None
            nearest_distance = None

            for cnt in contours:
                if cv2.contourArea(cnt) > 500:  # Filter small contours
                    # Reshape contour points for easier processing.
                    cnt_points = cnt.reshape(-1, 2)
                    # Create homogeneous coordinates for all contour points.
                    ones = np.ones((cnt_points.shape[0], 1))
                    points_homog = np.hstack([cnt_points, ones])
                    # Apply the homography to transform to ground coordinates.
                    ground_points = (self.homography @ points_homog.T).T
                    ground_points /= ground_points[:, 2][:, np.newaxis]  # Normalize

                    # Compute the Euclidean distance from the origin.
                    distances = np.sqrt(ground_points[:, 0]**2 + ground_points[:, 1]**2)
                    # Find the index of the point with the minimum distance.
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    
                    # Only consider this contour if its nearest point is within 20 cm.
                    if min_distance < 0.4:
                        # If this is the first valid point or it's closer than previous ones, update.
                        if nearest_distance is None or min_distance < nearest_distance:
                            nearest_distance = min_distance
                            nearest_point = cnt_points[min_idx]
            
            if nearest_point is not None:
                # Draw a circle on the nearest edge of the lane.
                cv2.circle(output, tuple(nearest_point), 5, (0, 255, 0), -1)
                cv2.putText(
                    output,
                    f"{color}: {nearest_distance:.2f}m",
                    (nearest_point[0] + 10, nearest_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                detections[color] = (nearest_point[0], nearest_distance)

        # For error computation you might, for example, use the horizontal coordinate difference.
        error = None
        if "white" in detections and "yellow" in detections:
            white_u, white_distance = detections["white"]
            yellow_u, yellow_distance = detections["yellow"]
            error = yellow_distance - white_distance
        elif "yellow" in detections:
            yellow_u, yellow_distance = detections["yellow"]
            error = yellow_distance - 0.09
        elif "white" in detections:
            white_u, white_distance = detections["white"]
            error = 20*(white_distance-0.09)

        return output, detections, error
        

    
    def calculate_p_control(self, error, dt):
        return self.Kp * error

    def calculate_pd_control(self, error, dt):
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        rospy.loginfo(derivative)
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
        if control_output >0.3: control_output = min(control_output, 0.3) # Scale control output appropriately.
        if control_output <-0.3: control_output = max(control_output, -0.3) # Scale control output appropriately.
        left_speed = self.base_speed - control_output
        right_speed = self.base_speed + control_output
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
        # rospy.loginfo("Published wheel cmd: left=%.3f, right=%.3f", left_speed, right_speed)
    
    def callback(self, msg):
        current_time = rospy.Time.now()
        if self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
        else:
            dt = 0.1  # default dt value for the first callback
        self.last_time = current_time
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(cv_image)
        colour_image, detections, error = self.detect_lane_color(preprocessed)
        # height, width = colour_image.shape[:2]
        # cropped_image = colour_image[height // 2:height, :]
        # lane_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding="bgr8")
        # # self.pub_lane.publish(lane_msg)
        if error is not None:
            detection_str = ", ".join([f"{color}: ({data[0]:.1f}, {data[1]:.2f}m)" for color, data in detections.items()])
            rospy.loginfo("Control error: %.3f, %s", error, detection_str)
            control_output = self.get_control_output(error, dt)
            # rospy.loginfo("Control output: %.3f", control_output)

            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
