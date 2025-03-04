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
        self.Kp = -0.25
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
            # "white": (np.array([0, 0, 216]), np.array([179, 55, 255]))
            "white": (np.array([0, 0, 216*0.85]), np.array([179*1.1, 55*1.2, 255]))
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
    


        
    def detect_lane_with_edges(self, image):
        """
        Detect white lane lines using Canny edge detection and linear estimation.
        Returns the same format as detect_lane_color but uses edge detection.
        """
        # First, undistort the image
        undistorted = self.undistort_image(image)
        
        # Create a copy for visualization
        output = image.copy()
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for white color (similar to original function)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges["white"][0], self.hsv_ranges["white"][1])
        
        # Apply the mask to the grayscale image
        masked_gray = cv2.bitwise_and(gray, gray, mask=white_mask)
        
        # Apply Canny edge detection
        edges = cv2.Canny(masked_gray, 50, 150)
        
        # Define region of interest - focus on the bottom half of the image
        height, width = edges.shape
        roi_mask = np.zeros_like(edges)
        roi_vertices = np.array([[(0, height), (0, height/2), 
                                (width, height/2), (width, height)]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Find line segments using probabilistic Hough transform
        line_segments = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, 
                                    minLineLength=20, maxLineGap=5)
        
        detections = {}
        error = None
        
        # If lines were found
        if line_segments is not None and len(line_segments) > 0:
            # Calculate average slope and intercept
            slopes = []
            intercepts = []
            
            for line in line_segments:
                x1, y1, x2, y2 = line[0]
                # Skip horizontal lines (avoid division by zero)
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Only consider slopes that might be lane lines (filter out horizontal lines)
                    if abs(slope) > 0.3:
                        slopes.append(slope)
                        intercepts.append(intercept)
                        # Draw the line segment
                        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if slopes:
                # Calculate average slope and intercept
                avg_slope = np.mean(max(slopes, 3))
                avg_intercept = np.mean(intercepts)
                
                # Calculate points for the average line
                y1 = height  # Bottom of the image
                y2 = int(height * 0.6)  # 60% up the image
                x1 = int((y1 - avg_intercept) / avg_slope)
                x2 = int((y2 - avg_intercept) / avg_slope)
                
                # Draw the average line
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Calculate the nearest point on the line (use bottom of image)
                nearest_point = np.array([x1, y1])
                
                # Apply homography to get distance (similar to original function)
                point_homog = np.array([nearest_point[0], nearest_point[1], 1])
                ground_point = self.homography @ point_homog
                ground_point /= ground_point[2]  # Normalize
                distance = np.sqrt(ground_point[0]**2 + ground_point[1]**2)
                
                # Store detection similarly to original function
                detections["white"] = (nearest_point[0], distance)
                
                # Annotate the image
                cv2.circle(output, tuple(nearest_point), 5, (0, 255, 0), -1)
                cv2.putText(
                    output,
                    f"white: {distance:.2f}m, slope: {avg_slope:.2f}",
                    (nearest_point[0] + 10, nearest_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Calculate error based on slope
                # Adjust target_slope as needed - will tune this later as mentioned
                target_slope = 0  # Example value, adjust as needed
                error = avg_slope - target_slope


        lane_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        self.pub_lane.publish(lane_msg)
        
        # Return detections, error, and the visualization image
        return output, detections, error

    def detect_lane_fast(self, image):
        """
        Optimized fast lane detection using fixed-height scanning.
        Scans for white pixels at fixed heights in the bottom half of the image,
        checking only every third pixel for speed.
        """
        # Convert to HSV and get white mask - only process bottom half
        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        # Convert to HSV and create mask only for bottom half
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_ranges["white"][0], self.hsv_ranges["white"][1])
        
        # Get dimensions and center
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        
        # Create a simple output image (minimal drawing for speed)
        output = image.copy()
        bottom_half_out = output[half_height_start:full_height, :]
        
        # Base weights (will be redistributed if points are missing)
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        offsets = []
        valid_detections = []
        
        # Loop through each scan height
        for i, y in enumerate(scan_heights):
            # Scan from center to right edge every 3 pixels for speed
            for x in range(center_x, width, 3):
                # If we found a white pixel
                if white_mask[y, x] > 0:
                    offset = x - center_x
                    offsets.append(offset)
                    valid_detections.append(i)
                    
                    # Minimal visualization - just circles at detection points
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
                    break
        
        # Calculate error - redistribute weights for missing points
        error = None
        if offsets:
            # Find the bottom-most detection (should be the most reliable)
            bottom_idx = valid_detections.index(max([i for i in valid_detections]))
            bottom_offset = offsets[bottom_idx]
            
            # Filter out detections that are too far from the bottom detection
            # (likely background noise or other lane markers)
            max_deviation = 10  # maximum allowed pixel deviation from bottom point
            
            filtered_offsets = []
            filtered_detections = []
            
            for i, offset in enumerate(offsets):
                if offset - bottom_offset <= max_deviation:
                    filtered_offsets.append(offset)
                    filtered_detections.append(valid_detections[i])
                else:
                    # Draw rejected points in red
                    detection_idx = valid_detections[i]
                    y = scan_heights[detection_idx]
                    x = center_x + offset
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 0, 255), -1)
            
            # Use the filtered detections
            if filtered_offsets:
                # Create new weights based on which points were detected
                total_weight = sum([base_weights[i] for i in filtered_detections])
                normalized_weights = [base_weights[i] / total_weight for i in filtered_detections]
                
                # Calculate weighted sum of offsets
                weighted_offsets = [filtered_offsets[i] * normalized_weights[i] for i in range(len(filtered_offsets))]
                error_raw = sum(weighted_offsets)
                
                # Scale the error
                error = error_raw * 0.01 -1.2
            
            # Minimal text display
            if len(valid_detections) < 3:
                # If we have very few points, make text red as a warning
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                
            cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output, error


    
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
        output, error = self.detect_lane_fast(preprocessed)
        # height, width = colour_image.shape[:2]
        # cropped_image = colour_image[height // 2:height, :]
        # lane_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding="bgr8")
        # self.pub_lane.publish(lane_msg)
        
        # Publish the visualization image
        lane_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        self.pub_lane.publish(lane_msg)


        if error is not None:
            # detection_str = ", ".join([f"{color}: ({data[0]:.1f}, {data[1]:.2f}m)" for color, data in detections.items()])
            rospy.loginfo("Control error: %.3f", error)
            control_output = self.get_control_output(error, dt)
            # rospy.loginfo("Control output: %.3f", control_output)

            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
