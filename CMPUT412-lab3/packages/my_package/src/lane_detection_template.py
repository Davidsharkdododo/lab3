#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import Header, ColorRGBA
from duckietown.dtros import DTROS, NodeType

Wheel_rad = 0.0318

def compute_distance(ticks):
    rotations = ticks / 135
    return 2 * 3.1415 * Wheel_rad * rotations

def compute_ticks(distance):
    rotations = distance / (2 * 3.1415 * Wheel_rad)
    return rotations * 135

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"  
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"      
        
        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        self.hsv_ranges = {
            "blue": (np.array([100, 80, 130]), np.array([140, 255, 255])),
            "red":  (np.array([0, 70, 150]),   np.array([10, 255, 255])),
            "green": (np.array([50, 40, 100]),  np.array([80, 255, 255]))
        }
        
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=1)
        
        self.led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self._led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        self._ticks_left = 0
        self._ticks_right = 0

        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        self.publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        
        self.rate = rospy.Rate(3)
        
        # Initialize a queue to store detected lane colors
        self.q = []
        # Flag to prevent overlapping maneuvers
        self.maneuver_active = False

    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def compute_distance_homography(self, u, v):
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def detect_lane_color(self, image):
        """
        Processes the image and returns the annotated image, the detected color, and
        the computed distance from the detected lane (using the largest valid contour).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_color = None
        detected_distance = None
        output = image.copy()
        for color, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    u = x + w / 2
                    v = y + h
                    distance = self.compute_distance_homography(u, v)
                    cv2.putText(output, f"{color}: {distance:.2f}m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    detected_color = color
                    detected_distance = distance
        return output, detected_color, detected_distance

    def set_led_color(self, r, g, b):
        num_leds = 5
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()
        default_color = ColorRGBA(r=1, g=0, b=0, a=1)
        leds = [default_color for _ in range(num_leds)]
        left_indices = [0, 3]
        right_indices = [1, 4]
        if r == 1:
            for i in range(num_leds):
                leds[i] = ColorRGBA(r=1, g=0, b=0, a=1)
        if g == 1:
            for i in left_indices:
                leds[i] = ColorRGBA(r=0, g=1, b=0, a=1)
        if b == 1:
            for i in right_indices:
                leds[i] = ColorRGBA(r=0, g=0, b=1, a=1)
        pattern_msg.rgb_vals = leds
        self._led_pub.publish(pattern_msg)
    
    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def dynamic_motor_control(self, left_power, right_power, distance_left, distance_right):
        """
        Drives the robot until the left and right wheels have traveled the respective distances.
        """
        rate = rospy.Rate(100)
        msg = WheelsCmdStamped()
        msg.vel_left = left_power
        msg.vel_right = right_power

        init_ticks_left = self._ticks_left
        init_ticks_right = self._ticks_right

        while (not rospy.is_shutdown()) and \
              (abs(compute_distance(self._ticks_left - init_ticks_left)) < abs(distance_left)) and \
              (abs(compute_distance(self._ticks_right - init_ticks_right)) < abs(distance_right)):

            # Optional: adjust speeds based on error between wheel distances.
            dist_ratio = distance_left / distance_right if distance_right != 0 else 1
            left_dist = self._ticks_left - init_ticks_left
            right_dist = self._ticks_right - init_ticks_right
            diff = abs(left_dist) - abs(right_dist) * dist_ratio
            modifier = 50 * diff / 1000  # Tune as needed.

            msg.vel_left = left_power * (1 - modifier)
            msg.vel_right = right_power * (1 + modifier)
            msg.header.stamp = rospy.Time.now()
            self.publisher.publish(msg)
            rate.sleep()

        # Stop the robot.
        stop_msg = WheelsCmdStamped()
        stop_msg.header = Header()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.vel_left = 0
        stop_msg.vel_right = 0
        self.publisher.publish(stop_msg)

    def callback(self, msg):
        # If a maneuver is already active, skip processing further messages.
        if self.maneuver_active:
            return

        # Convert the incoming message to an OpenCV image.
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Undistort and preprocess the image.
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        # Detect lane colors and obtain lane distance.
        color_image, detected_color, lane_distance = self.detect_lane_color(preprocessed)
        # Publish the undistorted and annotated images.
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="bgr8")
        self.pub_undistorted.publish(undistorted_msg)
        lane_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        self.pub_lane.publish(lane_msg)

        # If a color lane is detected and within 20cm, add the color to the queue
        if detected_color is not None and lane_distance < 0.20:
            rospy.loginfo("Detected %s lane at %.3f m", detected_color, lane_distance)
            self.q.append(detected_color)
            self.maneuver_active = True  # Prevent further detections until maneuver is executed.
            return

    def run(self):
        """
        Main loop: if a lane color is queued, execute the corresponding maneuver.
        Otherwise, keep moving forward.
        """
        while not rospy.is_shutdown():
            if self.q:
                # Pop the first detected color from the queue.
                color = self.q.pop(0)
                # Send a stop command.
                stop_msg = WheelsCmdStamped()
                stop_msg.header.stamp = rospy.Time.now()
                stop_msg.vel_left = 0
                stop_msg.vel_right = 0
                self.publisher.publish(stop_msg)
                
                # Wait for a random duration between 3 and 5 seconds.
                rospy.sleep(3)
                rospy.loginfo("Executing maneuver for %s lane", color)
                if color == "blue":
                    # Blue lane maneuver: turn right.
                    self.set_led_color(0, 0, 1)
                    self.dynamic_motor_control(0.586, 0.4, 0.534, 0.377)
                    self.set_led_color(1, 0, 0)
                elif color == "red":
                    # Red lane maneuver: drive straight for at least 30 cm.
                    self.set_led_color(1, 0, 0)
                    self.dynamic_motor_control(0.5, 0.5, 0.3, 0.3)
                elif color == "green":
                    # Green lane maneuver: turn left.
                    self.set_led_color(0, 1, 0)
                    self.dynamic_motor_control(0.4, 0.586, 0.377, 0.534)
                    self.set_led_color(1, 0, 0)
                # Reset the flag to allow new detections.
                self.maneuver_active = False
            else:
                # If no lane color is queued, keep moving forward.
                self.dynamic_motor_control(0.5, 0.5, 0.05, 0.05)
            self.rate.sleep()

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
