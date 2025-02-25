#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import Header, ColorRGBA

Wheel_rad = 0.0318

def compute_distance(ticks):
    rotations = ticks/135
    return 2 * 3.1415 * Wheel_rad * rotations

def compute_ticks(distance):
    rotations = distance / (2 * 3.1415 * Wheel_rad)
    return rotations * 135


class WheelControlNode(DTROS):
    def __init__(self, node_name):
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"
        self._left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"

        self._ticks_left = 0
        self._ticks_right = 0


        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)

        self._led_pub = rospy.Publisher(self.led_topic,LEDPattern, queue_size=1)


    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def set_led_color(self, r, g, b):
        num_leds = 5  # Most Duckiebots have 5 LEDs. Adjust if yours differs!

        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()

        color_msg = ColorRGBA()
        color_msg.r = r
        color_msg.g = g
        color_msg.b = b
        color_msg.a = 1
        pattern_msg.rgb_vals = [color_msg] * num_leds
        # Publish
        self._led_pub.publish(pattern_msg)

    def dynamic_motor_control(self, left_power, right_power, distance_left, distance_right):
        rate = rospy.Rate(100)
        msg = WheelsCmdStamped()
        msg.vel_left = left_power
        msg.vel_right = right_power

        init_ticks_left = self._ticks_left
        init_ticks_right = self._ticks_right

        while (not rospy.is_shutdown()) and \
        (abs(compute_distance(self._ticks_left - init_ticks_left)) < abs(distance_left)) and \
        (abs(compute_distance(self._ticks_right - init_ticks_right)) < abs(distance_right)):

            dist_ratio = distance_left/distance_right

            left_dist = self._ticks_left - init_ticks_left
            right_dist = self._ticks_right - init_ticks_right
            
            diff = abs(left_dist) - abs(right_dist)*dist_ratio
            modifier = 0 * diff/1000

            msg.vel_left = left_power * (1 - modifier)
            msg.vel_right = right_power * (1 + modifier)


            msg.header.stamp = rospy.Time.now()
            self._publisher.publish(msg)
            rate.sleep()

        stop_msg = WheelsCmdStamped()
        stop_msg.header = Header()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.vel_left = 0
        stop_msg.vel_right = 0
        self._publisher.publish(stop_msg)



    def run(self):
        #-----------------
        #part 2
        #-----------------
        rospy.sleep(1)
        self.dynamic_motor_control(0.5, 0.5, 1.25, 1.25)
        rospy.sleep(1)
        self.dynamic_motor_control(-0.5, -0.5, 0.9, 0.9)
        rospy.sleep(1)
        self.dynamic_motor_control(0.8, -0.8, 0.1, 0.1) # 90 deg right rotation
        rospy.sleep(1)
        self.dynamic_motor_control(-0.8, 0.8, 0.1, 0.1) # 90 deg left rotation


        #-----------------
        #part 3
        #-----------------
        # rospy.sleep(1)
        # self.set_led_color(0, 1, 0)
        # rospy.sleep(5)
        # self.set_led_color(1, 0, 0)
        # self.dynamic_motor_control(0.5, 0.5, 1.12, 1.12)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.6, -0.6, 0.061, 0.061)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.5, 0.5, 0.85, 0.85)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.586, 0.4, 0.534, 0.377)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.5, 0.5, 0.57, 0.57)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.586, 0.4, 0.534, 0.377)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.5, 0.5, 0.85, 0.85)
        # rospy.sleep(1)
        # self.dynamic_motor_control(0.7, -0.7, 0.07, 0.07)
        # self.set_led_color(0, 1, 0)
        # rospy.sleep(5)

if __name__ == '__main__':
    node = WheelControlNode(node_name='wheel_control_node')
    node.run()
    rospy.spin()
