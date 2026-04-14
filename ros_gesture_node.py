#!/usr/bin/env python3
"""
ROS Node for Gesture-Based Robot Control
Project: BA-25-1058 | Student: Mmesoma Kenneth (202307951)

Integrates gesture recognition with ROS navigation.
Publishes Twist messages to /cmd_vel based on detected gestures.
"""

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gesture_classifier import RuleBasedClassifier, CNNClassifier, TemporalSmoother, GESTURE_COMMANDS


class GestureROSNode:
    """
    ROS node that converts hand gestures to robot movement commands.
    """

    def __init__(self, mode='rule', model_path=None):
        """
        Initialize ROS node and gesture classifier.

        Args:
            mode: 'rule' for rule-based or 'cnn' for deep learning
            model_path: Path to trained CNN model (if mode='cnn')
        """
        rospy.init_node('gesture_control_node', anonymous=True)

        # Load classifier
        if mode == 'rule':
            self.classifier = RuleBasedClassifier()
            rospy.loginfo("[GestureNode] Using Rule-Based Classifier")
        elif mode == 'cnn':
            if model_path is None:
                rospy.logerr("[GestureNode] CNN mode requires model_path!")
                raise ValueError("model_path required for CNN mode")
            self.classifier = CNNClassifier(model_path)
            rospy.loginfo(f"[GestureNode] Using CNN Classifier: {model_path}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode
        self.smoother = TemporalSmoother(window=7, debounce_s=0.8)

        # ROS Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.gesture_pub = rospy.Publisher('/gesture_detected', String, queue_size=10)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Movement parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.3)  # m/s
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)  # rad/s

        # Control loop rate
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("[GestureNode] Initialization complete!")

    def gesture_to_twist(self, gesture):
        """
        Convert gesture to Twist command.

        Args:
            gesture: Gesture name (e.g., 'open_hand', 'closed_fist')

        Returns:
            Twist message with velocities
        """
        twist = Twist()

        if gesture is None:
            return twist

        command = GESTURE_COMMANDS.get(gesture, None)

        if command == "FORWARD":
            twist.linear.x = self.linear_speed
        elif command == "BACKWARD":
            twist.linear.x = -self.linear_speed
        elif command == "LEFT":
            twist.angular.z = self.angular_speed
        elif command == "RIGHT":
            twist.angular.z = -self.angular_speed
        elif command == "STOP":
            pass  # All zeros (default)

        return twist

    def process_frame(self):
        """
        Capture frame, detect gesture, publish commands.
        """
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("[GestureNode] Failed to read frame")
            return

        # Gesture detection
        if self.mode == 'rule':
            features, annotated = self.classifier.process_frame(frame)
            gesture = self.classifier.classify(features)
        else:  # CNN
            gesture, confidence = self.classifier.predict(frame)
            annotated = frame

        # Temporal smoothing
        stable_gesture = self.smoother.update(gesture)

        if stable_gesture:
            # Publish gesture
            gesture_msg = String()
            gesture_msg.data = stable_gesture
            self.gesture_pub.publish(gesture_msg)

            # Publish robot command
            twist = self.gesture_to_twist(stable_gesture)
            self.cmd_vel_pub.publish(twist)

            command = GESTURE_COMMANDS.get(stable_gesture, "UNKNOWN")
            rospy.loginfo(f"[GestureNode] {stable_gesture} -> {command}")

    def run(self):
        """
        Main control loop.
        """
        rospy.loginfo("[GestureNode] Starting gesture control loop...")

        try:
            while not rospy.is_shutdown():
                self.process_frame()
                self.rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("[GestureNode] Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Release resources.
        """
        self.cap.release()
        if self.mode == 'rule':
            self.classifier.close()

        # Stop robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)

        rospy.loginfo("[GestureNode] Cleanup complete")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Gesture-based ROS robot control")
    parser.add_argument('--mode', choices=['rule', 'cnn'], default='rule',
                       help="Classifier mode: rule-based or CNN")
    parser.add_argument('--model', type=str, help="Path to trained CNN model")
    args = parser.parse_args()

    try:
        node = GestureROSNode(mode=args.mode, model_path=args.model)
        node.run()
    except rospy.ROSInterruptException:
        pass
