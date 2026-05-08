#!/usr/bin/env python3
"""
Gesture Publisher ROS Node
Publishes detected gestures and robot commands to ROS topics.

Author: Mmesoma Kenneth (202307951)
Project: AI-Powered Gesture Control for Robot Navigation (BA-25-1058)
"""

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import sys

from gesture_control_ros.gesture_classifier import (
    CNNClassifier,
    GESTURE_COMMANDS,
    RuleBasedClassifier,
    TemporalSmoother,
)
import cv2


class GesturePublisherNode:
    """
    ROS node that captures webcam input, detects gestures,
    and publishes them to ROS topics.
    """

    def __init__(self, mode='rule', model_path=None, rate_hz=10):
        """
        Initialize the gesture publisher node.

        Args:
            mode: 'rule' or 'cnn' classification mode
            model_path: path to trained CNN model (if mode='cnn')
            rate_hz: publishing rate in Hz
        """
        rospy.init_node('gesture_publisher', anonymous=False)

        # Allow roslaunch parameters to override constructor defaults
        mode = rospy.get_param('~mode', mode)
        model_path = rospy.get_param('~model_path', model_path)
        rate_hz = rospy.get_param('~rate', rate_hz)

        # Initialize classifier
        if mode == 'rule':
            rospy.loginfo("Using rule-based classifier")
            self.classifier = RuleBasedClassifier()
        elif mode == 'cnn':
            rospy.loginfo(f"Using CNN classifier: {model_path}")
            self.classifier = CNNClassifier(model_path)
        else:
            rospy.logerr(f"Unknown mode: {mode}")
            sys.exit(1)

        # Initialize smoother for temporal filtering
        self.smoother = TemporalSmoother(window=10, debounce_s=0.5)

        # Safety: if we lose stable gestures for a while, force STOP
        self._no_stable_frames = 0
        self._max_no_stable_frames = rospy.get_param('~max_no_stable_frames', 8)

        # ROS Publishers
        self.gesture_pub = rospy.Publisher('/gesture/detected', String, queue_size=10)
        self.command_pub = rospy.Publisher('/gesture/command', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Publishing rate
        self.rate = rospy.Rate(rate_hz)

        # Stats
        self.frame_count = 0
        self.last_gesture = None

        # Velocity parameters (ROS params)
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)

        rospy.loginfo("Gesture Publisher Node initialized")
        rospy.loginfo(f"Publishing at {rate_hz} Hz")
        rospy.loginfo("Topics: /gesture/detected, /gesture/command, /cmd_vel")

    def gesture_to_twist(self, command: str) -> Twist:
        """
        Convert gesture command to Twist message for robot control.

        Args:
            command: Robot command (FORWARD, BACKWARD, LEFT, RIGHT, STOP)

        Returns:
            Twist message with linear and angular velocities
        """
        twist = Twist()

        if command == "FORWARD":
            twist.linear.x = self.linear_speed
        elif command == "BACKWARD":
            twist.linear.x = -self.linear_speed
        elif command == "LEFT":
            twist.angular.z = self.angular_speed
        elif command == "RIGHT":
            twist.angular.z = -self.angular_speed
        elif command == "STOP":
            # All zeros (default)
            pass

        return twist

    def process_frame(self):
        """
        Capture and process one frame from webcam.
        Detect gesture and publish to ROS topics.
        """
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to read frame from webcam")
            return False

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect gesture
        if hasattr(self.classifier, 'process_frame'):
            # Rule-based classifier
            features, annotated = self.classifier.process_frame(frame)
            gesture = self.classifier.classify(features)
            _ = 0.95 if gesture else 0.0
        else:
            # CNN classifier
            gesture, confidence, _ = self.classifier.predict(frame)
            annotated = frame

        # Apply temporal smoothing
        stable_gesture = self.smoother.update(gesture)

        # Publish if we have a stable gesture
        if stable_gesture:
            self._no_stable_frames = 0
            command = GESTURE_COMMANDS.get(stable_gesture, "STOP")

            # Publish gesture name
            self.gesture_pub.publish(String(data=stable_gesture))

            # Publish command
            self.command_pub.publish(String(data=command))

            # Publish Twist for robot control
            twist = self.gesture_to_twist(command)
            self.cmd_vel_pub.publish(twist)

            # Log if gesture changed
            if stable_gesture != self.last_gesture:
                rospy.loginfo(f"Gesture: {stable_gesture} -> Command: {command}")
                self.last_gesture = stable_gesture
        else:
            self._no_stable_frames += 1
            if self._no_stable_frames >= self._max_no_stable_frames:
                # Force stop to prevent stale /cmd_vel when hand leaves frame
                self.cmd_vel_pub.publish(Twist())
                self._no_stable_frames = 0

        self.frame_count += 1
        return True

    def run(self):
        """Main loop - capture frames and publish gestures."""
        rospy.loginfo("Starting gesture publisher loop...")
        rospy.loginfo("Press Ctrl+C to stop")

        try:
            while not rospy.is_shutdown():
                success = self.process_frame()
                if not success:
                    rospy.logwarn("Frame processing failed, retrying...")

                self.rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("Shutting down gesture publisher...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        if hasattr(self.classifier, 'close'):
            self.classifier.close()

        # Send stop command
        self.cmd_vel_pub.publish(Twist())

        rospy.loginfo(f"Processed {self.frame_count} frames")
        rospy.loginfo("Gesture publisher shutdown complete")


def main():
    """Main function."""
    import argparse
    import rospy

    parser = argparse.ArgumentParser(description="Gesture Publisher ROS Node")
    parser.add_argument('--mode', choices=['rule', 'cnn'], default='rule',
                        help="Classification mode")
    parser.add_argument('--model', type=str, default='models/gesture_model.h5',
                        help="Path to trained CNN model")
    parser.add_argument('--rate', type=int, default=10,
                        help="Publishing rate in Hz")
    parser.add_argument('--max-no-stable-frames', type=int, default=None,
                        help="Force STOP after this many frames with no stable gesture")
    ros_argv = rospy.myargv(argv=sys.argv)
    args, _unknown = parser.parse_known_args(ros_argv[1:])

    try:
        if args.max_no_stable_frames is not None:
            rospy.set_param('~max_no_stable_frames', int(args.max_no_stable_frames))
        node = GesturePublisherNode(
            mode=args.mode,
            model_path=args.model if args.mode == 'cnn' else None,
            rate_hz=args.rate
        )
        node.run()

    except Exception as e:
        rospy.logerr(f"Error in gesture publisher: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
