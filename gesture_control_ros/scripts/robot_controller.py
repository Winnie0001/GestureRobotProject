#!/usr/bin/env python3
"""
Robot Controller ROS Node
Subscribes to gesture commands and controls robot movement.

Author: Mmesoma Kenneth (202307951)
Project: AI-Powered Gesture Control for Robot Navigation (BA-25-1058)
"""

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class RobotControllerNode:
    """
    ROS node that subscribes to gesture commands and
    controls robot movement in Gazebo simulation.
    """

    def __init__(self):
        """Initialize robot controller node."""
        rospy.init_node('robot_controller', anonymous=False)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.nav_status_pub = rospy.Publisher('/gesture/nav_status', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('/gesture/command', String, self.command_callback)
        rospy.Subscriber('/gesture/detected', String, self.gesture_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Robot state
        self.current_position = None
        self.current_orientation = None
        self.current_command = "STOP"
        self.last_gesture = None

        # Control mode
        # teleop: gesture command -> /cmd_vel
        # nav:    gesture command -> move_base goal (waypoints)
        self.control_mode = rospy.get_param('~control_mode', 'teleop')

        # move_base (nav mode)
        self._mb_client = None
        self._last_goal_id = None
        self._last_goal_status = None

        # Control parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)  # m/s
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)  # rad/s

        # Waypoints (nav mode)
        # You can tune these per Gazebo world. Defaults are safe placeholders.
        self.wp1_x = rospy.get_param('~wp1_x', 1.0)
        self.wp1_y = rospy.get_param('~wp1_y', 0.0)
        self.wp1_yaw = rospy.get_param('~wp1_yaw', 0.0)
        self.wp2_x = rospy.get_param('~wp2_x', 0.0)
        self.wp2_y = rospy.get_param('~wp2_y', 1.0)
        self.wp2_yaw = rospy.get_param('~wp2_yaw', 1.57)
        self.wp3_x = rospy.get_param('~wp3_x', -1.0)
        self.wp3_y = rospy.get_param('~wp3_y', 0.0)
        self.wp3_yaw = rospy.get_param('~wp3_yaw', 3.14)

        # Safety parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s

        # Statistics
        self.command_count = 0
        self.total_distance = 0.0

        rospy.loginfo("Robot Controller Node initialized")
        rospy.loginfo(f"Control mode: {self.control_mode}")
        rospy.loginfo(f"Linear speed: {self.linear_speed} m/s")
        rospy.loginfo(f"Angular speed: {self.angular_speed} rad/s")

        if self.control_mode == 'nav':
            self._init_move_base()

    def _init_move_base(self):
        self._mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not self._mb_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logwarn("move_base action server not available (timeout). Nav mode may not work.")
        else:
            rospy.loginfo("Connected to move_base action server")

    def _send_waypoint_goal(self, x: float, y: float, yaw: float, goal_id: str) -> None:
        if self._mb_client is None:
            self._init_move_base()
        if self._mb_client is None:
            return

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        goal.target_pose.pose.orientation = _yaw_to_quat(float(yaw))

        self._mb_client.send_goal(
            goal,
            done_cb=lambda status, result: self._on_goal_done(goal_id, status, result),
            active_cb=lambda: self._on_goal_active(goal_id),
            feedback_cb=lambda feedback: self._on_goal_feedback(goal_id, feedback),
        )
        self._last_goal_id = goal_id
        self._publish_nav_status(f"SENT:{goal_id}")
        rospy.loginfo(f"[NAV] Sent goal '{goal_id}' -> x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def _cancel_nav_goals(self) -> None:
        if self._mb_client is not None:
            self._mb_client.cancel_all_goals()
            rospy.loginfo("[NAV] Canceled all goals")
            self._publish_nav_status("CANCELED")

    def _publish_nav_status(self, status: str) -> None:
        try:
            self.nav_status_pub.publish(String(data=status))
        except Exception:
            pass

    def _on_goal_active(self, goal_id: str) -> None:
        self._publish_nav_status(f"ACTIVE:{goal_id}")
        rospy.loginfo(f"[NAV] Goal active: {goal_id}")

    def _on_goal_feedback(self, goal_id: str, feedback) -> None:
        # Keep feedback lightweight to avoid spamming logs.
        if self._last_goal_status != f"FEEDBACK:{goal_id}":
            self._last_goal_status = f"FEEDBACK:{goal_id}"
            self._publish_nav_status(self._last_goal_status)

    def _on_goal_done(self, goal_id: str, status: int, result) -> None:
        status_map = {
            0: "PENDING",
            1: "ACTIVE",
            2: "PREEMPTED",
            3: "SUCCEEDED",
            4: "ABORTED",
            5: "REJECTED",
            6: "PREEMPTING",
            7: "RECALLING",
            8: "RECALLED",
            9: "LOST",
        }
        label = status_map.get(int(status), str(status))
        self._publish_nav_status(f"DONE:{goal_id}:{label}")
        rospy.loginfo(f"[NAV] Goal done: {goal_id} -> {label}")

    def gesture_callback(self, msg):
        """
        Callback for detected gestures.

        Args:
            msg: String message containing gesture name
        """
        gesture = msg.data
        if gesture != self.last_gesture:
            rospy.loginfo(f"Detected gesture: {gesture}")
            self.last_gesture = gesture

    def command_callback(self, msg):
        """
        Callback for gesture commands.
        Publishes corresponding Twist message to control robot.

        Args:
            msg: String message containing command (FORWARD, BACKWARD, LEFT, RIGHT, STOP)
        """
        command = msg.data

        # NAV mode: map commands to waypoints and use move_base
        if self.control_mode == 'nav':
            if command == 'FORWARD':
                self._send_waypoint_goal(self.wp1_x, self.wp1_y, self.wp1_yaw, goal_id='WP1')
            elif command == 'LEFT':
                self._send_waypoint_goal(self.wp2_x, self.wp2_y, self.wp2_yaw, goal_id='WP2')
            elif command == 'RIGHT':
                self._send_waypoint_goal(self.wp3_x, self.wp3_y, self.wp3_yaw, goal_id='WP3')
            elif command == 'BACKWARD':
                # Simple semantic: treat BACKWARD as "cancel + stop" in nav mode
                self._cancel_nav_goals()
                self.cmd_vel_pub.publish(Twist())
            elif command == 'STOP':
                self._cancel_nav_goals()
                self.cmd_vel_pub.publish(Twist())
            else:
                rospy.logwarn(f"Unknown command: {command}")
                return

            if command != self.current_command:
                rospy.loginfo(f"Command: {self.current_command} → {command}")
                self.current_command = command
                self.command_count += 1
            return

        # Create Twist message
        twist = Twist()

        if command == "FORWARD":
            twist.linear.x = self.linear_speed
            rospy.logdebug("Moving forward")
        elif command == "BACKWARD":
            twist.linear.x = -self.linear_speed
            rospy.logdebug("Moving backward")
        elif command == "LEFT":
            twist.angular.z = self.angular_speed
            rospy.logdebug("Turning left")
        elif command == "RIGHT":
            twist.angular.z = -self.angular_speed
            rospy.logdebug("Turning right")
        elif command == "STOP":
            # All zeros (default)
            rospy.logdebug("Stopping")
        else:
            rospy.logwarn(f"Unknown command: {command}")
            return

        # Safety checks
        twist.linear.x = max(min(twist.linear.x, self.max_linear_speed), -self.max_linear_speed)
        twist.angular.z = max(min(twist.angular.z, self.max_angular_speed), -self.max_angular_speed)

        # Publish command
        self.cmd_vel_pub.publish(twist)

        # Update stats
        if command != self.current_command:
            rospy.loginfo(f"Command: {self.current_command} → {command}")
            self.current_command = command
            self.command_count += 1

    def odom_callback(self, msg):
        """
        Callback for odometry data (robot position/orientation).

        Args:
            msg: Odometry message
        """
        # Extract position
        position = msg.pose.pose.position
        new_position = (position.x, position.y, position.z)

        # Calculate distance traveled
        if self.current_position is not None:
            dx = new_position[0] - self.current_position[0]
            dy = new_position[1] - self.current_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            self.total_distance += distance

        self.current_position = new_position

        # Extract orientation (quaternion -> euler)
        orientation = msg.pose.pose.orientation
        self.current_orientation = orientation

    def emergency_stop(self):
        """Emergency stop - publish zero velocities."""
        rospy.logwarn("Emergency stop!")
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)

    def print_stats(self):
        """Print robot controller statistics."""
        rospy.loginfo("="*60)
        rospy.loginfo("ROBOT CONTROLLER STATISTICS")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Total commands received: {self.command_count}")
        rospy.loginfo(f"Total distance traveled: {self.total_distance:.2f} m")
        if self.current_position:
            rospy.loginfo(f"Current position: ({self.current_position[0]:.2f}, "
                         f"{self.current_position[1]:.2f}, {self.current_position[2]:.2f})")
        rospy.loginfo("="*60)

    def run(self):
        """Main loop - keep node alive."""
        rospy.loginfo("Robot controller running...")
        rospy.loginfo("Listening for gesture commands on /gesture/command")
        rospy.loginfo("Press Ctrl+C to stop")

        # Print stats every 30 seconds
        stats_timer = rospy.Timer(rospy.Duration(30), lambda event: self.print_stats())

        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("Shutting down robot controller...")
        finally:
            self.emergency_stop()
            self.print_stats()


def main():
    """Main function."""
    try:
        node = RobotControllerNode()
        node.run()
    except Exception as e:
        rospy.logerr(f"Error in robot controller: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
