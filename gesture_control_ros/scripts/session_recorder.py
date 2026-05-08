#!/usr/bin/env python3

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


@dataclass
class Event:
    t_s: float
    topic: str
    data: Dict[str, Any]


class SessionRecorder:
    def __init__(self) -> None:
        self._t0 = time.time()
        self._events: List[Event] = []

        self.output_path = rospy.get_param('~output', '')
        self.max_events = int(rospy.get_param('~max_events', 20000))

        if not self.output_path:
            ts = time.strftime('%Y%m%d_%H%M%S')
            self.output_path = f'gesture_session_{ts}.json'

        self._last_gesture: Optional[str] = None
        self._last_command: Optional[str] = None
        self._last_nav_status: Optional[str] = None

        rospy.Subscriber('/gesture/detected', String, self._on_gesture)
        rospy.Subscriber('/gesture/command', String, self._on_command)
        rospy.Subscriber('/gesture/nav_status', String, self._on_nav_status)
        rospy.Subscriber('/cmd_vel', Twist, self._on_cmd_vel)
        rospy.Subscriber('/odom', Odometry, self._on_odom)

        rospy.on_shutdown(self._flush)
        rospy.loginfo(f"[Recorder] Logging to {self.output_path}")

    def _now_rel(self) -> float:
        return time.time() - self._t0

    def _append(self, topic: str, data: Dict[str, Any]) -> None:
        if len(self._events) >= self.max_events:
            return
        self._events.append(Event(t_s=self._now_rel(), topic=topic, data=data))

    def _on_gesture(self, msg: String) -> None:
        if msg.data == self._last_gesture:
            return
        self._last_gesture = msg.data
        self._append('/gesture/detected', {'gesture': msg.data})

    def _on_command(self, msg: String) -> None:
        if msg.data == self._last_command:
            return
        self._last_command = msg.data
        self._append('/gesture/command', {'command': msg.data})

    def _on_nav_status(self, msg: String) -> None:
        if msg.data == self._last_nav_status:
            return
        self._last_nav_status = msg.data
        self._append('/gesture/nav_status', {'status': msg.data})

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._append('/cmd_vel', {
            'linear_x': float(msg.linear.x),
            'linear_y': float(msg.linear.y),
            'linear_z': float(msg.linear.z),
            'angular_x': float(msg.angular.x),
            'angular_y': float(msg.angular.y),
            'angular_z': float(msg.angular.z),
        })

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._append('/odom', {
            'x': float(p.x),
            'y': float(p.y),
            'z': float(p.z),
            'qx': float(o.x),
            'qy': float(o.y),
            'qz': float(o.z),
            'qw': float(o.w),
        })

    def _flush(self) -> None:
        try:
            payload = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'duration_s': round(self._now_rel(), 3),
                'event_count': len(self._events),
                'events': [asdict(e) for e in self._events],
            }
            with open(self.output_path, 'w') as f:
                json.dump(payload, f, indent=2)
            rospy.loginfo(f"[Recorder] Saved {len(self._events)} events -> {self.output_path}")
        except Exception as exc:
            rospy.logwarn(f"[Recorder] Failed to save session: {exc}")


def main() -> None:
    rospy.init_node('gesture_session_recorder', anonymous=True)
    SessionRecorder()
    rospy.spin()


if __name__ == '__main__':
    main()
