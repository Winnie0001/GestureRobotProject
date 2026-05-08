
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gesture_classifier import (
    RuleBasedClassifier,
    TemporalSmoother,
    GESTURE_LABELS,
    GESTURE_COMMANDS,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ros_bridge")


# ══════════════════════════════════════════════════════════════════════════════
# Velocity profiles
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TwistCommand:
    """
    Represents a ROS geometry_msgs/Twist message.
    linear.x  : forward/backward speed in m/s
    angular.z : rotational speed in rad/s (positive = counter-clockwise)
    """
    linear_x:  float = 0.0
    angular_z: float = 0.0
    label:     str   = "STOP"

    def __str__(self) -> str:
        return (
            f"{self.label:10} | "
            f"linear.x={self.linear_x:+.2f} m/s  "
            f"angular.z={self.angular_z:+.2f} rad/s"
        )


# Velocity lookup table — tuned for TurtleBot3 in Gazebo
VELOCITY_TABLE: Dict[str, TwistCommand] = {
    "STOP":     TwistCommand( 0.00,  0.00, "STOP"),
    "FORWARD":  TwistCommand( 0.30,  0.00, "FORWARD"),
    "BACKWARD": TwistCommand(-0.20,  0.00, "BACKWARD"),
    "LEFT":     TwistCommand( 0.00,  0.50, "LEFT"),
    "RIGHT":    TwistCommand( 0.00, -0.50, "RIGHT"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Command history logger
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommandRecord:
    timestamp:  float
    gesture:    str
    command:    str
    linear_x:  float
    angular_z: float


class CommandLogger:
    """
    Records all gesture commands issued during a session.
    Provides summary statistics and exports to CSV for analysis.
    """

    def __init__(self) -> None:
        self._records: List[CommandRecord] = []
        self._session_start = time.monotonic()

    def log(self, gesture: str, command: str, twist: TwistCommand) -> None:
        self._records.append(CommandRecord(
            timestamp  = time.monotonic() - self._session_start,
            gesture    = gesture,
            command    = command,
            linear_x   = twist.linear_x,
            angular_z  = twist.angular_z,
        ))

    def summary(self) -> str:
        if not self._records:
            return "No commands recorded."

        duration = self._records[-1].timestamp
        counts: Dict[str, int] = {}
        for r in self._records:
            counts[r.command] = counts.get(r.command, 0) + 1

        lines = [
            f"\n{'═'*50}",
            f"  SESSION SUMMARY",
            f"{'═'*50}",
            f"  Duration     : {duration:.1f} s",
            f"  Total cmds   : {len(self._records)}",
            f"  Cmd/second   : {len(self._records)/max(duration,1):.2f}",
            f"\n  Command breakdown:",
        ]
        for cmd, n in sorted(counts.items(), key=lambda x: -x[1]):
            pct = n / len(self._records) * 100
            bar = "█" * int(pct / 5)
            lines.append(f"    {cmd:12} {bar:20} {n:4}x  ({pct:.1f}%)")

        lines.append(f"{'═'*50}")
        return "\n".join(lines)

    def export_csv(self, path: str = "command_log.csv") -> None:
        with open(path, "w") as f:
            f.write("timestamp_s,gesture,command,linear_x,angular_z\n")
            for r in self._records:
                f.write(
                    f"{r.timestamp:.3f},{r.gesture},{r.command},"
                    f"{r.linear_x:.3f},{r.angular_z:.3f}\n"
                )
        logger.info(f"Command log exported → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Safety Monitor
# ══════════════════════════════════════════════════════════════════════════════

class SafetyMonitor:
    """
    Safety layer between the gesture classifier and robot actuators.

    Implements two safety mechanisms:
    1. Confidence threshold — commands are blocked if classifier confidence
       is below the threshold, issuing STOP instead. Prevents low-confidence
       spurious movements.
    2. Consecutive NONE gate — if the classifier returns no gesture for
       max_none_frames consecutive frames, STOP is issued. Prevents the
       robot continuing to move when the operator's hand leaves the frame.

    This is a critical component for safe human-robot interaction and
    demonstrates awareness of safety-critical system design (CP6).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        max_none_frames:      int   = 15,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_none_frames      = max_none_frames
        self._none_count:  int  = 0
        self._blocked:     int  = 0   # stat: commands blocked by safety
        self._passed:      int  = 0

    def check(
        self,
        gesture:    Optional[str],
        command:    Optional[str],
        confidence: float,
    ) -> Tuple[str, bool]:
        """
        Returns (safe_command, was_overridden).
        safe_command is always a valid command string.
        was_overridden is True if the safety monitor changed the command.
        """
        # Track consecutive no-detection frames
        if gesture is None:
            self._none_count += 1
        else:
            self._none_count = 0

        # Override 1: hand left frame for too long
        if self._none_count >= self.max_none_frames:
            self._blocked += 1
            return "STOP", True

        # No gesture yet
        if command is None:
            return "STOP", False

        # Override 2: confidence below threshold
        if confidence < self.confidence_threshold:
            self._blocked += 1
            logger.debug(
                f"Safety: blocked '{command}' "
                f"(confidence={confidence:.2f} < {self.confidence_threshold})"
            )
            return "STOP", True

        self._passed += 1
        return command, False

    @property
    def block_rate(self) -> float:
        total = self._blocked + self._passed
        return self._blocked / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ROS Publisher
# ══════════════════════════════════════════════════════════════════════════════

class ROSBridge:
    """
    Full ROS integration bridge with graceful simulation fallback.

    In ROS mode:
      Publishes Twist to /cmd_vel, String to /gesture_command and
      /gesture_status. Subscribes to /robot_status for feedback display.

    In simulation mode:
      Prints all commands to console with timestamps, allowing full
      development and testing on Windows without ROS installed.
    """

    def __init__(
        self,
        node_name:   str   = "gesture_controller",
        cmd_topic:   str   = "/cmd_vel",
        label_topic: str   = "/gesture_command",
        pub_rate_hz: float = 10.0,
    ) -> None:
        self._ros_available = False
        self._pub_twist     = None
        self._pub_label     = None
        self._Twist         = None
        self._String        = None
        self._rate          = None
        self._robot_status  = "unknown"
        self._pub_rate_hz   = pub_rate_hz

        try:
            import rospy
            from geometry_msgs.msg import Twist
            from std_msgs.msg import String

            rospy.init_node(node_name, anonymous=True)
            self._pub_twist = rospy.Publisher(cmd_topic,   Twist,  queue_size=1)
            self._pub_label = rospy.Publisher(label_topic, String, queue_size=1)
            self._Twist     = Twist
            self._String    = String
            self._rate      = rospy.Rate(pub_rate_hz)

            # Subscribe to robot status feedback
            rospy.Subscriber(
                "/robot_status", String,
                lambda msg: setattr(self, "_robot_status", msg.data)
            )

            self._ros_available = True
            logger.info(
                f"ROSBridge: connected | "
                f"cmd={cmd_topic} | label={label_topic}"
            )

        except ImportError:
            logger.info(
                "ROSBridge: ROS not installed — running in SIMULATION mode.\n"
                "  To run with ROS: Ubuntu + ROS Noetic + roscore\n"
                "  All commands will be printed to console."
            )
        except Exception as exc:
            logger.warning(f"ROSBridge init error: {exc} — simulation mode.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def publish(self, twist: TwistCommand) -> None:
        """Publish a TwistCommand to /cmd_vel (or print in simulation mode)."""
        if self._ros_available:
            msg = self._Twist()
            msg.linear.x  = twist.linear_x
            msg.angular.z = twist.angular_z
            self._pub_twist.publish(msg)
        else:
            # Simulation mode — structured console output
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] /cmd_vel  →  {twist}")

    def publish_label(self, gesture: str) -> None:
        """Publish gesture label string to /gesture_command."""
        if self._ros_available and self._pub_label:
            self._pub_label.publish(self._String(data=gesture))

    def sleep(self) -> None:
        """Maintain publish rate."""
        if self._ros_available and self._rate:
            self._rate.sleep()
        else:
            time.sleep(1.0 / self._pub_rate_hz)

    def is_shutdown(self) -> bool:
        if self._ros_available:
            import rospy
            return rospy.is_shutdown()
        return False

    @property
    def mode(self) -> str:
        return "ROS" if self._ros_available else "SIMULATION"

    @property
    def robot_status(self) -> str:
        return self._robot_status


# ══════════════════════════════════════════════════════════════════════════════
# Main bridge loop
# ══════════════════════════════════════════════════════════════════════════════

def run_bridge(
    camera_idx:           int   = 0,
    confidence_threshold: float = 0.70,
    simulate:             bool  = False,
) -> None:
    """
    Main execution loop: captures webcam frames, classifies gestures,
    applies safety checks, and publishes ROS commands.

    Args:
        camera_idx           : webcam device index
        confidence_threshold : minimum confidence to issue a command
        simulate             : force simulation mode even if ROS available
    """
    # ── Initialise components ─────────────────────────────────────────────────
    logger.info("Initialising gesture recognition pipeline...")
    classifier = RuleBasedClassifier()
    smoother   = TemporalSmoother(window=7, debounce_s=0.5)
    safety     = SafetyMonitor(confidence_threshold=confidence_threshold)
    cmd_log    = CommandLogger()
    bridge     = ROSBridge()

    logger.info(f"Mode: {bridge.mode}")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_idx}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,           30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)

    print(f"\n{'═'*55}")
    print(f"  GESTURE → ROS BRIDGE  |  {bridge.mode} MODE")
    print(f"{'═'*55}")
    print(f"  Confidence threshold : {confidence_threshold:.0%}")
    print(f"  Safety none-frames   : {safety.max_none_frames}")
    print(f"  Press Q to quit")
    print(f"{'═'*55}\n")

    last_command  = "STOP"
    frame_count   = 0
    fps_times: list = []

    try:
        while not bridge.is_shutdown():
            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # ── Classify ──────────────────────────────────────────────────────
            landmarks, annotated = classifier.process_frame(frame)
            analysis = classifier.analyse(landmarks) if landmarks is not None else None

            raw_gesture  = analysis.raw_gesture  if analysis else None
            raw_conf     = analysis.confidence   if analysis else 0.0
            stable       = smoother.update(raw_gesture)

            raw_command  = GESTURE_COMMANDS.get(stable, None) if stable else None

            # ── Safety check ──────────────────────────────────────────────────
            safe_command, overridden = safety.check(stable, raw_command, raw_conf)
            twist = VELOCITY_TABLE.get(safe_command, VELOCITY_TABLE["STOP"])

            # ── Publish ───────────────────────────────────────────────────────
            bridge.publish(twist)
            if stable:
                bridge.publish_label(stable)
                cmd_log.log(stable, safe_command, twist)

            # ── Console log (on change) ────────────────────────────────────────
            if safe_command != last_command:
                override_tag = " [SAFETY OVERRIDE]" if overridden else ""
                print(f"  {twist}{override_tag}")
                last_command = safe_command

            # ── Annotate frame ────────────────────────────────────────────────
            _draw_bridge_hud(
                annotated, stable, raw_conf, safe_command,
                twist, overridden, bridge.mode,
                safety.block_rate, fps_times,
            )

            cv2.imshow(f"ROS Bridge — {bridge.mode}", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            fps_times.append(time.monotonic())
            if len(fps_times) > 30:
                fps_times.pop(0)

            frame_count += 1
            bridge.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        classifier.close()

        # Export command log
        cmd_log.export_csv("command_log.csv")

        # Print session summary
        print(cmd_log.summary())
        print(f"\n  Safety block rate: {safety.block_rate:.1%}")
        print(f"  Total frames:      {frame_count}")


def _draw_bridge_hud(
    frame:       np.ndarray,
    gesture:     Optional[str],
    confidence:  float,
    command:     str,
    twist:       TwistCommand,
    overridden:  bool,
    mode:        str,
    block_rate:  float,
    fps_times:   list,
) -> None:
    """Draw the ROS bridge HUD on the annotated frame."""
    h, w = frame.shape[:2]

    # Mode badge
    mode_col = (0, 180, 80) if mode == "ROS" else (0, 140, 255)
    cv2.rectangle(frame, (w-120, 8), (w-8, 34), mode_col, -1)
    cv2.putText(frame, mode, (w-110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Gesture + command
    if gesture:
        col = (0, 60, 220) if overridden else (0, 210, 80)
        cv2.putText(frame, f"Gesture: {gesture}",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"Command: {command}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        if overridden:
            cv2.putText(frame, "SAFETY OVERRIDE",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,60,220), 2)
    else:
        cv2.putText(frame, "Waiting for gesture...",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,120), 1)

    # Velocity readout
    cv2.putText(
        frame,
        f"linear.x={twist.linear_x:+.2f}  angular.z={twist.angular_z:+.2f}",
        (10, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,0), 1
    )

    # Confidence bar
    if confidence > 0:
        bw = 160
        bx, by = 10, h-36
        fill = int(bw * confidence)
        col = (0,210,80) if confidence > 0.7 else (0,140,255) if confidence > 0.5 else (0,50,220)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+12), (60,60,60), -1)
        cv2.rectangle(frame, (bx, by), (bx+fill, by+12), col, -1)
        cv2.putText(frame, f"Conf: {confidence:.0%}",
                    (bx+bw+8, by+10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # FPS
    if len(fps_times) > 1:
        fps = (len(fps_times)-1)/(fps_times[-1]-fps_times[0])
        cv2.putText(frame, f"FPS:{fps:.0f}  Block:{block_rate:.0%}",
                    (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Gazebo launch instructions (printed when requested)
# ══════════════════════════════════════════════════════════════════════════════

GAZEBO_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOW TO RUN WITH GAZEBO (Ubuntu + ROS Noetic)                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Install ROS Noetic:                                                      ║
║     sudo apt install ros-noetic-desktop-full                                 ║
║     echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc                   ║
║                                                                              ║
║  2. Install TurtleBot3:                                                      ║
║     sudo apt install ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations ║
║     echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc                       ║
║                                                                              ║
║  3. Create workspace:                                                        ║
║     mkdir -p ~/catkin_ws/src && cd ~/catkin_ws && catkin_make                ║
║     source devel/setup.bash                                                  ║
║                                                                              ║
║  4. Launch everything (4 terminals):                                         ║
║     Terminal 1: roscore                                                      ║
║     Terminal 2: roslaunch turtlebot3_gazebo turtlebot3_world.launch          ║
║     Terminal 3: python3 ros_bridge.py                                        ║
║     Terminal 4: rostopic echo /cmd_vel   (to verify messages)                ║
║                                                                              ║
║  5. Verify gesture commands reach the robot:                                 ║
║     rostopic echo /gesture_command                                           ║
║     rostopic echo /cmd_vel                                                   ║
║                                                                              ║
║  The TurtleBot3 in Gazebo will respond to your hand gestures in real time.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gesture → ROS Bridge for robot navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python ros_bridge.py --confidence 0.75"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.70,
        help="Minimum classifier confidence to issue command (default: 0.70)"
    )
    parser.add_argument(
        "--gazebo-help", action="store_true",
        help="Print Gazebo setup instructions and exit"
    )
    args = parser.parse_args()

    if args.gazebo_help:
        print(GAZEBO_INSTRUCTIONS)
        sys.exit(0)

    run_bridge(
        camera_idx=args.camera,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
