#!/usr/bin/env python3

import json
import socket
import threading
from typing import Optional, Tuple

import rospy
from std_msgs.msg import String


_VALID_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (gesture, command). Accepts either raw command or JSON."""
    s = line.strip()
    if not s:
        return None, None

    # Raw command
    upper = s.upper()
    if upper in _VALID_COMMANDS:
        return None, upper

    # JSON
    try:
        obj = json.loads(s)
        gesture = obj.get("gesture")
        cmd = obj.get("command")
        if isinstance(cmd, str):
            cmd_u = cmd.upper()
            if cmd_u in _VALID_COMMANDS:
                return gesture if isinstance(gesture, str) else None, cmd_u
    except Exception:
        pass

    return None, None


class TcpCommandBridge:
    def __init__(self):
        rospy.init_node("tcp_command_bridge", anonymous=False)

        self.host = rospy.get_param("~host", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 5005))
        self.publish_detected = bool(rospy.get_param("~publish_detected", True))

        self.command_pub = rospy.Publisher("/gesture/command", String, queue_size=10)
        self.gesture_pub = rospy.Publisher("/gesture/detected", String, queue_size=10)

        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)

        rospy.loginfo(f"[tcp_command_bridge] Listening on {self.host}:{self.port}")

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

        rospy.loginfo("[tcp_command_bridge] Publishing to /gesture/command (and /gesture/detected)")
        rospy.spin()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not rospy.is_shutdown():
            try:
                conn, addr = self._server.accept()
            except Exception:
                continue

            rospy.loginfo(f"[tcp_command_bridge] Client connected: {addr[0]}:{addr[1]}")
            try:
                self._handle_client(conn)
            except Exception as e:
                rospy.logwarn(f"[tcp_command_bridge] Client error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                rospy.loginfo("[tcp_command_bridge] Client disconnected")

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = b""

        while not rospy.is_shutdown():
            try:
                data = conn.recv(4096)
                if not data:
                    return
                buf += data
            except socket.timeout:
                continue

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    continue

                gesture, cmd = _parse_line(text)
                if cmd is None:
                    continue

                try:
                    self.command_pub.publish(String(data=cmd))
                    if self.publish_detected and gesture:
                        self.gesture_pub.publish(String(data=gesture))
                except Exception:
                    pass


def main() -> None:
    TcpCommandBridge().start()


if __name__ == "__main__":
    main()
