
import argparse
import sys
import time
import numpy as np
import cv2
import pyautogui
import socket

from gesture_classifier import (
    RuleBasedClassifier, CNNClassifier,
    TemporalSmoother, GESTURE_COMMANDS,
)

# ── PyAutoGUI safety ──────────────────────────────────────────────────────────
# Disable the corner fail-safe — we use screen-edge clamping instead,
# so the program never crashes when the mouse reaches a screen corner.
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0


class GestureControlDemo:
    """Real-time gesture recognition with mouse control."""

    MOVE_SPEED = 15   # pixels per command tick

    def __init__(
        self,
        mode: str = "rule",
        model_path: str = None,
        camera_index: int = 0,
        tcp_host: str = None,
        tcp_port: int = None,
        smooth_window: int = 7,
        debounce_s: float = 0.8,
        stop_after_no_stable_frames: int = 6,
        stop_grace_s: float = 0.0,
        tcp_log_every_s: float = 3.0,
    ):
        self.mode = mode

        if mode == "rule":
            self.classifier = RuleBasedClassifier()
        elif mode == "cnn":
            if model_path is None:
                print("[ERROR] --model is required for CNN mode.")
                sys.exit(1)
            self.classifier = CNNClassifier(model_path=model_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.smoother = TemporalSmoother(window=int(smooth_window), debounce_s=float(debounce_s))

        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._tcp_sock = None
        self._last_sent_cmd = None
        self._no_stable_frames = 0
        self._stop_after_no_stable_frames = int(stop_after_no_stable_frames)
        self._stop_grace_s = float(stop_grace_s)
        self._last_motion_t = 0.0
        self._tcp_log_every_s = float(tcp_log_every_s)
        self._last_tcp_log_t = 0.0

        # Screen size for boundary clamping
        self.screen_w, self.screen_h = pyautogui.size()

        # Webcam setup
        self.cap = cv2.VideoCapture(int(camera_index))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,           30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)

        self.fps_history = []
        self.frame_count = 0

    def _tcp_connect_if_needed(self) -> None:
        if not self._tcp_host or not self._tcp_port:
            return
        if self._tcp_sock is not None:
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self._tcp_host, int(self._tcp_port)))
            s.settimeout(None)
            self._tcp_sock = s
            print(f"[TCP] Connected to {self._tcp_host}:{self._tcp_port}")
        except Exception:
            now = time.time()
            if now - self._last_tcp_log_t >= self._tcp_log_every_s:
                print(f"[TCP] Waiting for bridge at {self._tcp_host}:{self._tcp_port} ...")
                self._last_tcp_log_t = now
            self._tcp_sock = None

    def _tcp_send_command(self, cmd: str) -> None:
        if not cmd:
            return
        self._tcp_connect_if_needed()
        if self._tcp_sock is None:
            return
        if cmd == self._last_sent_cmd:
            return

        try:
            self._tcp_sock.sendall((cmd + "\n").encode("utf-8"))
            self._last_sent_cmd = cmd
        except Exception:
            try:
                self._tcp_sock.close()
            except Exception:
                pass
            self._tcp_sock = None

    # ── Mouse control ──────────────────────────────────────────────────────────

    def _apply_command(self, command: str) -> None:
        """
        Translate a gesture command into a mouse movement.
        Clamps coordinates to screen bounds with a 5px margin so the
        cursor never reaches a corner (which previously triggered the
        PyAutoGUI FailSafeException).
        """
        if command == "STOP":
            return

        try:
            curr_x, curr_y = pyautogui.position()
        except Exception:
            return

        deltas = {
            "FORWARD":  (0,                -self.MOVE_SPEED),
            "BACKWARD": (0,                +self.MOVE_SPEED),
            "LEFT":     (-self.MOVE_SPEED,  0),
            "RIGHT":    (+self.MOVE_SPEED,  0),
        }
        dx, dy = deltas.get(command, (0, 0))

        # Clamp: 5px margin from every edge
        margin = 5
        new_x  = int(max(margin, min(self.screen_w - margin, curr_x + dx)))
        new_y  = int(max(margin, min(self.screen_h - margin, curr_y + dy)))

        try:
            pyautogui.moveTo(new_x, new_y, duration=0)
        except Exception:
            pass   # absorb any remaining edge-case errors silently

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, gesture=None, confidence=0.0, latency_ms=0.0):
        h, w = frame.shape[:2]
        avg_fps = np.mean(self.fps_history[-30:]) if self.fps_history else 0

        cv2.putText(frame, f"Mode: {self.mode.upper()}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}   Latency: {latency_ms:.1f} ms",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        if gesture:
            cmd = GESTURE_COMMANDS.get(gesture, "")
            cv2.putText(frame, f"Gesture: {gesture}",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Command: {cmd}",
                        (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

            # Confidence bar
            bx, by, bw = 10, 122, 180
            fill = int(bw * confidence)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (70, 70, 70), 2)
            cv2.rectangle(frame, (bx, by), (bx + fill, by + 14), (0, 210, 80), -1)
            cv2.putText(frame, f"{confidence:.0%}", (bx + bw + 8, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 80), 1)

        return frame

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n[START] Gesture Control ({self.mode.upper()} mode)")
        print("  SPACE = toggle mouse   Q = quit   R = reset smoother\n")

        if self._tcp_host and self._tcp_port:
            print(f"  TCP forward: {self._tcp_host}:{self._tcp_port}")

        mouse_enabled = False

        try:
            while True:
                t0 = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)   # mirror

                # Classify
                if self.mode == "rule":
                    landmarks, annotated = self.classifier.process_frame(frame)
                    gesture    = self.classifier.classify(landmarks)
                    confidence = 0.9 if gesture else 0.0
                else:
                    annotated  = frame.copy()
                    gesture, confidence, _ = self.classifier.predict(frame)

                # Smooth
                stable = self.smoother.update(gesture)

                # Forward to ROS bridge (Windows -> WSL)
                if stable:
                    self._no_stable_frames = 0
                    cmd = GESTURE_COMMANDS.get(stable, "STOP")
                    self._tcp_send_command(cmd)

                    if cmd != "STOP":
                        self._last_motion_t = time.time()
                else:
                    self._no_stable_frames += 1
                    if self._no_stable_frames >= self._stop_after_no_stable_frames:
                        if self._stop_grace_s > 0.0:
                            now = time.time()
                            if self._last_motion_t <= 0.0 or (now - self._last_motion_t) >= self._stop_grace_s:
                                self._tcp_send_command("STOP")
                                self._no_stable_frames = 0
                        else:
                            self._tcp_send_command("STOP")
                            self._no_stable_frames = 0

                # Act
                if stable and mouse_enabled:
                    cmd = GESTURE_COMMANDS.get(stable, "STOP")
                    self._apply_command(cmd)

                # Draw HUD
                latency_ms = (time.time() - t0) * 1000
                display    = self._draw_hud(annotated, stable, confidence, latency_ms)

                status_col = (0, 210, 80) if mouse_enabled else (100, 100, 255)
                cv2.putText(
                    display,
                    f"Mouse: {'ON' if mouse_enabled else 'OFF'}  (SPACE toggle) | Q quit | R reset",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 1,
                )

                cv2.imshow("Gesture Control — BA-25-1058", display)

                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    mouse_enabled = not mouse_enabled
                    print(f"  Mouse {'ENABLED' if mouse_enabled else 'DISABLED'}")
                elif key == ord('r'):
                    self.smoother.reset()
                    print("  Smoother reset.")

                self.fps_history.append(1.0 / (time.time() - t0 + 1e-9))
                self.frame_count += 1

        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.mode == "rule":
            self.classifier.close()
        if self._tcp_sock is not None:
            try:
                self._tcp_sock.close()
            except Exception:
                pass
        avg = np.mean(self.fps_history) if self.fps_history else 0
        print(f"\n[END] Processed {self.frame_count} frames")
        print(f"      Average FPS: {avg:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time gesture demo")
    parser.add_argument("--mode",   choices=["rule", "cnn"], default="rule")
    parser.add_argument("--model",  type=str,  default=None)
    parser.add_argument("--camera", type=int,  default=0)
    parser.add_argument("--tcp-host", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--debounce-s", type=float, default=0.8)
    parser.add_argument("--stop-after-no-stable-frames", type=int, default=6)
    parser.add_argument("--stop-grace-s", type=float, default=0.0)
    parser.add_argument("--tcp-log-every-s", type=float, default=3.0)
    args = parser.parse_args()

    demo = GestureControlDemo(
        mode=args.mode,
        model_path=args.model,
        camera_index=args.camera,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        smooth_window=args.smooth_window,
        debounce_s=args.debounce_s,
        stop_after_no_stable_frames=args.stop_after_no_stable_frames,
        stop_grace_s=args.stop_grace_s,
        tcp_log_every_s=args.tcp_log_every_s,
    )
    demo.run()


if __name__ == "__main__":
    main()