"""
Real-time gesture inference with webcam demo and mouse control.
"""

import cv2
import numpy as np
import time
import argparse
from gesture_classifier import RuleBasedClassifier, CNNClassifier, TemporalSmoother, GESTURE_COMMANDS
import pyautogui

class GestureControlDemo:
    """Real-time gesture recognition with mouse control."""

    def __init__(self, mode: str = "rule", model_path: str = None):
        """
        Args:
            mode: "rule" for rule-based, "cnn" for neural network
            model_path: path to trained CNN model (for mode="cnn")
        """
        self.mode = mode
        self.classifier = None

        if mode == "rule":
            self.classifier = RuleBasedClassifier()
        elif mode == "cnn":
            self.classifier = CNNClassifier(model_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.smoother = TemporalSmoother(window=7, debounce_s=0.8)

        # Mouse control parameters
        self.screen_width, self.screen_height = pyautogui.size()
        self.move_speed = 15  # pixels per frame
        self.last_command = None

        # Frame capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Performance tracking
        self.fps_history = []
        self.frame_count = 0

    def _draw_hud(self, frame, gesture: str = None, confidence: float = 0.0,
                  latency_ms: float = 0.0):
        """Draw heads-up display on frame."""
        h, w = frame.shape[:2]

        # Top-left: gesture info
        y_offset = 30
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if gesture:
            cmd = GESTURE_COMMANDS.get(gesture, "???")
            cv2.putText(frame, f"Gesture: {gesture}", (10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Command: {cmd}", (10, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # Confidence bar
            bar_width = 200
            bar_x, bar_y = 10, y_offset + 95
            bar_fill = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                         (200, 200, 200), 2)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 20),
                         (0, 255, 0), -1)
            cv2.putText(frame, f"{confidence:.2f}", (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Bottom-right: FPS + latency
        avg_fps = np.mean(self.fps_history[-30:]) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 200, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (w - 200, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        return frame

    def _apply_command(self, command: str):
        """Execute mouse control command."""
        curr_x, curr_y = pyautogui.position()

        if command == "FORWARD":
            pyautogui.moveTo(curr_x, curr_y - self.move_speed, duration=0.05)
        elif command == "BACKWARD":
            pyautogui.moveTo(curr_x, curr_y + self.move_speed, duration=0.05)
        elif command == "LEFT":
            pyautogui.moveTo(curr_x - self.move_speed, curr_y, duration=0.05)
        elif command == "RIGHT":
            pyautogui.moveTo(curr_x + self.move_speed, curr_y, duration=0.05)
        elif command == "STOP":
            pass  # Pause movement

        self.last_command = command

    def run(self):
        """Main loop."""
        print(f"\n[START] Gesture Control Demo ({self.mode.upper()} mode)")
        print("Press 'Q' to quit, SPACE to toggle mouse control\n")

        mouse_enabled = False

        try:
            while True:
                t_start = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)  # Mirror for intuitive control

                # Inference
                if self.mode == "rule":
                    features, annotated = self.classifier.process_frame(frame)
                    gesture = self.classifier.classify(features)
                    confidence = 0.95 if gesture else 0.0
                else:
                    try:
                        gesture, confidence = self.classifier.predict(frame)
                    except:
                        gesture, confidence = None, 0.0
                    annotated = frame

                # Temporal smoothing
                stable_gesture = self.smoother.update(gesture)

                # Apply command if smoothed gesture ready
                if stable_gesture and mouse_enabled:
                    cmd = GESTURE_COMMANDS.get(stable_gesture, "???")
                    self._apply_command(cmd)
                    stable_gesture_display = stable_gesture
                else:
                    stable_gesture_display = stable_gesture

                # Draw HUD
                latency_ms = (time.time() - t_start) * 1000
                display = self._draw_hud(annotated, stable_gesture_display, confidence, latency_ms)

                # Show instruction
                status = "ON" if mouse_enabled else "OFF"
                cv2.putText(display, f"Mouse Control: {status} (SPACE to toggle)",
                           (10, display.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

                cv2.imshow("Gesture Control", display)

                # Performance tracking
                self.fps_history.append(1.0 / (time.time() - t_start + 1e-6))
                self.frame_count += 1

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    mouse_enabled = not mouse_enabled
                    status = "ENABLED" if mouse_enabled else "DISABLED"
                    print(f"\n>>> Mouse control {status}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.mode == "rule":
            self.classifier.close()
        print(f"\n[END] Processed {self.frame_count} frames")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled mouse demo")
    parser.add_argument("--mode", choices=["rule", "cnn"], default="rule",
                       help="Classification mode: rule-based or CNN")
    parser.add_argument("--model", type=str, help="Path to trained CNN model")
    args = parser.parse_args()

    demo = GestureControlDemo(mode=args.mode, model_path=args.model)
    demo.run()


if __name__ == "__main__":
    main()
