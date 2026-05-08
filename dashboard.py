
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gesture_classifier import (
    RuleBasedClassifier,
    TemporalSmoother,
    GESTURE_LABELS,
    GESTURE_COMMANDS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Colour palette (BGR)
# ══════════════════════════════════════════════════════════════════════════════

# One colour per gesture — consistent across bars and labels
GESTURE_COLOURS = {
    "closed_fist": (0,   60, 220),   # red    → STOP
    "open_hand":   (0,  210,  80),   # green  → FORWARD
    "thumbs_up":   (0,  170, 240),   # amber  → LEFT
    "peace_sign":  (200, 200,  0),   # cyan   → RIGHT
    "pointing":    (200,  80, 200),  # purple → BACKWARD
}

_DARK  = (25,  25,  25)
_PANEL = (35,  35,  35)
_WHITE = (240, 240, 240)
_GREY  = (120, 120, 120)
_GREEN = (0,   210,  80)
_RED   = (0,    60, 220)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard renderer
# ══════════════════════════════════════════════════════════════════════════════

class Dashboard:
    """
    Real-time analytics dashboard rendered as an OpenCV window.

    Displays:
    - Per-gesture confidence bar chart (probability distribution)
    - Rolling FPS line graph
    - Session statistics (dominant gesture, uptime, total detections)
    - Command history strip
    """

    DASH_W = 680
    DASH_H = 500
    FPS_HISTORY = 90     # frames to keep in FPS graph
    CMD_HISTORY = 8      # recent commands to show in strip

    def __init__(self) -> None:
        # Confidence smoothing — exponential moving average per gesture
        self._conf_ema: Dict[str, float] = {g: 0.0 for g in GESTURE_LABELS}
        self._ema_alpha = 0.25           # smoothing factor

        # FPS tracking
        self._fps_buffer: deque = deque(maxlen=self.FPS_HISTORY)
        self._frame_times: deque = deque(maxlen=self.FPS_HISTORY)

        # Session stats
        self._session_start = time.monotonic()
        self._gesture_counts: Dict[str, int] = {g: 0 for g in GESTURE_LABELS}
        self._total_detections = 0
        self._cmd_history: deque = deque(maxlen=self.CMD_HISTORY)

        # Canvas
        self._canvas = np.zeros((self.DASH_H, self.DASH_W, 3), dtype=np.uint8)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        gesture:     Optional[str],
        confidences: Dict[str, float],
        fps:         float,
    ) -> np.ndarray:
        """
        Update dashboard state and return rendered frame.

        Args:
            gesture     : current stable gesture label or None
            confidences : per-gesture confidence dict {label: float}
            fps         : current FPS reading

        Returns:
            BGR image of shape (DASH_H, DASH_W, 3)
        """
        # Update EMA confidences
        for g in GESTURE_LABELS:
            raw = confidences.get(g, 0.0)
            self._conf_ema[g] = (
                self._ema_alpha * raw +
                (1 - self._ema_alpha) * self._conf_ema[g]
            )

        # Update stats
        self._fps_buffer.append(fps)
        self._frame_times.append(time.monotonic())

        if gesture:
            self._gesture_counts[gesture] += 1
            self._total_detections += 1
            cmd = GESTURE_COMMANDS.get(gesture, "")
            if not self._cmd_history or self._cmd_history[-1] != cmd:
                self._cmd_history.append(cmd)

        # Render
        self._canvas[:] = _DARK
        self._draw_confidence_bars()
        self._draw_fps_graph()
        self._draw_session_stats(gesture)
        self._draw_command_strip()
        self._draw_title()

        return self._canvas.copy()

    # ── Private: rendering ─────────────────────────────────────────────────────

    def _draw_title(self) -> None:
        cv2.putText(
            self._canvas,
            "BA-25-1058  |  Real-Time Gesture Analytics",
            (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREY, 1
        )
        cv2.line(self._canvas, (0, 32), (self.DASH_W, 32), _PANEL, 1)

    def _draw_confidence_bars(self) -> None:
        """Left panel: horizontal bar chart of per-gesture confidence."""
        x0, y0   = 16, 50
        bar_h    = 32
        bar_gap  = 12
        max_w    = 290
        label_w  = 110

        # Panel background
        panel_h = len(GESTURE_LABELS) * (bar_h + bar_gap) + 20
        cv2.rectangle(
            self._canvas,
            (x0-8, y0-8), (x0 + label_w + max_w + 60, y0 + panel_h),
            _PANEL, -1
        )

        cv2.putText(
            self._canvas, "Gesture confidence",
            (x0, y0+4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GREY, 1
        )

        for i, gesture in enumerate(GESTURE_LABELS):
            y = y0 + 22 + i * (bar_h + bar_gap)
            conf = self._conf_ema[gesture]
            colour = GESTURE_COLOURS[gesture]

            # Label
            cv2.putText(
                self._canvas,
                gesture.replace("_", " "),
                (x0, y + bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1
            )

            # Background track
            bx = x0 + label_w
            cv2.rectangle(
                self._canvas,
                (bx, y), (bx + max_w, y + bar_h),
                (55, 55, 55), -1
            )

            # Filled bar
            fill_w = int(max_w * conf)
            if fill_w > 0:
                cv2.rectangle(
                    self._canvas,
                    (bx, y), (bx + fill_w, y + bar_h),
                    colour, -1
                )

            # Percentage label
            pct_text = f"{conf*100:4.1f}%"
            cv2.putText(
                self._canvas, pct_text,
                (bx + max_w + 8, y + bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour if conf > 0.1 else _GREY, 1
            )

            # Command badge on active gesture
            cmd = GESTURE_COMMANDS.get(gesture, "")
            if conf > 0.5:
                badge_x = bx + fill_w - 60
                if badge_x > bx + 5:
                    cv2.putText(
                        self._canvas, cmd,
                        (badge_x, y + bar_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2
                    )
                    cv2.putText(
                        self._canvas, cmd,
                        (badge_x, y + bar_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _WHITE, 1
                    )

    def _draw_fps_graph(self) -> None:
        """Right panel: rolling FPS line graph."""
        gx, gy = 440, 50
        gw, gh = 224, 200

        # Background
        cv2.rectangle(
            self._canvas, (gx-8, gy-8), (gx+gw+8, gy+gh+30),
            _PANEL, -1
        )
        cv2.putText(
            self._canvas, "FPS (last 90 frames)",
            (gx, gy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1
        )

        # Grid lines at 10, 20, 30 FPS
        for fps_line in [10, 20, 30]:
            ly = gy + gh - int(gh * fps_line / 40)
            cv2.line(self._canvas, (gx, ly), (gx+gw, ly), (55,55,55), 1)
            cv2.putText(
                self._canvas, str(fps_line),
                (gx-26, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, _GREY, 1
            )

        # FPS line
        fps_vals = list(self._fps_buffer)
        if len(fps_vals) > 1:
            pts = []
            for i, f in enumerate(fps_vals):
                px = gx + int(i * gw / max(len(fps_vals)-1, 1))
                py = gy + gh - int(gh * min(f, 40) / 40)
                pts.append((px, py))

            for i in range(len(pts) - 1):
                col = _GREEN if fps_vals[i] > 20 else (0, 140, 255)
                cv2.line(self._canvas, pts[i], pts[i+1], col, 2, cv2.LINE_AA)

        # Current FPS
        cur_fps = fps_vals[-1] if fps_vals else 0
        cv2.putText(
            self._canvas, f"{cur_fps:.1f} FPS",
            (gx, gy + gh + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            _GREEN if cur_fps > 20 else (0, 140, 255), 1
        )

    def _draw_session_stats(self, gesture: Optional[str]) -> None:
        """Bottom-left: session statistics."""
        sx, sy = 16, 310
        uptime = time.monotonic() - self._session_start

        cv2.putText(
            self._canvas, "Session stats",
            (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GREY, 1
        )

        # Dominant gesture
        if self._total_detections > 0:
            dominant = max(self._gesture_counts, key=self._gesture_counts.get)
            dom_pct  = self._gesture_counts[dominant] / self._total_detections * 100
        else:
            dominant, dom_pct = "—", 0

        stats = [
            f"Uptime          : {uptime:6.1f} s",
            f"Total detected  : {self._total_detections:6d}",
            f"Dominant gesture: {dominant} ({dom_pct:.0f}%)",
        ]
        for i, line in enumerate(stats):
            cv2.putText(
                self._canvas, line,
                (sx, sy + 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, _WHITE, 1
            )

        # Per-gesture count mini-bar chart
        by = sy + 100
        bw_max = 200
        max_count = max(self._gesture_counts.values()) or 1
        for i, g in enumerate(GESTURE_LABELS):
            n    = self._gesture_counts[g]
            bw   = int(bw_max * n / max_count)
            col  = GESTURE_COLOURS[g]
            bar_y = by + i * 22
            cv2.rectangle(
                self._canvas,
                (sx + 115, bar_y),
                (sx + 115 + bw_max, bar_y + 16),
                (50, 50, 50), -1
            )
            if bw > 0:
                cv2.rectangle(
                    self._canvas,
                    (sx + 115, bar_y),
                    (sx + 115 + bw, bar_y + 16),
                    col, -1
                )
            cv2.putText(
                self._canvas,
                f"{g.replace('_',' ')[:12]:12} {n:4}",
                (sx, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col if n > 0 else _GREY, 1
            )

    def _draw_command_strip(self) -> None:
        """Bottom-right: recent command history strip."""
        cx, cy = 440, 310
        cv2.putText(
            self._canvas, "Command history",
            (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GREY, 1
        )

        cmds = list(self._cmd_history)
        for i, cmd in enumerate(cmds):
            alpha = 0.3 + 0.7 * (i / max(len(cmds)-1, 1))
            col_raw = _GREEN if cmd != "STOP" else _RED
            col = tuple(int(c * alpha) for c in col_raw)
            cv2.putText(
                self._canvas, cmd,
                (cx, cy + 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1
            )

        if not cmds:
            cv2.putText(
                self._canvas, "No commands yet",
                (cx, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _GREY, 1
            )


# ══════════════════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════════════════

def run(camera_idx: int = 0) -> None:
    """Run the full system with dashboard window."""
    classifier = RuleBasedClassifier()
    smoother   = TemporalSmoother(window=7, debounce_s=0.8)
    dashboard  = Dashboard()

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,           30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)

    print("\n[Dashboard] Running — press Q to quit")
    print("  Two windows: webcam feed + analytics dashboard\n")

    fps_times: deque = deque(maxlen=30)
    frame_count = 0

    try:
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            landmarks, annotated = classifier.process_frame(frame)
            analysis = classifier.analyse(landmarks) if landmarks is not None else None

            raw_gesture = analysis.raw_gesture if analysis else None
            raw_conf    = analysis.confidence  if analysis else 0.0
            stable      = smoother.update(raw_gesture)

            # Build per-gesture confidence dict
            # When a gesture is detected, assign its confidence; others get 0
            confidences = {g: 0.0 for g in GESTURE_LABELS}
            if stable and raw_conf > 0:
                confidences[stable] = raw_conf

            # FPS
            fps_times.append(time.monotonic())
            fps = (
                (len(fps_times)-1) / (fps_times[-1] - fps_times[0])
                if len(fps_times) > 1 else 0
            )

            # Show gesture on webcam window
            if stable:
                cmd = GESTURE_COMMANDS.get(stable, "")
                cv2.putText(annotated, f"{stable}  →  {cmd}",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0,210,80), 2)

            # Update and show dashboard
            dash_frame = dashboard.update(stable, confidences, fps)

            cv2.imshow("Webcam — BA-25-1058", annotated)
            cv2.imshow("Analytics Dashboard", dash_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        classifier.close()
        print(f"\n[Dashboard] Session ended — {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture analytics dashboard")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(camera_idx=args.camera)


if __name__ == "__main__":
    main()
