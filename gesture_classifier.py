"""
AI-Powered Gesture Control for Robot Navigation
Project: BA-25-1058 | Student: Mmesoma Kenneth | ID: 202307951

Simplified Gesture Classifier (works with MediaPipe 0.10.33+)
Uses hand contour detection via OpenCV for gesture classification.
"""

import cv2
import numpy as np
import time
from collections import deque, Counter
from typing import Optional, Tuple, List

# ── Constants ─────────────────────────────────────────────────────────────────

GESTURE_LABELS = {
    0: "closed_fist",
    1: "open_hand",
    2: "thumbs_up",
    3: "peace_sign",
    4: "pointing",
}

GESTURE_COMMANDS = {
    "closed_fist":  "STOP",
    "open_hand":    "FORWARD",
    "thumbs_up":    "LEFT",
    "peace_sign":   "RIGHT",
    "pointing":     "BACKWARD",
}


# ── Simplified Rule-Based Classifier ──────────────────────────────────────────

class RuleBasedClassifier:
    """
    Simplified hand gesture classifier using OpenCV contour detection.
    Detects hand shape and estimates gesture.
    """

    def __init__(self):
        # HSV range for skin color (adjust for different skin tones)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def _detect_hand(self, frame_hsv: np.ndarray) -> Optional[np.ndarray]:
        """Detect hand region via skin color."""
        mask = cv2.inRange(frame_hsv, self.lower_skin, self.upper_skin)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _get_contour_features(self, contour) -> dict:
        """Extract features from hand contour."""
        area = cv2.contourArea(contour)
        if area < 1000:
            return None

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Hull points (extrema)
        hull = cv2.convexHull(contour)

        # Approximate contour
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'hull': hull,
            'approx': approx,
            'contour': contour,
        }

    def classify(self, features: Optional[dict]) -> Optional[str]:
        """
        Classify gesture based on hand contour features.
        Simple heuristics:
        - High circularity + small hull points → closed_fist
        - Low circularity + many hull points → open_hand
        - etc.
        """
        if not features:
            return None

        circularity = features.get('circularity', 0)
        hull = features.get('hull', [])
        hull_len = len(hull) if hull is not None else 0

        # Simple classification
        if hull_len < 6:
            return "closed_fist"
        elif hull_len > 12:
            return "open_hand"
        elif hull_len >= 8 and hull_len <= 12:
            # Could be peace sign or thumbs up - use circular heuristic
            if circularity > 0.5:
                return "thumbs_up"
            else:
                return "peace_sign"
        else:
            return "pointing"

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Detect hand and extract features.
        Returns (features_dict, annotated_frame).
        """
        # Convert to HSV for better skin detection
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Detect hand region
        mask = self._detect_hand(frame_hsv)

        annotated = frame_bgr.copy()
        features = None

        if mask is not None:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour (should be hand)
                largest_contour = max(contours, key=cv2.contourArea)
                features = self._get_contour_features(largest_contour)

                if features:
                    # Draw hand region
                    cv2.drawContours(annotated, [features['contour']], 0, (0, 255, 0), 2)
                    cv2.drawContours(annotated, [features['hull']], 0, (255, 0, 0), 2)

        return features, annotated

    def close(self):
        """Cleanup (no resources to release)."""
        pass


# ── CNN Classifier (Placeholder) ──────────────────────────────────────────────

class CNNClassifier:
    """
    Placeholder CNN classifier (MobileNetV2).
    Will be implemented when TensorFlow is available.
    """

    IMG_SIZE = (224, 224)
    NUM_CLASSES = 5

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        print("[CNNClassifier] CNN mode not available (TensorFlow not installed)")
        print("[CNNClassifier] For training, use Python 3.11 or 3.12 with TensorFlow support")

    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        """Dummy predict (returns random gesture)."""
        raise RuntimeError("CNN model not loaded. Use --mode rule instead.")

    def save(self, path: str):
        pass


# ── Temporal Smoother ─────────────────────────────────────────────────────────

class TemporalSmoother:
    """Reduces gesture flickering via majority vote + debounce."""

    def __init__(self, window: int = 7, debounce_s: float = 0.8):
        self.window = window
        self.debounce_s = debounce_s
        self._buffer: deque = deque(maxlen=window)
        self._last_stable: Optional[str] = None
        self._last_trigger_time: float = 0.0

    def update(self, raw_gesture: Optional[str]) -> Optional[str]:
        """Feed raw gesture; returns stable gesture if consensus + debounce met."""
        self._buffer.append(raw_gesture)

        if len(self._buffer) < self.window:
            return None

        counts = Counter(g for g in self._buffer if g is not None)
        if not counts:
            return None

        top_gesture, top_count = counts.most_common(1)[0]
        if top_count < self.window * 0.7:
            return None

        now = time.time()
        if top_gesture == self._last_stable:
            if now - self._last_trigger_time < self.debounce_s:
                return None

        self._last_stable = top_gesture
        self._last_trigger_time = now
        return top_gesture

    def reset(self):
        self._buffer.clear()
        self._last_stable = None
        self._last_trigger_time = 0.0
