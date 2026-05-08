"""
AI-Powered Gesture Control for Robot Navigation
Project: BA-25-1058 | Student: Mmesoma Winnie Kenneth (202307951)
University of Hull | Honours Stage Project

gesture_classifier.py
--------------------
Core classification module using the MediaPipe Tasks API (0.10.30+).

  1. RuleBasedClassifier  - MediaPipe HandLandmarker in VIDEO mode.
                            VIDEO mode gives temporal context across frames,
                            eliminating skeleton flicker and the NORM_RECT
                            warning seen with IMAGE mode.

                            Finger extension uses a dual-joint check:
                              tip.y < pip.y  AND  tip.y < dip.y
                            This prevents misclassification when hand
                            curvature makes tip appear above PIP on a
                            partially curled finger.

                            Thumb IP joint angle is used to distinguish
                            thumbs_up from closed_fist:
                              IP angle > 150 deg  =>  thumbs_up
                              IP angle <= 150 deg =>  closed_fist
                            This is rotation-invariant and scale-invariant.

  2. CNNClassifier        - MobileNetV2 transfer-learning wrapper.
                            Two-phase fine-tuning on HaGRID dataset.
                            model_path=None builds architecture with
                            ImageNet weights for unit testing.

  3. TemporalSmoother     - Recency-weighted majority vote + debounce.
                            Newer frames weighted more than older ones.
                            Emits only when winner holds >= 60% of
                            weighted votes and debounce has elapsed.

Gesture -> Robot Command Mapping
---------------------------------
  closed_fist  ->  STOP
  open_hand    ->  FORWARD
  thumbs_up    ->  LEFT
  peace_sign   ->  RIGHT
  pointing     ->  BACKWARD

ROS2 Velocity Profiles
-----------------------
  FORWARD:  linear.x = +0.30 m/s
  BACKWARD: linear.x = -0.20 m/s
  LEFT:     angular.z = +0.50 rad/s
  RIGHT:    angular.z = -0.50 rad/s
  STOP:     all zero

References
----------
  Zhang et al. (2020) MediaPipe Hands. CVPR Workshops.
  Howard et al. (2017) MobileNets. arXiv:1704.04861.
  Kapitanov et al. (2022) HaGRID. WACV pp.4186-4195.
  Haddadin et al. (2016) On making robots understand safety. IJRR 31(13).
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

GESTURE_LABELS: List[str] = [
    "closed_fist",
    "open_hand",
    "thumbs_up",
    "peace_sign",
    "pointing",
]

GESTURE_COMMANDS: Dict[str, str] = {
    "closed_fist": "STOP",
    "open_hand":   "FORWARD",
    "thumbs_up":   "LEFT",
    "peace_sign":  "RIGHT",
    "pointing":    "BACKWARD",
}

# ROS2 Twist velocity profiles - tuned for TurtleBot3 Burger in Gazebo
# linear.x in m/s, angular.z in rad/s
ROS_VELOCITY_PROFILES: Dict[str, Dict[str, float]] = {
    "FORWARD":  {"linear_x":  0.30, "angular_z":  0.00},
    "BACKWARD": {"linear_x": -0.20, "angular_z":  0.00},
    "LEFT":     {"linear_x":  0.00, "angular_z":  0.50},
    "RIGHT":    {"linear_x":  0.00, "angular_z": -0.50},
    "STOP":     {"linear_x":  0.00, "angular_z":  0.00},
}

# -- MediaPipe landmark indices ------------------------------------------------
# Hand skeleton topology (MediaPipe canonical model):
#
#   0 = WRIST
#   1=THUMB_CMC  2=THUMB_MCP  3=THUMB_IP   4=THUMB_TIP
#   5=INDEX_MCP  6=INDEX_PIP  7=INDEX_DIP  8=INDEX_TIP
#   9=MID_MCP   10=MID_PIP   11=MID_DIP   12=MID_TIP
#  13=RING_MCP  14=RING_PIP  15=RING_DIP  16=RING_TIP
#  17=PINK_MCP  18=PINK_PIP  19=PINK_DIP  20=PINK_TIP

WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# (tip, pip, dip) triples for index through pinky - used in dual-joint check
_FINGER_TRIPLES: List[Tuple[int, int, int]] = [
    (INDEX_TIP,  INDEX_PIP,  INDEX_DIP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP),
    (RING_TIP,   RING_PIP,   RING_DIP),
    (PINKY_TIP,  PINKY_PIP,  PINKY_DIP),
]

# BGR colour constants for OpenCV drawing
_GREEN = (0, 210,  80)
_RED   = (0,  50, 220)
_CYAN  = (200, 200,  0)
_AMBER = (0, 170, 240)
_WHITE = (255, 255, 255)

# MediaPipe hand landmark model (downloaded once on first run)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = "hand_landmarker.task"

# Hand skeleton connection pairs for manual OpenCV drawing
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm transverse
]


# ══════════════════════════════════════════════════════════════════════════════
# Model download helper
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_model() -> str:
    """
    Download the MediaPipe HandLandmarker .task file if not already present.
    Returns the local path. Called once at RuleBasedClassifier construction.
    """
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    print("[Setup] Downloading hand landmark model (~9 MB)...")
    print(f"        {_MODEL_URL}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[Setup] Saved: {_MODEL_PATH}")
    except Exception as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            f"Download manually from:\n  {_MODEL_URL}\n"
            f"Place in project directory as: {_MODEL_PATH}"
        ) from exc
    return _MODEL_PATH


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HandAnalysis:
    """
    Full anatomical analysis of a detected hand from 21 MediaPipe landmarks.

    Attributes
    ----------
    landmarks      : (21, 3) float32 array of normalised [x, y, z] coords
    finger_states  : bool[5] - [thumb, index, middle, ring, pinky] extended
    extended_count : total number of extended fingers (0-5)
    thumb_up       : True when thumb tip is clearly above knuckle row
    thumb_ip_angle : IP joint flexion angle in degrees (used for fist/thumbs_up)
    raw_gesture    : classified gesture label string or None
    confidence     : heuristic confidence score [0, 1]
    joint_angles   : dict of named joint flexion angles in degrees
    """
    landmarks:      np.ndarray
    finger_states:  List[bool]       = field(default_factory=list)
    extended_count: int              = 0
    thumb_up:       bool             = False
    thumb_ip_angle: float            = 0.0
    raw_gesture:    Optional[str]    = None
    confidence:     float            = 0.0
    joint_angles:   Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Rule-Based Classifier
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedClassifier:
    """
    Anatomically-grounded hand gesture classifier using MediaPipe HandLandmarker.

    Uses the Tasks API in VIDEO running mode which maintains temporal state
    across frames via monotonic timestamps, producing smooth stable tracking
    without the per-frame jitter seen in IMAGE mode.

    Finger Extension Model
    ----------------------
    A finger is extended when BOTH conditions hold:
        tip.y < pip.y   (tip above PIP joint in image space)
        tip.y < dip.y   (tip above DIP joint in image space)
    The dual-joint check prevents misclassification when slight hand
    curvature places the tip marginally above only the PIP joint.

    Thumb uses x-axis lateral displacement (not y-axis) because the thumb
    moves laterally rather than vertically.

    Thumbs-Up vs Closed-Fist Disambiguation
    ----------------------------------------
    When no main fingers are extended, the IP (interphalangeal) joint
    flexion angle of the thumb distinguishes the two poses:
        IP angle > 150 degrees  =>  thumbs_up  (thumb genuinely extended)
        IP angle <= 150 degrees =>  closed_fist (thumb curled)
    This metric is rotation-invariant and scale-invariant, computed from
    the THUMB_MCP, THUMB_IP, THUMB_TIP landmark triplet.

    Parameters
    ----------
    min_detection_confidence : float - minimum hand detection confidence
    min_tracking_confidence  : float - minimum tracking confidence
    """

    # IP angle threshold separating thumbs_up from closed_fist
    _THUMBS_UP_IP_THRESHOLD: float = 150.0

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence:  float = 0.6,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import HandLandmarkerOptions

        self._mp = mp
        model_path = _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Monotonic clock origin for VIDEO mode timestamps (must be ms)
        self._t0 = time.monotonic()

        logger.info(
            "RuleBasedClassifier ready "
            f"(VIDEO mode, det={min_detection_confidence}, "
            f"track={min_tracking_confidence})"
        )

    # Public API

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Run MediaPipe HandLandmarker on a BGR webcam frame.

        Parameters
        ----------
        frame : BGR image from cv2.VideoCapture

        Returns
        -------
        landmarks : (21, 3) float32 array of normalised [x, y, z], or None
        annotated : BGR frame with skeleton, landmark dots, and status text
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Convert BGR -> RGB and wrap in MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires monotonically increasing timestamp in ms
        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        if not result.hand_landmarks:
            cv2.putText(
                annotated,
                "No hand detected - position hand in frame",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, _RED, 2, cv2.LINE_AA,
            )
            return None, annotated

        # Extract (21, 3) landmark array for first detected hand
        hand = result.hand_landmarks[0]
        lm_array = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32,
        )

        # Draw skeleton connections
        for a_idx, b_idx in _CONNECTIONS:
            ax, ay = int(lm_array[a_idx, 0] * w), int(lm_array[a_idx, 1] * h)
            bx, by = int(lm_array[b_idx, 0] * w), int(lm_array[b_idx, 1] * h)
            cv2.line(annotated, (ax, ay), (bx, by), _CYAN, 2, cv2.LINE_AA)

        # Draw landmark dots
        for lm in lm_array:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(annotated, (cx, cy), 5, _AMBER, -1)
            cv2.circle(annotated, (cx, cy), 3, _GREEN,  1)

        # Handedness label near wrist
        if result.handedness:
            side = result.handedness[0][0].display_name
            wx = int(lm_array[WRIST, 0] * w)
            wy = int(lm_array[WRIST, 1] * h) + 30
            cv2.putText(
                annotated, side, (wx, wy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _AMBER, 2,
            )

        return lm_array, annotated

    def analyse(self, landmarks: np.ndarray) -> HandAnalysis:
        """
        Perform full anatomical analysis of a (21, 3) landmark array.

        Computes finger extension states, thumb orientation, thumb IP angle,
        joint angles, and applies the classification rules. All intermediate
        values are stored in the returned HandAnalysis dataclass for
        transparency and logging.

        Parameters
        ----------
        landmarks : (21, 3) float32 array from process_frame()

        Returns
        -------
        HandAnalysis with all computed features and gesture label
        """
        if landmarks is None or landmarks.shape != (21, 3):
            return HandAnalysis(landmarks=np.zeros((21, 3), dtype=np.float32))

        a = HandAnalysis(landmarks=landmarks)
        hand_w = float(landmarks[:, 0].max() - landmarks[:, 0].min()) + 1e-6

        a.finger_states  = self._finger_states(landmarks, hand_w)
        a.extended_count = sum(a.finger_states)
        a.thumb_up       = self._is_thumb_up(landmarks)
        a.thumb_ip_angle = self._thumb_ip_angle(landmarks)
        a.joint_angles   = self._joint_angles(landmarks)
        a.raw_gesture, a.confidence = self._classify(a)

        return a

    def classify(self, landmarks: Optional[np.ndarray]) -> Optional[str]:
        """
        Convenience wrapper returning gesture label string or None.
        Compatible with the evaluate_system.py and realtime_inference.py
        interfaces.
        """
        if landmarks is None:
            return None
        return self.analyse(landmarks).raw_gesture

    def close(self) -> None:
        """Release MediaPipe graph resources."""
        self._landmarker.close()
        logger.info("RuleBasedClassifier closed.")

    # Private: geometry

    def _finger_states(self, lm: np.ndarray, hand_w: float) -> List[bool]:
        """
        Compute extension state for all five fingers.

        Returns bool[5]: [thumb, index, middle, ring, pinky]

        Thumb: extended when tip x-distance from wrist exceeds 1.3 times
               the MCP x-distance from wrist (lateral displacement metric).

        Index to Pinky: extended when BOTH tip.y < pip.y AND tip.y < dip.y.
        The dual-joint condition prevents misclassification from hand
        curvature placing the tip marginally above only the PIP joint.
        """
        # Thumb - lateral x-axis check
        thumb_ext = (
            abs(lm[THUMB_TIP, 0] - lm[WRIST, 0]) >
            abs(lm[THUMB_MCP, 0] - lm[WRIST, 0]) * 1.3
        )

        # Index through pinky - dual-joint y-axis check
        finger_ext = [
            float(lm[tip, 1]) < float(lm[pip, 1]) and
            float(lm[tip, 1]) < float(lm[dip, 1])
            for tip, pip, dip in _FINGER_TRIPLES
        ]

        return [thumb_ext] + finger_ext

    def _is_thumb_up(self, lm: np.ndarray) -> bool:
        """
        True when the thumb tip is clearly above the knuckle reference line.

        The reference line is the average y-coordinate of the index and
        middle MCP knuckles. The thumb tip must be at least 0.05 normalised
        units above this line, providing scale-invariant detection.
        """
        thumb_tip_y  = lm[THUMB_TIP,  1]
        index_mcp_y  = lm[INDEX_MCP,  1]
        middle_mcp_y = lm[MIDDLE_MCP, 1]
        knuckle_y    = (index_mcp_y + middle_mcp_y) / 2.0
        return float(thumb_tip_y) < float(knuckle_y) - 0.05

    def _thumb_ip_angle(self, lm: np.ndarray) -> float:
        """
        Compute the thumb IP (interphalangeal) joint flexion angle in degrees.

        Uses the law of cosines on the THUMB_MCP -> THUMB_IP -> THUMB_TIP
        landmark triplet. This angle is:
            Large (> 150 deg) for an extended thumb (thumbs_up)
            Small (< 120 deg) for a curled thumb (closed_fist)

        The metric is rotation-invariant and scale-invariant because it
        is derived from the relative positions of the three joints rather
        than their absolute coordinates.
        """
        v1 = lm[THUMB_MCP] - lm[THUMB_IP]
        v2 = lm[THUMB_TIP] - lm[THUMB_IP]
        cos_t = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
        return float(math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0))))

    def _joint_angles(self, lm: np.ndarray) -> Dict[str, float]:
        """
        Compute key joint flexion angles in degrees via law of cosines.
        Used for diagnostics, logging, and potential future features.
        """
        def angle(a: int, b: int, c: int) -> float:
            v1 = lm[a] - lm[b]
            v2 = lm[c] - lm[b]
            cos_t = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            return float(math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0))))

        return {
            "index_pip":  angle(INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
            "middle_pip": angle(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            "ring_pip":   angle(RING_MCP,   RING_PIP,   RING_TIP),
            "pinky_pip":  angle(PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
            "thumb_ip":   self._thumb_ip_angle(lm),
        }

    # Private: classification rules

    def _classify(self, a: HandAnalysis) -> Tuple[Optional[str], float]:
        """
        Priority-ordered rule engine mapping hand pose to gesture label.

        Rules are ordered most-specific to least-specific to prevent a
        general rule from capturing a more specific gesture.

        Priority order:
          1. open_hand   (4+ fingers extended)
          2. open_hand   (3 fingers - robust to little finger occlusion)
          3. peace_sign  (index + middle only)
          4. pointing    (index only)
          5. thumbs_up   (no main fingers, IP angle > 150 deg)
          6. closed_fist (no main fingers, IP angle <= 150 deg)

        The IP angle threshold at step 5/6 is the key improvement over
        the knuckle-reference approach: it directly measures thumb joint
        extension rather than inferring it from tip elevation, making it
        robust to variations in hand orientation and thumb resting position.
        """
        index  = a.finger_states[1]
        middle = a.finger_states[2]
        ring   = a.finger_states[3]
        pinky  = a.finger_states[4]
        thumb  = a.finger_states[0]

        fingers_up = sum([index, middle, ring, pinky])

        # open_hand: all four main fingers extended
        if index and middle and ring and pinky:
            return "open_hand", 0.95 if thumb else 0.88

        # open_hand: three of four extended (robust to pinky occlusion)
        if fingers_up == 3 and index and middle and ring:
            return "open_hand", 0.80

        # peace_sign: index and middle only
        if index and middle and not ring and not pinky:
            return "peace_sign", 0.92

        # pointing: index only
        if index and not middle and not ring and not pinky:
            return "pointing", 0.90

        # No main fingers extended: use IP angle to separate thumbs_up / fist
        if not index and not middle and not ring and not pinky:
            if a.thumb_ip_angle > self._THUMBS_UP_IP_THRESHOLD:
                return "thumbs_up", 0.93
            else:
                return "closed_fist", 0.91

        # Ambiguous pose - no rule matched cleanly
        logger.debug(
            f"Ambiguous: fingers={a.finger_states}, "
            f"ip_angle={a.thumb_ip_angle:.1f}, "
            f"thumb_up={a.thumb_up}"
        )
        return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CNN Classifier
# ══════════════════════════════════════════════════════════════════════════════

class CNNClassifier:
    """
    MobileNetV2-based gesture classifier using transfer learning.

    Architecture
    ------------
    MobileNetV2 (ImageNet weights, frozen during Phase 1)
      -> GlobalAveragePooling2D
      -> Dense(256, ReLU) + Dropout(0.3)
      -> Dense(128, ReLU) + Dropout(0.2)
      -> Dense(5, Softmax)

    Training Strategy
    -----------------
    Phase 1 (10 epochs, lr=1e-4): base frozen, classification head only.
    Phase 2 (20 epochs, lr=1e-5): top 30 base layers unfrozen, full fine-tune.

    Parameters
    ----------
    model_path : path to trained .h5 checkpoint, or None to build with
                 ImageNet weights (useful for unit testing without a
                 trained model).
    """

    IMG_SIZE    = (224, 224)
    NUM_CLASSES = 5

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model      = self._load_or_build(model_path)

    # Public API

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Prepare a BGR webcam frame for MobileNetV2 inference.

        Pipeline: BGR -> resize(224x224) -> RGB -> float32 -> normalise [0,1]
                  -> add batch dimension

        Returns
        -------
        np.ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 1]
        """
        img = cv2.resize(frame_bgr, self.IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Run inference on a raw BGR frame.

        Returns
        -------
        label      : predicted gesture label string, or None if unavailable
        confidence : probability of top class [0, 1]
        probs      : full (5,) softmax probability vector
        """
        if self.model is None:
            return None, 0.0, np.zeros(self.NUM_CLASSES, dtype=np.float32)
        probs = self.model.predict(self.preprocess(frame_bgr), verbose=0)[0]
        idx   = int(np.argmax(probs))
        return GESTURE_LABELS[idx], float(probs[idx]), probs

    def classify(self, frame_bgr: np.ndarray) -> Optional[str]:
        """Convenience wrapper returning label string only."""
        label, _, _ = self.predict(frame_bgr)
        return label

    def save(self, path: str) -> None:
        """Persist model weights to disk."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"CNNClassifier saved -> {path}")

    # Private

    def _load_or_build(self, model_path: Optional[str]):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.applications import MobileNetV2

            if model_path is not None:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"CNNClassifier: loaded from {model_path}")
                return model

            # Build with frozen base and random head (for unit testing)
            logger.info("CNNClassifier: building MobileNetV2 architecture...")
            base = MobileNetV2(
                input_shape=(*self.IMG_SIZE, 3),
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False

            inputs  = tf.keras.Input(shape=(*self.IMG_SIZE, 3))
            x       = base(inputs, training=False)
            x       = layers.GlobalAveragePooling2D(name="gap")(x)
            x       = layers.Dense(256, activation="relu")(x)
            x       = layers.Dropout(0.3)(x)
            x       = layers.Dense(128, activation="relu")(x)
            x       = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)
            model   = Model(inputs, outputs, name="gesture_mobilenetv2")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info(f"CNNClassifier: built - {model.count_params():,} params")
            return model

        except ImportError:
            logger.warning("TensorFlow not installed - CNNClassifier.model is None.")
            return None
        except Exception as exc:
            logger.error(f"CNNClassifier build error: {exc}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Temporal Smoother
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSmoother:
    """
    Stabilises noisy per-frame predictions for reliable robot commands.

    Algorithm
    ---------
    1. Maintain a sliding window of the last `window` predictions.
    2. Assign recency weights: oldest frame = 1, newest = window (linear ramp).
    3. Compute weighted vote totals for each non-None label.
    4. Emit the winning label only when it holds >= 60% of total weight.
    5. Suppress re-emission of the same label within debounce_s seconds.

    The recency weighting provides faster response to genuine gesture
    changes than uniform weighting: new frames accumulate weight faster
    than the trailing frames from the previous gesture.

    The 60% consensus threshold prevents emission during transitions where
    votes are split between the ending and starting gestures.

    The debounce prevents command flooding during a sustained hold.

    Parameters
    ----------
    window     : sliding window size in frames (default 7)
    debounce_s : minimum seconds between same-label emissions (default 0.8)
    """

    def __init__(self, window: int = 7, debounce_s: float = 0.8) -> None:
        self.window      = window
        self.debounce_s  = debounce_s
        self._buffer:     deque = deque(maxlen=window)
        self._last_label: Optional[str] = None
        self._last_time:  float = 0.0
        # Recency weight vector: position 0 = oldest (weight 1), -1 = newest (weight window)
        self._weights = np.arange(1, window + 1, dtype=np.float32)

    def update(self, label: Optional[str]) -> Optional[str]:
        """
        Feed one frame prediction. Returns stable gesture label or None.

        Parameters
        ----------
        label : raw gesture string from classifier, or None if no hand

        Returns
        -------
        Stable gesture label when consensus + debounce conditions are met,
        otherwise None.
        """
        self._buffer.append(label)

        if len(self._buffer) < self.window:
            return None

        # Accumulate weighted votes
        weighted: Dict[str, float] = {}
        for i, lbl in enumerate(self._buffer):
            if lbl is not None:
                weighted[lbl] = weighted.get(lbl, 0.0) + self._weights[i]

        if not weighted:
            return None

        total  = float(self._weights.sum())
        winner = max(weighted, key=weighted.get)

        # Require >= 60% weighted consensus
        if weighted[winner] / total < 0.60:
            return None

        # Debounce: suppress same label within debounce window
        now = time.monotonic()
        if winner == self._last_label and (now - self._last_time) < self.debounce_s:
            return None

        self._last_label = winner
        self._last_time  = now
        return winner

    def reset(self) -> None:
        """Clear buffer and debounce state."""
        self._buffer.clear()
        self._last_label = None
        self._last_time  = 0.0

    @property
    def last_gesture(self) -> Optional[str]:
        """Last emitted gesture label."""
        return self._last_label

    @property
    def buffer_fill(self) -> float:
        """Buffer fill ratio [0, 1] - used by dashboard warm-up indicator."""
        return len(self._buffer) / self.window


# ══════════════════════════════════════════════════════════════════════════════
# ROS Publisher stub
# ══════════════════════════════════════════════════════════════════════════════

class ROSPublisher:
    """
    Optional ROS integration publishing geometry_msgs/Twist to /cmd_vel.

    Degrades gracefully to a no-op when ROS is not installed (e.g. on Windows),
    allowing the rest of the pipeline to function normally with mouse control.

    ROS2 Twist velocity mapping:
        FORWARD  -> linear.x  = +0.30 m/s
        BACKWARD -> linear.x  = -0.20 m/s
        LEFT     -> angular.z = +0.50 rad/s  (counter-clockwise)
        RIGHT    -> angular.z = -0.50 rad/s  (clockwise)
        STOP     -> all zero
    """

    def __init__(
        self,
        node_name: str = "gesture_controller",
        topic:     str = "/cmd_vel",
    ) -> None:
        self._available = False
        self._pub       = None
        self._Twist     = None

        try:
            import rospy
            from geometry_msgs.msg import Twist
            rospy.init_node(node_name, anonymous=True)
            self._pub       = rospy.Publisher(topic, Twist, queue_size=1)
            self._Twist     = Twist
            self._available = True
            logger.info(f"ROSPublisher: active on {topic}")
        except ImportError:
            logger.info("ROSPublisher: ROS not available - simulation mode.")
        except Exception as exc:
            logger.warning(f"ROSPublisher init error: {exc}")

    def publish(self, command: Optional[str]) -> bool:
        """
        Publish a Twist message for the given command string.
        Returns True if published, False if ROS unavailable or command is None.
        """
        if not self._available or command is None:
            return False
        profile = ROS_VELOCITY_PROFILES.get(command, ROS_VELOCITY_PROFILES["STOP"])
        twist = self._Twist()
        twist.linear.x  = profile["linear_x"]
        twist.angular.z = profile["angular_z"]
        self._pub.publish(twist)
        return True

    @property
    def is_available(self) -> bool:
        return self._available


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 58)
    print("  gesture_classifier.py  --  self test")
    print("=" * 58)

    print("\n[1] RuleBasedClassifier init...")
    clf = RuleBasedClassifier()
    print("    OK")

    print("\n[2] Dummy frame (no hand expected)...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    lm, ann = clf.process_frame(dummy)
    assert ann.shape == (480, 640, 3), "Annotated frame shape mismatch"
    print(f"    OK  landmarks={'detected' if lm is not None else 'none (expected on blank frame)'}")

    print("\n[3] IP angle on zero array (degenerate case)...")
    zero_lm = np.zeros((21, 3), dtype=np.float32)
    angle = clf._thumb_ip_angle(zero_lm)
    assert isinstance(angle, float), "IP angle must be float"
    print(f"    OK  angle={angle:.1f} deg")

    print("\n[4] CNN preprocessing...")
    cnn  = CNNClassifier()
    rand = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    t    = cnn.preprocess(rand)
    assert t.shape == (1, 224, 224, 3), f"Shape mismatch: {t.shape}"
    assert 0.0 <= t.min() and t.max() <= 1.0, "Values out of [0, 1]"
    print(f"    OK  shape={t.shape}, range=[{t.min():.2f}, {t.max():.2f}]")

    print("\n[5] Temporal smoother...")
    sm = TemporalSmoother(window=7, debounce_s=0.8)
    assert sm.update("closed_fist") is None, "Should be None before buffer full"
    for _ in range(6):
        sm.update("open_hand")
    r = sm.update("open_hand")
    assert r == "open_hand", f"Expected 'open_hand', got {r}"
    print(f"    OK  majority vote -> '{r}'")

    print("\n[6] Constants...")
    assert len(GESTURE_LABELS) == 5
    assert GESTURE_LABELS[0]           == "closed_fist"
    assert GESTURE_LABELS[1]           == "open_hand"
    assert GESTURE_COMMANDS["closed_fist"] == "STOP"
    assert GESTURE_COMMANDS["open_hand"]   == "FORWARD"
    assert GESTURE_COMMANDS["thumbs_up"]   == "LEFT"
    print(f"    OK  {len(GESTURE_LABELS)} labels, {len(GESTURE_COMMANDS)} commands")

    clf.close()
    print("\n" + "=" * 58)
    print("  All tests passed")
    print("=" * 58 + "\n")
    sys.exit(0)