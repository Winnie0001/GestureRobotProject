# This file intentionally mirrors the project-root gesture_classifier.py so
# it can be imported cleanly from within the ROS package.
#
# Keep this module as the canonical import target for ROS nodes:
#   from gesture_control_ros.gesture_classifier import ...

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

GESTURE_LABELS: List[str] = [
    "closed_fist",
    "open_hand",
    "thumbs_up",
    "peace_sign",
    "pointing",
]

GESTURE_COMMANDS: Dict[str, str] = {
    "closed_fist": "STOP",
    "open_hand": "FORWARD",
    "thumbs_up": "LEFT",
    "peace_sign": "RIGHT",
    "pointing": "BACKWARD",
}

ROS_VELOCITY_PROFILES: Dict[str, Dict[str, float]] = {
    "FORWARD": {"linear_x": 0.30, "angular_z": 0.00},
    "BACKWARD": {"linear_x": -0.20, "angular_z": 0.00},
    "LEFT": {"linear_x": 0.00, "angular_z": 0.50},
    "RIGHT": {"linear_x": 0.00, "angular_z": -0.50},
    "STOP": {"linear_x": 0.00, "angular_z": 0.00},
}

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

_GREEN = (0, 210, 80)
_RED = (0, 50, 220)
_CYAN = (200, 200, 0)
_AMBER = (0, 170, 240)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = "hand_landmarker.task"

_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def _ensure_model() -> str:
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    print(f"[Setup] Downloading hand landmark model (~9 MB)...")
    print(f"        {_MODEL_URL}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[Setup] Saved: {_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            f"Download manually from:\n  {_MODEL_URL}\n"
            f"Save as: {_MODEL_PATH}"
        )
    return _MODEL_PATH


@dataclass
class HandAnalysis:
    landmarks: np.ndarray
    finger_states: List[bool] = field(default_factory=list)
    extended_count: int = 0
    thumb_up: bool = False
    raw_gesture: Optional[str] = None
    confidence: float = 0.0
    joint_angles: Dict[str, float] = field(default_factory=dict)


class RuleBasedClassifier:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
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
        self._t0 = time.monotonic()

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        if not result.hand_landmarks:
            cv2.putText(
                annotated,
                "No hand detected - show hand to camera",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                _RED,
                2,
                cv2.LINE_AA,
            )
            return None, annotated

        hand = result.hand_landmarks[0]
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

        for a_idx, b_idx in _CONNECTIONS:
            ax = int(lm_array[a_idx, 0] * w)
            ay = int(lm_array[a_idx, 1] * h)
            bx = int(lm_array[b_idx, 0] * w)
            by = int(lm_array[b_idx, 1] * h)
            cv2.line(annotated, (ax, ay), (bx, by), _CYAN, 2, cv2.LINE_AA)

        for lm in lm_array:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(annotated, (cx, cy), 5, _AMBER, -1)
            cv2.circle(annotated, (cx, cy), 3, _GREEN, 1)

        return lm_array, annotated

    def analyse(self, landmarks: np.ndarray) -> HandAnalysis:
        if landmarks is None or landmarks.shape != (21, 3):
            return HandAnalysis(landmarks=np.zeros((21, 3), dtype=np.float32))

        a = HandAnalysis(landmarks=landmarks)
        hand_w = float(landmarks[:, 0].max() - landmarks[:, 0].min()) + 1e-6
        a.finger_states = self._finger_states(landmarks, hand_w)
        a.extended_count = sum(a.finger_states)
        a.thumb_up = self._is_thumb_up(landmarks)
        a.joint_angles = self._joint_angles(landmarks)
        a.raw_gesture, a.confidence = self._classify(a)
        return a

    def classify(self, landmarks: Optional[np.ndarray]) -> Optional[str]:
        if landmarks is None:
            return None
        return self.analyse(landmarks).raw_gesture

    def close(self) -> None:
        self._landmarker.close()

    def _finger_states(self, lm: np.ndarray, hand_w: float) -> List[bool]:
        thumb_ext = (
            abs(lm[THUMB_TIP, 0] - lm[WRIST, 0])
            > abs(lm[THUMB_MCP, 0] - lm[WRIST, 0]) * 1.3
        )

        finger_pairs_with_dip = [
            (INDEX_TIP, INDEX_PIP, INDEX_DIP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP),
            (RING_TIP, RING_PIP, RING_DIP),
            (PINKY_TIP, PINKY_PIP, PINKY_DIP),
        ]
        finger_ext = [
            float(lm[tip, 1]) < float(lm[pip, 1]) and float(lm[tip, 1]) < float(lm[dip, 1])
            for tip, pip, dip in finger_pairs_with_dip
        ]

        return [thumb_ext] + finger_ext

    def _is_thumb_up(self, lm: np.ndarray) -> bool:
        wrist_y = lm[WRIST, 1]
        thumb_tip_y = lm[THUMB_TIP, 1]
        index_mcp_y = lm[INDEX_MCP, 1]
        hand_h = abs(wrist_y - index_mcp_y) + 1e-6
        return (wrist_y - thumb_tip_y) > hand_h * 1.0 and thumb_tip_y < index_mcp_y

    def _joint_angles(self, lm: np.ndarray) -> Dict[str, float]:
        def angle(a: int, b: int, c: int) -> float:
            v1 = lm[a] - lm[b]
            v2 = lm[c] - lm[b]
            cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return float(math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0))))

        return {
            "index_pip": angle(INDEX_MCP, INDEX_PIP, INDEX_TIP),
            "middle_pip": angle(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            "ring_pip": angle(RING_MCP, RING_PIP, RING_TIP),
            "pinky_pip": angle(PINKY_MCP, PINKY_PIP, PINKY_TIP),
            "thumb_ip": angle(THUMB_MCP, THUMB_IP, THUMB_TIP),
        }

    def _classify(self, a: HandAnalysis) -> Tuple[Optional[str], float]:
        index = a.finger_states[1]
        middle = a.finger_states[2]
        ring = a.finger_states[3]
        pinky = a.finger_states[4]

        fingers_up = sum([index, middle, ring, pinky])

        if a.thumb_up and not index and not middle and not ring and not pinky:
            return "thumbs_up", 0.93

        if index and middle and ring and pinky:
            return "open_hand", 0.95 if a.finger_states[0] else 0.88

        if fingers_up == 3 and (index and middle and ring):
            return "open_hand", 0.80

        if index and middle and not ring and not pinky:
            return "peace_sign", 0.92

        if index and not middle and not ring and not pinky:
            return "pointing", 0.90

        if not index and not middle and not ring and not pinky:
            return "closed_fist", 0.91

        return None, 0.0


class CNNClassifier:
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 5

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model = self._load_or_build(model_path)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, self.IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
        if self.model is None:
            return None, 0.0, np.zeros(self.NUM_CLASSES, dtype=np.float32)
        probs = self.model.predict(self.preprocess(frame_bgr), verbose=0)[0]
        idx = int(np.argmax(probs))
        return GESTURE_LABELS[idx], float(probs[idx]), probs

    def classify(self, frame_bgr: np.ndarray) -> Optional[str]:
        label, _, _ = self.predict(frame_bgr)
        return label

    def _load_or_build(self, model_path: Optional[str]):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.applications import MobileNetV2

            if model_path is not None:
                return tf.keras.models.load_model(model_path)

            base = MobileNetV2(
                input_shape=(*self.IMG_SIZE, 3),
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False
            inputs = tf.keras.Input(shape=(*self.IMG_SIZE, 3))
            x = base(inputs, training=False)
            x = layers.GlobalAveragePooling2D(name="gap")(x)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)
            model = Model(inputs, outputs, name="gesture_mobilenetv2")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model

        except ImportError:
            logger.warning("TensorFlow not installed - CNNClassifier.model is None.")
            return None
        except Exception as exc:
            logger.error(f"CNNClassifier build error: {exc}")
            return None


class TemporalSmoother:
    def __init__(self, window: int = 7, debounce_s: float = 0.8) -> None:
        self.window = window
        self.debounce_s = debounce_s
        self._buffer: deque = deque(maxlen=window)
        self._last_label: Optional[str] = None
        self._last_time: float = 0.0
        self._weights = np.arange(1, window + 1, dtype=np.float32)

    def update(self, label: Optional[str]) -> Optional[str]:
        self._buffer.append(label)
        if len(self._buffer) < self.window:
            return None

        weighted: Dict[str, float] = {}
        for i, lbl in enumerate(self._buffer):
            if lbl is not None:
                weighted[lbl] = weighted.get(lbl, 0.0) + self._weights[i]

        if not weighted:
            return None

        total = float(self._weights.sum())
        winner = max(weighted, key=weighted.get)
        if weighted[winner] / total < 0.60:
            return None

        now = time.monotonic()
        if winner == self._last_label and (now - self._last_time) < self.debounce_s:
            return None

        self._last_label = winner
        self._last_time = now
        return winner

    def reset(self) -> None:
        self._buffer.clear()
        self._last_label = None
        self._last_time = 0.0

    @property
    def last_gesture(self) -> Optional[str]:
        return self._last_label

    @property
    def buffer_fill(self) -> float:
        return len(self._buffer) / self.window
