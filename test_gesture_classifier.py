import cv2
import numpy as np
import time
from collections import deque, Counter

# -------------------------------
# Gesture labels + commands
# -------------------------------
GESTURE_LABELS = [
    "closed_fist",
    "open_hand",
    "thumbs_up",
    "point_left",
    "point_right"
]

GESTURE_COMMANDS = {
    "closed_fist": "STOP",
    "open_hand": "FORWARD",
    "thumbs_up": "LEFT",
    "point_left": "LEFT",
    "point_right": "RIGHT"
}



# MediaPipe

def _init_mediapipe():
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    try:
        mp_styles = mp.solutions.drawing_styles
    except AttributeError:
        mp_styles = None

    return mp_hands, mp_draw, mp_styles


# -------------------------------
# Rule-Based Classifier
# -------------------------------
class RuleBasedClassifier:
    def __init__(self):
        mp_hands, self.mp_draw, self.mp_styles = _init_mediapipe()

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        gesture = None

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                gesture = self._classify(handLms.landmark)

                self.mp_draw.draw_landmarks(
                    frame,
                    handLms,
                    self._get_connections()
                )

        return gesture, frame

    def _get_connections(self):
        import mediapipe as mp
        return mp.solutions.hands.HAND_CONNECTIONS

    def _classify(self, landmarks):
        fingers = []

        # Thumb
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        tips = [8, 12, 16, 20]
        for tip in tips:
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        total = sum(fingers)

        if total == 0:
            return "closed_fist"
        elif total == 5:
            return "open_hand"
        elif total == 1:
            return "thumbs_up"
        elif total == 2:
            return "point_left"
        elif total == 3:
            return "point_right"
        else:
            return None

    def close(self):
        self.hands.close()


# -------------------------------
# Temporal Smoother
# -------------------------------
class TemporalSmoother:
    def __init__(self, window=7, debounce_s=0.8):
        self.window = deque(maxlen=window)
        self.last_output_time = 0
        self.debounce_s = debounce_s

    def update(self, gesture):
        self.window.append(gesture)

        if len(self.window) < self.window.maxlen:
            return None

        counts = Counter(self.window)
        most_common, count = counts.most_common(1)[0]

        if most_common is None:
            return None

        now = time.time()

        if now - self.last_output_time < self.debounce_s:
            return None

        self.last_output_time = now
        return most_common

    def reset(self):
        self.window.clear()
        self.last_output_time = 0


# -------------------------------
# CNN Classifier (basic placeholder)
# -------------------------------
class CNNClassifier:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(GESTURE_LABELS), activation='softmax')
        ])

        return model

    def preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        return np.expand_dims(img, axis=0)