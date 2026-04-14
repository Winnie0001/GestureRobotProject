"""
Gesture classifier unit tests.
Verifies rule-based and CNN classifiers work correctly.
"""

import unittest
import numpy as np
import cv2
from gesture_classifier import (
    RuleBasedClassifier, CNNClassifier, TemporalSmoother,
    GESTURE_LABELS, GESTURE_COMMANDS
)


class TestRuleBasedClassifier(unittest.TestCase):
    """Test rule-based classifier hand pose detection."""

    def setUp(self):
        self.classifier = RuleBasedClassifier()

    def tearDown(self):
        self.classifier.close()

    def create_dummy_frame(self, size=(640, 480)):
        """Create a dummy BGR frame."""
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def test_frame_processing(self):
        """Test that frame processing doesn't crash."""
        frame = self.create_dummy_frame()
        landmarks, annotated = self.classifier.process_frame(frame)
        self.assertEqual(annotated.shape, frame.shape)

    def test_gesture_labels(self):
        """Test gesture label mapping."""
        self.assertEqual(len(GESTURE_LABELS), 5)
        self.assertEqual(GESTURE_LABELS[0], "closed_fist")
        self.assertEqual(GESTURE_LABELS[1], "open_hand")

    def test_gesture_commands(self):
        """Test gesture-to-command mapping."""
        self.assertEqual(GESTURE_COMMANDS["closed_fist"], "STOP")
        self.assertEqual(GESTURE_COMMANDS["open_hand"], "FORWARD")
        self.assertEqual(GESTURE_COMMANDS["thumbs_up"], "LEFT")


class TestTemporalSmoother(unittest.TestCase):
    """Test temporal smoothing logic."""

    def setUp(self):
        self.smoother = TemporalSmoother(window=7, debounce_s=0.8)

    def test_initial_state(self):
        """Test smoother starts in neutral state."""
        result = self.smoother.update("closed_fist")
        self.assertIsNone(result)  # Need 7 frames

    def test_majority_vote(self):
        """Test majority voting."""
        gestures = ["closed_fist"] * 5 + [None] * 2
        for g in gestures:
            result = self.smoother.update(g)
        self.assertEqual(result, "closed_fist")

    def test_debounce(self):
        """Test debounce prevents rapid re-triggering."""
        # First gesture
        for _ in range(7):
            self.smoother.update("closed_fist")
        result1 = self.smoother.update("closed_fist")
        self.assertEqual(result1, "closed_fist")

        # Immediate re-trigger should be blocked
        for _ in range(7):
            self.smoother.update("closed_fist")
        result2 = self.smoother.update("closed_fist")
        self.assertIsNone(result2)

    def test_reset(self):
        """Test smoother reset."""
        for _ in range(7):
            self.smoother.update("closed_fist")
        self.smoother.reset()
        result = self.smoother.update("closed_fist")
        self.assertIsNone(result)


class TestCNNClassifier(unittest.TestCase):
    """Test CNN classifier (build only, no training)."""

    def test_model_creation(self):
        """Test that CNN model can be created."""
        classifier = CNNClassifier()
        self.assertIsNotNone(classifier.model)

    def test_preprocess(self):
        """Test image preprocessing."""
        classifier = CNNClassifier()
        dummy_frame = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
        processed = classifier.preprocess(dummy_frame)
        self.assertEqual(processed.shape, (1, 224, 224, 3))
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))


if __name__ == "__main__":
    print("Running gesture classifier tests...\n")
    unittest.main(verbosity=2)
