"""
Gesture Recognition System Evaluation
Project: BA-25-1058 | Student: Mmesoma Kenneth (202307951)

Evaluates rule-based gesture classifier with:
- Accuracy per gesture
- Precision, Recall, F1-score
- Confusion matrix
- Performance metrics (FPS, latency)
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from gesture_classifier import RuleBasedClassifier, TemporalSmoother, GESTURE_LABELS, GESTURE_COMMANDS
import matplotlib.pyplot as plt

class GestureEvaluator:
    """Systematic evaluation of gesture recognition system."""

    def __init__(self, model_type='rule'):
        self.model_type = model_type
        self.classifier = RuleBasedClassifier()
        self.smoother = TemporalSmoother(window=7, debounce_s=0.8)

        self.results = {
            'true_labels': [],
            'predicted_labels': [],
            'confidences': [],
            'latencies': [],
            'fps_readings': []
        }

    def interactive_test(self, gesture_name: str, duration_s: int = 5):
        """
        Test a single gesture for specified duration.
        User holds gesture in front of webcam, system records predictions.

        Args:
            gesture_name: One of ['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing']
            duration_s: How long to hold gesture (default 5 seconds)
        """
        print(f"\n{'='*60}")
        print(f"Testing: {gesture_name.upper()}")
        print(f"Hold the gesture for {duration_s} seconds")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return

        start_time = time.time()
        frame_count = 0
        predictions = []

        while time.time() - start_time < duration_s:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            # Process frame
            landmarks, annotated = self.classifier.process_frame(frame)

            if landmarks is None:
                predicted = None
            else:
                raw_pred = self.classifier.classify(landmarks)
                predicted = self.smoother.update(raw_pred)

            latency = (time.time() - frame_start) * 1000  # ms
            self.results['latencies'].append(latency)

            if predicted:
                predictions.append(predicted)
                self.results['predicted_labels'].append(predicted)
                self.results['true_labels'].append(gesture_name)

            # Draw on frame
            cv2.putText(annotated, f"Hold: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detected: {predicted or 'NONE'}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated, f"Time: {int(time.time() - start_time)}/{duration_s}s", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Evaluation", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Summary
        if predictions:
            accuracy = sum(1 for p in predictions if p == gesture_name) / len(predictions)
            print(f"\n[RESULT] {gesture_name}")
            print(f"  Detected: {len(predictions)} times")
            print(f"  Accuracy this test: {accuracy*100:.1f}%")
            print(f"  Most common prediction: {max(set(predictions), key=predictions.count)}")
        else:
            print(f"\n[WARNING] No detections for {gesture_name}")

    def run_full_evaluation(self):
        """Test all 5 gestures in sequence."""
        gestures = ['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing']

        print("\n" + "="*60)
        print("GESTURE RECOGNITION SYSTEM EVALUATION")
        print("="*60)
        print("\nInstructions:")
        print("1. When prompted, hold the gesture in front of your webcam")
        print("2. Keep your hand centered and still")
        print("3. The system will record detections for 5 seconds")
        print("4. Press Q at any time to skip")

        for gesture in gestures:
            input(f"\nPress Enter when ready to test '{gesture}'...")
            self.interactive_test(gesture, duration_s=5)

        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)

        self.generate_report()

    def generate_report(self):
        """Generate evaluation report with metrics."""
        if len(self.results['true_labels']) == 0:
            print("[ERROR] No data collected")
            return

        # Metrics
        true = self.results['true_labels']
        pred = self.results['predicted_labels']

        accuracy = accuracy_score(true, pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            true, pred, average='weighted', zero_division=0
        )

        cm = confusion_matrix(true, pred, labels=['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing'])

        avg_latency = np.mean(self.results['latencies']) if self.results['latencies'] else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0

        # Print report
        print("\n" + "="*70)
        print("EVALUATION REPORT")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Type: {self.model_type.upper()}")
        print(f"Total Samples: {len(true)}")

        print("\n[OVERALL METRICS]")
        print(f"  Accuracy:  {accuracy*100:.1f}%")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall:    {recall*100:.1f}%")
        print(f"  F1-Score:  {f1*100:.1f}%")

        print("\n[PERFORMANCE]")
        print(f"  Avg Latency: {avg_latency:.1f} ms")
        print(f"  Avg FPS: {fps:.1f}")

        print("\n[CONFUSION MATRIX]")
        print("(rows=true, cols=predicted)")
        gestures = ['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing']
        print("           ", " ".join(f"{g[:8]:>10}" for g in gestures))
        for i, g in enumerate(gestures):
            print(f"{g:10}", " ".join(f"{cm[i,j]:>10}" for j in range(len(gestures))))

        # Per-class metrics
        print("\n[PER-CLASS METRICS]")
        for gesture in gestures:
            mask = np.array(true) == gesture
            if mask.sum() > 0:
                acc = np.array(pred)[mask] == gesture
                acc_pct = acc.sum() / len(acc) * 100
                print(f"  {gesture:15} Accuracy: {acc_pct:5.1f}% (n={mask.sum()})")

        # Save results
        self._save_report(accuracy, precision, recall, f1, cm)

    def _save_report(self, accuracy, precision, recall, f1, cm):
        """Save results to JSON and generate plots."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'total_samples': len(self.results['true_labels']),
            'avg_latency_ms': float(np.mean(self.results['latencies'])) if self.results['latencies'] else 0,
        }

        with open('evaluation_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[SAVED] evaluation_results.json")

        # Plot confusion matrix
        self._plot_confusion_matrix(cm)

    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        gestures = ['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing']

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks(np.arange(len(gestures)))
        ax.set_yticks(np.arange(len(gestures)))
        ax.set_xticklabels(gestures, rotation=45, ha='right')
        ax.set_yticklabels(gestures)

        for i in range(len(gestures)):
            for j in range(len(gestures)):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix - Gesture Recognition')

        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        print("[SAVED] confusion_matrix.png")
        plt.close()


if __name__ == '__main__':
    evaluator = GestureEvaluator(model_type='rule')
    evaluator.run_full_evaluation()
