

import cv2
import numpy as np
import time
import json
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)
from gesture_classifier import (
    RuleBasedClassifier, TemporalSmoother,
    GESTURE_LABELS, GESTURE_COMMANDS
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


GESTURES = ['closed_fist', 'open_hand', 'thumbs_up', 'peace_sign', 'pointing']

# Tips for each gesture shown on screen during testing
GESTURE_TIPS = {
    'closed_fist': 'Curl ALL fingers tightly, thumb tucked in',
    'open_hand':   'Spread all 5 fingers wide, palm facing camera',
    'thumbs_up':   'Thumb pointing UP, all other fingers curled',
    'peace_sign':  'Index + middle UP only, ring + pinky curled',
    'pointing':    'Index finger only pointing UP, others curled',
}


class GestureEvaluator:
    """Systematic evaluation of gesture recognition system."""

    def __init__(self, model_type: str = 'rule'):
        self.model_type = model_type
        self.classifier = RuleBasedClassifier(
            min_detection_confidence=0.6,   # slightly lower for evaluation
            min_tracking_confidence=0.5,
        )
        # Shorter debounce for evaluation - we want more samples
        self.smoother = TemporalSmoother(window=5, debounce_s=0.3)

        self.results = {
            'true_labels':      [],
            'predicted_labels': [],
            'confidences':      [],
            'latencies':        [],   # only hand-detected frames
        }

    # -- Per-gesture test -------------------------------------------------------

    def interactive_test(self, gesture_name: str, duration_s: int = 8):
        """
        Test a single gesture for duration_s seconds.
        Records every frame-level prediction where confidence > 0.
        Shows countdown, tip, and live detection on screen.
        """
        print(f"\n{'='*60}")
        print(f"Testing: {gesture_name.upper()}")
        print(f"Tip: {GESTURE_TIPS.get(gesture_name, '')}")
        print(f"Hold the gesture clearly for {duration_s} seconds")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)

        # 2-second countdown before recording starts
        countdown_end = time.time() + 2.0
        while time.time() < countdown_end:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            remaining = countdown_end - time.time()
            cv2.putText(frame, f"GET READY: {remaining:.1f}s",
                        (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 200, 255), 3)
            cv2.putText(frame, gesture_name.upper(),
                        (180, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            cv2.imshow("Gesture Evaluation", frame)
            cv2.waitKey(1)

        # Recording phase
        start_time   = time.time()
        frame_count  = 0
        predictions  = []
        self.smoother.reset()

        while time.time() - start_time < duration_s:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            t_frame = time.time()

            # Get landmarks and full analysis
            landmarks, annotated = self.classifier.process_frame(frame)

            predicted  = None
            confidence = 0.0

            if landmarks is not None:
                analysis   = self.classifier.analyse(landmarks)
                raw_pred   = analysis.raw_gesture
                confidence = analysis.confidence

                # Record latency only when hand detected
                latency_ms = (time.time() - t_frame) * 1000
                self.results['latencies'].append(latency_ms)

                # Use smoother for stable output
                predicted = self.smoother.update(raw_pred)

                # But ALSO record raw confident predictions directly
                # so we get more samples in the evaluation window
                if raw_pred and confidence >= 0.7:
                    predictions.append(raw_pred)
                    self.results['predicted_labels'].append(raw_pred)
                    self.results['true_labels'].append(gesture_name)
                    self.results['confidences'].append(confidence)
            else:
                self.smoother.update(None)

            # Draw HUD
            elapsed   = time.time() - start_time
            remaining = max(0, duration_s - elapsed)

            # Progress bar
            bar_w = int(600 * elapsed / duration_s)
            cv2.rectangle(annotated, (20, 440), (620, 460), (60,60,60), -1)
            cv2.rectangle(annotated, (20, 440), (20 + bar_w, 460), (0,210,80), -1)

            # Labels
            col = (0, 210, 80) if predicted == gesture_name else (0, 100, 255)
            cv2.putText(annotated, f"HOLD: {gesture_name}",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255), 2)
            cv2.putText(annotated, f"Detected: {predicted or 'none'}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            cv2.putText(annotated, f"Samples: {len(predictions)}",
                        (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,200,0), 1)
            cv2.putText(annotated, f"Time left: {remaining:.1f}s",
                        (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,200,200), 1)
            cv2.putText(annotated, GESTURE_TIPS.get(gesture_name, ''),
                        (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1)

            cv2.imshow("Gesture Evaluation", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Per-gesture summary
        if predictions:
            correct  = sum(1 for p in predictions if p == gesture_name)
            accuracy = correct / len(predictions)
            most_common = max(set(predictions), key=predictions.count)
            print(f"\n[RESULT] {gesture_name}")
            print(f"  Samples collected : {len(predictions)}")
            print(f"  Correct           : {correct}")
            print(f"  Accuracy          : {accuracy*100:.1f}%")
            print(f"  Most predicted    : {most_common}")
        else:
            print(f"\n[WARNING] No detections for {gesture_name}")
            print(f"  Tips: ensure good lighting, hand centred, fingers clearly visible")

    # -- Full evaluation run ----------------------------------------------------

    def run_full_evaluation(self):
        """Test all 5 gestures in sequence then generate report."""
        print("\n" + "="*60)
        print("  GESTURE RECOGNITION SYSTEM - FULL EVALUATION")
        print("  Project: BA-25-1058 | Mmesoma Kenneth (202307951)")
        print("="*60)
        print("\nInstructions:")
        print("  1. Press Enter when ready for each gesture")
        print("  2. A 2-second countdown appears - get your hand ready")
        print("  3. Hold the gesture clearly for 8 seconds")
        print("  4. Good lighting + plain background = better results")
        print("  5. Press Q on the camera window to skip a gesture")
        print()

        for gesture in GESTURES:
            input(f"\n  >>> Press Enter when ready to test '{gesture}'...")
            self.interactive_test(gesture, duration_s=8)

        print("\n" + "="*60)
        print("  All gestures tested - generating report...")
        print("="*60)
        self.generate_report()

    # -- Report generation ------------------------------------------------------

    def generate_report(self):
        """Compute and print full evaluation metrics."""
        if not self.results['true_labels']:
            print("[ERROR] No data collected - ensure hand is visible during tests")
            return

        true = self.results['true_labels']
        pred = self.results['predicted_labels']

        accuracy = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, pred, average='weighted', zero_division=0
        )

        labels = GESTURES
        cm = confusion_matrix(true, pred, labels=labels)

        avg_latency = float(np.mean(self.results['latencies'])) \
                      if self.results['latencies'] else 0.0
        avg_fps     = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        avg_conf    = float(np.mean(self.results['confidences'])) \
                      if self.results['confidences'] else 0.0

        # -- Console report -----------------------------------------------------
        print("\n" + "="*70)
        print("  EVALUATION REPORT")
        print("="*70)
        print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model type  : {self.model_type.upper()} (MediaPipe HandLandmarker)")
        print(f"  Total samples: {len(true)}")

        print("\n  [OVERALL METRICS]")
        print(f"    Accuracy  : {accuracy*100:.1f}%")
        print(f"    Precision : {precision*100:.1f}%")
        print(f"    Recall    : {recall*100:.1f}%")
        print(f"    F1-Score  : {f1*100:.1f}%")
        print(f"    Avg Conf  : {avg_conf*100:.1f}%")

        print("\n  [PERFORMANCE]")
        print(f"    Avg Latency : {avg_latency:.1f} ms")
        print(f"    Avg FPS     : {avg_fps:.1f}")

        print("\n  [PER-CLASS ACCURACY]")
        for gesture in labels:
            mask = np.array(true) == gesture
            if mask.sum() > 0:
                correct = (np.array(pred)[mask] == gesture).sum()
                pct     = correct / mask.sum() * 100
                bar     = '#' * int(pct / 5)
                print(f"    {gesture:15} {bar:20} {pct:5.1f}%  (n={mask.sum()})")
            else:
                print(f"    {gesture:15} {'-':20} No samples")

        print("\n  [CONFUSION MATRIX]  (rows=true, cols=predicted)")
        header = "              " + "".join(f"{g[:7]:>10}" for g in labels)
        print(header)
        for i, g in enumerate(labels):
            row = f"  {g:12}" + "".join(f"{cm[i,j]:>10}" for j in range(len(labels)))
            print(row)

        # -- Save files ---------------------------------------------------------
        self._save_json(accuracy, precision, recall, f1, cm,
                        avg_latency, avg_fps, avg_conf)
        self._plot_confusion_matrix(cm)
        self._plot_per_class_accuracy(true, pred, labels)

    def _save_json(self, accuracy, precision, recall, f1,
                   cm, avg_latency, avg_fps, avg_conf):
        data = {
            'timestamp':        datetime.now().isoformat(),
            'model_type':       self.model_type,
            'classifier':       'MediaPipe HandLandmarker + geometric rules',
            'total_samples':    len(self.results['true_labels']),
            'accuracy':         round(float(accuracy),  4),
            'precision':        round(float(precision), 4),
            'recall':           round(float(recall),    4),
            'f1_score':         round(float(f1),        4),
            'avg_confidence':   round(float(avg_conf),  4),
            'avg_latency_ms':   round(float(avg_latency), 2),
            'avg_fps':          round(float(avg_fps),     2),
            'confusion_matrix': cm.tolist(),
            'gesture_labels':   GESTURES,
        }
        with open('evaluation_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("\n  [SAVED] evaluation_results.json")

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(GESTURES)))
        ax.set_yticks(np.arange(len(GESTURES)))
        ax.set_xticklabels(GESTURES, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(GESTURES, fontsize=10)

        # Annotate cells
        for i in range(len(GESTURES)):
            for j in range(len(GESTURES)):
                val = cm[i, j]
                colour = 'white' if val > cm.max() * 0.5 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        color=colour, fontsize=12, fontweight='bold')

        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(
            f'Confusion Matrix - Gesture Recognition\n'
            f'Project BA-25-1058 | Mmesoma Kenneth (202307951)',
            fontsize=11
        )
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("  [SAVED] confusion_matrix.png")
        plt.close()

    def _plot_per_class_accuracy(self, true, pred, labels):
        accs = []
        for g in labels:
            mask = np.array(true) == g
            if mask.sum() > 0:
                accs.append((np.array(pred)[mask] == g).sum() / mask.sum() * 100)
            else:
                accs.append(0.0)

        colours = ['#E24B4A', '#1D9E75', '#EF9F27', '#378ADD', '#7F77DD']
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, accs, color=colours, edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5,
                    f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        ax.axhline(y=85, color='red', linestyle='--', linewidth=1.5,
                   label='Target (85%)')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Gesture', fontsize=12)
        ax.set_title(
            'Per-Gesture Classification Accuracy\n'
            'Project BA-25-1058 | Mmesoma Kenneth (202307951)',
            fontsize=11
        )
        ax.legend(fontsize=10)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
        print("  [SAVED] per_class_accuracy.png")
        plt.close()


if __name__ == '__main__':
    evaluator = GestureEvaluator(model_type='rule')
    evaluator.run_full_evaluation()