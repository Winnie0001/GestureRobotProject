#!/usr/bin/env python3
"""
Quick verification script to test project setup.
Run this before starting to ensure all dependencies work.
"""

import sys

def check_dependency(name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✓ {name:20} OK")
        return True
    except ImportError as e:
        print(f"✗ {name:20} MISSING: {e}")
        return False


def main():
    print("="*60)
    print("PROJECT SETUP VERIFICATION")
    print("="*60 + "\n")

    deps = [
        ("MediaPipe", "mediapipe"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("TensorFlow", "tensorflow"),
        ("Scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("PyAutoGUI", "pyautogui"),
        ("PIL", "PIL"),
        ("Requests", "requests"),
        ("tqdm", "tqdm"),
    ]

    print("Checking dependencies...\n")
    results = [check_dependency(name, import_name) for name, import_name in deps]

    print("\n" + "="*60)
    if all(results):
        print("✓ All dependencies installed!")
        print("="*60)

        # Try importing project modules
        print("\nChecking project modules...\n")
        try:
            from gesture_classifier import (
                RuleBasedClassifier, CNNClassifier, TemporalSmoother
            )
            print("✓ gesture_classifier.py")

            from dataset_prep import HaGRIDDatasetPrep
            print("✓ dataset_prep.py")

            print("\n" + "="*60)
            print("✓ READY TO START!")
            print("="*60)
            print("\n→ Run: python realtime_inference.py --mode rule")
            print("→ Press SPACE to toggle mouse control, Q to quit")
            return 0
        except ImportError as e:
            print(f"✗ Project module error: {e}")
            return 1
    else:
        print("✗ Some dependencies missing")
        print("="*60)
        print("\nInstall with: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
