

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

class HaGRIDDatasetPrep:
    """Download and prepare HaGRID subset for training."""

    GESTURE_MAPPING = {
        "closed_fist": "fist",
        "open_hand": "palm",
        "thumbs_up": "thumbs_up",
        "peace_sign": "victory",
        "pointing": "pointer",
    }

    HAGRID_CLASSES = list(GESTURE_MAPPING.values())

    def __init__(self, output_dir: str = "data/hagrid"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_synthetic_dataset(self, num_samples_per_class: int = 200):
        """
        Create synthetic training dataset for quick testing.
        In production, use real HaGRID images from:
        https://github.com/hukenovs/hagrid
        """
        print(f"[DatasetPrep] Creating synthetic dataset ({num_samples_per_class} per class)...")

        splits = {"train": 0.7, "val": 0.15, "test": 0.15}

        for split, ratio in splits.items():
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)

            for gesture_id, gesture_name in enumerate(self.HAGRID_CLASSES):
                gesture_dir = split_dir / gesture_name
                gesture_dir.mkdir(exist_ok=True)

                n_samples = int(num_samples_per_class * ratio) if split != "train" else num_samples_per_class

                for i in range(n_samples):
                    # Generate random 224x224 RGB image
                    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

                    # Add some structure to make it more realistic
                    # Draw a hand-like shape (rough approximation)
                    center = (112, 112)
                    cv2.circle(img, center, 50, (200, 100, 50), -1)

                    # Different patterns per gesture
                    if gesture_name == "fist":
                        cv2.circle(img, center, 40, (100, 50, 20), -1)
                    elif gesture_name == "palm":
                        for dx in [-30, 0, 30]:
                            cv2.circle(img, (center[0] + dx, center[1] - 40), 15, (200, 100, 50), -1)
                    elif gesture_name == "thumbs_up":
                        cv2.rectangle(img, (90, 100), (135, 160), (200, 100, 50), -1)
                    elif gesture_name == "victory":
                        cv2.rectangle(img, (85, 110), (105, 160), (200, 100, 50), -1)
                        cv2.rectangle(img, (120, 110), (140, 160), (200, 100, 50), -1)
                    elif gesture_name == "pointer":
                        cv2.rectangle(img, (112, 80), (122, 160), (200, 100, 50), -1)

                    filename = gesture_dir / f"{gesture_name}_{i:04d}.jpg"
                    cv2.imwrite(str(filename), img)

                print(f"  [OK] {gesture_name} ({split}): {n_samples} samples")

        print(f"[DatasetPrep] Dataset created at {self.output_dir}\n")
        return self.output_dir

    def load_dataset(self, split: str = "train", img_size: tuple = (224, 224)):
        """Load images + labels from split directory."""
        split_dir = self.output_dir / split
        images, labels = [], []
        label_map = {name: idx for idx, name in enumerate(self.HAGRID_CLASSES)}

        for gesture_name, label_idx in label_map.items():
            gesture_dir = split_dir / gesture_name
            if not gesture_dir.exists():
                continue

            for img_file in sorted(gesture_dir.glob("*.jpg")):
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_idx)

        return np.array(images, dtype=np.uint8), np.array(labels)

    def get_splits(self):
        """Return paths to train/val/test splits."""
        return {
            "train": self.output_dir / "train",
            "val": self.output_dir / "val",
            "test": self.output_dir / "test",
        }


if __name__ == "__main__":
    prep = HaGRIDDatasetPrep(output_dir="data/hagrid")

    # Create synthetic dataset for quick testing
    prep.create_synthetic_dataset(num_samples_per_class=100)

    # Load and verify
    X_train, y_train = prep.load_dataset("train")
    print(f"Train set: {X_train.shape}, labels: {y_train.shape}")

    X_val, y_val = prep.load_dataset("val")
    print(f"Val set: {X_val.shape}, labels: {y_val.shape}")

    X_test, y_test = prep.load_dataset("test")
    print(f"Test set: {X_test.shape}, labels: {y_test.shape}")
