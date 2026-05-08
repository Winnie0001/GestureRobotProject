"""
Training script: Phase 1 (frozen base) + Phase 2 (fine-tune).
Targets 85% accuracy and <200ms latency per PDD.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from pathlib import Path

def train_model(dataset_dir: str = "data/hagrid", output_model: str = "models/gesture_model.h5",
                phase2_epochs: int = 20):
    """
    Train MobileNetV2 gesture classifier.
    Phase 1: Frozen base (10 epochs, lr=1e-4)
    Phase 2: Fine-tune top 30 layers (20 epochs, lr=1e-5)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
        return

    # Load dataset
    from dataset_prep import HaGRIDDatasetPrep
    prep = HaGRIDDatasetPrep(dataset_dir)
    splits = prep.get_splits()

    print("\n" + "="*60)
    print("PHASE 1: Training with Frozen Base (10 epochs)")
    print("="*60)

    # Data loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(splits["train"]),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    val_gen = val_datagen.flow_from_directory(
        str(splits["val"]),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    # Build model
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nModel architecture:")
    model.summary()

    # Phase 1 training
    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        verbose=1,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ]
    )

    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Top 30 Layers ({} epochs)".format(phase2_epochs))
    print("="*60)

    # Unfreeze top 30 layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_epochs,
        verbose=1,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ]
    )

    # Save model
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)
    print(f"\nModel saved to {output_model}")

    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    test_gen = val_datagen.flow_from_directory(
        str(splits["test"]),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Generate report
    y_pred = model.predict(test_gen)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    # Visualize
    combined_hist = {
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(combined_hist['loss'], label='Train Loss')
    axes[0].plot(combined_hist['val_loss'], label='Val Loss')
    axes[0].axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', alpha=0.5, label='Phase 2 start')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training & Validation Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(combined_hist['accuracy'], label='Train Accuracy')
    axes[1].plot(combined_hist['val_accuracy'], label='Val Accuracy')
    axes[1].axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', alpha=0.5, label='Phase 2 start')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150)
    print("\nTraining history saved to models/training_history.png")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument("--dataset", type=str, default="data/hagrid",
                       help="Path to HaGRID dataset")
    parser.add_argument("--output", type=str, default="models/gesture_model.h5",
                       help="Output model path")
    parser.add_argument("--phase2-epochs", type=int, default=20,
                       help="Epochs for phase 2 fine-tuning")
    args = parser.parse_args()

    train_model(dataset_dir=args.dataset, output_model=args.output,
                phase2_epochs=args.phase2_epochs)
