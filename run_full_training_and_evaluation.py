# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Machine Learning Engineering: Full Training (Phase 1 + Phase 2)
 and Strict Evaluation on Coffee Leaf Test Set
==============================================================================
 Pipeline Highlights:
   1. Data:
      - Train: D:\Coffee_Dataset_Sanitized\train (Targeted Augmentation, Cercospora 3x extreme)
      - Val:   D:\Coffee_Dataset_Sanitized\val (Standard preprocessed)
      - Test:  D:\Coffee_Dataset_Sanitized\test (612 standardized test images)
   2. Loss: Categorical Focal Loss (gamma=2.0, alpha=0.25)
   3. Class Weights: Sklearn balanced weights + 3x Cercospora penalty
   4. Phase 1: Frozen MobileNetV2 base, 10 epochs, Adam(lr=1e-3)
   5. Phase 2: Unfreeze last 30 layers (BatchNorm frozen), 15 epochs, Adam(lr=1e-5), EarlyStopping(patience=4)
   6. Test Evaluation:
      - Confusion Matrix with Cercospora row & column explicitly highlighted
      - Classification Report (Precision, Recall, F1-score)
      - Strict threshold check: Cercospora Recall > 75%
==============================================================================
"""

import os
import sys
import math
import time
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------------------
# 1. Constants & File Paths
# ------------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
EARLY_STOPPING_PATIENCE = 4

DATASET_TRAIN = r"D:\Coffee_Dataset_Sanitized\train"
DATASET_VAL = r"D:\Coffee_Dataset_Sanitized\val"
DATASET_TEST = r"D:\Coffee_Dataset_Sanitized\test"

CLASS_NAMES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]

MODEL_PHASE1_CHECKPOINT = "best_coffee_model_phase1_targeted.h5"
MODEL_FINAL_CHECKPOINT = "best_coffee_model_full.h5"
CONFUSION_MATRIX_OUTPUT = "confusion_matrix_cercospora_highlighted.png"
EVALUATION_REPORT_JSON = "test_evaluation_report.json"


# ------------------------------------------------------------------------------
# 2. Categorical Focal Loss Implementation
# ------------------------------------------------------------------------------
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma)
        loss = tf.math.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return tf.math.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ------------------------------------------------------------------------------
# 3. Targeted Augmentation Sequence
# ------------------------------------------------------------------------------
class TargetedAugmentationSequence(tf.keras.utils.Sequence):
    def __init__(self, directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, cercospora_factor=3):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cercospora_factor = cercospora_factor

        self.class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.cercospora_idx = self.class_indices.get("Cerscospora", 0)

        # Extreme augmentation generator for Cercospora
        self.cercospora_datagen = ImageDataGenerator(
            rotation_range=45,
            zoom_range=0.3,
            brightness_range=[0.6, 1.4],
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Standard mild augmentation generator for other classes
        self.standard_datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            brightness_range=[0.85, 1.15],
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        self.samples = []
        self.raw_counts = {c: 0 for c in self.class_names}
        self.raw_classes_list = []
        self.augmented_counts = {c: 0 for c in self.class_names}

        for cls_name in self.class_names:
            cls_dir = os.path.join(directory, cls_name)
            cls_idx = self.class_indices[cls_name]
            fnames = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.raw_counts[cls_name] = len(fnames)
            self.raw_classes_list.extend([cls_idx] * len(fnames))

            repeat = self.cercospora_factor if cls_idx == self.cercospora_idx else 1
            for fname in fnames:
                fpath = os.path.join(cls_dir, fname)
                for _ in range(repeat):
                    self.samples.append((fpath, cls_idx))
                    self.augmented_counts[cls_name] += 1

        self.samples = np.array(self.samples, dtype=object)
        self.classes = np.array([s[1] for s in self.samples], dtype=int)
        self.raw_classes = np.array(self.raw_classes_list, dtype=int)
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_samples = self.samples[batch_indexes]

        batch_x = []
        batch_y = []

        for fpath, cls_idx in batch_samples:
            img = Image.open(fpath).convert("RGB")
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.BILINEAR)
            x = np.array(img, dtype=np.float32)

            if cls_idx == self.cercospora_idx:
                x = self.cercospora_datagen.random_transform(x)
            else:
                x = self.standard_datagen.random_transform(x)

            x = preprocess_input(x)

            y = np.zeros(self.num_classes, dtype=np.float32)
            y[cls_idx] = 1.0

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ------------------------------------------------------------------------------
# 4. Data Generators
# ------------------------------------------------------------------------------
def get_dataset_generators():
    print(f"[1/5] Loading datasets:\n      Train: {DATASET_TRAIN}\n      Val:   {DATASET_VAL}\n      Test:  {DATASET_TEST}")

    train_generator = TargetedAugmentationSequence(
        DATASET_TRAIN,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        cercospora_factor=3
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    val_generator = eval_datagen.flow_from_directory(
        DATASET_VAL,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = eval_datagen.flow_from_directory(
        DATASET_TEST,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator


# ------------------------------------------------------------------------------
# 5. Class Weights Calculation
# ------------------------------------------------------------------------------
def compute_training_class_weights(train_generator, cercospora_multiplier=3.0):
    print("[2/5] Computing balanced class weights with 3x Cercospora penalty...")
    classes = getattr(train_generator, "raw_classes", train_generator.classes)
    unique_classes = np.unique(classes)

    base_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes
    )

    class_weights_dict = {}
    for idx, weight in zip(unique_classes, base_weights):
        class_weights_dict[int(idx)] = float(weight)

    class_indices = train_generator.class_indices
    cercospora_idx = class_indices.get("Cerscospora", 0)
    original_w = class_weights_dict[cercospora_idx]
    class_weights_dict[cercospora_idx] *= cercospora_multiplier

    print("=" * 65)
    print("⚖️  EXACT CLASS WEIGHTS DICTIONARY:")
    for c_name, c_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        w = class_weights_dict[c_idx]
        note = f" <-- (3x Multiplier: {original_w:.4f} -> {w:.4f})" if c_idx == cercospora_idx else ""
        print(f" • Class {c_idx} [{c_name:<11}]: weight = {w:.4f}{note}")
    print("=" * 65)

    return class_weights_dict


# ------------------------------------------------------------------------------
# 6. Model Architecture Builder
# ------------------------------------------------------------------------------
def build_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=5):
    print("[3/5] Building MobileNetV2 Architecture...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    predictions = layers.Dense(num_classes, activation="softmax", name="softmax_out")(x)

    model = models.Model(inputs=base_model.input, outputs=predictions, name="Coffee_MobileNetV2_Focal")
    return model, base_model


# ------------------------------------------------------------------------------
# 7. Two-Phase Training Execution
# ------------------------------------------------------------------------------
def train_two_phases(model, base_model, train_gen, val_gen, class_weights):
    focal_loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.25)

    # ------------------ PHASE 1: Frozen Base ------------------
    print("\n" + "=" * 75)
    print(f"🚀 [Phase 1/2] Training Head Only ({PHASE1_EPOCHS} Epochs, Adam lr={PHASE1_LR})...")
    print("=" * 75)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=focal_loss_fn,
        metrics=["accuracy"]
    )

    cb_p1 = [
        ModelCheckpoint(MODEL_PHASE1_CHECKPOINT, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=cb_p1
    )

    best_p1_val_loss = min(history_p1.history["val_loss"])
    print(f"  --> Phase 1 Complete. Best val_loss: {best_p1_val_loss:.4f}")

    # ------------------ PHASE 2: Fine-Tuning Last 30 Layers ------------------
    print("\n" + "=" * 75)
    print(f"🔓 [Phase 2/2] Fine-Tuning: Unfreezing Last 30 Layers (Max {PHASE2_EPOCHS} Epochs, Adam lr={PHASE2_LR})...")
    print("=" * 75)

    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # For the last 30 layers, keep BatchNormalization frozen to preserve running statistics
    for layer in base_model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(len(layer.weights) for layer in model.layers if layer.trainable)
    print(f"  • Fine-tuning configured: Last 30 layers active (Trainable layer blocks: {trainable_count})")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=focal_loss_fn,
        metrics=["accuracy"]
    )

    cb_p2 = [
        ModelCheckpoint(MODEL_FINAL_CHECKPOINT, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    ]

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=cb_p2
    )

    best_p2_val_loss = min(history_p2.history["val_loss"])
    print(f"  --> Phase 2 Complete. Best val_loss: {best_p2_val_loss:.4f}")

    return model, history_p1, history_p2


# ------------------------------------------------------------------------------
# 8. Evaluation, Highlighted Confusion Matrix, and Strict Reporting
# ------------------------------------------------------------------------------
def evaluate_and_report(model, test_gen):
    print("\n" + "=" * 75)
    print("[5/5] Running Inference and Generating Strict Evaluation Report on Test Set...")
    print("=" * 75)

    test_gen.reset()
    y_true = test_gen.classes
    num_samples = len(y_true)

    print(f"  • Running inference on {num_samples} test images...")
    start_eval = time.time()
    y_pred_probs = model.predict(test_gen, steps=math.ceil(num_samples / BATCH_SIZE), verbose=1)
    eval_time = time.time() - start_eval
    print(f"  • Inference finished in {eval_time:.2f}s ({num_samples / eval_time:.1f} images/s)")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 2. Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    # 3. Generate Highlighted Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    # Base heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"}
    )

    # Explicit highlighting of Cercospora row (True cases) and column (Predicted cases)
    # Row 0: True Cercospora
    rect_row = patches.Rectangle(
        (0.02, 0.02), 4.96, 0.96,
        linewidth=3.5,
        edgecolor="#D32F2F", # Vivid Red
        facecolor="none",
        linestyle="--",
        label="Cercospora Row (True Ground Truth)"
    )
    ax.add_patch(rect_row)

    # Column 0: Predicted Cercospora
    rect_col = patches.Rectangle(
        (0.02, 0.02), 0.96, 4.96,
        linewidth=3.5,
        edgecolor="#FF8F00", # Vivid Amber
        facecolor="none",
        linestyle=":",
        label="Cercospora Column (Model Predictions)"
    )
    ax.add_patch(rect_col)

    ax.set_title("Confusion Matrix (Test Set)\n[Cercospora Row & Column Explicitly Highlighted]", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model Predicted Label", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig.savefig(CONFUSION_MATRIX_OUTPUT, dpi=300)
    plt.close(fig)
    print(f"  • Highlighted Confusion Matrix saved to: {os.path.abspath(CONFUSION_MATRIX_OUTPUT)}")

    # Copy to artifact directory if available
    artifact_dir = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(CONFUSION_MATRIX_OUTPUT, os.path.join(artifact_dir, CONFUSION_MATRIX_OUTPUT))
        print(f"  • Mirrored Confusion Matrix image to artifact directory: {artifact_dir}")

    # 4. Strict Metrics Extraction for Cercospora
    cercospora_metrics = report_dict["Cerscospora"]
    precision = cercospora_metrics["precision"]
    recall = cercospora_metrics["recall"]
    f1 = cercospora_metrics["f1-score"]
    support = int(cercospora_metrics["support"])

    overall_accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    recall_exceeds_75 = recall > 0.75

    # 5. Raw Output Printing (No embellishment, strictly engineering data)
    print("\n" + "=" * 75)
    print("📊 RAW CONFUSION MATRIX ARRAY (Shape 5x5):")
    print("=" * 75)
    print(f"{'':>12} | " + " | ".join([f"{c[:10]:>10}" for c in CLASS_NAMES]))
    print("-" * 75)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>12} | " + " | ".join([f"{val:>10d}" for val in row]))
    print("=" * 75)

    print("\n" + "=" * 75)
    print("📋 FULL CLASSIFICATION REPORT (RAW DATA):")
    print("=" * 75)
    print(report_text)
    print("=" * 75)

    print("\n" + "=" * 75)
    print("🎯 CERCOSPORA SPECIFIC EVALUATION (STRICT AUDIT):")
    print("=" * 75)
    print(f" • Class Name:         Cerscospora (Index 0)")
    print(f" • Ground Truth Count: {support} images")
    print(f" • Correct (TP):       {cm[0, 0]} images")
    print(f" • False Negatives:    {support - cm[0, 0]} images (Misclassified as other diseases)")
    print(f" • False Positives:    {int(np.sum(cm[:, 0]) - cm[0, 0])} images (Other diseases mistaken for Cercospora)")
    print(f" • Precision:          {precision:.4f} ({precision * 100:.2f}%)")
    print(f" • Recall:             {recall:.4f} ({recall * 100:.2f}%)")
    print(f" • F1-Score:           {f1:.4f}")
    print(f" • Target Check:       Recall > 75.0%: {'PASSED ✅' if recall_exceeds_75 else 'FAILED ❌'}")
    print(f" • Overall Test Acc:   {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    print(f" • Macro Avg F1:       {macro_f1:.4f}")
    print("=" * 75 + "\n")

    # Export report to JSON
    report_data = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "cercospora": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(support),
            "recall_exceeds_75_percent": bool(recall_exceeds_75)
        },
        "overall_accuracy": float(overall_accuracy),
        "macro_avg_f1": float(macro_f1)
    }
    with open(EVALUATION_REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data


# ------------------------------------------------------------------------------
# 9. Main Pipeline Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("☕ AGY COFFEE LEAF CLASSIFIER: END-TO-END TRAINING & STRICT EVALUATION")
    print("=" * 80)

    # 1. Load Data
    train_gen, val_gen, test_gen = get_dataset_generators()

    # 2. Weights
    class_weights = compute_training_class_weights(train_gen, cercospora_multiplier=3.0)

    # 3. Model
    model, base_model = build_mobilenetv2_model(num_classes=NUM_CLASSES)

    # 4. Train Phase 1 & Phase 2
    model, h1, h2 = train_two_phases(model, base_model, train_gen, val_gen, class_weights)

    # 5. Load Best Checkpoint for Evaluation
    if os.path.exists(MODEL_FINAL_CHECKPOINT):
        print(f"\n📦 Loading best checkpoint from {MODEL_FINAL_CHECKPOINT} for testing...")
        model = models.load_model(
            MODEL_FINAL_CHECKPOINT,
            custom_objects={"CategoricalFocalLoss": CategoricalFocalLoss, "focal_loss": CategoricalFocalLoss()}
        )

    # 6. Strict Evaluation
    report_data = evaluate_and_report(model, test_gen)
    print("✅ Pipeline Execution Finished.")


if __name__ == "__main__":
    main()
