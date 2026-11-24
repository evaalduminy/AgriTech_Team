# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Edge-AI: Few-Shot Domain Adaptation (Local Fine-Tuning)
==============================================================================
 Target Domain: Ugandan Coffee Leaves (as proxy for local field deployment)
 Strategy:
   1. Load baseline model (best_coffee_model_4class_final.h5 - 88% in Brazil).
   2. Sample exactly 100 few-shot local images (34 Healthy, 33 Rust, 33 Phoma).
   3. Retain the remaining 3,120 images strictly for out-of-sample testing.
   4. Unfreeze the last 30 layers (BatchNormalization frozen).
   5. Fine-tune for 5 epochs only with Adam(lr=1e-5) and mild augmentation.
   6. Strictly evaluate on all 3,120 unseen test images (Confusion Matrix + Report).
   7. Export to coffee_model_adapted.tflite (Optimize.DEFAULT).
==============================================================================
"""

import os
import sys
import math
import time
import json
import random
import shutil
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------------------
# 1. Paths & Configuration
# ------------------------------------------------------------------------------
WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"

BASE_MODEL_PATH = os.path.join(WORKSPACE, "best_coffee_model_4class_final.h5")
ADAPTED_H5_PATH = os.path.join(WORKSPACE, "best_coffee_model_adapted.h5")
ADAPTED_TFLITE_PATH = os.path.join(WORKSPACE, "coffee_model_adapted.tflite")
REPORT_JSON_PATH = os.path.join(WORKSPACE, "few_shot_adaptation_report.json")

UGANDA_ROOT = r"C:\Users\Hawaa\Downloads\A Machine  Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda\coffee dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 4
ADAPTATION_EPOCHS = 5
ADAPTATION_LR = 1e-5

CLASS_NAMES = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

# Seed for strictly reproducible 100-sample partition
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------------------------
# 2. Data Partitioning: 100 Few-Shot Adapt vs. 3,120 Test
# ------------------------------------------------------------------------------
def partition_uganda_dataset():
    print("=" * 80)
    print("Partitioning Ugandan Dataset: 100 Few-Shot Adaptation vs. 3,120 Unseen Test")
    print(f"Source Directory: {UGANDA_ROOT}")
    print("=" * 80)

    target_folders = {
        "Health leaves": ("Healthy", 34),
        "leaf rust":     ("Fungal_Disease", 33),
        "phoma":         ("Phoma", 33)
    }

    adapt_samples = []
    test_samples = []

    for folder_name, (gt_label, n_adapt) in target_folders.items():
        folder_path = os.path.join(UGANDA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
        all_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts) and os.path.getsize(os.path.join(folder_path, f)) > 1000
        ]

        # Deterministic shuffle
        random.shuffle(all_files)

        adapt_files = all_files[:n_adapt]
        test_files = all_files[n_adapt:]

        for f in adapt_files:
            adapt_samples.append((f, gt_label))
        for f in test_files:
            test_samples.append((f, gt_label))

        print(f" • Class [{gt_label:<15}]: {len(all_files)} total valid images -> {len(adapt_files)} Adapt (Few-Shot), {len(test_files)} Test")

    print("-" * 80)
    print(f"Total Few-Shot Adaptation Samples: {len(adapt_samples)}")
    print(f"Total Unseen Test Samples:         {len(test_samples)}")
    print("=" * 80)

    return adapt_samples, test_samples


# ------------------------------------------------------------------------------
# 3. Preprocessing & Augmentation
# ------------------------------------------------------------------------------
def preprocess_image_1to1(img_path, target_size=IMAGE_SIZE):
    """Standard 1:1 center-crop and MobileNetV2 normalization [-1.0, 1.0]."""
    with Image.open(img_path) as raw_img:
        rgb = raw_img.convert("RGB")
        w, h = rgb.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = rgb.crop((left, top, left + min_dim, top + min_dim))
        resized = cropped.resize(target_size, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        return preprocess_input(arr)


class FewShotSequence(tf.keras.utils.Sequence):
    """Feeds the 100 adaptation images with mild data augmentation."""
    def __init__(self, samples, batch_size=BATCH_SIZE, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_indices = {c: i for i, c in enumerate(CLASS_NAMES)}

        self.datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Preload the 100 images into memory
        self.cached_x = []
        self.cached_y = []
        for fpath, label in self.samples:
            tensor = preprocess_image_1to1(fpath)
            y = np.zeros(NUM_CLASSES, dtype=np.float32)
            y[self.class_indices[label]] = 1.0
            self.cached_x.append(tensor)
            self.cached_y.append(y)

        self.cached_x = np.array(self.cached_x, dtype=np.float32)
        self.cached_y = np.array(self.cached_y, dtype=np.float32)
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        bx = []
        by = []
        for i in batch_idx:
            # Mild transform
            x = self.datagen.random_transform(self.cached_x[i])
            bx.append(x)
            by.append(self.cached_y[i])
        return np.array(bx, dtype=np.float32), np.array(by, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ------------------------------------------------------------------------------
# 4. Model Loading & Fine-Tuning Setup
# ------------------------------------------------------------------------------
def setup_few_shot_model():
    print(f"\n[1/4] Loading Baseline Model from: {BASE_MODEL_PATH}...")
    model = tf.keras.models.load_model(BASE_MODEL_PATH)

    # Unfreeze the last 30 layers, keep BatchNormalization frozen
    print("  • Configuring layer trainability (Unfreezing last 30 layers, freezing BatchNorm)...")
    for layer in model.layers[:-30]:
        layer.trainable = False

    unfrozen_count = 0
    for layer in model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_count += 1

    print(f"  • Unfrozen layers: {unfrozen_count} (out of {len(model.layers)} total layers)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ADAPTATION_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------------------------------------
# 5. Fine-Tuning Execution
# ------------------------------------------------------------------------------
def run_adaptation(model, adapt_gen):
    print("\n" + "=" * 80)
    print(f"[2/4] Executing Few-Shot Domain Adaptation: {ADAPTATION_EPOCHS} Epochs on 100 Samples...")
    print(f"      Optimizer: Adam(lr={ADAPTATION_LR}), Loss: Categorical Crossentropy")
    print("=" * 80)

    t0 = time.time()
    history = model.fit(
        adapt_gen,
        epochs=ADAPTATION_EPOCHS,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nAdaptation finished in {elapsed:.2f} seconds ({elapsed/ADAPTATION_EPOCHS:.2f} s/epoch).")
    print(f"Final Adaptation Accuracy on 100 samples: {history.history['accuracy'][-1]*100:.2f}% (Loss: {history.history['loss'][-1]:.4f})")

    # Save H5
    model.save(ADAPTED_H5_PATH)
    print(f"Adapted model saved to: {ADAPTED_H5_PATH}")
    return model


# ------------------------------------------------------------------------------
# 6. Comprehensive Evaluation on 3,120 Unseen Ugandan Images
# ------------------------------------------------------------------------------
def evaluate_adapted_model(model, test_samples):
    print("\n" + "=" * 80)
    print(f"[3/4] Evaluating Adapted Model on Remaining {len(test_samples)} Unseen Ugandan Images...")
    print("=" * 80)

    EVAL_BATCH_SIZE = 64
    total_samples = len(test_samples)

    y_true = []
    y_pred = []
    y_conf = []

    t0 = time.time()

    for i in range(0, total_samples, EVAL_BATCH_SIZE):
        batch_slice = test_samples[i : i + EVAL_BATCH_SIZE]
        bx = []
        by = []

        for fpath, gt_label in batch_slice:
            try:
                x = preprocess_image_1to1(fpath)
                bx.append(x)
                by.append(gt_label)
            except Exception:
                continue

        if not bx:
            continue

        batch_arr = np.array(bx, dtype=np.float32)
        preds = model.predict(batch_arr, verbose=0)

        for p_vec, gt in zip(preds, by):
            p_idx = int(np.argmax(p_vec))
            pred_class = CLASS_NAMES[p_idx]
            conf = float(p_vec[p_idx])

            y_true.append(gt)
            y_pred.append(pred_class)
            y_conf.append(conf)

        if (i // EVAL_BATCH_SIZE) % 10 == 0 or i + EVAL_BATCH_SIZE >= total_samples:
            done = min(i + EVAL_BATCH_SIZE, total_samples)
            rate = done / max(time.time() - t0, 0.001)
            print(f"  Processed {done}/{total_samples} test images ({rate:.1f} img/s)...")

    eval_time = time.time() - t0
    eval_count = len(y_true)
    print(f"\nInference completed in {eval_time:.1f}s ({eval_count/eval_time:.1f} img/s).")

    # Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    adapted_accuracy = (correct / eval_count) * 100.0 if eval_count > 0 else 0.0

    gt_unique = ["Fungal_Disease", "Healthy", "Phoma"]
    pred_unique = CLASS_NAMES

    # Confusion Matrix
    cm = np.zeros((len(gt_unique), len(pred_unique)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        r = gt_unique.index(yt)
        c = pred_unique.index(yp)
        cm[r, c] += 1

    print("\n" + "=" * 80)
    print(f"ADAPTED UGANDAN CONFUSION MATRIX (N = {eval_count} Test Images):")
    print("=" * 80)
    header = f"{'Ground Truth':<16} | " + " | ".join([f"{c:>15}" for c in pred_unique]) + " | Total"
    print(header)
    print("-" * len(header))
    for r_idx, gt in enumerate(gt_unique):
        row_str = f"{gt:<16} | " + " | ".join([f"{cm[r_idx, c_idx]:>15}" for c_idx in range(len(pred_unique))])
        row_str += f" | {np.sum(cm[r_idx]):>5}"
        print(row_str)
    print("-" * len(header))
    col_totals = f"{'Total Predicted':<16} | " + " | ".join([f"{np.sum(cm[:, c_idx]):>15}" for c_idx in range(len(pred_unique))]) + f" | {eval_count:>5}"
    print(col_totals)
    print("=" * 80)

    print("\nPER-CLASS ACCURACY (RECALL) AFTER FEW-SHOT ADAPTATION:")
    for r_idx, gt in enumerate(gt_unique):
        row_sum = np.sum(cm[r_idx])
        correct_in_class = cm[r_idx, pred_unique.index(gt)]
        class_acc = (correct_in_class / row_sum) * 100 if row_sum > 0 else 0
        print(f" • {gt:<15}: {class_acc:6.2f}% ({correct_in_class}/{row_sum})")

    print("\nADAPTED CLASSIFICATION REPORT:")
    rep_dict = classification_report(y_true, y_pred, labels=gt_unique, target_names=gt_unique, digits=4, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=gt_unique, target_names=gt_unique, digits=4, zero_division=0))

    baseline_acc = 45.50
    cutmix_acc = 34.50
    delta_baseline = adapted_accuracy - baseline_acc
    delta_cutmix = adapted_accuracy - cutmix_acc

    print("=" * 80)
    print("FEW-SHOT DOMAIN ADAPTATION BENCHMARK COMPARISON:")
    print(f" • Baseline Model (Zero-Shot Brazil Model):             {baseline_acc:.2f}%")
    print(f" • Domain Randomization (CLAHE + CutMix Attempt):       {cutmix_acc:.2f}%")
    print(f" • Few-Shot Adapted Model (100 Local Samples, 5 Epochs): {adapted_accuracy:.2f}%")
    print(f" • Gain over Zero-Shot Baseline:                        {'+' if delta_baseline >= 0 else ''}{delta_baseline:.2f}%")
    print(f" • Target Goal (> 75.0%):                               {'PASSED' if adapted_accuracy >= 75.0 else 'UNMET'}")
    print("=" * 80)

    report_payload = {
        "baseline_accuracy": baseline_acc,
        "cutmix_accuracy": cutmix_acc,
        "adapted_accuracy": adapted_accuracy,
        "gain_over_baseline": delta_baseline,
        "goal_passed": bool(adapted_accuracy >= 75.0),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep_dict,
        "test_sample_count": eval_count
    }
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report_payload, f, indent=2)

    return adapted_accuracy, cm, rep_dict


# ------------------------------------------------------------------------------
# 7. TFLite Export
# ------------------------------------------------------------------------------
def export_adapted_tflite(model):
    print("\n" + "=" * 80)
    print("[4/4] Exporting Adapted Model to TFLite (Optimize.DEFAULT)...")
    print("=" * 80)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open(ADAPTED_TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes)

    size_bytes = os.path.getsize(ADAPTED_TFLITE_PATH)
    size_mb = size_bytes / (1024 * 1024)
    print(f" • Adapted TFLite exported: {ADAPTED_TFLITE_PATH}")
    print(f" • Exact Size:              {size_bytes:,} bytes ({size_mb:.2f} MB)")

    if os.path.exists(ARTIFACT_DIR):
        artifact_tflite = os.path.join(ARTIFACT_DIR, "coffee_model_adapted.tflite")
        shutil.copy2(ADAPTED_TFLITE_PATH, artifact_tflite)
        shutil.copy2(REPORT_JSON_PATH, os.path.join(ARTIFACT_DIR, "few_shot_adaptation_report.json"))
        print(f" • Mirrored to Artifact:    {artifact_tflite}")

    return size_mb, size_bytes


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    print("FEW-SHOT DOMAIN ADAPTATION: UGANDAN DATASET (LOCAL FIELD CALIBRATION)")

    # 1. Partition data: 100 adapt, 3,120 test
    adapt_samples, test_samples = partition_uganda_dataset()

    # 2. Setup sequence
    adapt_gen = FewShotSequence(adapt_samples, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup model
    model = setup_few_shot_model()

    # 4. Fine-tune for 5 epochs
    model = run_adaptation(model, adapt_gen)

    # 5. Evaluate on 3,120 test samples
    acc, cm, rep = evaluate_adapted_model(model, test_samples)

    # 6. Export TFLite
    export_adapted_tflite(model)

    print("\nFew-Shot Domain Adaptation Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
