# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Deep Learning Research: Domain Generalization Pipeline
 CLAHE Lightness Normalization + CutMix Regularization
==============================================================================
 Research Objectives:
 1. Alleviate Cross-Domain Lighting Shift using CIELAB CLAHE on L-channel.
 2. Destroy Background Context Overfitting using 50% CutMix Patch Augmentation.
 3. Two-Phase Transfer Learning on MobileNetV2 (Phase 1: 10 epochs, Phase 2: 15 epochs).
 4. Strict Evaluation on:
    - In-Domain Source Test Set: Brazilian Coffee Leaves (612 images)
    - Out-of-Domain Target Test Set: Ugandan Coffee Leaves (3,220 images)
 5. Benchmark Comparison against baseline (45.50% Uganda accuracy).
 6. Export to quantized TFLite.
==============================================================================
"""

import os
import sys
import math
import time
import json
import random
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"

DATASET_TRAIN = r"D:\Coffee_Dataset_4Class\train"
DATASET_VAL   = r"D:\Coffee_Dataset_4Class\val"
DATASET_TEST  = r"D:\Coffee_Dataset_4Class\test"
UGANDA_ROOT   = r"C:\Users\Hawaa\Downloads\A Machine  Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda\coffee dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
EARLY_STOPPING_PATIENCE = 4

MODEL_PHASE1_PATH = os.path.join(WORKSPACE, "best_coffee_model_clahe_cutmix_phase1.h5")
MODEL_FINAL_PATH  = os.path.join(WORKSPACE, "best_coffee_model_clahe_cutmix_final.h5")
TFLITE_PATH       = os.path.join(WORKSPACE, "coffee_model_clahe_cutmix.tflite")
UGANDA_REPORT_JSON = os.path.join(WORKSPACE, "uganda_generalization_report_clahe_cutmix.json")

# Pre-instantiate CLAHE object for fast processing
clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ------------------------------------------------------------------------------
# 2. Fast CLAHE Preprocessing Function
# ------------------------------------------------------------------------------
def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Applies CLAHE exclusively to the L channel in CIELAB, preserving colors."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe_engine.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def clahe_and_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Full preprocessing: CLAHE followed by MobileNetV2 normalization [-1.0, 1.0]."""
    clahe_img = apply_clahe(img_rgb.astype(np.uint8))
    return preprocess_input(clahe_img.astype(np.float32))


# ------------------------------------------------------------------------------
# 3. Fast In-Memory Training Sequence with CutMix & CLAHE
# ------------------------------------------------------------------------------
class FastInMemoryCutMixSequence(tf.keras.utils.Sequence):
    """
    In-memory caching sequence:
    1. Pre-caches all images in RAM with CLAHE already applied (eliminates CPU overhead).
    2. Applies spatial augmentations (rotation, zoom, flips).
    3. With 50% probability, applies CutMix between two random samples in the dataset.
    4. Generates soft target probability vectors matching the CutMix area proportion.
    """
    def __init__(self, directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, cutmix_prob=0.5, shuffle=True):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.cutmix_prob = cutmix_prob
        self.shuffle = shuffle

        self.class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        print(f"  • Preloading images into RAM and applying CLAHE from: {directory}...")
        t0 = time.time()
        self.raw_images_cache = {}
        self.samples = []
        self.raw_counts = {c: 0 for c in self.class_names}
        self.raw_classes_list = []

        for cls_name in self.class_names:
            cls_dir = os.path.join(directory, cls_name)
            cls_idx = self.class_indices[cls_name]
            fnames = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.raw_counts[cls_name] = len(fnames)
            self.raw_classes_list.extend([cls_idx] * len(fnames))

            for fname in fnames:
                fpath = os.path.join(cls_dir, fname)
                if fpath not in self.raw_images_cache:
                    with Image.open(fpath) as img:
                        rgb = img.convert("RGB")
                        if rgb.size != self.target_size:
                            rgb = rgb.resize(self.target_size, Image.BILINEAR)
                        arr_rgb = np.array(rgb, dtype=np.uint8)
                        # Apply CLAHE directly during RAM preloading
                        arr_clahe = apply_clahe(arr_rgb)
                        self.raw_images_cache[fpath] = arr_clahe

                self.samples.append((fpath, cls_idx))

        t_cache = time.time() - t0
        cache_mb = sum(v.nbytes for v in self.raw_images_cache.values()) / (1024 * 1024)
        print(f"  • Cached {len(self.raw_images_cache)} CLAHE-enhanced images in {t_cache:.2f}s ({cache_mb:.1f} MB RAM).")

        self.samples = np.array(self.samples, dtype=object)
        self.classes = np.array([s[1] for s in self.samples], dtype=int)
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def _rand_bbox(self, width, height, lam):
        """Generates random bounding box with area proportional to (1 - lam)."""
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        return bbx1, bby1, bbx2, bby2

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_samples = self.samples[batch_indexes]

        batch_x = []
        batch_y = []

        W, H = self.target_size

        for fpath, cls_idx in batch_samples:
            raw_arr_1 = self.raw_images_cache[fpath]
            x1 = self.datagen.random_transform(raw_arr_1.astype(np.float32))

            y1 = np.zeros(self.num_classes, dtype=np.float32)
            y1[cls_idx] = 1.0

            # Apply CutMix with specified probability (50%)
            if np.random.rand() < self.cutmix_prob:
                # Pick a random second image from the entire dataset
                rand_idx = np.random.randint(len(self.samples))
                rand_fpath, rand_cls_idx = self.samples[rand_idx]
                raw_arr_2 = self.raw_images_cache[rand_fpath]
                x2 = self.datagen.random_transform(raw_arr_2.astype(np.float32))

                y2 = np.zeros(self.num_classes, dtype=np.float32)
                y2[rand_cls_idx] = 1.0

                # Sample lambda from Beta(1.0, 1.0) -> Uniform [0, 1]
                lam = np.random.uniform(0.3, 0.7)
                bbx1, bby1, bbx2, bby2 = self._rand_bbox(W, H, lam)

                # Paste patch from x2 into x1
                x1[bby1:bby2, bbx1:bbx2, :] = x2[bby1:bby2, bbx1:bbx2, :]

                # Compute actual lambda based on pixel area
                actual_area = (bbx2 - bbx1) * (bby2 - bby1)
                actual_lam = 1.0 - (actual_area / float(W * H))

                # Soft target distribution
                y = (actual_lam * y1) + ((1.0 - actual_lam) * y2)
            else:
                y = y1

            # MobileNetV2 normalization [-1.0, 1.0]
            x_norm = preprocess_input(x1)
            batch_x.append(x_norm)
            batch_y.append(y)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ------------------------------------------------------------------------------
# 4. Model Architecture Builder
# ------------------------------------------------------------------------------
def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=4):
    print("Building MobileNetV2 Architecture...")
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    predictions = layers.Dense(num_classes, activation="softmax", name="softmax_out")(x)

    model = models.Model(inputs=base_model.input, outputs=predictions, name="Coffee_MobileNetV2_CLAHE_CutMix")
    return model, base_model


# ------------------------------------------------------------------------------
# 5. Two-Phase Training Routine
# ------------------------------------------------------------------------------
def train_two_phases(model, base_model, train_gen, val_gen, class_weights):
    loss_fn = "categorical_crossentropy"

    # ── Phase 1: Frozen Base ───────────────────────────────────────────────────
    print(f"\n[Phase 1/2] Training Classifier Head ({PHASE1_EPOCHS} Epochs, Adam lr={PHASE1_LR})...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR), loss=loss_fn, metrics=["accuracy"])

    cb_p1 = [ModelCheckpoint(MODEL_PHASE1_PATH, monitor="val_loss", save_best_only=True, verbose=1)]
    history_p1 = model.fit(train_gen, validation_data=val_gen, epochs=PHASE1_EPOCHS, class_weight=class_weights, callbacks=cb_p1)

    print(f"Phase 1 Complete. Best val_loss: {min(history_p1.history['val_loss']):.4f}")

    # ── Phase 2: Fine-Tuning Last 30 Layers ────────────────────────────────────
    print(f"\n[Phase 2/2] Fine-Tuning Last 30 Layers (Max {PHASE2_EPOCHS} Epochs, Adam lr={PHASE2_LR})...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR), loss=loss_fn, metrics=["accuracy"])

    cb_p2 = [
        ModelCheckpoint(MODEL_FINAL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    ]
    history_p2 = model.fit(train_gen, validation_data=val_gen, epochs=PHASE2_EPOCHS, class_weight=class_weights, callbacks=cb_p2)

    print(f"Phase 2 Complete. Best val_loss: {min(history_p2.history['val_loss']):.4f}")
    return model


# ------------------------------------------------------------------------------
# 6. Evaluation Routine on Ugandan Dataset
# ------------------------------------------------------------------------------
def evaluate_uganda_dataset(model):
    print("=" * 80)
    print("CRUCIAL BENCHMARK: Evaluating on Unseen Ugandan Dataset with CLAHE")
    print(f"Uganda Path: {UGANDA_ROOT}")
    print("=" * 80)

    target_classes = {
        "Health leaves": "Healthy",
        "leaf rust": "Fungal_Disease",
        "phoma": "Phoma"
    }

    samples = []
    for folder_name, gt_label in target_classes.items():
        folder_path = os.path.join(UGANDA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(valid_exts) and os.path.getsize(os.path.join(folder_path, f)) > 1000]
        for f in files:
            samples.append((f, gt_label))

    print(f"Total valid Ugandan test samples: {len(samples)}")

    y_true = []
    y_pred = []
    y_conf = []

    # Batch processing for maximum speed (64 images per batch)
    EVAL_BATCH_SIZE = 64
    total_samples = len(samples)
    t0 = time.time()

    for i in range(0, total_samples, EVAL_BATCH_SIZE):
        batch_slice = samples[i : i + EVAL_BATCH_SIZE]
        batch_imgs = []
        batch_gt = []

        for img_path, gt_label in batch_slice:
            try:
                with Image.open(img_path) as raw_img:
                    rgb = raw_img.convert("RGB")
                    w, h = rgb.size
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    cropped = rgb.crop((left, top, left + min_dim, top + min_dim))
                    resized = cropped.resize(IMAGE_SIZE, Image.BILINEAR)
                    arr_rgb = np.array(resized, dtype=np.uint8)
                    norm_tensor = clahe_and_preprocess(arr_rgb)
                    batch_imgs.append(norm_tensor)
                    batch_gt.append(gt_label)
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch_arr = np.array(batch_imgs, dtype=np.float32)
        preds = model.predict(batch_arr, verbose=0)

        for p_vec, gt in zip(preds, batch_gt):
            p_idx = int(np.argmax(p_vec))
            pred_class = CLASS_NAMES[p_idx]
            conf = float(p_vec[p_idx])

            y_true.append(gt)
            y_pred.append(pred_class)
            y_conf.append(conf)

        if (i // EVAL_BATCH_SIZE) % 10 == 0 or i + EVAL_BATCH_SIZE >= total_samples:
            done = min(i + EVAL_BATCH_SIZE, total_samples)
            rate = done / max(time.time() - t0, 0.001)
            print(f"  Processed {done}/{total_samples} Ugandan images ({rate:.1f} img/s)...")

    eval_time = time.time() - t0
    valid_count = len(y_true)
    print(f"\nUgandan Inference completed in {eval_time:.1f}s ({valid_count/eval_time:.1f} img/s).")

    # Metrics computation
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    uganda_accuracy = (correct / valid_count) * 100.0 if valid_count > 0 else 0.0

    gt_unique = ["Fungal_Disease", "Healthy", "Phoma"]
    pred_unique = CLASS_NAMES

    cm = np.zeros((len(gt_unique), len(pred_unique)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        r = gt_unique.index(yt)
        c = pred_unique.index(yp)
        cm[r, c] += 1

    print("\n" + "=" * 80)
    print("UGANDAN CONFUSION MATRIX (Rows: Ground Truth, Columns: Predicted):")
    print("=" * 80)
    header = f"{'Ground Truth':<16} | " + " | ".join([f"{c:>15}" for c in pred_unique]) + " | Total"
    print(header)
    print("-" * len(header))
    for r_idx, gt in enumerate(gt_unique):
        row_str = f"{gt:<16} | " + " | ".join([f"{cm[r_idx, c_idx]:>15}" for c_idx in range(len(pred_unique))])
        row_str += f" | {np.sum(cm[r_idx]):>5}"
        print(row_str)
    print("-" * len(header))
    col_totals = f"{'Total Predicted':<16} | " + " | ".join([f"{np.sum(cm[:, c_idx]):>15}" for c_idx in range(len(pred_unique))]) + f" | {valid_count:>5}"
    print(col_totals)
    print("=" * 80)

    print("\nPER-CLASS ACCURACY (RECALL) ON UGANDA:")
    for r_idx, gt in enumerate(gt_unique):
        row_sum = np.sum(cm[r_idx])
        correct_in_class = cm[r_idx, pred_unique.index(gt)]
        class_acc = (correct_in_class / row_sum) * 100 if row_sum > 0 else 0
        print(f" • {gt:<15}: {class_acc:6.2f}% ({correct_in_class}/{row_sum})")

    print("\nUGANDAN CLASSIFICATION REPORT:")
    report_dict = classification_report(y_true, y_pred, labels=gt_unique, target_names=gt_unique, digits=4, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=gt_unique, target_names=gt_unique, digits=4, zero_division=0))

    # Benchmark comparison
    baseline_acc = 45.50
    delta = uganda_accuracy - baseline_acc
    print("=" * 80)
    print("DOMAIN GENERALIZATION ACCURACY BENCHMARK:")
    print(f" • Baseline Model Accuracy (Before CLAHE + CutMix): {baseline_acc:.2f}%")
    print(f" • New Pipeline Accuracy   (With CLAHE + CutMix):   {uganda_accuracy:.2f}%")
    print(f" • Net Accuracy Change:                             {'+' if delta >= 0 else ''}{delta:.2f}%")
    print(f" • Goal Status (> 70.0%):                           {'PASSED' if uganda_accuracy >= 70.0 else 'UNMET'}")
    print("=" * 80)

    report_payload = {
        "baseline_accuracy": baseline_acc,
        "uganda_accuracy": uganda_accuracy,
        "delta": delta,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "valid_count": valid_count
    }
    with open(UGANDA_REPORT_JSON, "w") as f:
        json.dump(report_payload, f, indent=2)

    return uganda_accuracy, cm, report_dict


# ------------------------------------------------------------------------------
# 7. Main Pipeline Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("COFFEE LEAF CLASSIFIER: DOMAIN GENERALIZATION PIPELINE (CLAHE + CUTMIX)")
    print("=" * 80)

    # 1. Dataset generators
    train_gen = FastInMemoryCutMixSequence(
        DATASET_TRAIN,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        cutmix_prob=0.50,
        shuffle=True
    )

    eval_datagen = ImageDataGenerator(preprocessing_function=clahe_and_preprocess)

    val_gen = eval_datagen.flow_from_directory(
        DATASET_VAL,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = eval_datagen.flow_from_directory(
        DATASET_TEST,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # 2. Balanced class weights
    classes = train_gen.classes
    unique_classes = np.unique(classes)
    base_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=classes)
    class_weights = {int(idx): float(w) for idx, w in zip(unique_classes, base_weights)}

    # 3. Build model
    model, base_model = build_mobilenetv2(num_classes=NUM_CLASSES)

    # 4. Train Phase 1 and Phase 2
    model = train_two_phases(model, base_model, train_gen, val_gen, class_weights)

    # Load best checkpoint
    if os.path.exists(MODEL_FINAL_PATH):
        print(f"\nLoading best checkpoint: {MODEL_FINAL_PATH}")
        model = models.load_model(MODEL_FINAL_PATH)

    # 5. Evaluate on Brazilian In-Domain Test Set
    print("\n" + "=" * 80)
    print("Evaluating In-Domain Brazilian Test Set (612 images with CLAHE)...")
    test_gen.reset()
    eval_brazil = model.evaluate(test_gen, verbose=1)
    print(f"Brazilian In-Domain Test Loss: {eval_brazil[0]:.4f} | Accuracy: {eval_brazil[1]*100:.2f}%")

    # 6. Evaluate on Ugandan Out-of-Domain Test Set (The Real Test)
    uganda_acc, cm, rep = evaluate_uganda_dataset(model)

    # 7. Export TFLite
    print("\nExporting to Quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes)

    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"TFLite exported: {TFLITE_PATH} ({size_mb:.2f} MB)")

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(TFLITE_PATH, os.path.join(ARTIFACT_DIR, "coffee_model_clahe_cutmix.tflite"))
        shutil.copy2(UGANDA_REPORT_JSON, os.path.join(ARTIFACT_DIR, "uganda_generalization_report_clahe_cutmix.json"))

    print("\nAll Training and Cross-Domain Generalization Steps Completed Successfully.")


if __name__ == "__main__":
    main()
