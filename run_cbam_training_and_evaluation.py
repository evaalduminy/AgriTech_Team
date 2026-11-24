# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Edge-AI Engineering: MobileNetV2 with Pre-GAP CBAM Attention Module
 (Channel Attention + Spatial Attention) & Categorical Crossentropy
==============================================================================
 Key Architecture Elements:
   1. Base: MobileNetV2 (ImageNet weights, input 224x224x3, output 7x7x1280)
   2. Pre-GAP Attention (CBAM):
      - Channel Attention: GAP + GMP -> Shared MLP(ratio=16) -> Sigmoid -> Multiply (7x7x1280)
      - Spatial Attention: Channel Mean + Channel Max -> Conv2D(7x7, sigmoid) -> Multiply (7x7x1280)
      * Forces network to highlight lesion spatial boundaries BEFORE spatial destruction by GAP *
   3. GlobalAveragePooling2D: applied to spatially calibrated 7x7x1280 feature maps
   4. Classification Head: Dense(256, relu) -> Dropout(0.5) -> Dense(5, softmax)
 Loss: Standard Categorical Crossentropy (Fixing Rust -> Healthy Over-Conservatism)
 Class Weights: Aggressive (Cerscospora at 11.2649)
 Data Pipeline: Fast In-Memory Targeted Augmentation (Cerscospora 3x extreme)
 Evaluation:
   - Full Test Set (612 images) Confusion Matrix & Classification Report
   - Inference on problematic 'Euro Green' & 'images.jfif' images
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
# 1. Hyperparameters & Directory Setup
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

MODEL_PHASE1_CHECKPOINT = "best_coffee_model_phase1_cbam.h5"
MODEL_FINAL_CHECKPOINT = "best_coffee_model_cbam_full.h5"
CONFUSION_MATRIX_OUTPUT = "confusion_matrix_cbam.png"
EVALUATION_REPORT_JSON = "test_evaluation_report_cbam.json"


# ------------------------------------------------------------------------------
# 2. Custom CBAM Layers (Channel Attention & Spatial Attention)
# ------------------------------------------------------------------------------
class ChannelAttention(layers.Layer):
    """
    Channel Attention Module (CAM):
    Aggregates spatial dimensions using both Average Pooling and Max Pooling,
    then forwards through a shared MLP to generate channel importance weights.
    """
    def __init__(self, channels=1280, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(max(channels // ratio, 32), activation="relu", kernel_initializer="he_normal")
        self.dense2 = layers.Dense(channels, kernel_initializer="he_normal")

    def call(self, inputs):
        avg_pool = self.dense2(self.dense1(self.gap(inputs)))
        max_pool = self.dense2(self.dense1(self.gmp(inputs)))
        attention = tf.sigmoid(avg_pool + max_pool)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        return inputs * attention

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"channels": self.channels, "ratio": self.ratio})
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module (SAM):
    Aggregates channel information using mean and max pooling along the channel axis,
    then applies a 7x7 convolution to generate a 2D spatial attention mask.
    This acts as a soft lesion mask highlighting diseased regions before GAP.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, padding="same", activation="sigmoid", use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


# ------------------------------------------------------------------------------
# 3. Fast In-Memory Targeted Augmentation Sequence
# ------------------------------------------------------------------------------
class FastTargetedAugmentationSequence(tf.keras.utils.Sequence):
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

        # Extreme augmentation for Cercospora
        self.cercospora_datagen = ImageDataGenerator(
            rotation_range=45,
            zoom_range=0.3,
            brightness_range=[0.6, 1.4],
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Standard mild augmentation for other classes
        self.standard_datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            brightness_range=[0.85, 1.15],
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Preload and cache images in RAM
        print(f"  • Preloading images into RAM from {directory}...")
        t0 = time.time()
        self.raw_images_cache = {}
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
                if fpath not in self.raw_images_cache:
                    with Image.open(fpath) as img:
                        rgb = img.convert("RGB")
                        if rgb.size != self.target_size:
                            rgb = rgb.resize(self.target_size, Image.BILINEAR)
                        self.raw_images_cache[fpath] = np.array(rgb, dtype=np.uint8)

                for _ in range(repeat):
                    self.samples.append((fpath, cls_idx))
                    self.augmented_counts[cls_name] += 1

        t_cache = time.time() - t0
        print(f"  • Cached {len(self.raw_images_cache)} images in {t_cache:.2f}s ({sum(v.nbytes for v in self.raw_images_cache.values()) / (1024*1024):.1f} MB RAM)")

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
            raw_arr = self.raw_images_cache[fpath]
            x = raw_arr.astype(np.float32)

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
# 4. Data Pipeline Factory
# ------------------------------------------------------------------------------
def get_dataset_generators():
    print(f"[1/5] Loading datasets:\n      Train: {DATASET_TRAIN}\n      Val:   {DATASET_VAL}\n      Test:  {DATASET_TEST}")

    train_generator = FastTargetedAugmentationSequence(
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
# 6. Model Architecture Builder with Pre-GAP CBAM Module
# ------------------------------------------------------------------------------
def build_disease_model(input_shape=(224, 224, 3), num_classes=5, reduction_ratio=16):
    """
    MobileNetV2 + Pre-GAP CBAM Attention Architecture:
      Input (224, 224, 3)
         ↓
      MobileNetV2 Backbone (frozen in Phase 1) -> Output: (7, 7, 1280)
         ↓
      Channel Attention Module (CAM): Squeeze & Excitation on (7, 7, 1280) -> (7, 7, 1280)
         ↓
      Spatial Attention Module (SAM): 7x7 Conv on Channel-Pooled Maps -> (7, 7, 1280)
         ↓  [Lesions highlighted before spatial pooling]
      GlobalAveragePooling2D -> (1280)
         ↓
      Dense(256, relu) -> Dropout(0.5) -> Dense(5, softmax)
    """
    print("[3/5] Building MobileNetV2 Architecture with Pre-GAP CBAM Module...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    feat_map = base_model.output # Shape: (None, 7, 7, 1280)

    # 1. Pre-GAP Channel Attention (SE-Block on 3D Feature Map)
    x = ChannelAttention(channels=1280, ratio=reduction_ratio, name="cbam_channel_attention")(feat_map)

    # 2. Pre-GAP Spatial Attention (Soft Lesion Mask)
    x = SpatialAttention(kernel_size=7, name="cbam_spatial_attention")(x)

    # 3. Global Average Pooling (Aggregates calibrated lesion features)
    x_gap = layers.GlobalAveragePooling2D(name="gap")(x)

    # 4. Dense Classification Head
    x = layers.Dense(256, activation="relu", name="dense_256")(x_gap)
    x = layers.Dropout(0.5, name="dropout")(x)
    predictions = layers.Dense(num_classes, activation="softmax", name="softmax_out")(x)

    model = models.Model(inputs=base_model.input, outputs=predictions, name="Coffee_MobileNetV2_CBAM")
    return model, base_model


# ------------------------------------------------------------------------------
# 7. Two-Phase Training Execution (Standard Categorical Crossentropy)
# ------------------------------------------------------------------------------
def train_two_phases(model, base_model, train_gen, val_gen, class_weights):
    loss_fn = "categorical_crossentropy"

    # ------------------ PHASE 1: Frozen Base ------------------
    print("\n" + "=" * 75)
    print(f"🚀 [Phase 1/2] Training Head + CBAM Block ({PHASE1_EPOCHS} Epochs, Adam lr={PHASE1_LR}, Crossentropy)...")
    print("=" * 75)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=loss_fn,
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
    print(f"🔓 [Phase 2/2] Fine-Tuning: Unfreezing Last 30 Layers (Max {PHASE2_EPOCHS} Epochs, Adam lr={PHASE2_LR}, Crossentropy)...")
    print("=" * 75)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=loss_fn,
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
# 8. Evaluation, Confusion Matrix, and Single-Image Inference
# ------------------------------------------------------------------------------
def evaluate_and_report(model, test_gen):
    print("\n" + "=" * 75)
    print("[5/5] Running Inference and Generating Strict Evaluation Report on Test Set...")
    print("=" * 75)

    test_gen.reset()
    y_true = test_gen.classes
    num_samples = len(y_true)

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

    # 3. Highlighted Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

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

    # Highlight Cercospora row
    rect_cercospora_row = patches.Rectangle(
        (0.02, 0.02), 4.96, 0.96,
        linewidth=3.5,
        edgecolor="#D32F2F",
        facecolor="none",
        linestyle="--",
        label="Cercospora Row (Target > 75%)"
    )
    ax.add_patch(rect_cercospora_row)

    # Highlight Leaf Rust -> Healthy cell [Row 2 (Leaf rust), Col 1 (Healthy)]
    rect_rust_healthy = patches.Rectangle(
        (1.02, 2.02), 0.96, 0.96,
        linewidth=3.5,
        edgecolor="#FF6F00",
        facecolor="none",
        linestyle="-",
        label="Rust -> Healthy Error Cell"
    )
    ax.add_patch(rect_rust_healthy)

    ax.set_title("Confusion Matrix (CBAM Attention + Crossentropy)\n[Pre-GAP Channel & Spatial Attention]", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model Predicted Label", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig.savefig(CONFUSION_MATRIX_OUTPUT, dpi=300)
    plt.close(fig)
    print(f"  • Highlighted Confusion Matrix saved to: {os.path.abspath(CONFUSION_MATRIX_OUTPUT)}")

    # Copy to artifact directory
    artifact_dir = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(CONFUSION_MATRIX_OUTPUT, os.path.join(artifact_dir, CONFUSION_MATRIX_OUTPUT))
        print(f"  • Mirrored Confusion Matrix image to artifact directory: {artifact_dir}")

    # 4. Strict Metrics Extraction
    cercospora_metrics = report_dict["Cerscospora"]
    precision_c = cercospora_metrics["precision"]
    recall_c = cercospora_metrics["recall"]
    f1_c = cercospora_metrics["f1-score"]
    support_c = int(cercospora_metrics["support"])

    overall_accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    rust_to_healthy_errors = int(cm[2, 1])

    # 5. Raw Output Printing
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
    print("🎯 CERCOSPORA & RUST METRICS:")
    print("=" * 75)
    print(f" • Cercospora Recall:         {recall_c * 100:.2f}% ({cm[0, 0]}/{support_c} images)")
    print(f" • Cercospora Precision:      {precision_c * 100:.2f}%")
    print(f" • Cercospora F1-score:       {f1_c:.4f}")
    print(f" • Rust -> Healthy Errors:    {rust_to_healthy_errors} images")
    print(f" • Overall Test Accuracy:     {overall_accuracy * 100:.2f}%")
    print("=" * 75 + "\n")

    report_data = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "cercospora": {
            "precision": float(precision_c),
            "recall": float(recall_c),
            "f1_score": float(f1_c),
            "support": int(support_c)
        },
        "rust_to_healthy_errors": rust_to_healthy_errors,
        "overall_accuracy": float(overall_accuracy)
    }
    with open(EVALUATION_REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data


# ------------------------------------------------------------------------------
# 9. Single-Image Inference on Problematic 'Euro Green' Images
# ------------------------------------------------------------------------------
def run_single_image_eval(model):
    print("=" * 75)
    print("🔍 INFERENCE ON PROBLEMATIC 'EURO GREEN' & TEST IMAGES")
    print("=" * 75)

    def predict_single(img_path, center_crop=True):
        with Image.open(img_path) as raw_img:
            rgb = raw_img.convert("RGB")
            w, h = rgb.size
            if center_crop:
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                rgb = rgb.crop((left, top, left + min_dim, top + min_dim))
            rgb = rgb.resize(IMAGE_SIZE, Image.BILINEAR)
            x = np.array(rgb, dtype=np.float32)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
        probs = model.predict(x)[0]
        return probs

    test_images = [
        ("Euro Green 1 (Center Cropped 1:1)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg", True),
        ("Euro Green 1 (Full Image - Uncropped)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg", False),
        ("Euro Green 2 (Center Cropped 1:1)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg", True),
        ("images.jfif (Downloads)", r"C:\Users\Hawaa\Downloads\images.jfif", False)
    ]

    single_results = {}
    for label, path, crop in test_images:
        if os.path.exists(path):
            probs = predict_single(path, center_crop=crop)
            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]
            conf = float(probs[pred_idx] * 100)

            print(f"\n📸 Image: {label}")
            print(f"   Path:  {path}")
            print(f"   Center Crop: {crop}")
            print(f"   Probability Distribution:")
            for idx, c_name in enumerate(CLASS_NAMES):
                print(f"     • {c_name:<12}: {probs[idx]:.6f} ({probs[idx] * 100:6.2f}%)")
            print(f"   --> Predicted Class: {pred_name} (Confidence: {conf:.2f}%)")
            print(f"   --> Cercospora Prob: {probs[0] * 100:.2f}%")

            single_results[label] = {
                "path": path,
                "predicted": pred_name,
                "confidence": conf,
                "probabilities": {c_name: float(probs[idx]) for idx, c_name in enumerate(CLASS_NAMES)}
            }
        else:
            print(f"\n⚠️ File not found: {path}")

    print("=" * 75 + "\n")
    return single_results


# ------------------------------------------------------------------------------
# 10. Main Pipeline Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("☕ MOBILE-NET V2 WITH PRE-GAP CBAM ATTENTION & CROSSENTROPY PIPELINE")
    print("=" * 80)

    # 1. Dataset generators
    train_gen, val_gen, test_gen = get_dataset_generators()

    # 2. Aggressive Class Weights
    class_weights = compute_training_class_weights(train_gen, cercospora_multiplier=3.0)

    # 3. Model Architecture with Pre-GAP CBAM Attention
    model, base_model = build_disease_model(num_classes=NUM_CLASSES, reduction_ratio=16)

    # 4. Train Phase 1 and Phase 2
    model, h1, h2 = train_two_phases(model, base_model, train_gen, val_gen, class_weights)

    # 5. Load best checkpoint
    custom_objs = {"ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention}
    if os.path.exists(MODEL_FINAL_CHECKPOINT):
        print(f"\n📦 Loading best checkpoint from {MODEL_FINAL_CHECKPOINT} for testing...")
        model = models.load_model(MODEL_FINAL_CHECKPOINT, custom_objects=custom_objs)

    # 6. Evaluation and Reporting on Test Set
    report_data = evaluate_and_report(model, test_gen)

    # 7. Single-Image Inference on Problematic 'Euro Green' Image
    single_results = run_single_image_eval(model)

    print("✅ CBAM Pipeline Execution Finished.")


if __name__ == "__main__":
    main()
