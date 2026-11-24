# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Deep Learning: MobileNetV2 with Targeted Augmentation & Focal Loss
==============================================================================
 Data Pipeline:
   - Targeted Augmentation Sequence:
       * Cercospora: Extreme Augmentation (rotation 45°, zoom 0.3, brightness [0.6, 1.4], shifts 0.2)
       * Cercospora 3x Multiplier: Sampled 3x per epoch (151 -> 453 samples)
       * Healthy, Rust, Miner, Phoma: Standard Mild Augmentation (rotation 15°, zoom 0.1, brightness [0.85, 1.15])
       * Preprocessing: MobileNetV2 preprocess_input applied strictly AFTER augmentation
 Loss: Categorical Focal Loss (gamma=2.0, alpha=0.25)
 Class Weighting: Sklearn balanced weights with 3x manual multiplier for Cercospora
 Architecture: MobileNetV2 (frozen base in Phase 1) + Custom Head
==============================================================================
"""

import os
import sys
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------------------
# 1. Pipeline Constants & Hyperparameters
# ------------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

# Priority dataset path: Sanitized dataset (border text/watermark removed)
DATASET_TRAIN = r"D:\Coffee_Dataset_Sanitized\train"
DATASET_VAL = r"D:\Coffee_Dataset_Sanitized\val"

if not (os.path.exists(DATASET_TRAIN) and os.path.exists(DATASET_VAL)):
    DATASET_TRAIN = r"D:\Coffee_Dataset_Split\train"
    DATASET_VAL = r"D:\Coffee_Dataset_Split\val"

CLASS_NAMES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
MODEL_CHECKPOINT_PATH = "best_coffee_model_phase1_focal.h5"


# ------------------------------------------------------------------------------
# 2. Custom Focal Loss Implementation (gamma=2.0, alpha=0.25)
# ------------------------------------------------------------------------------
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Categorical Focal Loss for addressing extreme class imbalance and hard examples:
      FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified easy samples and focuses gradients on minority/hard
    samples (e.g. Cercospora lesions confused with Leaf Rust).
    """
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        # Clip predicted probabilities to prevent log(0) NaN instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy component: - y_true * log(y_pred)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Modulating factor: alpha * (1 - p_t)^gamma
        focal_weight = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma)
        
        # Focal loss per sample
        loss = tf.math.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return tf.math.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ------------------------------------------------------------------------------
# 3. Targeted Augmentation Sequence (Data Engineering Implementation)
# ------------------------------------------------------------------------------
class TargetedAugmentationSequence(tf.keras.utils.Sequence):
    """
    Data Engineering Pipeline: Targeted Conditional Augmentation
    -----------------------------------------------------------
    Requirements fulfilled:
      1. Extreme Augmentation (Cercospora):
           rotation_range=45, zoom_range=0.3, brightness_range=[0.6, 1.4],
           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
      2. Standard Mild Augmentation (Healthy, Rust, Miner, Phoma):
           rotation_range=15, zoom_range=0.1, brightness_range=[0.85, 1.15],
           width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
      3. Cercospora 3x Multiplier:
           Cercospora samples are repeated 3x with stochastic transforms per epoch.
      4. MobileNetV2 Preprocessing:
           Applied strictly AFTER augmentation (scaling pixels from [0, 255] to [-1.0, 1.0]).
    """
    def __init__(self, directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, cercospora_factor=3):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cercospora_factor = cercospora_factor

        # Discover classes sorted alphabetically (matching Keras flow_from_directory)
        self.class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.cercospora_idx = self.class_indices.get("Cerscospora", 0)

        # 1. Extreme augmentation generator for Cercospora
        self.cercospora_datagen = ImageDataGenerator(
            rotation_range=45,
            zoom_range=0.3,
            brightness_range=[0.6, 1.4],
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # 2. Standard mild augmentation generator for other classes
        self.standard_datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            brightness_range=[0.85, 1.15],
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        # Build sample registry
        self.samples = []
        self.raw_counts = {c: 0 for c in self.class_names}
        self.raw_classes_list = []
        self.augmented_counts = {c: 0 for c in self.class_names}
        self.actual_augmented_counter = {c: 0 for c in self.class_names}

        for cls_name in self.class_names:
            cls_dir = os.path.join(directory, cls_name)
            cls_idx = self.class_indices[cls_name]
            fnames = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.raw_counts[cls_name] = len(fnames)
            self.raw_classes_list.extend([cls_idx] * len(fnames))

            # Cercospora gets 3x oversampling with extreme stochastic augmentation
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

            # Conditional targeted augmentation
            cls_name = self.class_names[cls_idx]
            if cls_idx == self.cercospora_idx:
                x = self.cercospora_datagen.random_transform(x)
            else:
                x = self.standard_datagen.random_transform(x)

            self.actual_augmented_counter[cls_name] += 1

            # MobileNetV2 preprocessing applied strictly AFTER augmentation
            x = preprocess_input(x)

            # One-hot label
            y = np.zeros(self.num_classes, dtype=np.float32)
            y[cls_idx] = 1.0

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ------------------------------------------------------------------------------
# 4. Data Generators Factory
# ------------------------------------------------------------------------------
def get_data_generators(train_dir, val_dir):
    print(f"[1/4] Loading dataset with Targeted Augmentation Pipeline:\n      Train: {train_dir}\n      Val:   {val_dir}")
    
    train_generator = TargetedAugmentationSequence(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        cercospora_factor=3
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print(f"\n📊 DATA PIPELINE SUMMARY:")
    print(f" • Total training batches per epoch: {len(train_generator)}")
    print(f" • Raw training images: {sum(train_generator.raw_counts.values())}")
    print(f" • Effective augmented training images per epoch: {len(train_generator.samples)}")
    print(f" • Cercospora raw vs augmented: {train_generator.raw_counts['Cerscospora']} -> {train_generator.augmented_counts['Cerscospora']} (3.0x multiplier)")

    return train_generator, val_generator


# ------------------------------------------------------------------------------
# 5. Aggressive Class Weights Calculation (3x Cercospora Penalty)
# ------------------------------------------------------------------------------
def compute_aggressive_class_weights(train_generator, cercospora_multiplier=3.0):
    """
    Computes sklearn balanced class weights and applies aggressive 3x multiplier
    to the Cercospora class weight to eliminate false-negative misclassifications.
    """
    print("[2/4] Computing balanced class weights with 3x Cercospora penalty...")
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
    
    original_cercospora_weight = class_weights_dict[cercospora_idx]
    class_weights_dict[cercospora_idx] *= cercospora_multiplier

    print("\n" + "=" * 65)
    print("⚖️  EXACT CLASS WEIGHTS DICTIONARY (PRE-TRAINING VERIFICATION):")
    print("=" * 65)
    for c_name, c_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        w = class_weights_dict[c_idx]
        note = f" <-- (3x Multiplier Applied: {original_cercospora_weight:.4f} -> {w:.4f})" if c_idx == cercospora_idx else ""
        print(f" • Class {c_idx} [{c_name:<11}]: weight = {w:.4f}{note}")
    print("=" * 65 + "\n")

    return class_weights_dict


# ------------------------------------------------------------------------------
# 6. Model Architecture (MobileNetV2 Backbone + Custom Head)
# ------------------------------------------------------------------------------
def build_mobilenetv2_focal(input_shape=(224, 224, 3), num_classes=5):
    print("[3/4] Initializing MobileNetV2 architecture with Focal Loss...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Freeze the backbone for Phase 1
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    predictions = layers.Dense(num_classes, activation="softmax", name="softmax_out")(x)

    model = models.Model(inputs=base_model.input, outputs=predictions, name="MobileNetV2_Focal")
    return model, base_model


# ------------------------------------------------------------------------------
# 7. Phase 1 Training Execution
# ------------------------------------------------------------------------------
def run_phase1_training():
    print("=" * 75)
    print("🔬 Deep Learning Execution: Phase 1 Training with Focal Loss & 3x Weights")
    print("=" * 75)

    # 1. Generators
    train_gen, val_gen = get_data_generators(DATASET_TRAIN, DATASET_VAL)

    # 2. Aggressive Class Weights
    class_weights = compute_aggressive_class_weights(train_gen, cercospora_multiplier=3.0)

    # 3. Model Architecture
    model, base_model = build_mobilenetv2_focal(num_classes=NUM_CLASSES)

    # 4. Compile with Custom Focal Loss (gamma=2.0, alpha=0.25)
    focal_loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.25)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=focal_loss_fn,
        metrics=["accuracy"]
    )

    # 5. Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # 6. Train Phase 1
    print("\n" + "-" * 75)
    print(f"🚀 Training Phase 1: Frozen Base, 10 Epochs, Adam(lr={PHASE1_LR})...")
    print("-" * 75)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb]
    )

    # 7. Validation Loss Reporting
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print("\n" + "=" * 75)
    print("🏁 PHASE 1 TRAINING SUMMARY (FOCAL LOSS EVALUATION)")
    print("=" * 75)
    print(f" • Best Validation Loss:    {best_val_loss:.4f} (Epoch {history.history['val_loss'].index(best_val_loss) + 1})")
    print(f" • Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f" • Final Validation Loss:   {final_val_loss:.4f}")
    print(f" • Final Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f" • Checkpoint Saved To:     {os.path.abspath(MODEL_CHECKPOINT_PATH)}")
    print("=" * 75 + "\n")

    return history


if __name__ == "__main__":
    run_phase1_training()
