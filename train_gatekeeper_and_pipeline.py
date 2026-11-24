# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Edge-AI: Binary Gatekeeper Filter (Leaf vs. Not_Leaf) Training
 and Complete Cascaded 2-Stage Inference Pipeline
==============================================================================
 Pipeline Architecture:
   Stage 1: Binary Gatekeeper Filter (gatekeeper_leaf_filter.tflite)
            MobileNetV2 + Sigmoid -> P(Leaf)
            Threshold = 0.50
            Rejects Out-of-Distribution (OOD) objects, human faces, logos, etc.
   Stage 2: 4-Class Disease Classifier (coffee_model_final.tflite)
            Classes: ["Fungal_Disease", "Healthy", "Miner", "Phoma"]
            Threshold = 0.85
            Rejects unclear/ambiguous images ("Smart Rejection")
==============================================================================
"""

import os
import sys
import math
import time
import json
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ------------------------------------------------------------------------------
# 1. Constants & File Paths
# ------------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
GATEKEEPER_EPOCHS = 5
GATEKEEPER_LR = 1e-3

GATEKEEPER_DATASET_DIR = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier\Gatekeeper_Dataset"
TRAIN_DIR = os.path.join(GATEKEEPER_DATASET_DIR, "train")
VAL_DIR = os.path.join(GATEKEEPER_DATASET_DIR, "val")

GATEKEEPER_H5_PATH = "gatekeeper_leaf_filter.h5"
GATEKEEPER_TFLITE_PATH = "gatekeeper_leaf_filter.tflite"

DISEASE_MODEL_TFLITE_PATH = "coffee_model_final.tflite"
DISEASE_CLASS_NAMES = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"


# ------------------------------------------------------------------------------
# 2. Gatekeeper Data Generators
# ------------------------------------------------------------------------------
def get_gatekeeper_generators():
    print(f"[1/5] Loading Gatekeeper dataset from: {GATEKEEPER_DATASET_DIR}")
    
    # Mild augmentation for binary robustness
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Explicit mapping: Not_Leaf = 0, Leaf = 1
    # This guarantees sigmoid output represents P(Leaf)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=["Not_Leaf", "Leaf"],
        class_mode="binary",
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=["Not_Leaf", "Leaf"],
        class_mode="binary",
        shuffle=False
    )
    
    print(f"      • Class Mapping: {train_generator.class_indices}")
    print(f"      • Train Samples: {train_generator.samples} (Leaf: 200, Not_Leaf: 200)")
    print(f"      • Val Samples:   {val_generator.samples} (Leaf: 50, Not_Leaf: 50)")
    
    return train_generator, val_generator


# ------------------------------------------------------------------------------
# 3. Build & Train Gatekeeper Model (MobileNetV2 + Sigmoid)
# ------------------------------------------------------------------------------
def train_gatekeeper_model(train_gen, val_gen):
    print("\n[2/5] Building MobileNetV2 Binary Gatekeeper (Frozen Base + Sigmoid)...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    output = layers.Dense(1, activation="sigmoid", name="leaf_probability")(x)
    
    model = models.Model(inputs=base_model.input, outputs=output, name="Gatekeeper_Leaf_Filter")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=GATEKEEPER_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"\n[3/5] Training Gatekeeper for {GATEKEEPER_EPOCHS} Epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=GATEKEEPER_EPOCHS,
        verbose=1
    )
    
    # Save H5
    model.save(GATEKEEPER_H5_PATH)
    print(f"      • Gatekeeper Model saved to: {os.path.abspath(GATEKEEPER_H5_PATH)}")
    
    final_val_acc = history.history["val_accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    print(f"      • Final Validation Accuracy: {final_val_acc * 100:.2f}% | Loss: {final_val_loss:.4f}")
    
    return model


# ------------------------------------------------------------------------------
# 4. Export Gatekeeper to TFLite (Optimize.DEFAULT)
# ------------------------------------------------------------------------------
def export_gatekeeper_tflite(model):
    print("\n[4/5] Exporting Gatekeeper to TFLite (Optimize.DEFAULT)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    
    with open(GATEKEEPER_TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes)
        
    size_bytes = os.path.getsize(GATEKEEPER_TFLITE_PATH)
    size_mb = size_bytes / (1024 * 1024)
    
    # Mirror to artifact directory
    if os.path.exists(ARTIFACT_DIR):
        artifact_tflite = os.path.join(ARTIFACT_DIR, GATEKEEPER_TFLITE_PATH)
        with open(artifact_tflite, "wb") as f:
            f.write(tflite_bytes)
        print(f"      • Mirrored to Artifact Dir: {artifact_tflite}")
        
    print(f"      • Gatekeeper TFLite Path:   {os.path.abspath(GATEKEEPER_TFLITE_PATH)}")
    print(f"      • Exact TFLite File Size:   {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    return size_mb, size_bytes


# ------------------------------------------------------------------------------
# 5. Cascaded Inference Pipeline (Gatekeeper + Smart Rejection + Disease)
# ------------------------------------------------------------------------------
class TFLiteClassifier:
    """Wrapper for fast TFLite model execution."""
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


def preprocess_image_1to1(image_path, target_size=IMAGE_SIZE):
    """
    Simulates production mobile camera preprocessing:
    1. Square (1:1) center-crop matching the square viewfinder.
    2. High-quality resize to 224x224.
    3. MobileNetV2 normalization [-1.0, 1.0].
    """
    with Image.open(image_path) as raw_img:
        rgb = raw_img.convert("RGB")
        w, h = rgb.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = rgb.crop((left, top, left + min_dim, top + min_dim))
        resized = cropped.resize(target_size, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        norm_arr = preprocess_input(arr)
        batch = np.expand_dims(norm_arr, axis=0)
    return batch


def predict_with_gatekeeper(image_path,
                            gatekeeper_tflite_path=GATEKEEPER_TFLITE_PATH,
                            disease_tflite_path=DISEASE_MODEL_TFLITE_PATH,
                            gatekeeper_threshold=0.50,
                            confidence_threshold=0.85):
    """
    Production Cascaded Inference Pipeline:
    --------------------------------------
    Step 1: Gatekeeper Filter
      - If P(Leaf) < 0.50: Short-circuit rejection (Out-of-Distribution).
    Step 2: 4-Class Disease Classifier
      - If max_confidence < 0.85: Smart Rejection (Image unclear or early-stage).
      - If max_confidence >= 0.85: Confirmed diagnosis.
    """
    if not os.path.exists(image_path):
        return {"status": "Error", "message": f"Image file not found: {image_path}"}
        
    # Preprocess once
    tensor = preprocess_image_1to1(image_path)
    
    # ------------------ STAGE 1: GATEKEEPER ------------------
    gk_engine = TFLiteClassifier(gatekeeper_tflite_path)
    gk_pred = gk_engine.predict(tensor)
    gatekeeper_score = float(gk_pred[0][0])
    
    if gatekeeper_score < gatekeeper_threshold:
        return {
            "status": "Rejected",
            "stage": "Gatekeeper_OOD_Filter",
            "decision": "Not_A_Leaf",
            "gatekeeper_score": gatekeeper_score,
            "gatekeeper_score_percent": f"{gatekeeper_score * 100:.2f}%",
            "message": "⚠️ الصورة لا تحتوي على ورقة نبات. يرجى توجيه الكاميرا نحو ورقة البن."
        }
        
    # ------------------ STAGE 2: 4-CLASS DISEASE MODEL ------------------
    disease_engine = TFLiteClassifier(disease_tflite_path)
    disease_probs = disease_engine.predict(tensor)[0]
    
    pred_idx = int(np.argmax(disease_probs))
    pred_class = DISEASE_CLASS_NAMES[pred_idx]
    max_confidence = float(disease_probs[pred_idx])
    
    all_probs = {DISEASE_CLASS_NAMES[i]: float(disease_probs[i]) for i in range(len(DISEASE_CLASS_NAMES))}
    
    # Smart Rejection Guard (threshold = 0.85)
    if max_confidence < confidence_threshold:
        return {
            "status": "Rejected",
            "stage": "Smart_Rejection_Threshold",
            "decision": "Uncertain",
            "gatekeeper_score": gatekeeper_score,
            "top_candidate": pred_class,
            "max_confidence": max_confidence,
            "confidence_percent": f"{max_confidence * 100:.2f}%",
            "probabilities": all_probs,
            "message": "⚠️ الصورة غير واضحة أو أن المرض في مرحلة مبكرة جداً. يرجى إعادة التصوير بإضاءة أفضل والاقتراب من الورقة."
        }
        
    return {
        "status": "Accepted",
        "stage": "Confirmed_Diagnosis",
        "decision": pred_class,
        "gatekeeper_score": gatekeeper_score,
        "confidence": max_confidence,
        "confidence_percent": f"{max_confidence * 100:.2f}%",
        "probabilities": all_probs,
        "message": f"✅ تم تشخيص الورقة بنجاح: {pred_class} بنسبة ثقة {max_confidence * 100:.2f}%."
    }


# ------------------------------------------------------------------------------
# 6. Live Testing on Diverse Test Suite
# ------------------------------------------------------------------------------
def run_pipeline_verification():
    print("\n[5/5] Testing Cascaded Pipeline on Test Suite...")
    print("=" * 75)
    
    test_cases = [
        # 1. Botanical Leaf Images (Should Pass Gatekeeper)
        ("Euro Green 1 (Coffee Leaf)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg"),
        ("Euro Green 2 (Coffee Leaf)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg"),
        ("images.jfif (Web Leaf)", r"C:\Users\Hawaa\Downloads\images.jfif"),
        
        # 2. Non-Botanical OOD Images (Should Be Blocked by Gatekeeper)
        ("Foreign OOD: 2.png", r"D:\probe_foreign\2.png"),
        ("Foreign OOD: 1764100816328~2.jpg", r"D:\probe_foreign\1764100816328~2.jpg"),
        ("Foreign OOD: download (6).jfif", r"D:\probe_foreign\download (6).jfif")
    ]
    
    results = []
    for label, path in test_cases:
        if os.path.exists(path):
            res = predict_with_gatekeeper(path)
            res["test_label"] = label
            res["image_path"] = path
            results.append(res)
            
            print(f"\n🧪 Test Case: {label}")
            print(f"   Status:           {res['status']}")
            print(f"   Stage:            {res['stage']}")
            print(f"   Gatekeeper Score: {res.get('gatekeeper_score', 0):.4f} (Leaf Prob)")
            if res["status"] == "Accepted":
                print(f"   Diagnosis:        {res['decision']} ({res['confidence_percent']})")
            elif res["stage"] == "Smart_Rejection_Threshold":
                print(f"   Top Candidate:    {res['top_candidate']} ({res['confidence_percent']}) [Below 85%]")
            print(f"   Message:          {res['message']}")
        else:
            print(f"⚠️ Skipped missing file: {path}")
            
    print("\n" + "=" * 75)
    
    with open("cascaded_pipeline_test_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    return results


# ------------------------------------------------------------------------------
# 7. Main Driver
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🌿 BINARY GATEKEEPER & CASCADED EDGE-AI PIPELINE")
    print("=" * 80)
    
    # 1. Dataset
    train_gen, val_gen = get_gatekeeper_generators()
    
    # 2. Train
    model = train_gatekeeper_model(train_gen, val_gen)
    
    # 3. Export TFLite
    size_mb, size_bytes = export_gatekeeper_tflite(model)
    
    # 4. Verify Cascaded Pipeline
    run_pipeline_verification()
    
    print("\n🎉 Gatekeeper and Pipeline Implementation Complete!")


if __name__ == "__main__":
    main()
