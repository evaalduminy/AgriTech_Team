# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Edge-AI: Evaluation & Smart Rejection Benchmark
==============================================================================
 Computes:
 1. Standard Test Accuracy & Raw Confusion Matrix (All Test Samples)
 2. Post-Filter Performance with Smart Rejection (Confidence >= 0.85 vs < 0.85)
 3. Analyzes specific error reduction between Healthy and Leaf rust
==============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
MODEL_PATH = "best_coffee_model.h5"
if not os.path.exists(MODEL_PATH) and os.path.exists("final.h5"):
    MODEL_PATH = "final.h5"

DATASET_DIR = os.environ.get("DATASET_DIR", "Coffee_Dataset_Split")
if not os.path.exists(DATASET_DIR):
    for candidate in ["/content/Coffee_Dataset_Split", "D:\\Coffee_OOD_Run\\Coffee_Dataset_Split"]:
        if os.path.exists(candidate):
            DATASET_DIR = candidate
            break

TEST_DIR = os.path.join(DATASET_DIR, "test")


def evaluate_pipeline(model_path=MODEL_PATH, test_dir=TEST_DIR, threshold=0.85):
    print(f"🔍 Loading model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"📂 Collecting test samples from: {test_dir} ...")
    all_y_true = []
    all_y_pred = []
    all_confidences = []
    
    accepted_y_true = []
    accepted_y_pred = []
    rejected_count = 0

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        file_list = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        for fname in file_list:
            fpath = os.path.join(class_folder, fname)
            
            # 1. Center crop + resize to (224, 224)
            img = image.load_img(fpath)
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
            img_resized = img_cropped.resize((224, 224))
            
            # 2. Normalize [-1, 1]
            img_arr = (np.array(img_resized, dtype=np.float32) / 127.5) - 1.0
            batch = np.expand_dims(img_arr, axis=0)

            # 3. Predict
            probs = model.predict(batch, verbose=0)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))

            all_y_true.append(class_idx)
            all_y_pred.append(pred_idx)
            all_confidences.append(max_prob)

            # 4. Check Smart Rejection threshold
            if max_prob >= threshold:
                accepted_y_true.append(class_idx)
                accepted_y_pred.append(pred_idx)
            else:
                rejected_count += 1

    total_samples = len(all_y_true)
    accepted_samples = len(accepted_y_true)
    
    print("\n" + "=" * 70)
    print("📊 نتائج التقييم الشامل ومقارنة الرفض الذكي (Smart Rejection Benchmark)")
    print("=" * 70)
    print(f"• إجمالي عينات الاختبار المفحوصة:     {total_samples} صورة")
    print(f"• العينات المقبولة بثقة عالية (≥ 0.85): {accepted_samples} صورة ({accepted_samples/total_samples*100:.1f}%)")
    print(f"• العينات المحجوبة للتشكيك (< 0.85):   {rejected_count} صورة ({rejected_count/total_samples*100:.1f}%)")

    # Raw metrics
    raw_acc = np.mean(np.array(all_y_true) == np.array(all_y_pred)) * 100.0
    print(f"\n1. الدقة الخام قبل الرفض الذكي (جميع العينات): {raw_acc:.2f}%")

    # Filtered metrics
    if accepted_samples > 0:
        filtered_acc = np.mean(np.array(accepted_y_true) == np.array(accepted_y_pred)) * 100.0
        print(f"2. دقة التشخيص الميداني بعد الرفض الذكي (العينات الواضحة فقط): {filtered_acc:.2f}% 🚀")
        print(f"   (ارتفاع الدقة الميدانية بمقدار: +{filtered_acc - raw_acc:.2f}%)")

    # Plotting Confusion Matrix
    cm_raw = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Raw Confusion Matrix (Total: {total_samples}, Acc: {raw_acc:.2f}%)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('new_confusion_matrix.png', dpi=300)
    print("✓ تم حفظ مصفوفة الارتباك في: new_confusion_matrix.png")


if __name__ == "__main__":
    evaluate_pipeline()
