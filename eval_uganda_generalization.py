# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech Edge-AI: Generalization Evaluation on Ugandan Coffee Leaf Dataset
==============================================================================
 Evaluates Stage 2 (4-Class Disease Classifier: coffee_model_final.tflite)
 on the unseen Ugandan Dataset domain:
   - "Health leaves" -> Ground Truth: "Healthy"
   - "leaf rust"     -> Ground Truth: "Fungal_Disease"
   - "phoma"         -> Ground Truth: "Phoma"
 (Miner is not in ground truth, but model may predict it)

 Model Classes: ["Fungal_Disease", "Healthy", "Miner", "Phoma"]
 Preprocessing: 1:1 center-crop, resize to (224, 224), normalize [-1.0, 1.0].
==============================================================================
"""

import os
import glob
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
MODEL_PATH = os.path.join(WORKSPACE, "coffee_model_final.tflite")
UGANDA_ROOT = r"C:\Users\Hawaa\Downloads\A Machine  Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda\coffee dataset"

TARGET_CLASSES = {
    "Health leaves": "Healthy",
    "leaf rust": "Fungal_Disease",
    "phoma": "Phoma"
}

CLASS_NAMES_4 = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES_4)}

def load_tflite_model(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    return interp, in_idx, out_idx

def preprocess_image(image_path, target_size=(224, 224)):
    with Image.open(image_path) as raw_img:
        rgb = raw_img.convert("RGB")
        w, h = rgb.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = rgb.crop((left, top, left + min_dim, top + min_dim))
        resized = cropped.resize(target_size, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        norm = (arr / 127.5) - 1.0
        return np.expand_dims(norm, axis=0)

def main():
    print("=" * 80)
    print("🌍 Domain Generalization Evaluation: Ugandan Coffee Leaf Dataset")
    print(f"Dataset root: {UGANDA_ROOT}")
    print(f"Model path:   {MODEL_PATH}")
    print("=" * 80)

    interp, in_idx, out_idx = load_tflite_model(MODEL_PATH)

    # Collect images
    samples = []
    for folder_name, gt_label in TARGET_CLASSES.items():
        folder_path = os.path.join(UGANDA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory not found: {folder_path}")
            continue
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        print(f"  • Ground Truth [{gt_label:<14}] ('{folder_name}'): {len(files)} images")
        for f in files:
            samples.append((f, gt_label))

    total_samples = len(samples)
    print(f"\nTotal test samples: {total_samples}")
    print("-" * 80)

    y_true = []
    y_pred = []
    y_conf = []
    corrupted_files = []

    start_time = time.time()
    for idx, (img_path, gt_label) in enumerate(samples, 1):
        try:
            tensor = preprocess_image(img_path)
        except Exception as e:
            corrupted_files.append((img_path, str(e)))
            continue

        interp.set_tensor(in_idx, tensor)
        interp.invoke()
        probs = interp.get_tensor(out_idx)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES_4[pred_idx]
        confidence = float(probs[pred_idx])

        y_true.append(gt_label)
        y_pred.append(pred_label)
        y_conf.append(confidence)

        if idx % 500 == 0 or idx == total_samples:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            print(f"  Processed {idx}/{total_samples} images ({rate:.1f} img/s)...")

    valid_samples = len(y_true)
    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds ({valid_samples/total_time:.1f} img/s).")
    if corrupted_files:
        print(f"⚠️ Skipped {len(corrupted_files)} unreadable/corrupted images.")
    print("=" * 80)

    # Compute overall accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / valid_samples if valid_samples > 0 else 0

    print("\n📊 OVERALL GENERALIZATION ACCURACY:")
    print(f"   Accuracy: {accuracy * 100:.2f}% ({correct}/{valid_samples})")
    print(f"   Mean Confidence: {np.mean(y_conf) * 100:.2f}%")
    print("=" * 80)

    # Confusion matrix
    # Ground truth labels present in test set: ["Fungal_Disease", "Healthy", "Phoma"]
    # Model predictions can be any of 4: ["Fungal_Disease", "Healthy", "Miner", "Phoma"]
    gt_unique = ["Fungal_Disease", "Healthy", "Phoma"]
    pred_unique = CLASS_NAMES_4

    # Build 3x4 confusion matrix
    cm = np.zeros((len(gt_unique), len(pred_unique)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        r = gt_unique.index(yt)
        c = pred_unique.index(yp)
        cm[r, c] += 1

    print("\n🔍 CONFUSION MATRIX (Rows: Ground Truth, Columns: Predicted):")
    header = f"{'Ground Truth':<16} | " + " | ".join([f"{c:>15}" for c in pred_unique]) + " | Total"
    print(header)
    print("-" * len(header))
    for r_idx, gt in enumerate(gt_unique):
        row_str = f"{gt:<16} | " + " | ".join([f"{cm[r_idx, c_idx]:>15}" for c_idx in range(len(pred_unique))])
        row_str += f" | {np.sum(cm[r_idx]):>5}"
        print(row_str)
    print("-" * len(header))
    col_totals = f"{'Total Predicted':<16} | " + " | ".join([f"{np.sum(cm[:, c_idx]):>15}" for c_idx in range(len(pred_unique))]) + f" | {valid_samples:>5}"
    print(col_totals)

    # Per-class accuracy / recall
    print("\n📈 PER-CLASS RECALL (ACCURACY WITHIN GROUND TRUTH):")
    for r_idx, gt in enumerate(gt_unique):
        row_sum = np.sum(cm[r_idx])
        correct_in_class = cm[r_idx, pred_unique.index(gt)]
        class_acc = (correct_in_class / row_sum) * 100 if row_sum > 0 else 0
        print(f" • {gt:<15}: {class_acc:6.2f}% ({correct_in_class}/{row_sum})")

    # Classification report (evaluating on the 3 present ground truth classes)
    print("\n📋 CLASSIFICATION REPORT:")
    report = classification_report(
        y_true,
        y_pred,
        labels=gt_unique,
        target_names=gt_unique,
        digits=4,
        zero_division=0
    )
    print(report)

    # Detailed domain shift error analysis
    print("=" * 80)
    print("🔬 DOMAIN SHIFT ERROR BREAKDOWN:")
    for r_idx, gt in enumerate(gt_unique):
        row_sum = np.sum(cm[r_idx])
        print(f"\n[Ground Truth: {gt} (N={row_sum})]")
        for c_idx, pred in enumerate(pred_unique):
            cnt = cm[r_idx, c_idx]
            pct = (cnt / row_sum) * 100
            note = "CORRECT" if gt == pred else "MISCLASSIFIED"
            print(f"   -> Predicted {pred:<15}: {cnt:>5} ({pct:5.1f}%) [{note}]")
    print("=" * 80)

if __name__ == "__main__":
    main()
