# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech: Production Training Pipeline (train_colab.py)
==============================================================================
تصنيف وتشخيص أمراض أوراق البن (5 فئات معتمدة) - مخصص لبيئة Google Colab (Linux)
تم تطبيق كافة ملاحظات التدقيق (N1 - N8) + البذور العشوائية + التحصينات الأربعة لكولاب:
1. TF_USE_LEGACY_KERAS = '1' لمنع تعارض Keras 3 في بيئات Colab الحديثة.
2. توكيد صارم (Assertions) لأعداد العينات النظيفة (2,835 / 606 / 612).
3. بند إجهاض صارم لبصمة المعاملات (1,054,661) لحماية أوزان ImageNet.
4. نسخ احتياطي تلقائي لكافة الأرتيفكتات إلى Google Drive فور اكتمال التدريب.

 سياسة الأرتيفكتات: نتائج التدريب على Windows CPU هي الكانونية المعتمدة في outputs/
وتشغيل Colab مخصص للتوثيق والقياس المقارن على معالجات GPU السحابية.
==============================================================================
"""

import os
# [تحصين كولاب 1] إجبار بيئة كولاب على استخدام Keras 2 لتفادي تشوهات Keras 3
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import random
import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. البذور العشوائية (Reproducibility Seeds)
# ------------------------------------------------------------------------------
def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

set_global_seeds(42)

# ------------------------------------------------------------------------------
# 1. إعدادات الثوابت والمعايير (مسار بيئة Google Colab)
# ------------------------------------------------------------------------------
DATASET_DIR = "/content/Coffee_Dataset_Split"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# إنشاء مجلد للإخراجات لتنظيم الملفات (N8)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"] 

LR_PHASE1 = 1e-3
EPOCHS_PHASE1 = 20
PATIENCE_PHASE1 = 5

LR_PHASE2 = 1e-5
EPOCHS_PHASE2 = 15
PATIENCE_PHASE2 = 7
UNFREEZE_LAST_N_LAYERS = 10

MODEL_PHASE1_PATH = os.path.join(OUTPUT_DIR, "best_coffee_model_phase1.h5")
MODEL_FINAL_PATH = os.path.join(OUTPUT_DIR, "best_coffee_model_final.h5")
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_evaluation_report.json")

# ------------------------------------------------------------------------------
# 2. فحص سلامة البيانات الدفاعي (N4 & N8)
# ------------------------------------------------------------------------------
def validate_dataset(min_imgs=5):
    for split in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not os.path.isdir(split):
            raise FileNotFoundError(f"مسار مفقود: {split}")
        for cls in CLASSES:
            cls_path = os.path.join(split, cls)
            if not os.path.isdir(cls_path):
                raise RuntimeError(f"مجلد فئة مفقود: {cls_path}")
            n = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            if n < min_imgs:
                raise RuntimeError(f"فئة ناقصة أو فارغة: {cls_path} ({n} صورة)")

# ------------------------------------------------------------------------------
# 3. مولدات البيانات مع التوكيد الصارم لأعداد العينات (تحصين كولاب 2)
# ------------------------------------------------------------------------------
def create_data_generators():
    print("=" * 70)
    print(" إعداد مولدات البيانات والتعزيز الموجه (Data Augmentation)...")
    print("=" * 70)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.15, horizontal_flip=True, vertical_flip=False,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=True, seed=42
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )

    # [تحصين كولاب 2] توكيد صارم لمنع تشغيل أي نسخة قديمة أو ملوثة من البيانات
    assert train_generator.samples == 2835, f" خطأ: عينات التدريب {train_generator.samples} != 2835"
    assert val_generator.samples == 606, f" خطأ: عينات التحقق {val_generator.samples} != 606"
    assert test_generator.samples == 612, f" خطأ: عينات الاختبار {test_generator.samples} != 612"
    print("✓ تم التحقق الصارم من صحة العينات المنظفة: 2,835 تدريب | 606 تحقق | 612 اختبار")

    return train_generator, val_generator, test_generator

# ------------------------------------------------------------------------------
# 4. حساب أوزان الفئات بأمان (N4)
# ------------------------------------------------------------------------------
def compute_training_class_weights(train_generator):
    classes = train_generator.classes
    unique_classes = np.unique(classes)
    weights = compute_class_weight("balanced", classes=unique_classes, y=classes)
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    
    print("\n أوزان الفئات المحسوبة حصرياً من مجلد التدريب:")
    for idx, cls_name in enumerate(CLASSES):
        w = class_weights_dict.get(idx)
        if w is not None:
            print(f"   • [{cls_name:<12}]: {w:.4f}")
        else:
            print(f"   • [{cls_name:<12}]: مفقودة! (تأكد من وجود صور للفئة)")
    return class_weights_dict

# ------------------------------------------------------------------------------
# 5. بناء النموذج والعدّاد الحقيقي (N3)
# ------------------------------------------------------------------------------
def build_model():
    print("\n" + "=" * 70)
    print(" بناء معمارية النموذج: MobileNetV2 Transfer Learning...")
    print("=" * 70)

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name="AgriTech_MobileNetV2")
    return model, base_model

# ------------------------------------------------------------------------------
# 6. المرحلة الأولى (Phase 1)
# ------------------------------------------------------------------------------
def train_phase1(model, train_gen, val_gen, class_weights):
    print("\n بدء المرحلة 1: Feature Extraction")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE1), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_PHASE1, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PHASE1_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history_p1 = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1,
        class_weight=class_weights, callbacks=callbacks
    )
    best_val_loss = min(history_p1.history["val_loss"])
    print(f"✓ اكتملت المرحلة 1 | Best Val Loss: {best_val_loss:.4f}")
    return history_p1, best_val_loss

# ------------------------------------------------------------------------------
# 7. المرحلة الثانية (Phase 2) - مع تحصين N1, N2 وبند الإجهاض
# ------------------------------------------------------------------------------
def train_phase2(model, base_model, train_gen, val_gen, class_weights, baseline_loss):
    print("\n بدء المرحلة 2: Fine-Tuning")
    
    # (N2) فك تجميد آخر 10 طبقات مع تجميد BatchNormalization
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAST_N_LAYERS:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    # (N3) العدّاد الحقيقي
    base_trainable = sum(1 for l in base_model.layers if l.trainable)
    head_trainable = sum(1 for l in model.layers if l is not base_model and len(l.weights) > 0)
    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"   عدد الطبقات القابلة للتدريب: {base_trainable + head_trainable} | معاملات: {trainable_params:,}")

    # [تحصين كولاب 3 - بند الإجهاض الصارم]
    assert trainable_params == 1054661, f" ABORT: بصمة المعاملات {trainable_params:,} لا تطابق 1,054,661 بالضبط!"

    model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE2), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_PHASE2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_FINAL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history_p2 = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2,
        class_weight=class_weights, callbacks=callbacks
    )

    phase2_best_val_loss = min(history_p2.history["val_loss"])
    guard_decision = "phase2_accepted"

    # (N1) الحارس الصارم وإعادة الحفظ على القرص
    if phase2_best_val_loss >= baseline_loss:
        print(f" Phase 2 Guard: خسارة المرحلة 2 ({phase2_best_val_loss:.4f}) ليست أفضل من المرحلة 1 ({baseline_loss:.4f}).")
        print(f"   استرجاع أوزان المرحلة 1 وإعادة حفظ النموذج على القرص...")
        model.load_weights(MODEL_PHASE1_PATH)
        model.save(MODEL_FINAL_PATH)
        guard_decision = "rollback_to_phase1"
    else:
        print(f"✓ تحسّن فعلي في المرحلة 2: {baseline_loss:.4f} → {phase2_best_val_loss:.4f}")

    return history_p2, guard_decision

# ------------------------------------------------------------------------------
# 8. التقييم النهائي (N4 & N6)
# ------------------------------------------------------------------------------
def evaluate_final(model, test_gen, guard_decision):
    print("\n التقييم النهائي المستقل على Test Set...")
    test_gen.reset()
    
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    loss, acc = model.evaluate(test_gen, verbose=0)
    
    cls_report_dict = classification_report(
        y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASSES,
        digits=4, output_dict=True, zero_division=0
    )
    print("\n تقرير التصنيف:")
    print(classification_report(y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASSES, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guard_decision": guard_decision,
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "classification_report": cls_report_dict,
        "confusion_matrix": cm.tolist()
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n✓ تم حفظ التقرير في: {REPORT_OUTPUT_PATH}")
    return cm

# ------------------------------------------------------------------------------
# 9. الرسوم البيانية (N7)
# ------------------------------------------------------------------------------
def plot_and_save_visualizations(history_p1, history_p2, cm, guard_decision):
    print("\nتوليد الرسوم البيانية (300 DPI)...")

    train_acc = history_p1.history["accuracy"] + history_p2.history["accuracy"]
    val_acc = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
    train_loss = history_p1.history["loss"] + history_p2.history["loss"]
    val_loss = history_p1.history["val_loss"] + history_p2.history["val_loss"]

    epochs_p1 = len(history_p1.history["loss"])
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs_range, train_acc, label="Train Accuracy", color="#1f77b4", linewidth=2.5)
    plt.plot(epochs_range, val_acc, label="Val Accuracy", color="#ff7f0e", linewidth=2.5, linestyle="--")
    plt.axvline(x=epochs_p1, color="#2ca02c", linestyle=":", label="Start Phase 2", linewidth=2)
    if guard_decision == "rollback_to_phase1":
        plt.axvline(x=epochs_p1 + len(history_p2.history["loss"]), color="red", linestyle="-.", label="Rolled back to P1", linewidth=2)
    plt.title("Accuracy Curve", fontsize=14, fontweight="bold")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300); plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs_range, train_loss, label="Train Loss (Weighted)", color="#1f77b4", linewidth=2.5)
    plt.plot(epochs_range, val_loss, label="Val Loss (Unweighted)", color="#d62728", linewidth=2.5, linestyle="--")
    plt.axvline(x=epochs_p1, color="#2ca02c", linestyle=":", label="Start Phase 2", linewidth=2)
    plt.title("Loss Curve (Note: Train is weighted, Val is unweighted)", fontsize=12, fontweight="bold")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300); plt.close()

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
    plt.title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300); plt.close()

# ------------------------------------------------------------------------------
# 10. الدالة الرئيسية وحفظ Google Drive (تحصين كولاب 4)
# ------------------------------------------------------------------------------
def main():
    print(" بدء خط إنتاج التدريب (مُحصّن ضد التسرب والأخطاء - Google Colab)...")
    validate_dataset() # N8
    train_gen, val_gen, test_gen = create_data_generators()
    class_weights = compute_training_class_weights(train_gen)
    model, base_model = build_model()

    history_p1, best_val_loss = train_phase1(model, train_gen, val_gen, class_weights)
    history_p2, guard_decision = train_phase2(model, base_model, train_gen, val_gen, class_weights, best_val_loss)
    cm = evaluate_final(model, test_gen, guard_decision)
    plot_and_save_visualizations(history_p1, history_p2, cm, guard_decision)

    # [تحصين كولاب 4] نسخ احتياطي فوري للأرتيفكتات إلى Google Drive
    try:
        from google.colab import drive
        print("\n جاري حفظ نسخة احتياطية من المخرجات في Google Drive...")
        drive.mount('/content/drive')
        import shutil
        backup_dir = f'/content/drive/MyDrive/coffee_outputs_{time.strftime("%Y%m%d_%H%M%S")}'
        shutil.copytree(OUTPUT_DIR, backup_dir)
        print(f"✓ تم حفظ نسخة احتياطية كاملة في: {backup_dir}")
    except Exception as e:
        print(f"  تم تجاوز نسخ Drive (جلسة محلية أو غير تفاعلية): {e}")

    print("\n اكتملت جميع المراحل بنجاح تام على Google Colab!")

if __name__ == "__main__":
    main()
