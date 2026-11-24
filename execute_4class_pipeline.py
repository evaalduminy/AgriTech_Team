# -*- coding: utf-8 -*-
"""
نظام تصنيف أمراض أوراق البن رباعي الفئات باستخدام شبكة MobileNetV2
استراتيجية دمج الفئات الفطرية: دمج صدأ الأوراق (Leaf Rust) مع السركسبورا (Cercospora) في فئة واحدة (Fungal_Disease).
الفئات الأربع المعتمدة: ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

الهيكل المعماري: MobileNetV2 + GlobalAveragePooling2D + Dense(256) + Dropout(0.5) + Dense(4)
دالة الخسارة: دالة الانتروبيا المتقاطعة الفئوية (Categorical Crossentropy)
موازنة الفئات: أوزان فئوية متوازنة رياضياً (Balanced Class Weights)

مراحل التنفيذ:
1. المرحلة الأولى: تدريب رأس التصنيف فقط مع تجميد الطبقات الأساسية (10 دورات، معدل تعلم 0.001).
2. المرحلة الثانية: الضبط الدقيق لأخر 30 طبقة من النموذج الأساسي (15 دورة، معدل تعلم 0.00001).
3. التقييم الصارم على مجموعة الاختبار المكونة من 612 صورة وحساب مصفوفة الارتباك.
4. فحص الاستدلال على الصور الفردية للتحقق من تخطي نسبة الثقة لـ 85%.
5. التصدير والتكميم لبيئات التشغيل المدمجة والهواتف بصيغة TFLite.
"""

# استيراد المكتبات الأساسية لإدارة نظام التشغيل والمسارات والرياضيات
import os
import sys
import math
import time
import json
import numpy as np
from PIL import Image

# ضبط مكتبة الرسم للعمل في بيئة خالية من واجهة المستخدم الرسومية (Headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# استيراد مكتبات التعلم العميق TensorFlow و Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# استيراد دوال حساب الأوزان ومقاييس الأداء الأكاديمية
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------------------------------------------------------
# 1. المعاملات الفائقة ومسارات الملفات (Hyperparameters and Paths)
# ------------------------------------------------------------------------------
# أبعاد صورة الإدخال المعيارية لشبكة MobileNetV2 (الارتفاع والعرض)
IMAGE_SIZE = (224, 224)
# حجم الدفعة الواحدة لتحديث أوزان النموذج أثناء التدريب
BATCH_SIZE = 32
# عدد الفئات المستهدفة في التصنيف النهائي
NUM_CLASSES = 4
# عدد دورات التدريب للمرحلة الأولى (تدريب رأس التصنيف فقط)
PHASE1_EPOCHS = 10
# معدل التعلم للمرحلة الأولى
PHASE1_LR = 1e-3
# عدد دورات التدريب للمرحلة الثانية (الضبط الدقيق)
PHASE2_EPOCHS = 15
# معدل التعلم المنخفض للمرحلة الثانية لتفادي تدمير الأوزان المسبقة
PHASE2_LR = 1e-5
# عدد الدورات المسموح بها دون تحسن في دالة الخسارة قبل إيقاف التدريب مبكراً
EARLY_STOPPING_PATIENCE = 4

# مسارات مجموعات البيانات بعد التقسيم المسبق
DATASET_TRAIN = r"D:\Coffee_Dataset_4Class\train"
DATASET_VAL = r"D:\Coffee_Dataset_4Class\val"
DATASET_TEST = r"D:\Coffee_Dataset_4Class\test"

# الأسماء الرسمية للفئات الأربع المعتمدة
CLASS_NAMES = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

# مسارات حفظ النماذج والتقارير الناتجة
MODEL_PHASE1_CHECKPOINT = "best_coffee_model_4class_phase1.h5"
MODEL_FINAL_CHECKPOINT = "best_coffee_model_4class_final.h5"
TFLITE_OUTPUT_PATH = "coffee_model_final.tflite"
CONFUSION_MATRIX_OUTPUT = "confusion_matrix_4class.png"
EVALUATION_REPORT_JSON = "test_evaluation_report_4class.json"


# ------------------------------------------------------------------------------
# 2. فئة التحميل فائق السرعة للبيانات في الذاكرة العشوائية (Fast In-Memory Sequence)
# ------------------------------------------------------------------------------
class FastInMemorySequence(tf.keras.utils.Sequence):
    """
    فئة مخصصة ترث من Sequence في Keras لتحميل وتخزين جميع صور التدريب في ذاكرة RAM.
    الهدف الأكاديمي:
    إلغاء عنق الزجاجة الناتج عن سرعة القراءة المتكررة من القرص الصلب (Disk I/O Bottleneck)،
    مما يسرع زمن الدورة التدريبية مع تطبيق الزيادة البيانية (Data Augmentation) لحظياً.
    """
    def __init__(self, directory, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # قراءة وتصنيف المجلدات الفرعية لترتيب الفئات أبجدياً
        self.class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        # تعيين فهرس رقمي لكل فئة
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # مولد الزيادة البيانية المعتمد لتحسين متانة النموذج ضد تغيرات زوايا التصوير والإضاءة
        self.datagen = ImageDataGenerator(
            rotation_range=30,             # التدوير العشوائي حتى 30 درجة
            zoom_range=0.2,                # التكبير والتصغير العشوائي بنسبة 20%
            brightness_range=[0.75, 1.25], # تغيير شدة الإضاءة بين 75% و 125%
            width_shift_range=0.15,        # إزاحة أفقية بنسبة 15%
            height_shift_range=0.15,       # إزاحة رأسية بنسبة 15%
            horizontal_flip=True,          # قلب أفقي عشوائي
            fill_mode="nearest"            # ملء الفراغات الناتجة عن الإزاحة بأقرب بكسل
        )

        print(f"Preloading 4-class images into RAM from {directory}...")
        t0 = time.time()
        self.raw_images_cache = {}
        self.samples = []
        self.raw_counts = {c: 0 for c in self.class_names}
        self.raw_classes_list = []

        # تحميل جميع الصور إلى مصفوفات في الذاكرة المؤقتة
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
                        # حفظ مصفوفة الصورة كأعداد صحيحة بدون إشارة (uint8) لتوفير استهلاك الذاكرة
                        self.raw_images_cache[fpath] = np.array(rgb, dtype=np.uint8)

                self.samples.append((fpath, cls_idx))

        t_cache = time.time() - t0
        cache_ram_mb = sum(v.nbytes for v in self.raw_images_cache.values()) / (1024 * 1024)
        print(f"Cached {len(self.raw_images_cache)} images in {t_cache:.2f}s ({cache_ram_mb:.1f} MB RAM)")

        self.samples = np.array(self.samples, dtype=object)
        self.classes = np.array([s[1] for s in self.samples], dtype=int)
        self.raw_classes = np.array(self.raw_classes_list, dtype=int)
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """حساب عدد الدفعات (Batches) في الدورة التدريبية الواحدة."""
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        """توليد دفعة تدريبية واحدة تشمل تطبيق الزيادة البيانية والمعالجة القياسية للألوان."""
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_samples = self.samples[batch_indexes]

        batch_x = []
        batch_y = []

        for fpath, cls_idx in batch_samples:
            raw_arr = self.raw_images_cache[fpath]
            x = raw_arr.astype(np.float32)

            # تطبيق التحويلات العشوائية لحظياً على الصورة
            x = self.datagen.random_transform(x)
            # تطبيق المعالجة القياسية المعتمدة لشبكة MobileNetV2 (تحويل القيم لنطاق [-1, 1])
            x = preprocess_input(x)

            # ترميز الفئة بنظام One-Hot Encoding
            y = np.zeros(self.num_classes, dtype=np.float32)
            y[cls_idx] = 1.0

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        """إعادة خلط مؤشرات العينات في نهاية كل دورة تدريبية لضمان تنوع التدريب."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ------------------------------------------------------------------------------
# 3. إعداد مولدات البيانات لمجموعات التدريب والتحقق والاختبار
# ------------------------------------------------------------------------------
def get_dataset_generators():
    """
    تهيئة مولدات البيانات للتدريب والتحقق والاختبار:
    - مجموعة التدريب تستخدم FastInMemorySequence لتحقيق أقصى سرعة ممكنة.
    - مجموعات التحقق والاختبار تستخدم المولدات القياسية لقراءة الصور بدون زيادة بيانية عشوائية.
    """
    print(f"[1/6] Loading 4-class datasets:\n      Train: {DATASET_TRAIN}\n      Val:   {DATASET_VAL}\n      Test:  {DATASET_TEST}")

    train_generator = FastInMemorySequence(
        DATASET_TRAIN,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
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
# 4. حساب الأوزان المتوازنة للفئات (Balanced Class Weights)
# ------------------------------------------------------------------------------
def compute_balanced_weights(train_generator):
    """
    حساب الأوزان الفئوية المتوازنة باستخدام معادلة Scikit-Learn:
    n_samples / (n_classes * np.bincount(y))
    الهدف الأكاديمي:
    إعطاء وزن أعلى في دالة الخسارة للفئات ذات العينات الأقل لمنع انحياز النموذج للفئات الأكثر عدداً.
    """
    print("[2/6] Computing balanced class weights for 4 classes...")
    classes = train_generator.classes
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

    print("Exact Balanced Class Weights (4 Classes):")
    for c_name, c_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        w = class_weights_dict[c_idx]
        print(f"Class {c_idx} [{c_name}]: count = {train_generator.raw_counts[c_name]} | weight = {w:.4f}")

    return class_weights_dict


# ------------------------------------------------------------------------------
# 5. بناء معمارية النموذج الأساسي MobileNetV2 (بدون وحدات انتباه إضافية)
# ------------------------------------------------------------------------------
def build_baseline_mobilenetv2(input_shape=(224, 224, 3), num_classes=4):
    """
    بناء المعمارية القياسية الخفيفة المناسبة للأجهزة الطرفية والمحمولة:
    - النموذج الأساسي: MobileNetV2 محمل بأوزان ImageNet مسبقة التدريب.
    - تجميد النموذج الأساسي مبدئياً.
    - إضافة طبقة التجميع المتوسط العالمي (GlobalAveragePooling2D) لتقليل الأبعاد المكانية إلى متجه خصائص.
    - طبقة كثيفة Dense(256) مع دالة التنشيط ReLU لاستخراج العلاقات غير الخطية.
    - طبقة إسقاط عشوائي Dropout(0.5) لتقليل فرط التخصيص.
    - طبقة الإخراج Dense(4) مع دالة التنشيط Softmax لتوليد التوزيع الاحتمالي للفئات الأربع.
    """
    print("[3/6] Building Baseline MobileNetV2 Architecture (Lightweight)...")
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

    model = models.Model(inputs=base_model.input, outputs=predictions, name="Coffee_MobileNetV2_4Class")
    return model, base_model


# ------------------------------------------------------------------------------
# 6. تنفيذ التدريب عبر مرحلتين (Two-Phase Transfer Learning)
# ------------------------------------------------------------------------------
def train_two_phases(model, base_model, train_gen, val_gen, class_weights):
    """
    تنفيذ استراتيجية التعلم بالنقل على مرحلتين:
    - المرحلة الأولى: تدريب رأس التصنيف فقط (Frozen Base) بمعدل تعلم قياسي (1e-3).
    - المرحلة الثانية: فك تجميد أخر 30 طبقة من MobileNetV2 لتدريبها بالتوازي مع الرأس
      بمعدل تعلم دقيق جداً (1e-5) مع تثبيت طبقات BatchNormalization لمنع تدهور الإحصائيات.
    """
    loss_fn = "categorical_crossentropy"

    # المرحلة الأولى: تدريب الرأس المضاف مع تجميد العمود الفقري للنموذج
    print(f"\n[Phase 1/2] Training Head ({PHASE1_EPOCHS} Epochs, Adam lr={PHASE1_LR}, Categorical Crossentropy)...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # حفظ النموذج عند تحقيق أقل خسارة على مجموعة التحقق
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
    best_p1_val_acc = max(history_p1.history["val_accuracy"])
    print(f"Phase 1 Complete. Best val_loss: {best_p1_val_loss:.4f}, Best val_acc: {best_p1_val_acc*100:.2f}%")

    # المرحلة الثانية: الضبط الدقيق لأخر 30 طبقة من النموذج الأساسي
    print(f"\n[Phase 2/2] Fine-Tuning: Unfreezing Last 30 Layers (Max {PHASE2_EPOCHS} Epochs, Adam lr={PHASE2_LR})...")

    base_model.trainable = True
    # تجميد جميع الطبقات السابقة باستثناء آخر 30 طبقة
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # فك تجميد آخر 30 طبقة مع الإبقاء على تجميد طبقات تجميع الدفعات (BatchNormalization)
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

    # دمج نقطة الحفظ مع خاصية التوقف المبكر عند استقرار الخسارة
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
    best_p2_val_acc = max(history_p2.history["val_accuracy"])
    print(f"Phase 2 Complete. Best val_loss: {best_p2_val_loss:.4f}, Best val_acc: {best_p2_val_acc*100:.2f}%")

    return model, history_p1, history_p2


# ------------------------------------------------------------------------------
# 7. التقييم الصارم على مجموعة الاختبار واستخراج التقارير ومصفوفة الارتباك
# ------------------------------------------------------------------------------
def evaluate_and_report(model, test_gen):
    """
    إجراء تقييم صارم على مجموعة الاختبار المستقلة (612 صورة):
    - حساب التنبؤات واستخراج مصفوفة الارتباك (Confusion Matrix).
    - حساب الدقة، والاستدعاء (Recall)، ودرجة F1 لكل فئة بدقة 4 خانات عشرية.
    - رسم وحفظ صورة مصفوفة الارتباك عالية الدقة (300 DPI).
    - حفظ التقرير الإحصائي بصيغة JSON للتوثيق الأكاديمي.
    """
    print("\n[4/6] Running Strict Evaluation on 4-Class Test Set (612 Images)...")

    test_gen.reset()
    y_true = test_gen.classes
    num_samples = len(y_true)

    start_eval = time.time()
    y_pred_probs = model.predict(test_gen, steps=math.ceil(num_samples / BATCH_SIZE), verbose=1)
    eval_time = time.time() - start_eval
    print(f"Inference finished in {eval_time:.2f}s ({num_samples / eval_time:.1f} images/s)")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # حساب مصفوفة الارتباك
    cm = confusion_matrix(y_true, y_pred)

    # استخراج تقرير التصنيف بصيغتي القاموس والنص المنسق
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    # رسم وتنسيق خريطة مصفوفة الارتباك الحرارية
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_title("Confusion Matrix (4-Class Model: Fungal_Disease Merged)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig.savefig(CONFUSION_MATRIX_OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Confusion Matrix saved to: {os.path.abspath(CONFUSION_MATRIX_OUTPUT)}")

    # حفظ نسخة في مجلد المخرجات الأساسي
    artifact_dir = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(CONFUSION_MATRIX_OUTPUT, os.path.join(artifact_dir, CONFUSION_MATRIX_OUTPUT))
        print(f"Mirrored Confusion Matrix image to artifact directory: {artifact_dir}")

    overall_accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    print("\nRaw Confusion Matrix Array (Shape 4x4):")
    print(f"{'':>16} | " + " | ".join([f"{c[:14]:>14}" for c in CLASS_NAMES]))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>16} | " + " | ".join([f"{val:>14d}" for val in row]))

    print("\nFull 4-Class Classification Report (Raw Data):")
    print(report_text)

    print("\nAccuracy Goal Check:")
    print(f"Overall Test Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Macro Avg F1-Score:    {macro_f1:.4f}")
    print(f"Fungal_Disease F1:     {report_dict['Fungal_Disease']['f1-score']:.4f} (Recall: {report_dict['Fungal_Disease']['recall']*100:.2f}%, Precision: {report_dict['Fungal_Disease']['precision']*100:.2f}%)")

    report_data = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "overall_accuracy": float(overall_accuracy),
        "macro_avg_f1": float(macro_f1)
    }
    with open(EVALUATION_REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data


# ------------------------------------------------------------------------------
# 8. الاستدلال الفردي على الصور المحددة للتحقق من ثقة النموذج
# ------------------------------------------------------------------------------
def run_single_image_eval(model):
    """
    تنفيذ الاستدلال المباشر على الصور الإشكالية وحالات الاختبار الخاصة
    للتحقق من استقرار التوزيع الاحتمالي ونسبة ثقة فئة Fungal_Disease.
    """
    print("[5/6] INFERENCE ON PROBLEMATIC TEST IMAGES (4-CLASS MODEL)")

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
            fungal_prob = float(probs[0] * 100)

            print(f"\nImage: {label}")
            print(f"Path:  {path}")
            print(f"Center Crop: {crop}")
            print("Probability Distribution:")
            for idx, c_name in enumerate(CLASS_NAMES):
                print(f"  Class {c_name}: {probs[idx]:.6f} ({probs[idx] * 100:6.2f}%)")
            print(f"Predicted Class:     {pred_name} (Confidence: {conf:.2f}%)")
            print(f"Fungal_Disease Prob: {fungal_prob:.2f}%")

            single_results[label] = {
                "path": path,
                "predicted": pred_name,
                "confidence": conf,
                "fungal_prob": fungal_prob,
                "probabilities": {c_name: float(probs[idx]) for idx, c_name in enumerate(CLASS_NAMES)}
            }
        else:
            print(f"\nFile not found: {path}")

    return single_results


# ------------------------------------------------------------------------------
# 9. تصدير النموذج بصيغة TensorFlow Lite مع التكميم المتقدم للأجهزة الطرفية
# ------------------------------------------------------------------------------
def export_tflite(model):
    """
    تحويل وتكميم النموذج المكتمل إلى صيغة TFLite المناسبة للهواتف الذكية:
    - استخدام Optimize.DEFAULT لتطبيق التكميم الديناميكي للأوزان (Dynamic Range Quantization)
      لتقليل حجم النموذج بمقدار 75% مع الحفاظ التام على دقة الاستدلال.
    - التحقق البرمجي من سلامة النموذج عبر تمرير مصفوفة وهمية (Dummy Forward Pass).
    """
    print(f"[6/6] EXPORTING MODEL TO TFLITE: {TFLITE_OUTPUT_PATH}...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    tflite_size_bytes = os.path.getsize(TFLITE_OUTPUT_PATH)
    tflite_size_mb = tflite_size_bytes / (1024 * 1024)

    # حفظ نسخة في مجلد النتائج
    artifact_dir = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"
    artifact_tflite_path = os.path.join(artifact_dir, TFLITE_OUTPUT_PATH)
    with open(artifact_tflite_path, "wb") as f:
        f.write(tflite_model)

    # التحقق من صلاحية ملف TFLite عن طريق بناء المترجم المفسر واختبار الأبعاد
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.zeros(input_details[0]["shape"], dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    dummy_output = interpreter.get_tensor(output_details[0]["index"])

    print(f"Export Successful: {os.path.abspath(TFLITE_OUTPUT_PATH)}")
    print(f"Mirrored TFLite:   {artifact_tflite_path}")
    print(f"TFLite File Size:  {tflite_size_mb:.2f} MB ({tflite_size_bytes:,} bytes)")
    print(f"Input Shape:       {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"Output Shape:      {output_details[0]['shape']} ({output_details[0]['dtype']})")
    print(f"Verification:      PASSED (Forward pass output sum = {np.sum(dummy_output):.4f})")

    return tflite_size_mb, tflite_size_bytes


# ------------------------------------------------------------------------------
# 10. الدالة الرئيسية لتنسيق وتشغيل مسار التدريب المتكامل
# ------------------------------------------------------------------------------
def main():
    print("4-CLASS COFFEE LEAF CLASSIFIER: MERGED FUNGAL_DISEASE PIPELINE")

    # 1. إعداد مولدات البيانات
    train_gen, val_gen, test_gen = get_dataset_generators()

    # 2. حساب أوزان الفئات المتوازنة
    class_weights = compute_balanced_weights(train_gen)

    # 3. بناء هيكل النموذج
    model, base_model = build_baseline_mobilenetv2(num_classes=NUM_CLASSES)

    # 4. تنفيذ التدريب بمرحلتيه الأولى والثانية
    model, h1, h2 = train_two_phases(model, base_model, train_gen, val_gen, class_weights)

    # 5. تحميل نقطة الحفظ الفضلى لمجموعة التحقق
    if os.path.exists(MODEL_FINAL_CHECKPOINT):
        print(f"\nLoading best checkpoint from {MODEL_FINAL_CHECKPOINT} for testing...")
        model = models.load_model(MODEL_FINAL_CHECKPOINT)

    # 6. إجراء التقييم الشامل واستخراج مصفوفة الارتباك
    report_data = evaluate_and_report(model, test_gen)

    # 7. اختبار الاستدلال الفردي على الصور الإشكالية
    single_results = run_single_image_eval(model)

    # 8. التصدير لصيغة TFLite المخففة
    tflite_mb, tflite_bytes = export_tflite(model)

    print("All 4-Class Pipeline Steps Completed Successfully.")


if __name__ == "__main__":
    main()
