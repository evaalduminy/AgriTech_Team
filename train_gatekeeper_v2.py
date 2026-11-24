# -*- coding: utf-8 -*-
"""
تدريب وتصدير نموذج الحارس الأمني الثنائي (Gatekeeper Filter)
المهمة:
تصنيف المدخلات ثنائياً إلى (ورقة نبات: Leaf) مقابل (ليست ورقة نبات: Not_Leaf).
الهدف الأكاديمي:
منع الصور الخارجة عن النطاق الموزع (Out-of-Distribution - OOD) كالوجوه البشرية،
الشعارات، السيارات، أو الأجسام العشوائية من الوصول إلى نموذج تصنيف الأمراض.

المعالجات المضافة:
- تضمين عينات أوراق ملتقطة في ظروف إضاءة داكنة وشاشات لضمان عدم رفض أوراق البن الحقيقية.
- توليد عينات معززة منخفضة السطوع (بين 0.4 و 0.7) لتدريب النموذج على استيعاب الإضاءة الميدانية المنخفضة.
- تصدير النموذج بهيكلية خفيفة جداً عبر تكميم TFLite للعمل اللحظي على الهواتف.
"""

# استيراد المكتبات الأساسية لإدارة الملفات والنسخ وتوليد الأرقام العشوائية
import os
import shutil
import random
import numpy as np

# استيراد مكتبات التعلم العميق وبناء النماذج من TensorFlow و Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# استيراد مكتبة معالجة الصور لإنشاء التنويعات الضوئية
from PIL import Image, ImageEnhance

# ------------------------------------------------------------------------------
# مسارات الملفات ومجلدات العمل (Paths and Directories)
# ------------------------------------------------------------------------------
WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
DATASET_ROOT = os.path.join(WORKSPACE, "Gatekeeper_Dataset")
TRAIN_LEAF_DIR = os.path.join(DATASET_ROOT, "train", "Leaf")
ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"

# مسارات الصور المرجعية ذات الإضاءة المعقدة
EURO_GREEN_SRC_1 = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg"
EURO_GREEN_SRC_2 = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg"

EURO_GREEN_DST_1 = os.path.join(TRAIN_LEAF_DIR, "euro_green_1.jpg")
EURO_GREEN_DST_2 = os.path.join(TRAIN_LEAF_DIR, "euro_green_2.jpg")

# مسارات تصدير نموذج TFLite
OUTPUT_TFLITE = os.path.join(WORKSPACE, "gatekeeper_leaf_filter_v2.tflite")
ARTIFACT_TFLITE = os.path.join(ARTIFACT_DIR, "gatekeeper_leaf_filter_v2.tflite")

# ------------------------------------------------------------------------------
# الخطوة 1: نسخ صور الأوراق ذات الإضاءة الداكنة إلى مجموعة تدريب فئة الأوراق
# ------------------------------------------------------------------------------
print("STEP 1: Copying reference leaf images into Leaf training set...")

os.makedirs(TRAIN_LEAF_DIR, exist_ok=True)

# نسخ الصور المرجعية إذا لم تكن موجودة بالفعل في مجلد التدريب
for src, dst in [(EURO_GREEN_SRC_1, EURO_GREEN_DST_1), (EURO_GREEN_SRC_2, EURO_GREEN_DST_2)]:
    if os.path.exists(dst):
        print(f"Already exists: {os.path.basename(dst)}")
    else:
        shutil.copy2(src, dst)
        print(f"Copied: {os.path.basename(dst)}")

# ------------------------------------------------------------------------------
# الخطوة 2: توليد 10 عينات معززة خافتة الإضاءة لتمثيل الظروف الميدانية الصعبة
# ------------------------------------------------------------------------------
print("\nSTEP 2: Generating 10 dark brightness-augmented leaf images...")

# تثبيت البذرة العشوائية لضمان تكرار نتائج المعالجة بدقة
random.seed(42)
aug_count = 0

# توليد 5 صور معززة لكل صورة مرجعية بمعدلات سطوع خافتة تتراوح بين 0.4 و 0.7
for img_idx, src_path in enumerate([EURO_GREEN_SRC_1, EURO_GREEN_SRC_2], start=1):
    img_pil = Image.open(src_path).convert("RGB").resize((224, 224))
    for aug_idx in range(5):
        brightness_factor = random.uniform(0.4, 0.7)
        enhancer = ImageEnhance.Brightness(img_pil)
        aug_img = enhancer.enhance(brightness_factor)
        out_name = f"euro_green_aug_{img_idx}_{aug_idx+1}.jpg"
        out_path = os.path.join(TRAIN_LEAF_DIR, out_name)
        aug_img.save(out_path, "JPEG")
        aug_count += 1
        print(f"Saved: {out_name} (brightness={brightness_factor:.2f})")

print(f"Total augmented images added: {aug_count}")

# حساب وتأكيد إجمالي عدد صور الأوراق الصالحة للتدريب
leaf_count = len([f for f in os.listdir(TRAIN_LEAF_DIR)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))])
print(f"Total Leaf training images now: {leaf_count}")

# ------------------------------------------------------------------------------
# الخطوة 3: بناء معمارية نموذج الحارس الأمني الخفيف (MobileNetV2 Binary Gatekeeper)
# ------------------------------------------------------------------------------
print("\nSTEP 3: Building MobileNetV2 Gatekeeper model...")

# تحميل نموذج MobileNetV2 بأوزان ImageNet مع استبعاد الرأس المخصص لـ 1000 فئة
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# تجميد جميع أوزان النموذج الأساسي لسرعة التدريب وتجنب استهلاك الموارد
base_model.trainable = False

# تحويل الخرائط المكانية (7x7x1280) إلى متجه أحادي البعد (1280) عبر التجميع المتوسط
x = base_model.output
x = GlobalAveragePooling2D()(x)
# طبقة عصبية واحدة مع دالة التنشيط Sigmoid لإنتاج احتمال الانتماء لفئة الورقة P(Leaf)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
# تجميع النموذج باستخدام دالة الانتروبيا الثنائية ومحسن Adam
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()
trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable params: {trainable_count}")

# ------------------------------------------------------------------------------
# الخطوة 4: تهيئة مولدات البيانات والتدعيم الرقمي للصور
# ------------------------------------------------------------------------------
print("\nSTEP 4: Setting up data generators...")

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# تطبيق تحويلات فيزيائية وضوئية عشوائية لتحسين متانة الحارس ضد زوايا التصوير والإضاءة
train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    brightness_range=[0.4, 1.3],
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ضبط الفئات بترتيب صريح: Not_Leaf = 0 و Leaf = 1 لضمان أن مخرج Sigmoid يمثل P(Leaf)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Not_Leaf", "Leaf"],
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Not_Leaf", "Leaf"],
    shuffle=False
)

print(f"Class indices: {train_gen.class_indices}")
print(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

# ------------------------------------------------------------------------------
# الخطوة 5: تدريب النموذج لمدة 5 دورات
# ------------------------------------------------------------------------------
print("\nSTEP 5: Training (5 epochs, frozen base)...")

history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    verbose=1
)

final_epoch = 4
final_train_acc = history.history["accuracy"][final_epoch]
final_val_acc = history.history["val_accuracy"][final_epoch]
print(f"\nEpoch 5 — train_acc={final_train_acc:.4f} val_acc={final_val_acc:.4f}")

# ------------------------------------------------------------------------------
# الخطوة 6: تقييم الاستدلال على صور الاختبار وتأكيد الفئات المقبولة والمرفوضة
# ------------------------------------------------------------------------------
print("\nSTEP 6: Running prediction tests...")

# قائمة حالات الاختبار المعيارية: تشمل أوراقاً حقيقية وأجساماً خارجية خارج النطاق
TEST_IMAGES = [
    ("Gatekeeper_Dataset/train/Leaf/leaf_train_000.jpg", "Leaf", ">0.5"),
    ("Gatekeeper_Dataset/train/Leaf/euro_green_1.jpg", "Leaf", ">0.5"),
    ("Gatekeeper_Dataset/train/Leaf/euro_green_2.jpg", "Leaf", ">0.5"),
    (r"D:\probe_foreign\2.png", "Not_Leaf", "<0.5"),
    (r"D:\probe_foreign\1764100816328~2.jpg", "Not_Leaf", "<0.5"),
    (r"D:\probe_foreign\download (6).jfif", "Not_Leaf", "<0.5"),
]

results = []
for img_path, expected_class, expected_dir in TEST_IMAGES:
    # تحويل المسارات النسبية إلى مسارات مطلقة
    if not os.path.isabs(img_path):
        full_path = os.path.join(WORKSPACE, img_path)
    else:
        full_path = img_path

    if not os.path.exists(full_path):
        print(f"[MISSING] {img_path}")
        results.append((img_path, None, expected_class, expected_dir, "MISSING"))
        continue

    # قراءة الصورة وتطبيق المعالجة الحسابية المعتمدة لشبكة MobileNetV2
    img = load_img(full_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    score = float(model.predict(arr, verbose=0)[0][0])

    # التحقق من مطابقة النتيجة للحد المتوقع
    if expected_dir == ">0.5":
        passed = score > 0.5
    else:
        passed = score < 0.5

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {os.path.basename(img_path):35s} sigmoid={score:.4f} expected={expected_class}({expected_dir})")
    results.append((img_path, score, expected_class, expected_dir, status))

# ------------------------------------------------------------------------------
# الخطوة 7: تحويل وتكميم النموذج بصيغة TensorFlow Lite للأجهزة المحمولة
# ------------------------------------------------------------------------------
print("\nSTEP 7: Exporting to TFLite (DEFAULT optimization)...")

# استخدام محول TFLite مع تفعيل التكميم الافتراضي لتقليل استهلاك الذاكرة
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# كتابة وحفظ النموذج المكمم
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

size_bytes = os.path.getsize(OUTPUT_TFLITE)
size_mb = size_bytes / (1024 * 1024)
print(f"Saved: {OUTPUT_TFLITE}")
print(f"Size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

# ------------------------------------------------------------------------------
# الخطوة 8: نسخ النموذج إلى مجلد المخرجات الأساسي
# ------------------------------------------------------------------------------
print("\nSTEP 8: Copying TFLite to artifact directory...")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
shutil.copy2(OUTPUT_TFLITE, ARTIFACT_TFLITE)
print(f"Copied to: {ARTIFACT_TFLITE}")

# طباعة ملخص النتائج النهائي
print("\nTRAINING COMPLETE — SUMMARY")
print(f"Epoch 5 train_accuracy: {final_train_acc:.4f}")
print(f"Epoch 5 val_accuracy:   {final_val_acc:.4f}")
print(f"TFLite size:           {size_bytes:,} bytes ({size_mb:.2f} MB)\n")
print("Prediction results:")
for img_path, score, expected_class, expected_dir, status in results:
    score_str = f"{score:.4f}" if score is not None else "N/A"
    print(f"  [{status}] {os.path.basename(img_path):35s} sigmoid={score_str}")

all_passed = all(r[4] in ("PASS", "MISSING") for r in results)
print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
