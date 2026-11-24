# -*- coding: utf-8 -*-
"""
نظام معالجة وتطهير بيانات أوراق البن وإزالة الاختصارات البصرية
الغرض:
تنقية صور التدريب الخام للتخلص من حفظ العلامات المائية والنصوص الهامشية
مثل بطاقات الألوان، المساطر، والنصوص التوضيحية على الحواف.

مراحل المعالجة:
1. القص المركزي بنسبة عرض إلى ارتفاع 1:1 لحذف النصوص الجانبية والسفلية.
2. قص عشوائي خفيف بمقياس (0.85 إلى 1.0) لتنويع موضع حواف الورقة.
3. التعديل اللوني الضوئي عبر فضاء الألوان HSV ومحسنات PIL لفك ارتباط معالم المرض بإضاءة الماسح الضوئي.
4. إعادة تحجيم الصورة بدقة استيفاء عالية إلى الأبعاد المعيارية (224x224).
"""

# استيراد المكتبات الأساسية لإدارة نظام التشغيل والمسارات
import os
import sys
import time
import random

# استيراد مكتبة المعالجة الرقمية للمصفوفات
import numpy as np

# استيراد مكتبة الرؤية الحاسوبية OpenCV لمعالجة فضاءات الألوان وإعادة التحجيم
import cv2

# استيراد مكتبة معالجة الصور PIL للتعامل مع الصور وقنوات الألوان
from PIL import Image, ImageEnhance

# تثبيت البذور العشوائية لضمان إمكانية تكرار النتائج الرياضية والتجارب الأكاديمية بدقة
random.seed(42)
np.random.seed(42)


def apply_color_jitter(pil_img, brightness_range=(0.85, 1.15),
                       contrast_range=(0.85, 1.15),
                       saturation_range=(0.85, 1.15),
                       hue_delta_max=5):
    """
    تطبيق تشويش لوني طفيف باستخدام محسنات مكتبة PIL وفضاء الألوان HSV في OpenCV.
    الهدف الأكاديمي:
    إجبار الشبكة العصبية على التركيز على الأنماط النسيجية والهيكلية لبقع المرض
    بدلاً من الاعتماد على درجة حرارة اللون أو معايرة أجهزة المسح الضوئي.
    
    المعاملات:
    pil_img: صورة الإدخال من نوع PIL Image.
    brightness_range: نطاق معامل السطوع (بين 0.85 و 1.15).
    contrast_range: نطاق معامل التباين.
    saturation_range: نطاق معامل التشبع اللوني.
    hue_delta_max: الحد الأقصى لإزاحة زاوية تدرج اللون (درجات).
    """
    # 1. تطبيق التحسين الضوئي عبر مكتبة PIL للسطوع والتباين والتشبع اللوني
    # توليد معامل عشوائي لضبط السطوع وتطبيقه على الصورة
    b_factor = random.uniform(*brightness_range)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(b_factor)

    # توليد معامل عشوائي لضبط التباين وتطبيقه على الصورة
    c_factor = random.uniform(*contrast_range)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(c_factor)

    # توليد معامل عشوائي لضبط تشبع اللون وتطبيقه على الصورة
    s_factor = random.uniform(*saturation_range)
    pil_img = ImageEnhance.Color(pil_img).enhance(s_factor)

    # 2. إزاحة طفيفة لقناة تدرج اللون (Hue) عبر فضاء ألوان HSV في مكتبة OpenCV
    # تحويل الصورة من كائن PIL (RGB) إلى مصفوفة رقمية بنظام BGR المعتمد في OpenCV
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # تحويل فضاء الألوان من BGR إلى HSV للتحكم في صبغة اللون بمعزل عن شدة الإضاءة
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # توليد إزاحة عشوائية طفيفة لدرجة اللون ضمن النطاق المحدد [-5, +5] درجات
    hue_shift = random.uniform(-hue_delta_max, hue_delta_max)
    # تطبيق الإزاحة على القناة الأولى (قناة Hue) مع استخدام باقي القسمة على 180 (نطاق زوايا الألوان في OpenCV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    # تقييد القيم الناتجة لتكون محصورة بدقة بين 0 و 255 ثم تحويلها إلى أعداد صحيحة بدون إشارة (uint8)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    # إعادة التحويل من فضاء ألوان HSV إلى BGR
    img_jittered_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # تحويل الصورة النهائية من BGR إلى RGB لتتوافق مجدداً مع مكتبة PIL والشبكات العصبية
    img_jittered_rgb = cv2.cvtColor(img_jittered_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_jittered_rgb)


def apply_center_and_random_crop(pil_img, target_size=(224, 224), scale_range=(0.85, 1.0)):
    """
    تنفيذ الاقتصاص المركزي والقص العشوائي:
    1. القص المركزي المربع (1:1): إزالة الهوامش الجانبية والسفلية حيث توجد العلامات المائية.
    2. القص العشوائي الخفيف: اقتطاع منطقة جزئية بين 85% و 100% من مركز المربع لمنع الحفظ الأعمى للحواف.
    3. إعادة التحجيم إلى الدقة المستهدفة (224x224) باستخدام الاستيفاء الخطي الثنائي.
    
    المعاملات:
    pil_img: صورة الإدخال الأصلية.
    target_size: الأبعاد المستهدفة للنموذج (224, 224).
    scale_range: نطاق نسبة الاقتصاص العشوائي من الحجم المربع الأصلي.
    """
    w, h = pil_img.size

    # الخطوة الأولى: القص المركزي للحصول على مربع متساوي الأبعاد (نسبة 1:1)
    # حساب البعد الأصغر لاقتطاع أكبر مربع متمركز في وسط الصورة
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    center_square = pil_img.crop((left, top, left + min_dim, top + min_dim))

    # الخطوة الثانية: تطبيق اقتصاص عشوائي خفيف ضمن المربع المتمركز (1:1)
    # اختيار مقياس عشوائي للاقتصاص بين 85% و 100%
    crop_scale = random.uniform(*scale_range)
    crop_size = int(min_dim * crop_scale)

    # حساب الحد الأقصى للإزاحة الأفقية والعمودية المسموح بها للاقتصاص
    max_offset = min_dim - crop_size
    offset_x = random.randint(0, max_offset) if max_offset > 0 else 0
    offset_y = random.randint(0, max_offset) if max_offset > 0 else 0

    # اقتطاع الجزء المختار عشوائياً
    sub_cropped = center_square.crop((offset_x, offset_y, offset_x + crop_size, offset_y + crop_size))

    # الخطوة الثالثة: إعادة التحجيم عالية الجودة باستخدام مكتبة OpenCV
    # تحويل الصورة المقصوصة إلى مصفوفة رقمية
    np_crop = np.array(sub_cropped)
    # تغيير الحجم باستخدام الاستيفاء الخطي الثنائي (INTER_LINEAR)
    resized_bgr = cv2.resize(
        cv2.cvtColor(np_crop, cv2.COLOR_RGB2BGR),
        target_size,
        interpolation=cv2.INTER_LINEAR
    )
    # إعادة تحويل فضاء اللون إلى RGB
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(resized_rgb)


def sanitize_and_augment_dataset(input_dir, output_dir, target_size=(224, 224)):
    """
    نقطة الدخول الرئيسية لمعالجة وتطهير مجموعة البيانات:
    - المرور عبر جميع الصور الموجودة داخل المجلدات الفرعية في input_dir.
    - تطبيق القص المركزي المربع، والقص العشوائي، والتشويش اللوني.
    - حفظ الصور المنقاة والمعيارية في output_dir بنفس الهيكل التنظيمي للمجلدات.
    - طباعة تقرير مفصل بالأرقام الدقيقة للصور المعالجة لكل فئة.
    
    المعاملات:
    input_dir: مسار المجلد الرئيسي للبيانات الخام.
    output_dir: مسار المجلد الذي ستُحفظ فيه البيانات المعالجة.
    target_size: أبعاد الصور المستهدفة (224, 224).
    """
    print("Dataset Sanitization and Border Text Removal Pipeline")
    print("Input Directory: " + str(os.path.abspath(input_dir)))
    print("Output Directory: " + str(os.path.abspath(output_dir)))
    print("Target Resolution: " + str(target_size[0]) + "x" + str(target_size[1]) + " RGB")

    # التحقق من وجود المجلد المصدري للبيانات
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # إنشاء المجلد الهدف إذا لم يكن موجوداً بالفعل
    os.makedirs(output_dir, exist_ok=True)

    # استكشاف أسماء المجلدات الفرعية التي تمثل فئات الأمراض
    valid_extensions = ('.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.webp')
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # التعامل مع الحالة الاستثنائية إذا كانت الصور موجودة في المجلد الرئيسي مباشرة
    if not class_dirs:
        class_dirs = ["."]

    processed_counts = {}
    start_time = time.time()
    total_processed = 0

    # المرور على كل فئة مرضية على حدة
    for class_name in sorted(class_dirs):
        src_class_path = os.path.join(input_dir, class_name) if class_name != "." else input_dir
        dst_class_path = os.path.join(output_dir, class_name) if class_name != "." else output_dir
        os.makedirs(dst_class_path, exist_ok=True)

        # حصر جميع أسماء ملفات الصور الصالحة داخل مجلد الفئة
        files = [f for f in os.listdir(src_class_path) if f.lower().endswith(valid_extensions)]
        class_count = 0

        print("Processing Class: [" + str(class_name) + "] (" + str(len(files)) + " source images)")

        # معالجة كل صورة داخل الفئة
        for fname in files:
            src_file = os.path.join(src_class_path, fname)
            dst_file = os.path.join(dst_class_path, fname)

            try:
                # 1. فتح وقراءة الصورة بأمان وتحويلها إلى وضع RGB
                with Image.open(src_file) as raw_img:
                    img_rgb = raw_img.convert("RGB")

                    # 2. تطبيق القص المركزي المربع والقص العشوائي الخفيف
                    cleaned_spatial = apply_center_and_random_crop(
                        img_rgb,
                        target_size=target_size,
                        scale_range=(0.85, 1.0)
                    )

                    # 3. تطبيق التعديل اللوني الضوئي الخفيف لفك انحياز الإضاءة
                    sanitized_img = apply_color_jitter(
                        cleaned_spatial,
                        brightness_range=(0.85, 1.15),
                        contrast_range=(0.85, 1.15),
                        saturation_range=(0.85, 1.15),
                        hue_delta_max=4
                    )

                    # 4. حفظ الصورة المنقاة بجودة ضغط عالية تبلغ 95
                    sanitized_img.save(dst_file, quality=95)
                    class_count += 1
                    total_processed += 1

            except Exception as e:
                # تسجيل أي خطأ في حال كانت الصورة تالفة أو غير قابلة للقراءة
                print("Error processing " + str(src_file) + ": " + str(e))

        processed_counts[class_name] = class_count
        print("Completed [" + str(class_name) + "]: " + str(class_count) + " sanitized images saved.")

    elapsed = time.time() - start_time

    # طباعة ملخص إحصائي شامل لعملية المعالجة
    print("\nSanitization Execution Summary (Images Processed Per Class)")
    for c_name, count in processed_counts.items():
        print("Class " + str(c_name) + ": " + str(count) + " images")
    print("Total Sanitized Images: " + str(total_processed) + " images")
    print("Total Execution Time: " + str(f"{elapsed:.2f}") + " seconds (" + str(f"{total_processed / max(elapsed, 0.001):.1f}") + " img/sec)")
    print("Sanitized Dataset Path: " + str(os.path.abspath(output_dir)))

    return processed_counts


if __name__ == "__main__":
    # المسارات الافتراضية المعتمدة لمجموعة بيانات التدريب
    DEFAULT_INPUT_DIR = r"D:\Coffee_Dataset_Split\train"
    DEFAULT_OUTPUT_DIR = r"D:\Coffee_Dataset_Sanitized\train"

    # السماح بتمرير مسارات مخصصة عبر سطر الأوامر إن وُجدت
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    # بدء تشغيل خط معالجة البيانات
    sanitize_and_augment_dataset(input_path, output_path)
