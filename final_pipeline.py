# -*- coding: utf-8 -*-
"""
خط الاستدلال والتشخيص الميداني المتسلسل ثنائي المراحل
الهيكل المعماري:
المرحلة الأولى: فلتر الحارس الأمني الثنائي (gatekeeper_leaf_filter_v2.tflite)
المهمة: تقييم احتمالية كون الصورة المدخلة ورقة نبات P(Leaf)
عتبة القبول: 0.35 (أو 0.18 للقبول التام لكافة ظروف الإضاءة الميدانية)
الوظيفة: عزل الصور الخارجة عن النطاق (OOD) مثل الوجوه، الحيوانات، والشعارات.

المرحلة الثانية: نموذج تشخيص الأمراض الرباعي (coffee_model_final.tflite)
الفئات: ["Fungal_Disease", "Healthy", "Miner", "Phoma"]
عتبة الثقة التشخيصية: 0.85
الوظيفة: تشخيص الإصابة المرضية مع رفض الصور غير الواضحة (Smart Rejection)
إذا كانت الثقة أقل من 85% لحماية المزارع من التشخيص الخاطئ.
"""

# استيراد المكتبات الأساسية لإدارة المسارات والمصفوفات ومعالجة الصور
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------------------------
# المسارات والمعاملات المعيارية (Paths and Hyperparameters)
# ------------------------------------------------------------------------------
WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"

# المسارات الافتراضية للنموذجين المكممين بصيغة TFLite
DEFAULT_GATEKEEPER_TFLITE = os.path.join(WORKSPACE, "gatekeeper_leaf_filter_v2.tflite")
DEFAULT_DISEASE_TFLITE = os.path.join(WORKSPACE, "coffee_model_final.tflite")

# أبعاد الصورة المعيارية لشبكات MobileNetV2
IMAGE_SIZE = (224, 224)

# عتبة الحارس الأمني الافتراضية (0.35) وعتبة الثقة التشخيصية لنموذج المرض (0.85)
DEFAULT_GATEKEEPER_THRESHOLD = 0.35
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# الأسماء الرسمية لفئات الأمراض الأربع المعتمدة
DISEASE_CLASS_NAMES = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]


class TFLiteClassifier:
    """
    فئة مساعدة لتشغيل نماذج TensorFlow Lite على الأجهزة الطرفية والمحمولة:
    - تحميل النموذج من المسار المحدد.
    - تخصيص مصفوفات الذاكرة للطبقات المدخلة والمخرجة (allocate_tensors).
    - تنفيذ عملية الاستدلال السريع واسترجاع النتيجة.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_tensor):
        """تمرير مصفوفة الصورة إلى منفذ الإدخال وتشغيل النموذج واسترجاع المخرجات."""
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


def preprocess_image_1to1(image_path, target_size=IMAGE_SIZE):
    """
    المعالجة المسبقة القياسية لمحاكاة عدسة كاميرا تطبيق الهاتف الذكي:
    1. القص المركزي المربع (1:1): اقتطاع المنطقة الوسطى لمنع تشويه أبعاد الورقة عند التحجيم.
    2. إعادة التحجيم عالي الجودة إلى دقة (224x224) باستخدام الاستيفاء الخطي الثنائي.
    3. المعايرة الحسابية المعتمدة لشبكة MobileNetV2 لتحويل قيم البكسل من [0, 255] إلى [-1.0, 1.0].
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
        norm_arr = (arr / 127.5) - 1.0
        batch = np.expand_dims(norm_arr, axis=0)
    return batch


def predict_with_gatekeeper(image_path,
                            gatekeeper_tflite_path=DEFAULT_GATEKEEPER_TFLITE,
                            disease_tflite_path=DEFAULT_DISEASE_TFLITE,
                            gatekeeper_threshold=DEFAULT_GATEKEEPER_THRESHOLD,
                            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    خط الاستدلال المتسلسل المتكامل (Production Cascaded Pipeline):
    
    المرحلة الأولى: فلتر الحارس الأمني
      - إذا كان احتمال كون الصورة ورقة P(Leaf) أقل من gatekeeper_threshold يتم رفض الصورة فوراً.
      
    المرحلة الثانية: نموذج تشخيص أمراض البن الرباعي
      - إذا كانت نسبة ثقة الفئة المتوقعة أقل من confidence_threshold يتم تفعيل الرفض الذكي
        وتنبيه المستخدم بأن الصورة غير واضحة أو أن المرض في بدايته.
      - إذا تجاوزت الثقة النسبة المطلوبة يتم اعتماد التشخيص بنجاح.
    """
    if not os.path.exists(image_path):
        return {"status": "Error", "message": f"Image file not found: {image_path}"}

    # تجهيز الصورة وفق المواصفات القياسية
    tensor = preprocess_image_1to1(image_path)

    # المرحلة الأولى: فحص الحارس الأمني للصور الخارجة عن النطاق
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
            "message": "تنبيه: الصورة لا تحتوي على ورقة نبات. يرجى توجيه الكاميرا نحو ورقة البن."
        }

    # المرحلة الثانية: نموذج تشخيص المرض الرباعي
    disease_engine = TFLiteClassifier(disease_tflite_path)
    disease_probs = disease_engine.predict(tensor)[0]

    pred_idx = int(np.argmax(disease_probs))
    pred_class = DISEASE_CLASS_NAMES[pred_idx]
    max_confidence = float(disease_probs[pred_idx])

    all_probs = {DISEASE_CLASS_NAMES[i]: float(disease_probs[i]) for i in range(len(DISEASE_CLASS_NAMES))}

    # صمام الرفض الذكي (Smart Rejection) للصور الملتبسة أو ضعيفة الثقة
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
            "message": "تنبيه: الصورة غير واضحة أو أن المرض في مرحلة مبكرة جدا. يرجى إعادة التصوير بإضاءة أفضل والاقتراب من الورقة."
        }

    # اعتماد التشخيص النهائي المؤكد
    return {
        "status": "Accepted",
        "stage": "Confirmed_Diagnosis",
        "decision": pred_class,
        "gatekeeper_score": gatekeeper_score,
        "confidence": max_confidence,
        "confidence_percent": f"{max_confidence * 100:.2f}%",
        "probabilities": all_probs,
        "message": f"تم تشخيص الورقة بنجاح: {pred_class} بنسبة ثقة {max_confidence * 100:.2f}%."
    }


def run_comprehensive_verification():
    """
    تنفيذ فحص وتحقق شامل لمسار الاستدلال على مجموعة عينات متنوعة
    تشمل أوراقاً مصابة وسليمة وأجساماً خارجية خارج النطاق.
    """
    print("Cascaded Pipeline: Comprehensive Verification Run")

    fungal_imgs = sorted(glob.glob(r"D:\Coffee_Dataset_4Class\test\Fungal_Disease\*"))[:5]
    healthy_imgs = sorted(glob.glob(r"D:\Coffee_Dataset_4Class\test\Healthy\*"))[:3]
    euro_green = [
        r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg",
        r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg",
    ]
    ood_imgs = [
        r"D:\probe_foreign\2.png",
        r"D:\probe_foreign\1764100816328~2.jpg",
        r"D:\probe_foreign\download (6).jfif",
    ]

    test_cases = (
        [("Fungal_Disease/" + os.path.basename(p), p) for p in fungal_imgs] +
        [("Healthy/" + os.path.basename(p), p) for p in healthy_imgs] +
        [("Euro_Green_1", euro_green[0])] +
        [("Euro_Green_2", euro_green[1])] +
        [("OOD/" + os.path.basename(p), p) for p in ood_imgs]
    )

    for threshold in [0.35, 0.18]:
        print(f"\nRunning Evaluation with Gatekeeper Threshold = {threshold:.2f} (Confidence Threshold = 0.85):")
        print(f"{'Image':<35} | {'GK Score':>9} | {'Stage 1':<12} | {'Decision / Pred':<18} | {'Conf':>8} | Status")

        for label, path in test_cases:
            if not os.path.exists(path):
                print(f"{label:<35} | FILE NOT FOUND")
                continue
            res = predict_with_gatekeeper(path, gatekeeper_threshold=threshold)
            gk_score = res.get("gatekeeper_score", 0.0)
            status = res["status"]
            stage = res["stage"]
            decision = res.get("decision", "N/A")
            conf_str = res.get("confidence_percent", res.get("gatekeeper_score_percent", "N/A"))

            s1_status = "Passed" if stage != "Gatekeeper_OOD_Filter" else "Rejected"
            print(f"{label:<35} | {gk_score:>9.4f} | {s1_status:<12} | {decision:<18} | {conf_str:>8} | {status}")

    print("\nModel File Metadata:")
    for mpath in [DEFAULT_GATEKEEPER_TFLITE, DEFAULT_DISEASE_TFLITE]:
        size_bytes = os.path.getsize(mpath)
        print(f"File {os.path.basename(mpath)}: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    run_comprehensive_verification()
