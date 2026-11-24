# -*- coding: utf-8 -*-
"""
==============================================================================
🔍 AgriTech: Final Provenance Check (final_provenance_check.py)
==============================================================================
إعادة تحميل النموذج النهائي المعتمد مباشرة من القرص وفحص:
1. بصمة المعاملات القابلة للتدريب (Trainable Parameters == 1,054,661)
2. تجميد كافة طبقات BatchNormalization
3. إعادة تقييم مستقلة على Test Set للتحقق من تطابق الدقة 84.80% والخسارة 0.3680
4. مطابقة تقرير final_evaluation_report.json على القرص
==============================================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATASET_DIR = r"D:\Coffee_Dataset_Split"
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = "outputs"
MODEL_FINAL_PATH = os.path.join(OUTPUT_DIR, "best_coffee_model_final.h5")
REPORT_PATH = os.path.join(OUTPUT_DIR, "final_evaluation_report.json")
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]

def main():
    print("=" * 75)
    print("🔍 بدء الفحص الجنائي النهائي لإثبات الملكية والمطابقة (Provenance Check)...")
    print("=" * 75)

    assert os.path.exists(MODEL_FINAL_PATH), f"الملف مفقود: {MODEL_FINAL_PATH}"
    assert os.path.exists(REPORT_PATH), f"التقرير مفقود: {REPORT_PATH}"
    assert os.path.exists(TEST_DIR), f"مسار الاختبار مفقود: {TEST_DIR}"

    print(f"📦 حجم ملف النموذج على القرص: {os.path.getsize(MODEL_FINAL_PATH):,} بايت")

    # 1. تحميل النموذج من القرص
    print("🔄 إعادة تحميل النموذج best_coffee_model_final.h5 من القرص...")
    model = tf.keras.models.load_model(MODEL_FINAL_PATH)

    # تحديد Base Model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    # 2. فحص البصمة والمعاملات
    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"\n🔑 فحص بصمة المعاملات (Trainable Parameters): {trainable_params:,}")
    assert trainable_params == 1054661, f"❌ خطأ: المعاملات {trainable_params} لا تطابق 1,054,661!"
    print("   ✓ تطابق تام مع البصمة المستهدفة: 1,054,661 معامل")

    # 3. فحص تجميد طبقات BatchNormalization
    bn_layers = [l for l in (base_model.layers if base_model else model.layers) if isinstance(l, tf.keras.layers.BatchNormalization)]
    bn_trainable = [l for l in bn_layers if l.trainable]
    print(f"\n🛡️ فحص طبقات BatchNormalization ({len(bn_layers)} طبقة):")
    print(f"   عدد طبقات BN غير المجمدة: {len(bn_trainable)}")
    assert len(bn_trainable) == 0, "❌ تحذير: توجد طبقات BatchNormalization غير مجمدة!"
    print("   ✓ كافة طبقات BatchNormalization مجمدة بالكامل (N2 محصنة)")

    # 4. إعادة التقييم المستقل على Test Set
    print("\n📊 إجراء تقييم مستقل على مجموعة الاختبار (612 صورة)...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(224, 224), batch_size=16,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )
    assert test_gen.samples == 612, f"عدد عينات الاختبار {test_gen.samples} != 612"

    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\n🎯 نتائج التقييم المحسوبة من النموذج على القرص:")
    print(f"   • Test Accuracy: {acc*100:.2f}% (القيمة الدقيقة: {acc:.4f})")
    print(f"   • Test Loss:     {loss:.4f}")

    # 5. مطابقة التقرير final_evaluation_report.json
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    rep_acc = rep.get("test_accuracy", 0.0)
    rep_loss = rep.get("test_loss", 0.0)
    guard_dec = rep.get("guard_decision")

    print(f"\n📑 مطابقة تقرير final_evaluation_report.json:")
    print(f"   • Guard Decision: {guard_dec}")
    print(f"   • المسجل في JSON: دقة={rep_acc*100:.2f}%, خسارة={rep_loss:.4f}")

    assert abs(acc - rep_acc) < 1e-4, f"عدم تطابق في الدقة: {acc} != {rep_acc}"
    assert abs(loss - rep_loss) < 1e-4, f"عدم تطابق في الخسارة: {loss} != {rep_loss}"
    assert guard_dec == "phase2_accepted", f"قرار الحارس غير متوقع: {guard_dec}"

    print("\n" + "=" * 75)
    print("🏆 FINAL PROVENANCE CHECK RESULT: CLOSED ✅")
    print("   - النموذج على القرص = النموذج المُقيّم = التقرير المصدر المعتمد")
    print("=" * 75)

if __name__ == "__main__":
    main()
