# 🌿 AgriTech: Production Coffee Leaf Disease Classifier & Mobile Edge-AI Pipeline
## مصنف أمراض أوراق البن للإنتاج والحوسبة الطرفية المحمولة

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Edge--AI-green.svg)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

An end-to-end, cascaded Deep Learning system for real-time identification of coffee leaf diseases under severe field conditions and cross-continental domain shift. Designed for edge deployment (Mobile/Flutter/Android) with zero cloud latency.

---

## 📌 Architecture Overview | بنية النظام المتتالية

The inference pipeline employs a **Two-Stage Cascaded Architecture** to ensure reliable field deployment and prevent out-of-distribution (OOD) hallucinations:

```
[ Input Camera Image (224x224) ]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Binary Gatekeeper Leaf Filter      │
│  Model: gatekeeper_leaf_filter.tflite        │
│  Threshold: P(Leaf) >= 0.35                  │
└──────────────────────────────────────────────┘
         │                           │
    [ < 0.35 ]                  [ >= 0.35 ]
         │                           │
         ▼                           ▼
┌──────────────────┐    ┌──────────────────────────────────────────────┐
│  REJECT (Error)  │    │  Stage 2: 4-Class Disease Classifier         │
│  "Not Coffee     │    │  Model: coffee_model_adapted.tflite          │
│   Leaf"          │    │  Confidence Threshold: >= 0.85               │
└──────────────────┘    └──────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
               [ >= 0.85 ]                       [ < 0.85 ]
                    │                                 │
                    ▼                                 ▼
      ┌───────────────────────────┐      ┌─────────────────────────┐
      │  Confirmed Disease Result │      │  UNCERTAIN / REJECT     │
      │  • Healthy                │      │  "Low Confidence /      │
      │  • Fungal_Disease (Rust)  │      │   Uncertain Field Spot" │
      │  • Miner                  │      └─────────────────────────┘
      │  • Phoma                  │
      └───────────────────────────┘
```

---

## 🎯 Supported Classes | فئات التصنيف

1. **`Healthy`** (أوراق سليمة): Coffee leaves free from biological disease.
2. **`Fungal_Disease`** (أمراض فطرية): Merged class covering **Coffee Leaf Rust** (*Hemileia vastatrix*) and **Brown Eye Spot** (*Cercospora coffeicola*).
3. **`Miner`** (صانعة الأنفاق): Serpentine mines caused by *Leucoptera coffeella*.
4. **`Phoma`** (تبقع الفومة): Necrotic leaf spots caused by *Phoma costarricensis*.

---

## 🌍 Cross-Continental Domain Shift & Few-Shot Adaptation
### معالجة التباين البيئي والتكيف القليل للعينات (اليمن وأوغندا)

Field coffee leaves in developing regions (Uganda, Yemen) exhibit severe domain shift compared to clean lab/bench datasets:
- Complex cluttered agricultural backgrounds (weeds, soil, farmer fingers).
- Harsh direct tropical sunlight and specular glare.
- Variable mobile phone sensor colorimetry.

### Benchmark Comparison (3,120 Unseen Field Images):

| Metric | 1. Zero-Shot Baseline | 2. Synthetic (CutMix + CLAHE) | 3. Few-Shot Adaptation (100 Samples) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **45.50%** | 34.50% (-11.00%) | **73.24% (+27.74% Jump)** |
| **Healthy Recall** | 50.32% | 3.24% (Catastrophic) | **88.42% (Rescued)** |
| **Fungal Disease Recall** | 82.26% | 98.40% (Biased) | **79.16% (Balanced)** |
| **Phoma Recall** | 13.60% | 0.00% (Vanished) | **53.02% (4x Improvement)** |

> **Key Takeaway:** Unsupervised synthetic patch mixing (CutMix) distorted natural features (-11% drop). In contrast, **Few-Shot Domain Adaptation** using only **100 local field images** with gentle fine-tuning ($\text{lr} = 10^{-5}$, 5 epochs) elevated test accuracy to **73.24%** and rescued Healthy recall to **88.42%**, establishing field deployment readiness for Yemeni coffee plantations.

---

## 📂 Core Repository Structure | هيكلية المشروع الكانونية

```text
coffee_leaf_classifier/
├── final_pipeline.py                 # السكربت النهائي الشامل للاستدلال المتسلسل (Stage 1 + Stage 2)
├── execute_4class_pipeline.py        # سكربت التدريب الكانوني ثنائي المراحل (Phase 1 + Phase 2)
├── train_gatekeeper_v2.py            # تدريب وتصدير نموذج الحارس الثنائي (Gatekeeper)
├── few_shot_domain_adaptation.py     # سكربت التكيّف المحلي للبيئة الحقلية (Few-Shot Fine-Tuning)
├── sanitize_dataset.py               # الفحص الجنائي، إزالة التكرار (pHash) ومنع تسرب البيانات
│
├── coffee_model_adapted.tflite       # نموذج التصنيف الرباعي المكيف حافياً (2.52 MB)
├── coffee_model_final.tflite         # نموذج التصنيف الرباعي الأساسي (2.52 MB)
├── gatekeeper_leaf_filter.tflite     # نموذج الفلتر الثنائي (2.52 MB)
│
├── labels.txt                        # أسماء الفئات المعتمدة
├── requirements.txt                  # متطلبات التشغيل
├── Project_Defense_Guide.md          # الدليل الشامل لمناقشة أطروحة الماجستير والتخرج
└── README_Supervisor.md              # تقرير المتابعة الأكاديمية وبوابات القبول
```

---

## 🚀 Quick Start | دليل البدء السريع

### 1. Requirements & Environment Setup
```bash
git clone https://github.com/evaalduminy/Agrethec_Team.git
cd Agrethec_Team
pip install -r requirements.txt
```

### 2. Run Inference on a Single Image
```python
from final_pipeline import CascadedCoffeeClassifier

classifier = CascadedCoffeeClassifier(
    gatekeeper_path="gatekeeper_leaf_filter.tflite",
    disease_model_path="coffee_model_adapted.tflite"
)

result = classifier.predict("path/to/test_leaf.jpg")
print(result)
# Output:
# {
#   "status": "ACCEPTED",
#   "is_leaf": True,
#   "leaf_probability": 0.982,
#   "disease": "Healthy",
#   "confidence": 0.941,
#   "action": "Normal Maintenance"
# }
```

### 3. Edge / Mobile Deployment (Flutter & Android)
The exported `.tflite` models are optimized using `tf.lite.Optimize.DEFAULT` (float32 fallback with quantized constants), resulting in an exact size of **2.52 MB** per model:
- Total storage requirement on device: **< 6 MB**.
- Inference latency: **~18 ms** on modern mobile NPUs / CPUs.
- Zero network connectivity required (100% offline edge execution).

---

## 🎓 Academic Defense & Citation | التوثيق الأكاديمي

Developed as part of the AgriTech Precision Agriculture Initiative.  
For technical inquiries and forensic audit logs, refer to [Project_Defense_Guide.md](Project_Defense_Guide.md) and [outputs/final_evaluation_report.json](outputs/final_evaluation_report.json).
