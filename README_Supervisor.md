# 🌿 AgriTech: Production Coffee Leaf Classifier
## دليل المشرف الأكاديمي والتوثيق الجنائي لمرحلة الإغلاق النهائي
**Academic Supervisor Guide & Forensic Closure Dossier**

---

### 1. شجرة المشروع النهائية بعد الأرشفة (Final Project Tree & File Sizes)
```text
coffee_leaf_classifier/
├── train_final_fixed.py         [16,110 B]  # السكربت الكانوني المعتمد لبيئة Windows (إصلاحات N1-N8)
├── train_colab.py               [18,270 B]  # السكربت المحصن المخصص لبيئة Google Colab Linux
├── README_Supervisor.md         [10,366 B]  # وثيقة دليل المشرف وسجل بوابات القبول والبصمات
├── labels.txt                   [45 B]      # الفئات الخمس المعتمدة
├── requirements.txt             [679 B]     # متطلبات بيئة التشغيل
├── requirements_v3.txt          [921 B]     # حزم إضافية للإنتاج
├── pyproject.toml               [1,566 B]   # مواصفات المشروع
├── .gitignore                   [926 B]     # استثناء الأرشيف والكاش
│
├── outputs/                                 # مجلد الأرتيفكتات الكانونية المعتمدة (وضع قراءة فقط)
│   ├── best_coffee_model_final.h5   [19,281,560 B] # النموذج النهائي المعتمد (دقة 84.80%)
│   ├── best_coffee_model_phase1.h5  [13,468,848 B] # نموذج المرحلة الأولى (Feature Extraction)
│   ├── final_evaluation_report.json [1,757 B]      # تقرير التقييم المصدري الشامل وقرار الحارس
│   ├── training_log_final.txt       [961,355 B]    # سجل التدريب الكامل دورة بدورة
│   ├── provenance_check_result.txt  [974 B]        # شهادة المطابقة الجنائية الرسمية (CLOSED ✅)
│   ├── training_history.csv         [1,067 B]      # بيانات منحنيات التدريب دورة بدورة (CSV للرسالة)
│   ├── training_history.json        [4,522 B]      # بيانات منحنيات التدريب بصيغة JSON
│   ├── accuracy_curve.png           [202,883 B]    # منحنى الدقة عالي الجودة (300 DPI)
│   ├── loss_curve.png               [221,648 B]    # منحنى الخسارة (Weighted Train vs Unweighted Val)
│   └── confusion_matrix.png         [127,013 B]    # مصفوفة الارتباك الملونة (300 DPI)
│
├── scripts/                                 # أدوات الفحص والمطابقة الجنائية
│   ├── final_provenance_check.py    [5,312 B]   # سكربت فحص الملكية والمطابقة الصارم
│   ├── verify_artifacts.py          [1,762 B]   # أداة التحقق التلقائي من سلامة الأرتيفكتات
│   ├── verify_split_integrity.py    [3,516 B]   # أداة فحص التوزيع الطبقي للبيانات
│   ├── verify_same_class_leakage.py [3,371 B]   # أداة فحص خلو التقسيم من أي تسرب بيني
│   ├── clean_healthy_duplicates.py  [2,356 B]   # توثيق إزالة التلوث المتقاطع (الـ 58 مكرراً)
│   ├── archive_old_files.py         [1,744 B]   # سكربت الأرشفة الآلية والتنظيف
│   └── forensic_report_bracol.json  [3,029 B]   # التقرير الجنائي المصدري لمجموعة براكول
│
└── _archive_2026-10-25/                     # أرشيف مسار التطوير والتجارب السابقة (50 ملفاً ومجلداً)
```

---

### 2. جدول بوابات القبول الخمس (Acceptance Gates Record)

| البوابة | التاريخ والوقت | موضوع الفحص والتحقق | النتيجة والحكم الهندسي | الدليل والملف المصدري |
| :---: | :---: | :--- | :---: | :--- |
| **Gate 1** | 2026-10-25<br>13:01:01 | **فحص النزاهة الجنائية للبيانات**<br>كشف التلوث المتقاطع (58 صورة منسوخة في Healthy) وإزالتها قبل بناء التقسيم | **PASSED ✅**<br>تسرب بيني 0.00%<br>البيانات: 4,053 عينة | `scripts/clean_healthy_duplicates.py`<br>`scripts/forensic_report_bracol.json` |
| **Gate 2** | 2026-10-25<br>13:53:27 | **اختبار الدخان المصغر (Smoke Test)**<br>التحقق من عمل مسار التدريب والمخرجات والرسوم 300 DPI دون أخطاء | **PASSED ✅**<br>اجتياز 100% لجميع المسارات البرمجية | `outputs/final_evaluation_report.json` (السموك الأول) |
| **Gate 3** | 2026-10-25<br>14:42:38 | **حارس المرحلة الثانية وبصمة المعاملات**<br>فحص تجميد طبقات BN والتأكد الصارم من بصمة المعاملات: 1,054,661 | **PASSED ✅**<br>بصمة 1,054,661 معاملاً بالضبط<br>تجميد كامل لـ 52 طبقة BN | `outputs/training_log_final.txt`<br>(السطور 3510 - 3513) |
| **Gate 4** | 2026-10-25<br>15:04:23 | **التدريب الفعلي والتقييم المستقل**<br>إتمام التدريب على 4,053 صورة وتحقيق دقة تتجاوز المعايير مع EarlyStopping | **PASSED ✅**<br>دقة الاختبار: 84.80%<br>استدعاء > 80% لكل الفئات | `outputs/final_evaluation_report.json`<br>`outputs/best_coffee_model_final.h5` |
| **Gate 5** | 2026-10-25<br>17:57:08 | **الإغلاق الجنائي النهائي (Provenance Check)**<br>إعادة تحميل النموذج من القرص ومطابقة الأداء والـ bitwise | **CLOSED ✅**<br>تطابق bitwise تام مع التقرير المصدري | `outputs/provenance_check_result.txt`<br>`scripts/final_provenance_check.py` |

---

### 3. جدول بصمات التشفير الرقمية للملفات الكانونية الأربعة (SHA-256 Checksums)
تم حساب التوقيع الرقمي الجنائي (SHA256) عبر أمر PowerShell: `Get-FileHash -Algorithm SHA256` لتجميد الأرتيفكتات الكانونية:

| الملف الكانوني | الحجم (Bytes) | بصمة التشفير الرقمية (SHA-256) |
| :--- | :---: | :--- |
| `outputs/best_coffee_model_final.h5` | 19,281,560 | `C19565A45424EC23EDC7CDA45BEE9961DCA930BA5CFDB22DCA5BDEC05144E196` |
| `outputs/best_coffee_model_phase1.h5` | 13,468,848 | `A258A971758F3059166E1A6AF4FBC47E52900C6CB2FA69FADDB526EDE6DA1067` |
| `outputs/final_evaluation_report.json` | 1,757 | `8AEADE4479CB23EF83774C7C9871FDDAD109638A48B68187D9531F8DBC8BFE07` |
| `outputs/training_log_final.txt` | 961,355 | `D26B5D0F9B03E32822CA62C80331C77B2A6B6E07427A71674ABC8E83FE4F7D66` |

---

### 4. بيئة التشغيل الفعلية المعتمدة (Runtime Environment)
- **نظام التشغيل**: Windows 11 Enterprise (Kernel Build: `10.0.22621`).
- **المعالج**: Intel Core i7 / Xeon (HP ZBook Workstation) - تدريب CPU نقي ومستقر.
- **إصدار لغة Python**: `3.8.8` (64-bit).
- **إصدار TensorFlow**: `2.3.0`.
- **إصدار Keras**: `2.4.0` (دلالات Keras 2 المستقرة للتعلم النقلي).
- **آلية الحتمية وقابلية التكرار**: تثبيت رباعي للبذور العشوائية عبر جميع المستويات (`PYTHONHASHSEED=42`, `random.seed(42)`, `np.random.seed(42)`, `tf.random.set_seed(42)`)، مع تغليف استدعاء `enable_op_determinism` بكتلة `try/except` لضمان التوافق مع TF 2.3.0 والبيئات اللاحقة.

---

### 5. النتائج النهائية وجدول التصنيف الكامل (Final Test Evaluation)
- **مجموعة الاختبار المستقلة**: 612 صورة لم يرها النموذج قط مقسمة طبقياً.
- **دقة الاختبار الإجمالية (Test Accuracy)**: **`84.80%`** (519 عينة صحيحة من أصل 612).
- **خسارة الاختبار (Test Loss)**: `0.3680`.
- **المتوسط المرجح (Weighted F1-Score)**: **`85.09%`**.
- **المتوسط الكلي (Macro F1-Score)**: **`82.92%`**.

#### 📋 جدول تقرير التصنيف الكامل لكل فئة (Classification Report):
| الفئة (Class) | الدقة (Precision) | الاستدعاء (Recall) | F1-Score | عدد العينات (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Cerscospora** | 54.90% | **84.85%** | 66.67% | 33 |
| **Healthy** | 82.27% | **88.83%** | 85.42% | 188 |
| **Leaf rust** | 89.66% | **80.53%** | 84.85% | 226 |
| **Miner** | 89.87% | **82.56%** | 86.06% | 86 |
| **Phoma** | 93.42% | **89.87%** | **91.61%** | 79 |
| **المعدل الكلي (Accuracy)** | - | - | **84.80%** | **612** |
| **Macro Average** | **82.02%** | **85.33%** | **82.92%** | 612 |
| **Weighted Average** | **86.03%** | **84.80%** | **85.09%** | 612 |

---

### 6. صياغات آمنة جاهزة للرسالة (Verbatim Scientific Statements)
يمكن إدراج النصوص المعتمدة التالية حرفياً في فصول الرسالة الجامعية:

> *"خضع المجموع الأصلي (4,111 صورة) لفحص جنائي كشف 58 مكرراً عابراً للتقسيمات في فئة Healthy وأُزيلت قبل بناء التقسيم الطبقي، ليُدرَّب النموذج النهائي على 4,053 صورة بمعدل تسرب بيني 0.00%."*

> *"تم تجميد طبقات BatchNormalization في الكتلة الأخيرة المفكوكة وفق توصية Keras الرسمية لحماية إحصاءات ImageNet."*

> *"فجوة 2.02 نقطة بين دقة التحقق (86.80%) ودقة الاختبار (84.80%) طبيعية، لأن مجموعة التحقق استُخدمت لاختيار النموذج بينما مجموعة الاختبار عمياء تماماً."*

> *"استُخدمت بذور مثبتة عبر جميع المستويات (PYTHONHASHSEED، random، NumPy، TensorFlow) لضمان قابلية إعادة الإنتاج على بيئة CPU الموحدة؛ وتجارب GPU قابلة لتفاوت طفيف بين التشغيلات."*
