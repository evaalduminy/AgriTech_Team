# ==============================================================================
# 🌿 AgriTech Coffee Leaf Classifier - Phase 3 Docker Environment
# ==============================================================================
# بيئة التشغيل المعيارية المعتمدة: Python 3.10 + TensorFlow 2.15.0 الحقيقي
# ==============================================================================

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# تثبيت متطلبات النظام الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملفات الاعتماديات وتثبيتها
COPY requirements_v3.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# نسخ كود المشروع والاختبارات
COPY . /app

# فحص كود المشروع وتشغيل حزمة الاختبارات الشاملة (30+ اختباراً)
CMD ["python", "-m", "pytest", "tests/test_hardening_layer_v3.py", "-v"]
